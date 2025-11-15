#!/usr/bin/env python3
"""
AutoFix — Python Static & Runtime Analysis Tool
Now with runfile mode (directly run Python files without pytest)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Any


# -------------------------------
# Traceback parsing
# -------------------------------

TRACEBACK_FILE_RE = re.compile(r'^\s*File\s+"(?P<file>.+?)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>.+)$')
EXC_LASTLINE_RE = re.compile(r'^(?P<etype>[A-Za-z_][A-Za-z0-9_\.]*):\s*(?P<msg>.*)$')


@dataclass
class Frame:
    file: str
    line: int
    func: str


@dataclass
class ParsedTraceback:
    frames: List[Frame]
    etype: str
    msg: str
    raw: str


class TracebackParser:
    """Extract frames + exception summary from Python traceback text."""

    @staticmethod
    def parse(text: str) -> List[ParsedTraceback]:
        lines = text.splitlines()
        blocks: List[ParsedTraceback] = []
        i = 0

        while i < len(lines):
            m = EXC_LASTLINE_RE.match(lines[i])
            if m:
                frames = []
                j = i - 1
                while j >= 0:
                    m2 = TRACEBACK_FILE_RE.match(lines[j])
                    if m2:
                        frames.append(
                            Frame(
                                file=os.path.normpath(m2.group("file")),
                                line=int(m2.group("line")),
                                func=m2.group("func").strip(),
                            )
                        )
                        j -= 1
                        continue
                    j -= 1
                frames.reverse()
                blocks.append(
                    ParsedTraceback(
                        frames=frames,
                        etype=m.group("etype"),
                        msg=m.group("msg"),
                        raw="\n".join(lines[max(0, j): i + 1]),
                    )
                )
            i += 1

        return blocks


# -------------------------------
# AST-based static index
# -------------------------------

@dataclass
class FunctionSig:
    name: str
    file: str
    line: int
    args: List[str]
    defaults: int
    vararg: Optional[str]
    kwonly: List[str]
    kw_defaults: int
    kwvararg: Optional[str]


@dataclass
class ClassInfo:
    name: str
    file: str
    line: int
    methods: Dict[str, FunctionSig] = field(default_factory=dict)
    attributes: set = field(default_factory=set)


class SymbolIndex:
    def __init__(self) -> None:
        self.functions: Dict[str, List[FunctionSig]] = {}
        self.classes: Dict[str, List[ClassInfo]] = {}

    def index_paths(self, paths: Iterable[str]) -> None:
        for root in paths:
            p = Path(root)
            if p.is_file() and p.suffix == ".py":
                self._index_file(p)
            else:
                for f in p.rglob("*.py"):
                    self._index_file(f)

    def _index_file(self, file: Path) -> None:
        try:
            text = file.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(file))
        except Exception:
            return

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sig = self._fn_sig(node, str(file))
                self.functions.setdefault(node.name, []).append(sig)
            elif isinstance(node, ast.ClassDef):
                ci = ClassInfo(name=node.name, file=str(file), line=node.lineno)
                for b in node.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ci.methods[b.name] = self._fn_sig(b, str(file))
                    elif isinstance(b, ast.Assign):
                        for t in b.targets:
                            if isinstance(t, ast.Name):
                                ci.attributes.add(t.id)
                self.classes.setdefault(node.name, []).append(ci)

    @staticmethod
    def _fn_sig(node, file):
        a = node.args
        pos = [arg.arg for arg in a.args]
        vararg = a.vararg.arg if a.vararg else None
        kwonly = [arg.arg for arg in a.kwonlyargs]
        kwvararg = a.kwarg.arg if a.kwarg else None
        defaults = len(a.defaults)
        kw_defaults = sum(1 for d in a.kw_defaults if d is not None)
        return FunctionSig(
            name=node.name,
            file=file,
            line=node.lineno,
            args=pos,
            defaults=defaults,
            vararg=vararg,
            kwonly=kwonly,
            kw_defaults=kw_defaults,
            kwvararg=kwvararg,
        )


# -------------------------------
# Rule Engine
# -------------------------------

@dataclass
class Suggestion:
    title: str
    detail: str
    score: float = 0.6
    fixes: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)


class RuleEngine:
    def __init__(self, index: SymbolIndex):
        self.index = index

    def suggest(self, tb: ParsedTraceback) -> List[Suggestion]:
        out: List[Suggestion] = []

        # NameError missing import
        if tb.etype == "NameError":
            m = re.search(r"name '([^']+)' is not defined", tb.msg)
            if m:
                name = m.group(1)
                hits = []
                if name in self.index.functions:
                    for sig in self.index.functions[name]:
                        hits.append(f"(Maybe import?) defined in: {sig.file}:{sig.line}")
                if hits:
                    out.append(
                        Suggestion(
                            title=f"NameError: '{name}' not defined",
                            detail="\n".join(hits),
                            score=0.9,
                        )
                    )

        # AttributeError typo
        if tb.etype == "AttributeError":
            m = re.search(r"'[^']+' object has no attribute '([^']+)'", tb.msg)
            if m:
                attr = m.group(1)
                candidates = []
                for cls_list in self.index.classes.values():
                    for ci in cls_list:
                        for a in list(ci.attributes) + list(ci.methods.keys()):
                            if attr.lower() in a.lower() or a.lower() in attr.lower():
                                candidates.append(f"{ci.name}.{a} (in {ci.file}:{ci.line})")
                if candidates:
                    out.append(
                        Suggestion(
                            title=f"Attribute '{attr}' may be a typo",
                            detail="\n".join(candidates),
                            score=0.8,
                        )
                    )

        # TypeError mismatched args
        if tb.etype == "TypeError":
            # pattern: fn() takes 2 positional arguments but 3 were given
            m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\(\) takes (\d+) positional arguments but (\d+)", tb.msg)
            if m:
                fn = m.group(1)
                if fn in self.index.functions:
                    det = []
                    for s in self.index.functions[fn]:
                        det.append(f"{fn} defined at {s.file}:{s.line} with args {s.args}")
                    out.append(
                        Suggestion(
                            title="Function argument mismatch",
                            detail="\n".join(det),
                            score=0.85,
                        )
                    )

        return sorted(out, key=lambda s: s.score, reverse=True)


# -------------------------------
# Run Python file directly (NEW)
# -------------------------------

def run_python_file(path: str) -> str:
    cmd = [sys.executable, path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stderr


# -------------------------------
# Reporting
# -------------------------------

def make_report(blocks: List[ParsedTraceback], engine: RuleEngine) -> str:
    buf = []

    for i, tb in enumerate(blocks, 1):
        buf.append("\n" + "=" * 80)
        buf.append(f"Failure #{i}: {tb.etype}: {tb.msg}")

        if tb.frames:
            buf.append("Frames:")
            for fr in tb.frames:
                buf.append(f"  • {fr.file}:{fr.line} in {fr.func}")
        else:
            buf.append("(no frames parsed)")

        suggestions = engine.suggest(tb)
        buf.append("Suggestions (ranked):")
        if suggestions:
            for s in suggestions[:5]:
                buf.append(f"  - {s.title} [score={s.score:.2f}]")
                if s.detail:
                    buf.append(textwrap.indent(s.detail, "      "))
        else:
            buf.append("  (no suggestions)")

    return "\n".join(buf)


# -------------------------------
# CLI
# -------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="AutoFix")
    sub = p.add_subparsers(dest="cmd", required=True)

    # analyze
    pa = sub.add_parser("analyze")
    pa.add_argument("paths", nargs="+")

    # run via pytest (original)
    pr = sub.add_parser("run")
    pr.add_argument("paths", nargs="*")

    # NEW: runfile mode
    pf = sub.add_parser("runfile")
    pf.add_argument("script", help="Python script to run")
    pf.add_argument("paths", nargs="*", default=["."])

    args = p.parse_args(argv)

    # analyze
    if args.cmd == "analyze":
        idx = SymbolIndex()
        idx.index_paths(args.paths)
        print(f"Indexed: {len(idx.functions)} functions, {len(idx.classes)} classes")
        return 0

    # runfile (NEW)
    if args.cmd == "runfile":
        idx = SymbolIndex()
        idx.index_paths(args.paths)
        stderr = run_python_file(args.script)
        blocks = TracebackParser.parse(stderr)
        engine = RuleEngine(idx)
        print(make_report(blocks, engine))
        return 0

    # original pytest run
    if args.cmd == "run":
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"] + args.paths,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout

        idx = SymbolIndex()
        idx.index_paths(args.paths)
        blocks = TracebackParser.parse(out)
        engine = RuleEngine(idx)
        print(make_report(blocks, engine))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
