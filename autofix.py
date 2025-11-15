#!/usr/bin/env python3
"""
AutoFix — Improved Python Static & Runtime Error Analyzer
---------------------------------------------------------
Features:
 • Run Python scripts directly (runfile)
 • Extract real traceback data
 • Show exact error line + code snippet
 • AST-based index for smarter suggestions
 • Improved readability and error formatting
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
# Traceback Parsing
# -------------------------------

TRACEBACK_FILE_RE = re.compile(
    r'^\s*File\s+"(?P<file>.+?)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>.+)$'
)
EXC_LASTLINE_RE = re.compile(
    r'^(?P<etype>[A-Za-z_][A-Za-z0-9_\.]*):\s*(?P<msg>.*)$'
)


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
    @staticmethod
    def parse(text: str) -> List[ParsedTraceback]:
        lines = text.splitlines()
        blocks = []
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
# AST Indexing
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
    def __init__(self):
        self.functions = {}
        self.classes = {}

    def index_paths(self, paths: Iterable[str]):
        for root in paths:
            p = Path(root)
            if p.is_file() and p.suffix == ".py":
                self._index_file(p)
            else:
                for f in p.rglob("*.py"):
                    self._index_file(f)

    def _index_file(self, file: Path):
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


class RuleEngine:
    def __init__(self, index: SymbolIndex):
        self.index = index

    def suggest(self, tb: ParsedTraceback) -> List[Suggestion]:
        out = []

        # NameError
        if tb.etype == "NameError":
            m = re.search(r"name '([^']+)' is not defined", tb.msg)
            if m:
                name = m.group(1)
                locs = []
                if name in self.index.functions:
                    for sig in self.index.functions[name]:
                        locs.append(f"{sig.file}:{sig.line}")
                if locs:
                    out.append(
                        Suggestion(
                            title=f"NameError: '{name}' not defined",
                            detail="\n".join(locs),
                            score=0.9,
                        )
                    )

        # AttributeError
        if tb.etype == "AttributeError":
            m = re.search(r"object has no attribute '([^']+)'", tb.msg)
            if m:
                attr = m.group(1)
                candidates = []
                for cls_list in self.index.classes.values():
                    for ci in cls_list:
                        for a in ci.attributes.union(ci.methods.keys()):
                            if attr.lower() in a.lower() or a.lower() in attr.lower():
                                candidates.append(f"{ci.name}.{a}  ({ci.file}:{ci.line})")

                if candidates:
                    out.append(
                        Suggestion(
                            title=f"Possible typo: '{attr}'",
                            detail="\n".join(candidates),
                            score=0.85,
                        )
                    )

        # TypeError – wrong argument count
        if tb.etype == "TypeError":
            m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\(\) takes", tb.msg)
            if m:
                fn = m.group(1)
                if fn in self.index.functions:
                    details = []
                    for s in self.index.functions[fn]:
                        details.append(
                            f"{fn} defined at {s.file}:{s.line} with args {s.args}"
                        )
                    out.append(
                        Suggestion(
                            title="Function argument mismatch",
                            detail="\n".join(details),
                            score=0.8,
                        )
                    )

        return sorted(out, key=lambda x: x.score, reverse=True)


# -------------------------------
# Execute script
# -------------------------------

def run_python_file(path: str) -> str:
    cmd = [sys.executable, path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.stderr


# -------------------------------
# Reporting (beautiful output)
# -------------------------------

def make_report(blocks: List[ParsedTraceback], engine: RuleEngine) -> str:
    buf = []

    def read_snippet(file, line, context=2):
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            start = max(0, line - context - 1)
            end = min(len(lines), line + context)

            snippet = []
            for i in range(start, end):
                prefix = ">>" if (i + 1) == line else "  "
                snippet.append(f"{prefix} {i+1:4}: {lines[i].rstrip()}")

            return "\n".join(snippet)

        except Exception:
            return "(source unavailable)"

    for i, tb in enumerate(blocks, 1):
        buf.append("\n" + "=" * 80)
        buf.append(f"Failure #{i}: {tb.etype}: {tb.msg}")

        if tb.frames:
            buf.append("\nTraceback (most recent call last):")
            for fr in tb.frames:
                buf.append(f"  • File \"{fr.file}\", line {fr.line}, in {fr.func}")
                buf.append(textwrap.indent(read_snippet(fr.file, fr.line), "        "))
        else:
            buf.append("(no frames parsed)")

        suggestions = engine.suggest(tb)
        buf.append("\nSuggestions (ranked):")
        if suggestions:
            for s in suggestions:
                buf.append(f"  - {s.title} (score={s.score:.2f})")
                buf.append(textwrap.indent(s.detail, "        "))
        else:
            buf.append("  (no suggestions)")

    return "\n".join(buf)


# -------------------------------
# CLI
# -------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="AutoFix Error Analyzer")
    sub = p.add_subparsers(dest="cmd", required=True)

    # analyze
    pa = sub.add_parser("analyze")
    pa.add_argument("paths", nargs="+")

    # run through pytest (optional)
    pr = sub.add_parser("run")
    pr.add_argument("paths", nargs="*")

    # run a single file
    pf = sub.add_parser("runfile")
    pf.add_argument("script")
    pf.add_argument("paths", nargs="*", default=["."])

    args = p.parse_args(argv)

    # analyze
    if args.cmd == "analyze":
        idx = SymbolIndex()
        idx.index_paths(args.paths)
        print(f"Indexed: {len(idx.functions)} functions, {len(idx.classes)} classes")
        return 0

    # runfile
    if args.cmd == "runfile":
        idx = SymbolIndex()
        idx.index_paths(args.paths)
        stderr = run_python_file(args.script)
        blocks = TracebackParser.parse(stderr)
        engine = RuleEngine(idx)
        print(make_report(blocks, engine))
        return 0

    # pytest mode (optional)
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
