#!/usr/bin/env python3
"""
AutoFix — Python Static & Runtime Analysis Tool
------------------------------------------------
A single-file Python CLI that:
  • Parses your codebase with AST to build O(1) symbol lookups
  • Runs pytest (or ingests saved tracebacks) and parses failures
  • Applies pattern-matching rules to auto-suggest likely fixes

Usage examples:
  $ python autofix.py analyze src/ tests/
  $ python autofix.py run --pytest-args "-q -x" src/ tests/
  $ python autofix.py suggest --traceback tb.txt src/
  $ python autofix.py run --json out.json src/ tests/

Notes:
  • No external deps beyond the stdlib. Pytest must be available on PATH for `run`.
  • Designed as a starting point you can extend with more rules.
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
PYTEST_SUMMARY_RE = re.compile(r"=+\s*(?P<nfailed>\d+) failed.*=+")

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
    """Extract frames + exception summary from Python/pytest output."""

    @staticmethod
    def parse(text: str) -> List[ParsedTraceback]:
        # Split into blocks by blank lines that end with Exception line
        # A robust approach: find every exception tail and backtrack frames
        lines = text.splitlines()
        blocks: List[ParsedTraceback] = []

        i = 0
        while i < len(lines):
            line = lines[i]
            m = EXC_LASTLINE_RE.match(line)
            if m:
                # Backtrack to collect frames above until a blank line or start
                frames: List[Frame] = []
                j = i - 1
                while j >= 0:
                    m2 = TRACEBACK_FILE_RE.match(lines[j])
                    if m2:
                        frames.append(Frame(
                            file=os.path.normpath(m2.group('file')),
                            line=int(m2.group('line')),
                            func=m2.group('func').strip(),
                        ))
                        j -= 1
                        continue
                    # Stop when we hit a non-frame after seeing at least one frame
                    if frames and lines[j].strip().startswith('Traceback'):
                        break
                    j -= 1
                frames.reverse()
                blocks.append(ParsedTraceback(
                    frames=frames,
                    etype=m.group('etype'),
                    msg=m.group('msg'),
                    raw='\n'.join(lines[max(0, j):i+1])
                ))
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
    defaults: int  # number of defaulted params at end
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
    """Build O(1) lookups of functions/classes/attributes from a set of paths."""

    def __init__(self) -> None:
        self.functions: Dict[str, List[FunctionSig]] = {}
        self.classes: Dict[str, List[ClassInfo]] = {}
        self.modules: Dict[str, str] = {}  # module name -> file

    def index_paths(self, paths: Iterable[str]) -> None:
        for root in paths:
            root_path = Path(root)
            if root_path.is_file() and root_path.suffix == '.py':
                self._index_file(root_path)
            else:
                for py in root_path.rglob('*.py'):
                    self._index_file(py)

    def _index_file(self, file: Path) -> None:
        try:
            text = file.read_text(encoding='utf-8')
        except Exception:
            return
        try:
            tree = ast.parse(text, filename=str(file))
        except SyntaxError:
            return

        mod_name = self._module_name_from_path(file)
        self.modules[mod_name] = str(file)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                sig = self._fn_sig(node, str(file))
                self.functions.setdefault(node.name, []).append(sig)
            elif isinstance(node, ast.AsyncFunctionDef):
                sig = self._fn_sig(node, str(file))
                self.functions.setdefault(node.name, []).append(sig)
            elif isinstance(node, ast.ClassDef):
                ci = ClassInfo(name=node.name, file=str(file), line=node.lineno)
                for body in node.body:
                    if isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ci.methods[body.name] = self._fn_sig(body, str(file))
                    elif isinstance(body, ast.Assign):
                        for tgt in body.targets:
                            if isinstance(tgt, ast.Name):
                                ci.attributes.add(tgt.id)
                self.classes.setdefault(node.name, []).append(ci)

    @staticmethod
    def _module_name_from_path(file: Path) -> str:
        parts = []
        for p in file.with_suffix('').parts:
            parts.append(p)
        return '.'.join(parts)

    @staticmethod
    def _fn_sig(node: ast.AST, file: str) -> FunctionSig:
        assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
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
# Rule engine for suggestions
# -------------------------------

@dataclass
class Suggestion:
    title: str
    detail: str
    score: float = 0.6
    fixes: List[str] = field(default_factory=list)  # textual fix ideas / diffs TODO
    evidence: Dict[str, Any] = field(default_factory=dict)

Rule = Tuple[str, Any]  # name, callable(index, tb) -> List[Suggestion]

class RuleEngine:
    def __init__(self, index: SymbolIndex):
        self.index = index
        self.rules: List[Rule] = [
            ("NameErrorMissingImport", self.rule_nameerror_missing_import),
            ("AttributeErrorMaybeTypo", self.rule_attributeerror_typo),
            ("TypeErrorWrongArity", self.rule_typeerror_wrong_arity),
            ("ImportErrorModule", self.rule_importerror_module),
            ("KeyErrorDict", self.rule_keyerror_dict),
            ("IndexErrorBound", self.rule_indexerror_bound),
            ("AssertionErrorEquality", self.rule_assertion_hint),
        ]

    # ---- Individual rules ----

    def rule_nameerror_missing_import(self, tb: ParsedTraceback) -> List[Suggestion]:
        if tb.etype.endswith('NameError'):
            # Extract missing name
            m = re.search(r"name '([^']+)' is not defined", tb.msg)
            if m:
                name = m.group(1)
                # If a function/class exists elsewhere, propose import
                hits = []
                if name in self.index.functions:
                    for sig in self.index.functions[name]:
                        hits.append(f"from {Path(sig.file).with_suffix('').name} import {name}")
                if name in self.index.classes:
                    for ci in self.index.classes[name]:
                        hits.append(f"from {Path(ci.file).with_suffix('').name} import {name}")
                if hits:
                    return [Suggestion(
                        title=f"Add missing import for '{name}'",
                        detail=f"`{name}` exists elsewhere. Consider importing it.",
                        score=0.9,
                        fixes=hits,
                        evidence={"name": name},
                    )]
        return []

    def rule_attributeerror_typo(self, tb: ParsedTraceback) -> List[Suggestion]:
        if tb.etype.endswith('AttributeError'):
            m = re.search(r"object has no attribute '([^']+)'", tb.msg)
            if not m:
                m = re.search(r"has no attribute '([^']+)'", tb.msg)
            if m:
                attr = m.group(1)
                # Heuristic: look for similarly named attributes in classes indexed
                candidates = []
                for cls_list in self.index.classes.values():
                    for ci in cls_list:
                        for a in list(ci.attributes) + list(ci.methods.keys()):
                            if _levenshtein(attr, a) == 1 or (attr.lower() == a.lower() and attr != a):
                                candidates.append((ci.name, a, ci.file, ci.line))
                if candidates:
                    lines = [f"Did you mean `{cls}.{a}`? ({file}:{line})" for cls,a,file,line in candidates[:5]]
                    return [Suggestion(
                        title=f"Attribute '{attr}' not found; possible typo",
                        detail="\n".join(lines),
                        score=0.75,
                        evidence={"attr": attr, "candidates": candidates[:5]},
                    )]
        return []

    def rule_typeerror_wrong_arity(self, tb: ParsedTraceback) -> List[Suggestion]:
        if tb.etype.endswith('TypeError'):
            # Match typical message: myfn() takes 2 positional arguments but 3 were given
            m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\(\) takes (\d+) positional arguments but (\d+) were given", tb.msg)
            if m:
                fn, expected, given = m.group(1), int(m.group(2)), int(m.group(3))
                if fn in self.index.functions:
                    sigs = self.index.functions[fn]
                    hints = []
                    for s in sigs:
                        hints.append(f"`{fn}` at {s.file}:{s.line} — defined as {s.args} (+{s.kwonly} kwonly); defaults={s.defaults}")
                    return [Suggestion(
                        title="Mismatched function arity",
                        detail="\n".join(hints),
                        score=0.8,
                        evidence={"fn": fn, "expected": expected, "given": given},
                    )]
        return []

    def rule_importerror_module(self, tb: ParsedTraceback) -> List[Suggestion]:
        if tb.etype.endswith('ImportError') or tb.etype.endswith('ModuleNotFoundError'):
            m = re.search(r"No module named '([^']+)'", tb.msg)
            if m:
                mod = m.group(1)
                # Suggest pip install or relative import lift
                hints = [
                    f"If `{mod}` is third-party: run `python -m pip install {mod}`.",
                    f"If it's local: check PYTHONPATH or convert to relative import (e.g. `from . import {mod}` in a package).",
                ]
                if mod in self.index.modules:
                    hints.append(f"Found similarly named module path: {self.index.modules[mod]}")
                return [Suggestion(
                    title=f"Module '{mod}' not found",
                    detail="\n".join(hints),
                    score=0.85,
                    evidence={"module": mod},
                )]
        return []

    def rule_keyerror_dict(self, tb: ParsedTraceback) -> List[Suggestion]:
        if tb.etype.endswith('KeyError'):
            m = re.search(r"KeyError: (.*)", tb.msg)
            if m:
                return [Suggestion(
                    title="Key not present in dict",
                    detail="Guard access with `.get()` or `in` check; verify fixture/test data.",
                    score=0.6,
                    evidence={"key": m.group(1)},
                )]
        return []

    def rule_indexerror_bound(self, tb: ParsedTraceback) -> List[Suggestion]:
        if tb.etype.endswith('IndexError'):
            return [Suggestion(
                title="Index out of range",
                detail="Validate list length before indexing; prefer slicing or safe iteration.",
                score=0.6,
            )]
        return []

    def rule_assertion_hint(self, tb: ParsedTraceback) -> List[Suggestion]:
        if tb.etype.endswith('AssertionError'):
            return [Suggestion(
                title="Assertion failed",
                detail="Use `-vv` to show values; check that expected/actual types align; consider `pytest.approx` for floats.",
                score=0.55,
            )]
        return []

    # ---- Run all rules ----

    def suggest(self, tb: ParsedTraceback) -> List[Suggestion]:
        out: List[Suggestion] = []
        for _name, fn in self.rules:
            try:
                out.extend(fn(tb))
            except Exception as e:
                # Rules must be resilient; keep going
                out.append(Suggestion(
                    title="Rule crashed",
                    detail=f"{fn.__name__} raised {e.__class__.__name__}: {e}",
                    score=0.0,
                ))
        # Sort by score desc
        out.sort(key=lambda s: s.score, reverse=True)
        return out

# -------------------------------
# Utility: simple Levenshtein (distance 1/2 quick check)
# -------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 3  # early out
    # DP with small strings only
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(
                prev[j] + 1,      # deletion
                cur[j-1] + 1,     # insertion
                prev[j-1] + cost  # substitution
            ))
        prev = cur
    return prev[-1]

# -------------------------------
# Pytest runner
# -------------------------------

def run_pytest(pytest_args: str) -> str:
    cmd = [sys.executable, '-m', 'pytest']
    if pytest_args:
        cmd += pytest_args.split()
    env = os.environ.copy()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.stdout

# -------------------------------
# Reporting
# -------------------------------

def make_report(blocks: List[ParsedTraceback], engine: RuleEngine, as_json: bool = False) -> str:
    report: List[Dict[str, Any]] = []
    for i, tb in enumerate(blocks, 1):
        suggestions = engine.suggest(tb)
        report.append({
            "case": i,
            "etype": tb.etype,
            "message": tb.msg,
            "frames": [frame.__dict__ for frame in tb.frames],
            "suggestions": [s.__dict__ for s in suggestions],
        })

    if as_json:
        return json.dumps({"failures": report}, indent=2)

    # Human-readable
    buf = []
    for item in report:
        buf.append("\n" + "="*80)
        buf.append(f"Failure #{item['case']}: {item['etype']}: {item['message']}")
        if item['frames']:
            buf.append("Frames:")
            for fr in item['frames']:
                buf.append(f"  • {fr['file']}:{fr['line']} in {fr['func']}")
        else:
            buf.append("(no frames parsed)")
        buf.append("Suggestions (ranked):")
        if item['suggestions']:
            for s in item['suggestions'][:5]:
                buf.append(f"  - {s['title']}  [score={s['score']:.2f}]")
                if s.get('detail'):
                    buf.append(textwrap.indent(s['detail'], prefix='      '))
                if s.get('fixes'):
                    for fix in s['fixes'][:3]:
                        buf.append(f"      fix: {fix}")
        else:
            buf.append("  (no suggestions)")
    return "\n".join(buf) + "\n"

# -------------------------------
# CLI
# -------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="AutoFix — PyTest failure analyzer with AST + rules")
    sub = p.add_subparsers(dest='cmd', required=True)

    pa = sub.add_parser('analyze', help='Build AST index for given paths')
    pa.add_argument('paths', nargs='+')

    pr = sub.add_parser('run', help='Run pytest and analyze failures')
    pr.add_argument('paths', nargs='*', help='Source/test paths to index for suggestions')
    pr.add_argument('--pytest-args', default='-q', help='Args passed to pytest (e.g. "-q -x -k smoke")')
    pr.add_argument('--json', dest='json_out', help='Write JSON report to file')

    ps = sub.add_parser('suggest', help='Analyze a saved traceback file')
    ps.add_argument('--traceback', '-t', required=True, help='Path to a text file with pytest output or a raw traceback')
    ps.add_argument('paths', nargs='*', help='Source paths to index for suggestions')
    ps.add_argument('--json', dest='json_out', help='Write JSON report to file')

    args = p.parse_args(argv)

    if args.cmd == 'analyze':
        idx = SymbolIndex()
        idx.index_paths(args.paths)
        print(f"Indexed: {len(idx.functions)} functions, {len(idx.classes)} classes, {len(idx.modules)} modules")
        return 0

    if args.cmd == 'run':
        idx = SymbolIndex()
        paths = args.paths or ['.']
        idx.index_paths(paths)
        out = run_pytest(args.pytest_args)
        blocks = TracebackParser.parse(out)
        engine = RuleEngine(idx)
        report_txt = make_report(blocks, engine, as_json=bool(args.json_out))
        if args.json_out:
            Path(args.json_out).write_text(report_txt, encoding='utf-8')
            print(f"Wrote JSON report to {args.json_out}")
        else:
            print(report_txt)
        # Print quick summary of failures based on pytest footer if present
        m = PYTEST_SUMMARY_RE.search(out)
        if m:
            print(f"Summary: {m.group('nfailed')} failed")
        return 0

    if args.cmd == 'suggest':
        text = Path(args.traceback).read_text(encoding='utf-8')
        idx = SymbolIndex()
        idx.index_paths(args.paths or ['.'])
        blocks = TracebackParser.parse(text)
        engine = RuleEngine(idx)
        report_txt = make_report(blocks, engine, as_json=bool(args.json_out))
        if args.json_out:
            Path(args.json_out).write_text(report_txt, encoding='utf-8')
            print(f"Wrote JSON report to {args.json_out}")
        else:
            print(report_txt)
        return 0

    return 1


if __name__ == '__main__':
    raise SystemExit(main())
