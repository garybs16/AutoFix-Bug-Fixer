# AutoFix — Python Static & Runtime Analysis Tool

AutoFix is a **command-line tool** that accelerates debugging and test failure triage by combining **AST-based static analysis** and **runtime traceback parsing**. It scans your Python project, indexes all functions/classes for O(1) lookups, parses Pytest failures, and applies **pattern-matching logic** to automatically **suggest likely fixes**.

---

## 🚀 Features

- **AST-based Symbol Indexing**  
  Parses all `.py` files to build an in-memory index of functions, classes, methods, and attributes for O(1) lookups.

- **Runtime Traceback Parsing**  
  Extracts structured frame data and exception summaries from raw `pytest` output or plain Python tracebacks.

- **AI-like Suggestion Engine**  
  Applies heuristic and pattern-based rules to detect common issues and suggest targeted fixes.

- **Automatic Pytest Integration**  
  Runs `pytest`, captures failures, parses tracebacks, and ranks possible solutions—all in one command.

- **Extensible Rule Engine**  
  Add your own fix patterns for custom project needs. Rules are modular and isolated for easy extension.

- **Human & Machine Readable Reports**  
  Outputs clean human-readable text reports or structured JSON for integration with CI/CD dashboards.

---

## 📦 Installation

You can clone and use directly:

```bash
git clone https://github.com/yourusername/autofix.git
cd autofix
python3 autofix.py --help
```

No dependencies beyond Python’s standard library (requires Python 3.8+). You only need **Pytest** installed and accessible on your PATH.

---

## ⚙️ Usage

### 1. Analyze Codebase (Static Index)

Build an AST index for your project:

```bash
python autofix.py analyze src/ tests/
```

This scans your code and prints how many functions, classes, and modules were indexed.

### 2. Run Pytest + Auto Analysis

Run tests, parse traceback output, and get fix suggestions in one go:

```bash
python autofix.py run --pytest-args "-q -x" src/ tests/
```

Example output:
```
================================================================================
Failure #1: NameError: name 'DataFram' is not defined
Frames:
  • src/utils/data_loader.py:22 in load_dataset
Suggestions (ranked):
  - Add missing import for 'DataFrame'  [score=0.90]
      fix: from pandas import DataFrame
```

You can also export structured results for machine consumption:

```bash
python autofix.py run --json results.json src/ tests/
```

### 3. Analyze Saved Traceback Files

If you have saved a traceback or pytest output to a file (e.g., `tb.txt`):

```bash
python autofix.py suggest --traceback tb.txt src/
```

Export JSON output:
```bash
python autofix.py suggest --traceback tb.txt src/ --json suggestions.json
```

---

## 🧠 How It Works

### 🔍 1. Static Indexing via AST
AutoFix walks your source tree and parses `.py` files using the `ast` module. It collects:
- Function and method signatures (args, defaults, keywords)
- Class names, attributes, and methods
- Module-level imports

These are stored in dictionaries for O(1) lookups, enabling near-instant fix suggestions.

### ⚙️ 2. Runtime Traceback Parsing
AutoFix reads your pytest output or saved traceback text and extracts:
- Frame info: file, line number, function name
- Exception type & message

### 🤖 3. Rule-Based Suggestion Engine
Each rule is a small heuristic that matches specific error patterns. For example:

| Rule Name | Target Error | Suggestion |
|------------|---------------|-------------|
| `NameErrorMissingImport` | `NameError` | Suggests imports for known symbols |
| `AttributeErrorMaybeTypo` | `AttributeError` | Detects likely typos (via Levenshtein distance) |
| `TypeErrorWrongArity` | `TypeError` | Points to mismatched function signatures |
| `ImportErrorModule` | `ImportError` | Proposes pip install or relative import fix |
| `KeyErrorDict` | `KeyError` | Recommends `.get()` or membership checks |
| `IndexErrorBound` | `IndexError` | Suggests bound checking |
| `AssertionErrorEquality` | `AssertionError` | Reminds to use `pytest.approx` for floats |

---

## 🧩 Example Workflow

1. Run your tests:
   ```bash
   pytest -q --tb=short > tb.txt
   ```

2. Feed traceback into AutoFix:
   ```bash
   python autofix.py suggest -t tb.txt src/
   ```

3. Get clear, actionable suggestions in seconds.

---

## 🧱 Architecture Overview

```
+------------------------+
|   AutoFix CLI (argparse)   |
+------------------------+
           |
           v
+-----------------------+
|  SymbolIndex (AST)    |
|  → functions/classes  |
+-----------------------+
           |
           v
+-----------------------+
|  TracebackParser      |
|  → extract frames & errors |
+-----------------------+
           |
           v
+-----------------------+
|  RuleEngine           |
|  → apply fix rules    |
+-----------------------+
           |
           v
+-----------------------+
|  Reporter (text/json) |
+-----------------------+
```

---

## 🧰 Extending AutoFix

You can add new rules easily:

```python
def rule_custom(self, tb: ParsedTraceback):
    if 'ConnectionError' in tb.etype:
        return [Suggestion(title='Network Issue', detail='Check API connectivity.')]
```
Then register it inside `RuleEngine.__init__()`.

---

## 📊 Example JSON Output

```json
{
  "failures": [
    {
      "case": 1,
      "etype": "NameError",
      "message": "name 'DatFram' is not defined",
      "suggestions": [
        {
          "title": "Add missing import for 'DataFrame'",
          "detail": "`DataFrame` exists elsewhere. Consider importing it.",
          "fixes": ["from pandas import DataFrame"],
          "score": 0.9
        }
      ]
    }
  ]
}
```

---

## 🧩 Integration Ideas
- **CI/CD pipelines:** Automatically parse test failures and upload suggestions to build logs.
- **IDE plugins:** Hook into linting or test runners to suggest quick-fixes in real time.
- **LLM copilots:** Feed `--json` output into an LLM to generate automated PR patches.

---

## 🧑‍💻 Requirements
- Python **3.8+**
- `pytest` installed on PATH

---

---

## 🤝 Contributions
Pull requests are welcome — feel free to extend the rule set or improve the parsing accuracy.  
If you build something cool on top of AutoFix (e.g., a VSCode integration), let’s connect!

---


