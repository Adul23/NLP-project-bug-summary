"""Prepare unique, parseable Python functions without reformatting their bodies."""

from __future__ import annotations

import argparse
import ast
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data/raw/codesearchnet/python_functions.jsonl"
CLEAN_PATH = PROJECT_ROOT / "data/processed/clean_functions.csv"
CLEAN_COLUMNS = ["source_id", "code", "label", "mutation_type"]


def parse_python(code: str) -> ast.Module:
    """Parse source without executing it; suppress dataset escape-sequence warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", (SyntaxWarning, DeprecationWarning))
        return ast.parse(code)


def is_function(tree: ast.Module) -> bool:
    """Require one standalone (possibly decorated/async) function, not a fragment."""
    return len(tree.body) == 1 and isinstance(
        tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)
    )


def _remove_docstrings(code: str, tree: ast.Module) -> str:
    # AST columns are UTF-8 byte offsets, not character offsets.
    raw = code.encode("utf-8")
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        leading_strings = []
        for statement in node.body:
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                break
            leading_strings.append(statement)
        # Removing only the first string could expose a second docstring.
        for statement in leading_strings:
            start = offsets[statement.lineno - 1] + statement.col_offset
            end = offsets[statement.end_lineno - 1] + statement.end_col_offset
            if (
                len(leading_strings) == len(node.body)
                and statement is leading_strings[-1]
            ):
                replacement = b"pass"
            else:
                replacement = b""
                # In `def f(): "doc"; return x`, remove the separator too.
                cursor = end
                while cursor < len(raw) and raw[cursor : cursor + 1] in (b" ", b"\t"):
                    cursor += 1
                if raw[cursor : cursor + 1] == b";":
                    end = cursor + 1
            edits.append((start, end, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        raw = raw[:start] + replacement + raw[end:]
    return raw.decode("utf-8").strip()


def _prepare(raw: object) -> tuple[str | None, str, str | None]:
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty", None
    code = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    try:
        tree = parse_python(code)
        if not is_function(tree):
            return None, "non_function", None
        code = _remove_docstrings(code, tree)
        parse_python(code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        return None, "invalid", str(exc)
    return code, "valid", None


def prepare_code(raw: object) -> str | None:
    """Normalize and remove docstrings; return None for invalid/nonfunction input."""
    return _prepare(raw)[0]


@dataclass
class PreparationStats:
    loaded_samples: int = 0
    valid_python: int = 0
    invalid_python: int = 0
    empty_samples: int = 0
    non_function_samples: int = 0
    duplicates_removed: int = 0
    final_clean_functions: int = 0
    parse_error_examples: list[str] = field(default_factory=list)


class CleanCollector:
    """Incrementally accept unique functions and assign stable, padded source IDs."""

    def __init__(self) -> None:
        self.stats = PreparationStats()
        self.rows: list[dict] = []
        self._seen: set[str] = set()

    def add(self, raw: object) -> bool:
        self.stats.loaded_samples += 1
        code, status, error = _prepare(raw)
        if status == "empty":
            self.stats.empty_samples += 1
            return False
        if status == "invalid":
            self.stats.invalid_python += 1
            if len(self.stats.parse_error_examples) < 5:
                self.stats.parse_error_examples.append(error or "Unknown parse error")
            return False
        self.stats.valid_python += 1
        if status == "non_function":
            self.stats.non_function_samples += 1
            return False
        if code in self._seen:
            self.stats.duplicates_removed += 1
            return False
        self._seen.add(code)
        self.rows.append(
            {
                "source_id": f"{len(self.rows) + 1:06d}",
                "code": code,
                "label": 0,
                "mutation_type": "clean",
            }
        )
        self.stats.final_clean_functions = len(self.rows)
        return True


def print_stats(stats: PreparationStats) -> None:
    for title, value in [
        ("Loaded samples", stats.loaded_samples),
        ("Valid Python", stats.valid_python),
        ("Invalid Python", stats.invalid_python),
        ("Empty samples", stats.empty_samples),
        ("Non-function samples", stats.non_function_samples),
        ("Duplicates removed", stats.duplicates_removed),
        ("Final clean functions", stats.final_clean_functions),
    ]:
        print(f"{title}: {value}")


def read_raw(path: Path):
    """Iterate the download JSONL, retaining useful errors for malformed files."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or "code" not in row:
                raise ValueError(f"{path}:{number}: expected a JSON object with code")
            yield row


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete UTF-8 CSV only after a successful temporary write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def preprocess(
    input_path: Path, output_path: Path, target: int = 10_000
) -> CleanCollector:
    """Prepare exactly target functions, or fail before publishing an incomplete CSV."""
    if target <= 0:
        raise ValueError("target must be positive")
    collector = CleanCollector()
    for row in read_raw(input_path):
        collector.add(row["code"])
        if len(collector.rows) == target:
            break
    print_stats(collector.stats)
    if len(collector.rows) != target:
        raise ValueError(
            f"Only {len(collector.rows)} valid unique functions; need {target}. "
            "Download more input, or explicitly lower --target."
        )
    frame = pd.DataFrame(collector.rows, columns=CLEAN_COLUMNS)
    write_csv(frame, output_path)
    output_path.with_suffix(".stats.json").write_text(
        json.dumps(asdict(collector.stats), indent=2) + "\n", encoding="utf-8"
    )
    return collector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_PATH)
    parser.add_argument("--output", type=Path, default=CLEAN_PATH)
    parser.add_argument("--target", type=int, default=10_000)
    args = parser.parse_args()
    preprocess(args.input, args.output, args.target)


if __name__ == "__main__":
    main()
