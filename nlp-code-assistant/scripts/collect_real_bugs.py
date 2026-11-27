"""Real single-operator bug fixes from CTSSB-1M / TSSB-3M, as buggy/fixed function pairs.

Evaluation data only (never used for training). Each TSSB record is one statement changed by a
commit; the code itself is not in the dataset, so both file versions are downloaded from
raw.githubusercontent.com (parent_sha = buggy, commit_sha = fixed), with a pause between
requests and a disk cache. A pair is kept when exactly one function differs and the only
difference is one binary/comparison/boolean operator, both operators from our mutation set
(== != < > <= >= + - * / and or). `in_tables` says whether fixed -> buggy is one of the
mutations the models were trained on (e.g. <= -> <), or only a change inside the set
(e.g. < -> <=, which the synthetic data never contains).

    # see which pattern names the file uses (no network)
    python scripts/collect_real_bugs.py --input ctssb_data_1M.zip --list-patterns
    # collect up to 500 pairs
    python scripts/collect_real_bugs.py --input ctssb_data_1M.zip --limit 500

Output CSV (sample_id, source_id, code, label, mutation_type, repo, in_tables): label 1 is the
version before the fix, label 0 after it; both rows of a pair share source_id. The CSV holds
third-party code and stays out of git (data/real_bugs/ is ignored).
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import random
import re
import sys
import time
import tokenize
import urllib.error
import urllib.parse
import urllib.request
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.mutations import ARITHMETIC, BOOLEANS, COMPARISONS  # noqa: E402

# operator class -> (symbol, mutation type, class it mutates into in training)
OPERATORS = {
    cls: (text, kind, mutated)
    for kind, table in (
        ("comparison_operator", COMPARISONS),
        ("arithmetic_operator", ARITHMETIC),
        ("boolean_operator", BOOLEANS),
    )
    for cls, (mutated, text, _) in table.items()
}
SYMBOLS = {text for text, _, _ in OPERATORS.values()}
_ALL_OPS = [
    c.__name__ for base in (ast.operator, ast.cmpop, ast.boolop) for c in base.__subclasses__()
]
_OP_CALL = re.compile(r"\b(?:" + "|".join(_ALL_OPS) + r")\(\)")
_GITHUB = re.compile(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$")
USER_AGENT = "real-bug-collector/0.1 (research evaluation)"
MAX_FILE_BYTES = 2_000_000


def iter_records(path: Path):
    """JSON records from a .zip of .jsonl(.gz) files, a folder of them, or one such file."""

    def lines(handle, name):
        if name.endswith(".gz"):
            handle = gzip.open(handle)
        yield from io.TextIOWrapper(handle, encoding="utf-8")

    if path.is_dir():
        sources = [(p.name, p.open("rb")) for p in sorted(path.rglob("*.jsonl*"))]
    elif path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        names = sorted(n for n in archive.namelist() if ".jsonl" in n and not n.endswith("/"))
        sources = ((n, archive.open(n)) for n in names)
    else:
        sources = [(path.name, path.open("rb"))]
    for name, handle in sources:
        with handle:
            for line in lines(handle, name):
                if line.strip():
                    yield json.loads(line)


def norm_pattern(value) -> str:
    return str(value).strip().upper().replace(" ", "_")


def _tokens(statement: str):
    skip = {tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT, tokenize.INDENT,
            tokenize.DEDENT, tokenize.ENDMARKER}
    try:
        tokens = tokenize.generate_tokens(io.StringIO(statement.strip()).readline)
        return [t.string for t in tokens if t.type not in skip]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None


def statement_prefilter(before: str, after: str) -> bool:
    """Cheap check before any download: one token differs and both are our operators.

    Statements that do not tokenize on their own (e.g. an unclosed bracket) pass through;
    the function-level AST check decides for them.
    """
    a, b = _tokens(before or ""), _tokens(after or "")
    if a is None or b is None:
        return True
    if len(a) != len(b):
        return False
    diff = [(x, y) for x, y in zip(a, b) if x != y]
    return len(diff) == 1 and diff[0][0] in SYMBOLS and diff[0][1] in SYMBOLS


def raw_url(project_url: str, sha: str, file_path: str) -> str | None:
    match = _GITHUB.match(project_url or "")
    if not match:
        return None
    owner, repo = match.groups()
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{sha}/{urllib.parse.quote(file_path)}"


def fetch_text(url: str, cache: Path, pause: float) -> str | None:
    """Download once; 404s are cached as missing so they are never retried."""
    path = cache / hashlib.sha1(url.encode()).hexdigest()
    missing = path.with_suffix(".missing")
    if path.exists():
        return path.read_text(encoding="utf-8")
    if missing.exists():
        return None
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(MAX_FILE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410, 451):
            missing.touch()
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None  # transient: not cached
    finally:
        time.sleep(pause)
    if len(data) > MAX_FILE_BYTES:
        missing.touch()
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        missing.touch()
        return None
    path.write_text(text, encoding="utf-8")
    return text


def _parse(source: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(source)


def dedent_function(lines, width: int) -> str:
    # Remove up to `width` leading whitespace characters from every line.
    # textwrap.dedent gives up when a comment inside the function starts at a smaller
    # indentation (e.g. column 0); Python accepts such comments, so keep them as they are.
    out = []
    for line in lines:
        removed = len(line) - len(line.lstrip(' \t'))
        out.append(line[min(removed, width):])
    return ''.join(out)


def functions(source: str) -> dict[str, tuple[ast.AST, str]]:
    """Module-level functions and methods (any class depth): qualname -> (node, dedented code)."""
    lines = source.splitlines(keepends=True)
    found = {}

    def visit(body, prefix):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                code = dedent_function(lines[node.lineno - 1 : node.end_lineno], node.col_offset)
                found[prefix + node.name] = (node, code)
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f"{prefix}{node.name}.")

    visit(_parse(source).body, "")
    return found


def _op_sequence(node):
    return [n for n in ast.walk(node) if isinstance(n, (ast.operator, ast.cmpop, ast.boolop))]


def operator_change(buggy: ast.AST, fixed: ast.AST):
    """(buggy op class, fixed op class) if the trees differ in exactly one operator, else None."""
    if _OP_CALL.sub("OP()", ast.dump(buggy)) != _OP_CALL.sub("OP()", ast.dump(fixed)):
        return None
    diff = [
        (type(a), type(b))
        for a, b in zip(_op_sequence(buggy), _op_sequence(fixed))
        if type(a) is not type(b)
    ]
    return diff[0] if len(diff) == 1 else None


def extract_pair(buggy_source: str, fixed_source: str):
    """The single changed function, or (None, reason)."""
    try:
        before, after = functions(buggy_source), functions(fixed_source)
    except SyntaxError:
        return None, "parse_failed"
    changed = [
        name
        for name in before.keys() & after.keys()
        if ast.dump(before[name][0]) != ast.dump(after[name][0])
    ]
    if len(changed) != 1:
        return None, "not_one_changed_function"
    name = changed[0]
    change = operator_change(before[name][0], after[name][0])
    if change is None:
        return None, "not_one_operator"
    buggy_cls, fixed_cls = change
    if buggy_cls not in OPERATORS or fixed_cls not in OPERATORS:
        return None, "operator_outside_set"
    _, kind, mutated = OPERATORS[fixed_cls]
    return {
        "qualname": name,
        "buggy": before[name][1],
        "fixed": after[name][1],
        "mutation_type": kind,
        "in_tables": mutated is buggy_cls,
        "old": OPERATORS[fixed_cls][0],
        "new": OPERATORS[buggy_cls][0],
    }, None


def _norm(code: str) -> str:
    return " ".join(str(code).split())


def known_code(paths) -> set[str]:
    known = set()
    for path in paths:
        if Path(path).exists():
            known.update(_norm(c) for c in pd.read_csv(path, usecols=["code"]).code)
    return known


def collect(records, *, limit, pattern, fetch, known=frozenset(), max_per_repo=5, seed=42):
    """Rows for the output CSV and a Counter of why candidates were dropped."""
    stats = Counter()
    candidates = []
    for record in records:
        stats["records"] += 1
        if norm_pattern(record.get("sstub_pattern")) != pattern:
            continue
        stats["pattern_match"] += 1
        if record.get("in_function") is False or record.get("comodified") is True:
            stats["skip_not_in_function_or_comodified"] += 1
            continue
        if not statement_prefilter(record.get("before"), record.get("after")):
            stats["skip_statement_not_our_operators"] += 1
            continue
        candidates.append(record)
    stats["candidates"] = len(candidates)
    random.Random(seed).shuffle(candidates)

    rows, seen, per_repo = [], set(), Counter()
    for record in candidates:
        if len(rows) >= 2 * limit:
            break
        match = _GITHUB.match(record.get("project_url") or "")
        if not match:
            stats["skip_not_github"] += 1
            continue
        repo = "/".join(match.groups())
        if per_repo[repo] >= max_per_repo:
            stats["skip_repo_cap"] += 1
            continue
        path = record["file_path"]
        buggy_src = fetch(raw_url(record["project_url"], record["parent_sha"], path))
        fixed_src = buggy_src and fetch(raw_url(record["project_url"], record["commit_sha"], path))
        if not buggy_src or not fixed_src:
            stats["skip_download_failed"] += 1
            continue
        pair, reason = extract_pair(buggy_src, fixed_src)
        if pair is None:
            stats[f"skip_{reason}"] += 1
            continue
        key = (_norm(pair["buggy"]), _norm(pair["fixed"]))
        if key in seen:
            stats["skip_duplicate"] += 1
            continue
        if key[0] in known or key[1] in known:
            stats["skip_in_our_dataset"] += 1
            continue
        seen.add(key)
        per_repo[repo] += 1
        source_id = f"{record['commit_sha'][:12]}:{path}:{pair['qualname']}"
        for label, code, kind in ((1, pair["buggy"], pair["mutation_type"]), (0, pair["fixed"], "clean")):
            rows.append(
                {
                    "sample_id": len(rows) + 1,
                    "source_id": source_id,
                    "code": code,
                    "label": label,
                    "mutation_type": kind,
                    "repo": repo,
                    "in_tables": pair["in_tables"],
                }
            )
        stats["pairs"] += 1
        stats["pairs_in_tables"] += pair["in_tables"]
        if stats["pairs"] % 25 == 0:
            print(f"{stats['pairs']} pairs", flush=True)
    return rows, stats


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--input", type=Path, required=True,
                   help="ctssb_data_1M.zip / tssb_data_3M.zip, a folder of .jsonl.gz, or one file")
    p.add_argument("--output", type=Path, default=ROOT / "data/real_bugs/ctssb_binop.csv")
    p.add_argument("--cache", type=Path, default=ROOT / "data/real_bugs/cache")
    p.add_argument("--limit", type=int, default=500, help="maximum number of pairs")
    p.add_argument("--pattern", default="CHANGE_BINARY_OPERATOR")
    p.add_argument("--pause", type=float, default=0.5, help="seconds between downloads")
    p.add_argument("--max-per-repo", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude", type=Path, nargs="*",
                   default=[ROOT / "data/processed/bug_detection_dataset.csv"],
                   help="CSV files with a `code` column; functions found there are dropped")
    p.add_argument("--list-patterns", action="store_true",
                   help="print sstub_pattern counts and exit (no downloads)")
    a = p.parse_args()

    if a.list_patterns:
        counts = Counter(norm_pattern(r.get("sstub_pattern")) for r in iter_records(a.input))
        for name, n in counts.most_common():
            print(f"{n:>9}  {name}")
        return

    a.cache.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    rows, stats = collect(
        iter_records(a.input),
        limit=a.limit,
        pattern=norm_pattern(a.pattern),
        fetch=lambda url: fetch_text(url, a.cache, a.pause) if url else None,
        known=known_code(a.exclude),
        max_per_repo=a.max_per_repo,
        seed=a.seed,
    )
    pd.DataFrame(rows, columns=["sample_id", "source_id", "code", "label", "mutation_type",
                                "repo", "in_tables"]).to_csv(a.output, index=False)
    a.output.with_suffix(".stats.json").write_text(json.dumps(dict(stats), indent=2) + "\n")
    print(json.dumps(dict(stats), indent=2))
    print(f"wrote {stats['pairs']} pairs to {a.output}")


if __name__ == "__main__":
    main()
