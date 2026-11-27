"""scripts/collect_real_bugs.py on a tiny fake TSSB file; downloads are replaced by a dict."""

import gzip
import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_real_bugs.py"
spec = importlib.util.spec_from_file_location("collect_real_bugs", SCRIPT)
crb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(crb)

URL = "https://github.com/owner/repo"


def source(condition, extra=""):
    return (
        "import os\n\n\n"
        "class Box:\n"
        "    def clamp(self, x, lo):\n"
        f"        if {condition}:\n"
        "            return lo\n"
        f"        return x{extra}\n\n\n"
        "def other():\n"
        "    return 1\n"
    )


def record(sha, before, after, **extra):
    return {
        "project_url": URL,
        "commit_sha": sha + "f" * 30,
        "parent_sha": sha + "b" * 30,
        "file_path": "pkg/box.py",
        "before": f"if {before}:",
        "after": f"if {after}:",
        "in_function": True,
        "comodified": False,
        "sstub_pattern": "CHANGE_BINARY_OPERATOR",
        **extra,
    }


def scenario():
    """(records, files): each record's buggy/fixed file contents by URL."""
    cases = [
        # sha, buggy condition, fixed condition, buggy extra, record extras
        ("a1", "x < lo", "x <= lo", "", {}),  # <= -> < is a training mutation
        ("a2", "x <= lo", "x < lo", "", {}),  # < -> <= is not
        ("a3", "x == None", "x is None", "", {}),  # `is` is outside our operators
        ("a4", "x > lo", "x < lo", " + 1", {}),  # a second change in the same function
        ("a5", "x != lo", "x == lo", "", {"comodified": True}),
        ("a6", "x < lo", "x <= lo", "", {}),  # same pair as a1 from another commit
        ("a7", "x - lo", "x + lo", "", {"sstub_pattern": "CHANGE_OPERAND"}),
    ]
    records, files = [], {}
    for sha, buggy, fixed, extra, more in cases:
        r = record(sha, buggy, fixed, **more)
        records.append(r)
        files[crb.raw_url(URL, r["parent_sha"], r["file_path"])] = source(buggy, extra)
        files[crb.raw_url(URL, r["commit_sha"], r["file_path"])] = source(fixed)
    return records, files


def test_collect_keeps_single_operator_pairs():
    records, files = scenario()
    rows, stats = crb.collect(records, limit=10, pattern="CHANGE_BINARY_OPERATOR",
                              fetch=files.get, max_per_repo=10)
    assert stats["pattern_match"] == 6
    assert stats["skip_not_in_function_or_comodified"] == 1
    assert stats["skip_statement_not_our_operators"] == 1
    assert stats["skip_not_one_operator"] == 1
    assert stats["skip_duplicate"] == 1
    assert stats["pairs"] == 2 and stats["pairs_in_tables"] == 1

    by_source = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)
    assert len(by_source) == 2
    for pair in by_source.values():
        buggy, fixed = sorted(pair, key=lambda r: -r["label"])
        assert (buggy["label"], fixed["label"]) == (1, 0)
        assert buggy["mutation_type"] == "comparison_operator"
        assert fixed["mutation_type"] == "clean"
        assert buggy["repo"] == "owner/repo"
        assert buggy["code"].startswith("def clamp(self, x, lo):\n    if")  # dedented method
        assert buggy["source_id"].endswith(":pkg/box.py:Box.clamp")
    flags = {r["in_tables"] for pair in by_source.values() for r in pair
             if "x < lo" in r["code"] and r["label"] == 1}
    assert flags == {True}  # the a1 pair (<= -> <) is the training-table one


def test_limit_repo_cap_and_known_code():
    records, files = scenario()
    rows, stats = crb.collect(records, limit=1, pattern="CHANGE_BINARY_OPERATOR",
                              fetch=files.get, max_per_repo=10)
    assert stats["pairs"] == 1 and len(rows) == 2

    rows, stats = crb.collect(records, limit=10, pattern="CHANGE_BINARY_OPERATOR",
                              fetch=files.get, max_per_repo=1)
    assert stats["pairs"] == 1 and stats["skip_repo_cap"] >= 1

    fixed = crb.functions(source("x <= lo"))["Box.clamp"][1]
    rows, stats = crb.collect(records, limit=10, pattern="CHANGE_BINARY_OPERATOR",
                              fetch=files.get, known={crb._norm(fixed)}, max_per_repo=10)
    assert stats["skip_in_our_dataset"] >= 1
    assert all(crb._norm(r["code"]) != crb._norm(fixed) for r in rows)


def test_failed_downloads_are_skipped():
    records, _ = scenario()
    rows, stats = crb.collect(records, limit=10, pattern="CHANGE_BINARY_OPERATOR",
                              fetch=lambda url: None)
    assert rows == [] and stats["skip_download_failed"] == stats["candidates"]


@pytest.mark.parametrize(
    "before, after, expected",
    [
        ("if a < b:", "if a <= b:", True),
        ("x = a and b", "x = a or b", True),
        ("if a == None:", "if a is None:", False),
        ("x = a // b", "x = a / b", False),
        ("x = f(a + b", "x = f(a - b", True),  # does not tokenize alone: decided later
        ("x = a + b", "x = a + c", False),
    ],
)
def test_statement_prefilter(before, after, expected):
    assert crb.statement_prefilter(before, after) is expected


def test_operator_change():
    parse = lambda s: crb._parse(s).body[0]  # noqa: E731
    assert crb.operator_change(parse("a < b"), parse("a <= b")) is not None
    assert crb.operator_change(parse("a < b and c"), parse("a <= b or c")) is None  # two ops
    assert crb.operator_change(parse("a < b"), parse("a < c")) is None  # operand changed


def test_iter_records_reads_zip_of_jsonl_gz(tmp_path):
    records, _ = scenario()
    payload = gzip.compress("\n".join(json.dumps(r) for r in records).encode())
    archive = tmp_path / "ctssb.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("ctssb_data_1M/file-0.jsonl.gz", payload)
    assert list(crb.iter_records(archive)) == records
