"""Repair candidates for bug localization without reference code.

For every operator in the function, propose each operator that could have been mutated
into it (inverse of the mutation tables in src/data/mutations.py). Model scoring is a
separate step; this module only builds the candidate functions.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import tokenize
import warnings

from src.data.mutations import (
    ARITHMETIC,
    BOOLEANS,
    COMPARISONS,
    _BodyVisitor,
    _render_local,
)

# operator now in the code -> operators the original could have had.
# The tables are not symmetric: ">=" mutates to ">", so a ">" may be a broken ">=" or "<".
REPAIRS: dict[str, list[str]] = {}
for table in (COMPARISONS, ARITHMETIC, BOOLEANS):
    for _, original, mutated in table.values():
        REPAIRS.setdefault(mutated, []).append(original)


def _changed_line(old: str, new: str) -> int:
    for number, (a, b) in enumerate(zip(old.splitlines(), new.splitlines()), start=1):
        if a != b:
            return number
    raise ValueError("candidate does not change the code")


def _parse(code: str) -> ast.AST:
    # User code may contain invalid escape sequences; ast would warn about each one.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(code)


def repair_candidates(code: str) -> list[dict]:
    """Every single-operator repair of `code` that still parses."""
    tree = _parse(code)
    visitor = _BodyVisitor()
    visitor.visit(tree)
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    candidates = []
    for mutation_type, found in visitor.groups.items():
        for candidate in found:
            for fix in REPAIRS.get(candidate.old_text, []):
                fixed = _render_local(
                    code, dataclasses.replace(candidate, new_text=fix), tokens
                )
                if fixed is None or fixed == code:
                    continue
                try:
                    _parse(fixed)
                except SyntaxError:
                    continue
                candidates.append(
                    {
                        "code": fixed,
                        "line": _changed_line(code, fixed),
                        "old": candidate.old_text,
                        "new": fix,
                        "mutation_type": mutation_type,
                    }
                )
    return candidates


def localize(code: str, predictor, top_k: int = 3) -> dict:
    """Rank repair candidates by how much they lower the model's bug probability.

    `predictor` is anything with predict_batch(codes) -> [{"bug_probability": ...}, ...]
    (TfidfPredictor or LoraPredictor). The original and all candidates go in one batch.
    """
    candidates = repair_candidates(code)
    scores = predictor.predict_batch([code] + [c["code"] for c in candidates])
    before = scores[0]["bug_probability"]
    ranked = [
        {
            **c,
            "bug_probability_after": s["bug_probability"],
            "drop": before - s["bug_probability"],
        }
        for c, s in zip(candidates, scores[1:])
    ]
    ranked.sort(key=lambda c: c["drop"], reverse=True)
    return {
        "bug_probability": before,
        "n_candidates": len(ranked),
        "candidates": ranked[:top_k],
    }
