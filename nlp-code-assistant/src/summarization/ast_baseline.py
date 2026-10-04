"""AST/template baseline: a fixed sentence built from the function name and its return type.

No learning is involved. The function name gives the subject; the AST tells us
whether the function returns a value. This is the floor any learned model must beat.
"""

from __future__ import annotations

import ast
import warnings

# Names that already start with an action verb are used as imperative sentences.
_VERBS = frozenset(
    "get set create make build add remove delete update check is has load save parse "
    "convert compute calculate read write run init validate find search render format "
    "send handle process generate export import start stop open close reset register".split()
)


def _first_function(tree: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node
    return None


def _returns_value(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return any(
        isinstance(node, ast.Return) and node.value is not None for node in ast.walk(func)
    )


def predict(code: str) -> str:
    """Return a one-sentence summary for a Python function's source code."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(code)
    except SyntaxError:
        return "Perform the operation."

    func = _first_function(tree)
    if func is None:
        return "Perform the operation."

    if func.name == "__init__":
        return "Initialize the object."

    words = [word for word in func.name.lower().split("_") if word]
    if not words:
        return "Perform the operation."

    phrase = " ".join(words)
    if words[0] in _VERBS:
        sentence = phrase
    elif _returns_value(func):
        sentence = f"Return the {phrase}"
    else:
        sentence = phrase

    return sentence[0].upper() + sentence[1:] + "."
