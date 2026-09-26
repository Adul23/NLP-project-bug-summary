"""One AST operator mutation per function, with source-preserving rendering."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import io
import random
import tokenize

if __package__:
    from .preprocess import is_function, parse_python
else:
    from preprocess import is_function, parse_python

MUTATION_TYPES = ('comparison_operator', 'arithmetic_operator', 'boolean_operator')
COMPARISONS = {ast.Eq: (ast.NotEq, '==', '!='), ast.NotEq: (ast.Eq, '!=', '=='),
               ast.Gt: (ast.Lt, '>', '<'), ast.Lt: (ast.Gt, '<', '>'),
               ast.GtE: (ast.Gt, '>=', '>'), ast.LtE: (ast.Lt, '<=', '<')}
ARITHMETIC = {ast.Add: (ast.Sub, '+', '-'), ast.Sub: (ast.Add, '-', '+'),
              ast.Mult: (ast.Div, '*', '/'), ast.Div: (ast.Mult, '/', '*')}
BOOLEANS = {ast.And: (ast.Or, 'and', 'or'), ast.Or: (ast.And, 'or', 'and')}


@dataclass(frozen=True)
class MutationResult:
    code: str
    mutation_type: str


@dataclass
class MutationStats:
    invalid_mutations_skipped: int = 0
    duplicate_mutations_skipped: int = 0
    no_candidate_functions: int = 0
    functions_without_bug: int = 0


@dataclass
class _Candidate:
    node: ast.AST
    index: int | None
    replacement: type[ast.AST]
    old_text: str
    new_text: str
    left: ast.AST
    right: ast.AST


class _BodyVisitor(ast.NodeVisitor):
    """Avoid mutating defaults, annotations, decorators and class signatures."""

    def __init__(self) -> None:
        self.groups: dict[str, list[_Candidate]] = {kind: [] for kind in MUTATION_TYPES}

    def visit_FunctionDef(self, node):
        for statement in node.body:
            self.visit(statement)

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def visit_Lambda(self, node):
        self.visit(node.body)

    def visit_Compare(self, node):
        operands = [node.left, *node.comparators]
        for index, operator in enumerate(node.ops):
            if type(operator) in COMPARISONS:
                replacement, old, new = COMPARISONS[type(operator)]
                self.groups['comparison_operator'].append(
                    _Candidate(node, index, replacement, old, new, operands[index], operands[index + 1]))
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if type(node.op) in ARITHMETIC:
            replacement, old, new = ARITHMETIC[type(node.op)]
            self.groups['arithmetic_operator'].append(
                _Candidate(node, None, replacement, old, new, node.left, node.right))
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # A flat `a and b and c` has one AST op but TWO source operators.
        # Only binary nodes satisfy the strict one-operator-per-sample rule.
        if len(node.values) == 2:
            replacement, old, new = BOOLEANS[type(node.op)]
            self.groups['boolean_operator'].append(
                _Candidate(node, None, replacement, old, new, *node.values))
        self.generic_visit(node)


def _replace_operator(candidate: _Candidate, replacement: ast.AST) -> ast.AST:
    if candidate.index is None:
        previous = candidate.node.op
        candidate.node.op = replacement
    else:
        previous = candidate.node.ops[candidate.index]
        candidate.node.ops[candidate.index] = replacement
    return previous


def _render_local(code: str, candidate: _Candidate, tokens: list) -> str | None:
    # Unlike str.splitlines(), tokenize/AST count only physical LF line breaks.
    lines = io.StringIO(code).readlines()

    def position(line: int, byte_column: int) -> tuple[int, int]:
        return line, len(lines[line - 1].encode('utf-8')[:byte_column].decode('utf-8'))

    left_end = position(candidate.left.end_lineno, candidate.left.end_col_offset)
    right_start = position(candidate.right.lineno, candidate.right.col_offset)
    matches = [token for token in tokens
               if token.type in (tokenize.OP, tokenize.NAME) and token.string == candidate.old_text
               and left_end <= token.start and token.end <= right_start]
    if len(matches) != 1:
        return None
    token = matches[0]
    start = sum(map(len, lines[:token.start[0] - 1])) + token.start[1]
    end = sum(map(len, lines[:token.end[0] - 1])) + token.end[1]
    return code[:start] + candidate.new_text + code[end:]


def mutate_code(code: str, rng: random.Random, forbidden: set[str] | None = None,
                stats: MutationStats | None = None) -> MutationResult | None:
    """Choose a category randomly, then one valid AST-local operator substitution.

    Mutate AST nodes directly, unparse and reparse, then render the operator at
    its original token span. Accept only if that source has exactly the same
    AST as the mutated tree. This retains comments/spacing in both classes.
    Try remaining candidates on validation failure or a global code collision.
    """
    stats = stats if stats is not None else MutationStats()
    forbidden = forbidden if forbidden is not None else set()
    tree = parse_python(code)
    if not is_function(tree):
        raise ValueError('Expected one valid standalone function')
    visitor = _BodyVisitor()
    visitor.visit(tree)
    categories = [kind for kind, candidates in visitor.groups.items() if candidates]
    if not categories:
        stats.no_candidate_functions += 1
        stats.functions_without_bug += 1
        return None
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    rng.shuffle(categories)
    for kind in categories:
        candidates = visitor.groups[kind]
        rng.shuffle(candidates)
        for candidate in candidates:
            previous = _replace_operator(candidate, candidate.replacement())
            try:
                expected = ast.dump(tree, include_attributes=False)
                canonical = ast.unparse(tree)
                if ast.dump(parse_python(canonical), include_attributes=False) != expected:
                    stats.invalid_mutations_skipped += 1
                    continue
                buggy = _render_local(code, candidate, tokens)
                if (buggy is None or buggy == code
                        or ast.dump(parse_python(buggy), include_attributes=False) != expected):
                    stats.invalid_mutations_skipped += 1
                    continue
                if buggy in forbidden:
                    stats.duplicate_mutations_skipped += 1
                    continue
                return MutationResult(buggy, kind)
            except (SyntaxError, ValueError, RecursionError, tokenize.TokenError):
                stats.invalid_mutations_skipped += 1
            finally:
                _replace_operator(candidate, previous)
    stats.functions_without_bug += 1
    return None


if __name__ == '__main__':
    example = 'def check(x):\n    return x >= 10'
    result = mutate_code(example, random.Random(42))
    print('ORIGINAL:\n' + example)
    print('\nBUGGY:\n' + result.code)
