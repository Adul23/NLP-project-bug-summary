"""Mutation contracts: one operator, valid AST, unchanged strings and comments."""
import ast
import random
import unittest

try:
    from src.data.mutations import mutate_code
except ImportError:
    mutate_code = None


class MutationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(mutate_code, 'AST mutation is not implemented')

    def test_supported_operators(self):
        cases = [('==', '!=', 'comparison_operator'), ('!=', '==', 'comparison_operator'),
                 ('>', '<', 'comparison_operator'), ('<', '>', 'comparison_operator'),
                 ('>=', '>', 'comparison_operator'), ('<=', '<', 'comparison_operator'),
                 ('+', '-', 'arithmetic_operator'), ('-', '+', 'arithmetic_operator'),
                 ('*', '/', 'arithmetic_operator'), ('/', '*', 'arithmetic_operator'),
                 ('and', 'or', 'boolean_operator'), ('or', 'and', 'boolean_operator')]
        for old, new, kind in cases:
            source = f'def f(a, b):\n    return a {old} b'
            with self.subTest(old=old):
                result = mutate_code(source, random.Random(42))
                self.assertEqual(result.mutation_type, kind)
                self.assertEqual(result.code, f'def f(a, b):\n    return a {new} b')
                ast.parse(result.code)

    def test_only_one_operator_changes_in_chained_comparison(self):
        source = 'def f(a, b, c):\n    return a < b <= c'
        result = mutate_code(source, random.Random(42))
        self.assertIn(result.code, ['def f(a, b, c):\n    return a > b <= c',
                                   'def f(a, b, c):\n    return a < b < c'])

    def test_preserves_strings_comments_formatting_and_unicode(self):
        source = 'def f(а, b):\n    s = "and + >="  # + >= and\n    return  а+b'
        result = mutate_code(source, random.Random(42))
        self.assertEqual(result.code,
                         'def f(а, b):\n    s = "and + >="  # + >= and\n    return  а-b')

    def test_parentheses_and_operator_precedence_preserved(self):
        for source in ['def f(a,b,c): return (a+b)*c',
                       'def f(a,b,c): return a+b*c',
                       'def f(a,b,c): return (a and b) or c',
                       'def f(a,b,c): return a and (b or c)']:
            result = mutate_code(source, random.Random(42))
            self.assertIsNotNone(result)
            ast.parse(result.code)

    def test_seed_reproducibility(self):
        source = 'def f(x, y):\n    return (x + y > 1) and (x - y < 3)'
        first = random.Random(42)
        second = random.Random(42)
        self.assertEqual([mutate_code(source, first) for _ in range(20)],
                         [mutate_code(source, second) for _ in range(20)])

    def test_non_newline_separators_do_not_shift_operator_positions(self):
        for separator in ['\u2028', '\u2029', '\x0c']:
            source = f'def f(a,b):\n    s = "x{separator}y"\n    return a+b'
            with self.subTest(separator=repr(separator)):
                result = mutate_code(source, random.Random(42))
                self.assertIsNotNone(result)
                self.assertEqual(result.code,
                                 f'def f(a,b):\n    s = "x{separator}y"\n    return a-b')

    def test_no_supported_operator(self):
        for source in ['def f(): return "a+b"', 'def f(x): return -x',
                       'def f(a,b): return a is b', 'def f(a,b): return a // b',
                       'def f(a,b,c): return a and b and c']:
            self.assertIsNone(mutate_code(source, random.Random(42)))

    def test_no_changes_to_function_decorators_defaults_or_annotations(self):
        source = '@deco(1 + 2)\ndef f(x: A | B = 1 + 2):\n    return x'
        self.assertIsNone(mutate_code(source, random.Random(42)))


if __name__ == '__main__':
    unittest.main()
