"""Reject malformed inputs and preserve source meaning during preparation."""
import ast
import unittest

try:
    from src.data.preprocess import CleanCollector, prepare_code
except ImportError:
    CleanCollector = prepare_code = None


class PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(prepare_code, 'preprocessing is not implemented')

    def test_normalizes_without_reformatting(self):
        code = '\r\ndef f(x):\r\n    # keep this\r\n    return  x+1\r\n'
        self.assertEqual(prepare_code(code), 'def f(x):\n    # keep this\n    return  x+1')

    def test_rejects_empty_invalid_and_nonfunction_fragments(self):
        for code in [None, '', '  ', 'def broken(:', 'x = 1', 'f = lambda x: x',
                     'class A:\n    def f(self): pass', 'def f(): pass\nx = 1']:
            with self.subTest(code=code):
                self.assertIsNone(prepare_code(code))

    def test_accepts_async_and_decorated_functions(self):
        for code in ['async def f(x):\n    return await x', '@wrap\ndef f():\n    return 1']:
            self.assertEqual(prepare_code(code), code)

    def test_removes_docstrings_but_preserves_literals_and_unicode(self):
        source = 'def привет():\n    """Summary.\n    Details."""\n    return "a + b and c"'
        prepared = prepare_code(source)
        self.assertNotIn('Summary', prepared)
        self.assertIn('return "a + b and c"', prepared)
        self.assertIsNone(ast.get_docstring(ast.parse(prepared).body[0]))

    def test_docstring_only_and_inline_docstring_remain_valid(self):
        for source in ['def f(): "doc"', 'def f(): "doc"; return 1',
                       'def f():\n    "doc" # comment\n    return 1']:
            prepared = prepare_code(source)
            self.assertIsNotNone(prepared)
            self.assertIsNone(ast.get_docstring(ast.parse(prepared).body[0]))

    def test_consecutive_docstrings_do_not_expose_a_new_docstring(self):
        for source in ['def f():\n    "first"\n    "second"\n    return 1',
                       'def f(): "first"; "second"']:
            prepared = prepare_code(source)
            self.assertIsNotNone(prepared)
            self.assertIsNone(ast.get_docstring(ast.parse(prepared).body[0]))
            self.assertNotIn('second', prepared)

    def test_deduplication_stats_and_zero_padded_ids(self):
        collector = CleanCollector()
        for source in ['', 'def broken(:', 'x = 1', 'def f():\r\n    return 1\r\n',
                       'def f():\n    return 1', 'async def g():\n    return 2']:
            collector.add(source)
        self.assertEqual([r['source_id'] for r in collector.rows], ['000001', '000002'])
        self.assertEqual(collector.stats.loaded_samples, 6)
        self.assertEqual(collector.stats.invalid_python, 1)
        self.assertEqual(collector.stats.duplicates_removed, 1)
        self.assertEqual(collector.stats.valid_python, 4)
        self.assertTrue(all(r['label'] == 0 and r['mutation_type'] == 'clean'
                            for r in collector.rows))


if __name__ == '__main__':
    unittest.main()
