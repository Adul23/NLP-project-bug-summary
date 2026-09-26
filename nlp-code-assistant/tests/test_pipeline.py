"""Integration and quality checks using real local records and CLI subprocesses."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

try:
    from src.data.download_codesearchnet import collect_stream
    from src.data.build_dataset import build_dataset, validate_dataset, format_examples
except ImportError:
    collect_stream = build_dataset = validate_dataset = format_examples = None

ROOT = Path(__file__).resolve().parents[1]


def clean_frame(codes):
    return pd.DataFrame([{'source_id': f'{i:06d}', 'code': code, 'label': 0,
                          'mutation_type': 'clean'} for i, code in enumerate(codes, 1)])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(build_dataset, 'dataset pipeline is not implemented')

    def test_stream_stops_at_valid_unique_target(self):
        def records():
            for code in ['', 'def broken(:', 'def a(): return 1',
                         'def a(): return 1', 'def b(): return 2']:
                yield {'whole_func_string': code, 'language': 'python'}
            raise AssertionError('stream read past target')
        output = io.StringIO()
        collector = collect_stream(records(), output, target=2)
        self.assertEqual(len(collector.rows), 2)
        self.assertEqual(collector.stats.loaded_samples, 5)
        self.assertEqual(len(output.getvalue().splitlines()), 5)

    def test_stream_exhaustion_and_schema_errors_fail(self):
        for records in [[], [{'unexpected_field': 'x'}]]:
            with self.subTest(records=records), self.assertRaises(ValueError):
                collect_stream(records, io.StringIO(), target=2)

    def test_parquet_reader_stops_early_and_repeats_seed(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from src.data.download_codesearchnet import remote_records
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.parquet'
            rows = [{'whole_func_string': f'def f{i}(x): return x+1', 'language': 'python',
                     'repository_name': 'test/repo', 'func_path_in_repository': 'test.py',
                     'func_name': f'f{i}', 'func_code_url': 'https://example.com/test.py'}
                    for i in range(2000)]
            pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=100)
            outputs = []
            for _ in range(2):
                output = io.StringIO()
                collector = collect_stream(remote_records(42, 20, url=str(path)), output, target=3)
                self.assertEqual(len(collector.rows), 3)
                outputs.append(output.getvalue())
            self.assertEqual(outputs[0], outputs[1])

    def test_pair_linkage_category_counts_and_examples(self):
        clean = clean_frame(['def compare(x): return x >= 10',
                             'def add(a,b): return a+b',
                             'def both(a,b): return a and b',
                             'def identity(x): return x'])
        frame, stats = build_dataset(clean)
        self.assertEqual(frame.shape, (7, 5))
        self.assertEqual(frame['label'].value_counts().to_dict(), {0: 4, 1: 3})
        self.assertEqual(frame['sample_id'].tolist(), list(range(1, 8)))
        self.assertEqual(frame.query('label == 1')['source_id'].tolist(), ['000001', '000002', '000003'])
        self.assertEqual(stats.no_candidate_functions, 1)
        validate_dataset(frame)
        output = format_examples(frame, per_category=5)
        self.assertEqual(output.count('ORIGINAL:'), 3)
        self.assertEqual(output.count('BUGGY:'), 3)
        pd.testing.assert_frame_equal(frame, build_dataset(clean)[0])

    def test_mutation_collision_with_any_clean_function_is_skipped(self):
        clean = clean_frame(['def f(a,b): return a+b', 'def f(a,b): return a-b'])
        frame, stats = build_dataset(clean)
        self.assertEqual(len(frame), 2)
        self.assertEqual(stats.duplicate_mutations_skipped, 2)

    def test_quality_checks_reject_corrupt_rows(self):
        frame, _ = build_dataset(clean_frame(['def f(a,b): return a+b']))
        changes = [('label', 2), ('mutation_type', 'clean'),
                   ('mutation_type', 'unknown'), ('source_id', '999999'),
                   ('code', 'def bad(:'), ('code', frame.iloc[0]['code'])]
        for column, value in changes:
            corrupt = frame.copy()
            corrupt.loc[1, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                validate_dataset(corrupt)
        with self.assertRaises(ValueError):
            validate_dataset(pd.concat([frame, frame.iloc[[1]]], ignore_index=True))

    def test_all_cli_stages_and_leading_zero_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            fixture = folder / 'fixture.jsonl'
            raw, clean, combined = (folder / name for name in ['raw.jsonl', 'clean.csv', 'dataset.csv'])
            fixture.write_text('\n'.join(json.dumps({'code': code}) for code in [
                'def broken(:', 'def f(a,b): return a+b',
                'def g(a,b): return a and b', 'def h(a,b): return a >= b']), encoding='utf-8')
            commands = [
                ['download_codesearchnet.py', '--input', str(fixture), '--output', str(raw), '--target', '3'],
                ['preprocess.py', '--input', str(raw), '--output', str(clean), '--target', '3'],
                ['build_dataset.py', '--input', str(clean), '--output', str(combined)],
            ]
            for script, *args in commands:
                result = subprocess.run([sys.executable, str(ROOT / 'src/data' / script), *args],
                                        cwd=folder, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            frame = pd.read_csv(combined, dtype={'source_id': str})
            self.assertEqual(frame.shape, (6, 5))
            self.assertEqual(frame['source_id'].tolist(), ['000001', '000001', '000002',
                                                          '000002', '000003', '000003'])
            validate_dataset(frame)
            self.assertTrue(combined.with_suffix('.examples.txt').exists())

    def test_insufficient_input_does_not_publish_output(self):
        from src.data.preprocess import preprocess
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw = folder / 'raw.jsonl'
            raw.write_text('{"code": "def f(): return 1"}\n')
            with self.assertRaises(ValueError):
                preprocess(raw, folder / 'clean.csv', target=2)
            self.assertFalse((folder / 'clean.csv').exists())


if __name__ == '__main__':
    unittest.main()
