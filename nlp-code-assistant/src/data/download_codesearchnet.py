"""Stream only enough official CodeSearchNet Python records for the experiment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from typing import Iterable, TextIO

if __package__:
    from .preprocess import CleanCollector, PROJECT_ROOT, RAW_PATH, SEED, print_stats, read_raw
else:
    from preprocess import CleanCollector, PROJECT_ROOT, RAW_PATH, SEED, print_stats, read_raw

DATASET = 'code-search-net/code_search_net'
REVISION = '7e3332b8032ff895377dba3e57f0152fcb32ff7d'
PARQUET_URL = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/python/train/0000.parquet'
PROVENANCE_FIELDS = ('repository_name', 'func_path_in_repository', 'func_name', 'func_code_url')


def collect_stream(records: Iterable[dict], output: TextIO, target: int = 10_000,
                   max_scan: int = 100_000) -> CleanCollector:
    """Stop immediately at target unique valid functions; retain consumed raw rows."""
    if target <= 0 or max_scan < target:
        raise ValueError('Require target > 0 and max_scan >= target')
    collector = CleanCollector()
    for record in records:
        if record.get('language', 'python').lower() != 'python':
            raise ValueError('Input contains a non-Python record')
        field = next((key for key in ('whole_func_string', 'code', 'func_code_string')
                      if key in record), None)
        if field is None:
            raise ValueError('CodeSearchNet record has no source code field')
        code = record[field]
        accepted = collector.add(code)
        raw = {'code': code, 'source_id': collector.rows[-1]['source_id'] if accepted else None}
        raw.update({key: record[key] for key in PROVENANCE_FIELDS if key in record})
        output.write(json.dumps(raw, ensure_ascii=False) + '\n')
        if len(collector.rows) == target:
            return collector
        if collector.stats.loaded_samples >= max_scan:
            break
    raise ValueError(f'Stream ended or scan limit reached: {len(collector.rows)}/{target} '
                     f'valid unique functions after {collector.stats.loaded_samples} records.')


def _parquet_records(url: str):
    """Yield small synchronous batches; safely stop before the Parquet file ends."""
    import fsspec
    import pyarrow.parquet as pq

    # The Dataset scanner can leave native worker threads alive at interpreter
    # shutdown after early stopping. ParquetFile avoids that background scanner.
    with fsspec.open(url, 'rb', block_size=1 << 20) as handle:
        with pq.ParquetFile(handle, pre_buffer=False) as parquet:
            for batch in parquet.iter_batches(
                batch_size=256, columns=['whole_func_string', 'language', *PROVENANCE_FIELDS],
                use_threads=False,
            ):
                yield from batch.to_pylist()


def remote_records(seed: int, buffer_size: int, url: str = PARQUET_URL):
    """Stream selected columns with HF iterable shuffling, without a full download."""
    os.environ.setdefault('HF_HOME', str(PROJECT_ROOT / '.cache/huggingface'))
    from datasets import IterableDataset

    dataset = IterableDataset.from_generator(_parquet_records, gen_kwargs={'url': url})
    return dataset.shuffle(seed=seed, buffer_size=buffer_size)


def download(output_path: Path = RAW_PATH, target: int = 10_000, seed: int = SEED,
             buffer_size: int = 1_000, max_scan: int = 100_000,
             input_path: Path | None = None) -> CleanCollector:
    """Save a reproducible raw subset and its provenance manifest atomically."""
    if buffer_size <= 0:
        raise ValueError('buffer_size must be positive')
    if input_path is not None and input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output must differ')
    records = read_raw(input_path) if input_path else remote_records(seed, buffer_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            collector = collect_stream(records, handle, target, max_scan)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        'dataset': DATASET if input_path is None else str(input_path.resolve()),
        'revision': REVISION if input_path is None else None,
        'url': PARQUET_URL if input_path is None else None,
        'subset': 'python', 'upstream_split': 'train', 'seed': seed,
        'shuffle_buffer_size': buffer_size if input_path is None else 0,
        'target': target, 'max_scan': max_scan, 'python_version': platform.python_version(),
        'packages': {package: version(package) for package in ('pandas', 'datasets', 'pyarrow')},
        'preparation': asdict(collector.stats),
    }
    output_path.with_suffix('.manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print_stats(collector.stats)
    print(f'Raw subset saved: {output_path}')
    return collector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RAW_PATH)
    parser.add_argument('--input', type=Path, help='Optional local JSONL for offline runs')
    parser.add_argument('--target', type=int, default=10_000)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--buffer-size', type=int, default=1_000)
    parser.add_argument('--max-scan', type=int, default=100_000)
    args = parser.parse_args()
    download(args.output, args.target, args.seed, args.buffer_size, args.max_scan, args.input)


if __name__ == '__main__':
    main()
