"""Validate the baseline input and split original functions without group leakage."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / 'data/processed/bug_detection_dataset.csv'
REQUIRED_COLUMNS = ('sample_id', 'source_id', 'code', 'label', 'mutation_type')
MUTATION_TYPES = ('comparison_operator', 'arithmetic_operator', 'boolean_operator')
SPLIT_NAMES = ('train', 'validation', 'test')


def validate_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject invalid input and return a copy with exact duplicate rows removed."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    if frame.empty:
        raise ValueError('Dataset is empty')
    for column in ('code', 'source_id', 'mutation_type'):
        valid = frame[column].map(lambda value: isinstance(value, str) and bool(value.strip()))
        if not valid.all():
            raise ValueError(f'{column} must contain nonmissing, nonempty strings; '
                             f'invalid rows: {frame.index[~valid].tolist()[:5]}')
    if not frame['label'].isin([0, 1]).all():
        raise ValueError('label must contain only 0 and 1, without missing values')
    clean = frame['label'].eq(0)
    if not frame.loc[clean, 'mutation_type'].eq('clean').all():
        raise ValueError('Clean samples must have mutation_type == clean')
    if frame.loc[~clean, 'mutation_type'].eq('clean').any():
        raise ValueError('Buggy samples must not have mutation_type == clean')
    result = frame.drop_duplicates().reset_index(drop=True).copy()
    result['label'] = result['label'].astype(int)
    result.attrs['duplicates_removed'] = len(frame) - len(result)
    return result


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    # Preserve leading zeros and literal strings such as "NA" in source code/IDs.
    frame = pd.read_csv(path, dtype={'source_id': str, 'code': str, 'mutation_type': str},
                        keep_default_na=False)
    return validate_dataset(frame)


def dataset_statistics(frame: pd.DataFrame) -> dict:
    kinds = dict.fromkeys(('clean', *MUTATION_TYPES, *sorted(frame['mutation_type'].unique())))
    return {
        'rows': len(frame), 'unique_source_ids': int(frame['source_id'].nunique()),
        'clean_samples': int(frame['label'].eq(0).sum()),
        'buggy_samples': int(frame['label'].eq(1).sum()),
        'duplicates_removed': frame.attrs.get('duplicates_removed', 0),
        'mutation_distribution': {kind: int(frame['mutation_type'].eq(kind).sum()) for kind in kinds},
    }


def check_source_overlap(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict:
    a, b, c = (set(part['source_id']) for part in (train, validation, test))
    overlap = {'train/validation': len(a & b), 'train/test': len(a & c),
               'validation/test': len(b & c)}
    if any(overlap.values()):
        raise RuntimeError('source_id leakage detected')
    return overlap


def split_dataset(frame: pd.DataFrame, random_state: int = 42) -> tuple:
    """Split unique IDs 70/15/15; row percentages depend on group sizes."""
    frame = validate_dataset(frame)
    ids = sorted(frame['source_id'].unique())
    if len(ids) < 4:
        raise ValueError('At least 4 unique source_id values are required for a three-way split')
    train_ids, temporary_ids = train_test_split(ids, test_size=0.30, random_state=random_state)
    validation_ids, test_ids = train_test_split(temporary_ids, test_size=0.50,
                                               random_state=random_state)
    parts = tuple(frame[frame['source_id'].isin(group)].reset_index(drop=True)
                  for group in (train_ids, validation_ids, test_ids))
    check_source_overlap(*parts)
    return parts


def save_splits(parts: tuple, output_dir: Path) -> None:
    check_source_overlap(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(SPLIT_NAMES, parts):
        frame.to_csv(output_dir / f'{name}.csv', index=False)


def print_summary(frame: pd.DataFrame, parts: tuple) -> None:
    stats = dataset_statistics(frame)
    for title, key in [('Dataset rows', 'rows'), ('Unique source_ids', 'unique_source_ids'),
                       ('Clean samples', 'clean_samples'), ('Buggy samples', 'buggy_samples'),
                       ('Exact duplicate rows removed', 'duplicates_removed')]:
        print(f'{title}: {stats[key]}')
    print('\nMutation distribution:')
    for kind, count in stats['mutation_distribution'].items():
        print(f'{kind}: {count}')
    for name, part in zip(SPLIT_NAMES, parts):
        print(f'{name.title()} rows: {len(part)}')
        print(f'{name.title()} unique source_ids: {part.source_id.nunique()}')
    print('\nSource ID overlap:')
    for pair, count in check_source_overlap(*parts).items():
        print(f'{pair}: {count}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DATASET_PATH)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT / 'data/processed')
    args = parser.parse_args()
    frame = load_dataset(args.input)
    parts = split_dataset(frame)
    save_splits(parts, args.output_dir)
    print_summary(frame, parts)


if __name__ == '__main__':
    main()
