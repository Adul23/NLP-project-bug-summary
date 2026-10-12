"""Verified synthetic explanations from original/mutant pairs, never label-only guesses."""
import argparse
import ast
import io
import json
import tokenize
from pathlib import Path

from src.bug_detection.experiment_data import DEFAULT_SPLITS, load_splits, sha256
from src.bug_detection.split_dataset import ROOT, SPLIT_NAMES
from src.data.build_dataset import _check_single_mutation
from src.data.preprocess import parse_python, is_function
from src.data.mutations import COMPARISONS, ARITHMETIC, BOOLEANS

MAPPINGS = {'comparison_operator': COMPARISONS, 'arithmetic_operator': ARITHMETIC, 'boolean_operator': BOOLEANS}


def verified_change(original, buggy):
    """Require a supported single AST change AND a single changed source token."""
    try:
        left, right = parse_python(original), parse_python(buggy)
        if not is_function(left) or not is_function(right):
            raise ValueError('Expected standalone functions')
        old = list(tokenize.generate_tokens(io.StringIO(original).readline))
        new = list(tokenize.generate_tokens(io.StringIO(buggy).readline))
        if len(old) != len(new):
            raise ValueError('Token structure differs')
        changes = [(a, b) for a, b in zip(old, new) if (a.type, a.string) != (b.type, b.string)]
        if len(changes) != 1:
            raise ValueError('Expected exactly one changed token')
        a, b = changes[0]
        for category, mapping in MAPPINGS.items():
            if (a.string, b.string) in {(v[1], v[2]) for v in mapping.values()}:
                _check_single_mutation(left, right, category)
                return {'mutation_type': category, 'old_operator': a.string, 'new_operator': b.string,
                        'line': b.start[0], 'bug_type': category + '_change',
                        'explanation': f"Synthetic mutation: {category.replace('_', ' ')} changed from `{a.string}` to `{b.string}` on line {b.start[0]}. This may alter behavior; an actual failure is not established.",
                        'suggested_fix': None}
        raise ValueError('Unsupported operator change')
    except (SyntaxError, tokenize.TokenError, RecursionError) as exc:
        raise ValueError('Cannot verify the code pair') from exc


def prepare(split_dir, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty supervision directory')
    splits, manifest = load_splits(split_dir)
    prepared = {}
    for name, frame in zip(SPLIT_NAMES, splits):
        clean = frame[frame.label == 0].set_index('source_id')
        if not clean.index.is_unique:
            raise ValueError('Ambiguous clean originals')
        records = []
        for row in frame[frame.label == 1].itertuples():
            if row.source_id not in clean.index:
                raise ValueError(f'No original for {row.source_id}')
            original = clean.loc[row.source_id, 'code']
            change = verified_change(original, row.code)
            if change['mutation_type'] != row.mutation_type:
                raise ValueError('Mutation metadata contradicts the verified code change')
            records.append({'source_id': row.source_id, 'sample_id': int(row.sample_id),
                            'buggy_code': row.code, 'original_code': original,
                            'explanation': change['explanation'], 'change': change,
                            'supervision': 'verified_synthetic_pair'})
        prepared[name] = records
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in prepared.items():
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (output / 'manifest.json').write_text(json.dumps({'split_manifest': manifest,
        'counts': {name: len(rows) for name, rows in prepared.items()},
        'files': {name: sha256(output / f'{name}.jsonl') for name in SPLIT_NAMES},
        'supervision': 'verified_synthetic_pair', 'human_annotated': False}, indent=2))
    return {name: len(rows) for name, rows in prepared.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, default=DEFAULT_SPLITS)
    p.add_argument('--output', type=Path, default=ROOT / 'data/processed/summaries_v1')
    a = p.parse_args()
    print(prepare(a.split_dir, a.output))


if __name__ == '__main__':
    main()
