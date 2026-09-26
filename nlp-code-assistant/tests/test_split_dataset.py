"""Validation and group-split regression tests."""
import importlib

import pandas as pd
import pytest


def api():
    name = 'src.bug_detection.split_dataset'
    assert importlib.util.find_spec(name) is not None, 'split_dataset is not implemented'
    return importlib.import_module(name)


def frame():
    return pd.DataFrame([
        {'sample_id': 2*i+y, 'source_id': f'{i:06d}',
         'code': f'def f{i}(x): return x {"+" if y == 0 else "-"} 1',
         'label': y, 'mutation_type': 'clean' if y == 0 else 'arithmetic_operator'}
        for i in range(40) for y in (0, 1)
    ])


def test_group_split_no_leakage_complete_reproducible():
    module = api()
    data = frame()
    splits = module.split_dataset(data)
    again = module.split_dataset(data)
    assert [part.source_id.nunique() for part in splits] == [28, 6, 6]
    assert sum(map(len, splits)) == len(data)
    for i, part in enumerate(splits):
        assert set(part.label) == {0, 1}
        pd.testing.assert_frame_equal(part, again[i])
        for other in splits[i+1:]:
            assert set(part.source_id).isdisjoint(other.source_id)
    assert set(pd.concat(splits).sample_id) == set(data.sample_id)


@pytest.mark.parametrize('column,value', [
    ('code', None), ('code', ''), ('code', '  '), ('label', 2),
    ('label', None), ('source_id', None), ('source_id', '  '),
    ('mutation_type', None), ('mutation_type', 'clean'),
])
def test_invalid_rows_fail_clearly(column, value):
    data = frame()
    data.loc[1, column] = value
    with pytest.raises(ValueError, match=column):
        api().validate_dataset(data)


def test_clean_mutation_and_missing_columns_rejected():
    data = frame()
    data.loc[0, 'mutation_type'] = 'boolean_operator'
    with pytest.raises(ValueError, match='mutation_type'):
        api().validate_dataset(data)
    with pytest.raises(ValueError, match='sample_id'):
        api().validate_dataset(frame().drop(columns='sample_id'))


def test_exact_duplicates_removed_and_csv_ids_preserved(tmp_path):
    data = frame()
    path = tmp_path / 'data.csv'
    pd.concat([data, data.iloc[[0]]]).to_csv(path, index=False)
    loaded = api().load_dataset(path)
    assert len(loaded) == len(data)
    assert loaded.iloc[0].source_id == '000000'


def test_overlap_guard():
    module = api()
    with pytest.raises(RuntimeError, match='source_id leakage detected'):
        module.check_source_overlap(frame(), frame(), frame())
