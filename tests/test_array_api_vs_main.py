"""
The ported functions give what pvlib gave before the port, for the calls in
_array_api_cases with Python, NumPy and pandas inputs: the same values, the
same pandas types, index, names and columns, and the same dtype.

The results from before the port are in data/array_api_main_results.json;
_make_array_api_golden.py explains how to make them again.
"""
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ._array_api_cases import CASES, CONTAINER_CHANGES, run
from .conftest import TESTS_DATA_DIR, has_numba

MAIN = json.loads(
    (TESTS_DATA_DIR / 'array_api_main_results.json').read_text())
# dtypes follow NumPy's promotion rules, which changed in NumPy 2
SAME_NUMPY = (MAIN['meta']['numpy'].split('.')[0]
              == np.__version__.split('.')[0])


def _compared():
    for name, case in CASES.items():
        if MAIN['cases'][name]['type'] == 'error' and case.policy == 'match':
            continue  # raised before the port; the port may accept more
        yield name


@pytest.mark.parametrize('name', list(_compared()))
def test_same_as_before_port(name):
    case = CASES[name]
    if 'numba' in case.requires and not has_numba:
        pytest.skip('requires numba')
    before = MAIN['cases'][name]
    now = run(case)
    if case.policy == 'raise':
        assert before['type'] == 'error'
        assert now['type'] == 'error', f'returned {now["type"]}, raised ' \
            f'{before["error"]} before the port'
        assert now['error'] in ('TypeError', 'ValueError', 'UFuncTypeError')
        return
    assert now['type'] != 'error', f'raised {now["error"]}'
    _assert_same(now, before, name in CONTAINER_CHANGES, name)


def _assert_same(now, before, any_container, where, names=True):
    old, new = before['type'], now['type']
    if old == 'dict':
        assert new in ('dict', 'DataFrame'), f'{where}: {new}, not a dict'
        items = now['items'] if new == 'dict' else now['columns']
        assert sorted(items) == sorted(before['items']), where
        # a dict before the port may be a DataFrame now, as documented for
        # erbs: its columns are Series, named by their keys
        frame = new == 'DataFrame'
        for k, v in before['items'].items():
            _assert_same(items[k], v, any_container or frame,
                         f'{where}[{k}]', names=not frame)
        return
    if old == 'DataFrame':
        assert new == 'DataFrame', f'{where}: {new}, not a DataFrame'
        assert now['index'] == before['index'], f'{where}: index'
        assert now['order'] == before['order'], f'{where}: columns'
        for k, v in before['columns'].items():
            _assert_same(now['columns'][k], v, any_container,
                         f'{where}[{k}]')
        return
    if not any_container:
        assert new == old, f'{where}: {new}, not {old}'
    if old == 'Series' and new == 'Series':
        assert now['index'] == before['index'], f'{where}: index'
        if names:
            assert now['name'] == before['name'], f'{where}: name'
    if SAME_NUMPY:
        assert now['dtype'] == before['dtype'], f'{where}: dtype'
    values, expected = (np.asarray(now['values'], float),
                        np.asarray(before['values'], float))
    assert values.shape == expected.shape, f'{where}: shape'
    assert_allclose(values, expected, rtol=1e-10, atol=1e-10, equal_nan=True,
                    err_msg=where)
