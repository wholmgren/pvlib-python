"""
Input type contract for the functions that support array libraries: the
same values given as Python scalars, lists, NumPy arrays, masked arrays,
pandas Series and xarray DataArrays give the same results, in the container
type that came in.

The functions and their inputs are the entries of _supported_functions. A
function whose handling of input types changes shows up here, not in its own
tests, which use one input type each. Array API libraries (PyTorch, JAX,
...) are covered in test_array_api.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from ._supported_functions import INDEX, KEEP, SUPPORTED, function

MASKED = 1  # position masked in the masked array inputs

KINDS = ['scalar', 'list', 'ndarray', 'masked', 'series', 'xarray']


def _convert(kind, param, values):
    if kind == 'scalar':
        return values[0]
    if kind == 'list':
        return list(values)
    if kind == 'ndarray':
        return np.array(values)
    if kind == 'masked':
        mask = np.arange(len(values)) == MASKED
        return np.ma.masked_array(values, mask=mask)
    if kind == 'nan_at_masked':  # what masked values become inside pvlib
        values = np.array(values, dtype=float)
        values[MASKED] = np.nan
        return values
    if kind == 'series':
        return pd.Series(values, index=INDEX[:len(values)], name=param)
    if kind == 'xarray':
        xr = pytest.importorskip('xarray')
        return xr.DataArray(values, dims='time')


def _call(case, kind):
    entry = SUPPORTED[case]
    kwargs = {k: _convert(kind, k, v) for k, v in entry.arrays.items()}
    return function(case)(**kwargs, **entry.params)


def _check_container(out, kind, case):
    expected_name = SUPPORTED[case].name
    if kind == 'scalar':
        assert np.ndim(out) == 0 and not isinstance(out, np.ndarray), \
            f'{type(out)} for scalar input'
    elif kind in ('list', 'ndarray'):
        assert type(out) is np.ndarray
    elif kind == 'masked':
        assert isinstance(out, np.ma.MaskedArray)
        assert out.mask[MASKED]
    elif kind == 'series':
        assert isinstance(out, pd.Series)
        assert out.index.equals(INDEX[:len(out)])
        if expected_name is KEEP:
            expected_name = next(iter(SUPPORTED[case].arrays))
        assert out.name == expected_name
    elif kind == 'xarray':
        import xarray as xr
        assert isinstance(out, xr.DataArray)


def _values(out, kind):
    if kind == 'masked':
        return np.ma.filled(out.astype(float), np.nan)
    if isinstance(out, (pd.Series, pd.Index)) or hasattr(out, 'dims'):
        out = out.values  # pandas or xarray
    return np.asarray(out, dtype=float)


def _check(out, reference, kind, case, key=''):
    _check_container(out, kind, case)
    actual = _values(out, kind)
    expected = np.asarray(reference, dtype=float)
    if kind == 'scalar':
        expected = expected[0]
    if kind == 'masked':
        keep = np.arange(len(expected)) != MASKED
        actual, expected = actual[keep], expected[keep]
    assert_allclose(actual, expected, rtol=1e-12, equal_nan=True,
                    err_msg=f'{case} {key}')


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('case', list(SUPPORTED))
def test_input_types(case, kind):
    if kind == 'scalar' and SUPPORTED[case].one_d:
        pytest.skip('takes 1-D series')
    # masked values enter the calculation as nan, which matters to
    # functions that accumulate along time, such as soiling.hsu
    reference = _call(case, 'nan_at_masked' if kind == 'masked'
                      else 'ndarray')
    out = _call(case, kind)
    if not isinstance(reference, dict):
        _check(out, reference, kind, case)
        return
    # a dict of arrays: a DataFrame for pandas input, a Dataset for xarray
    if kind == 'series':
        assert isinstance(out, pd.DataFrame)
        assert out.index.equals(INDEX)
        out, kind = {k: out[k].to_numpy() for k in out}, 'ndarray'
    elif kind == 'xarray':
        import xarray as xr
        assert isinstance(out, xr.Dataset)
        out = {k: out[k] for k in out.data_vars}
    else:
        assert isinstance(out, dict)
    for key in reference:
        _check(out[key], reference[key], kind, case, key)
