"""
Input type contract for calls that mix input types, which test_input_types
doesn't: each argument has its own type, and every pair of types for every
pair of arguments appears in some call. Time arguments also take datetimes:
a DatetimeIndex, a Series of datetimes, a datetime64 array, a Timestamp and
xarray time.

Each call is compared with the same values as NumPy arrays (numbers for
time). Its result comes in the type of the inputs that carry the most: xarray
over pandas over masked arrays over NumPy arrays, and a NumPy scalar if all
inputs are scalars. The user guide's "Type in, type out" says the same.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pvlib import irradiance, soiling, solarposition

from .test_input_types import CASES as TYPE_CASES, INDEX, MASKED, _convert

NUM = ['scalar', 'list', 'ndarray', 'masked', 'series', 'xarray']
DATETIMES = ['dtindex', 'dt_series', 'dt64', 'timestamp', 'dt_xarray']
# kinds per category of parameter
KINDS = {
    'num': NUM,
    'num1d': [k for k in NUM if k != 'scalar'],
    'doy': NUM + DATETIMES,
    'unix': NUM + [k for k in DATETIMES if k != 'timestamp'],
    'unix1d': ['list', 'ndarray', 'series', 'xarray', 'dtindex', 'dt_series',
               'dt64', 'dt_xarray'],
}
XARRAY = {'xarray', 'dt_xarray'}
PANDAS = {'series', 'dt_series', 'dtindex'}

UTC = INDEX.tz_convert('UTC').tz_localize(None)
TIME_VALUES = {
    'doy': np.asarray(UTC.day_of_year, float),
    'unix': np.asarray((UTC - pd.Timestamp('1970-01-01'))
                       / pd.Timedelta('1s')),
}
TIME_VALUES['unix1d'] = TIME_VALUES['unix']

# name: (function, {parameter: (values, category)}, other arguments,
#        kinds that give a pandas result besides PANDAS)
FUNCTIONS = {
    name: (func, {k: (v, 'num') for k, v in arrays.items()}, params, set())
    for name, (func, arrays, params, _) in TYPE_CASES.items()
    if len(arrays) > 1 and name != 'irradiance.erbs'
}
FUNCTIONS.update({
    'irradiance.erbs': (
        irradiance.erbs,
        {'ghi': ([900., 700., 400., 100.], 'num'),
         'zenith': ([20., 40., 60., 80.], 'num'),
         'datetime_or_doy': (None, 'doy')}, {}, set()),
    'irradiance.get_extra_radiation': (
        irradiance.get_extra_radiation,
        {'datetime_or_doy': (None, 'doy'),
         'solar_constant': ([1361., 1366.1, 1367., 1370.], 'num')}, {},
        set()),
    # spa_python returns a DataFrame for any datetime-like time
    'solarposition.spa_python': (
        solarposition.spa_python,
        {'time': (None, 'unix'), 'latitude': ([30., 32., 34., 36.], 'num'),
         'longitude': ([-110., -111., -112., -113.], 'num')}, {},
        {'dt64'}),
    'soiling.hsu': (
        soiling.hsu,
        {'rainfall': ([0., 1.6, 0., 1.2], 'num1d'),
         'times': (None, 'unix1d'),
         'surface_tilt': ([10., 20., 30., 40.], 'num'),
         'pm2_5': ([1e-2, 1.5e-2, 2e-2, 2.5e-2], 'num'),
         'pm10': ([2e-2, 2.5e-2, 3e-2, 3.5e-2], 'num')},
        {'cleaning_threshold': 1.0}, set()),
})


def _pairwise(kinds):
    """Rows of kinds, one per parameter, such that every pair of kinds of
    every pair of parameters is in some row. Greedy and deterministic."""
    n = len(kinds)

    def pair(p, a, q, b):
        return (p, a, q, b) if p < q else (q, b, p, a)

    uncovered = {(p, a, q, b) for p in range(n) for q in range(p + 1, n)
                 for a in kinds[p] for b in kinds[q]}
    rows = []
    while uncovered:
        p, a, q, b = min(uncovered)
        row = [None] * n
        row[p], row[q] = a, b
        for r in range(n):
            if row[r] is None:
                row[r] = max(kinds[r], key=lambda k: sum(
                    pair(r, k, s, row[s]) in uncovered
                    for s in range(n) if row[s] is not None))
        rows.append(tuple(row))
        uncovered -= {(p, row[p], q, row[q])
                      for p in range(n) for q in range(p + 1, n)}
    return rows


def _calls():
    for name, (_, params, _, _) in FUNCTIONS.items():
        kinds = [KINDS[category] for _, category in params.values()]
        for row in _pairwise(kinds):
            label = ','.join(f'{p}={k}' for p, k in zip(params, row))
            yield pytest.param(name, dict(zip(params, row)),
                               id=f'{name}-{label}')


def _values(values, category):
    return TIME_VALUES[category] if values is None else np.asarray(values)


def _datetimes(kind, param):
    if kind == 'dtindex':
        return INDEX
    if kind == 'dt_series':
        return pd.Series(INDEX, index=INDEX, name=param)
    if kind == 'dt64':
        return UTC.values
    if kind == 'timestamp':
        return INDEX[0]
    xr = pytest.importorskip('xarray')
    return xr.DataArray(UTC.values, dims='time')


def _argument(param, kind, values, category):
    if kind in DATETIMES:
        return _datetimes(kind, param)
    return _convert(kind, param, list(_values(values, category)))


def _reference_argument(kind, values, category):
    # the same values as NumPy, numbers for times
    values = _values(values, category).astype(float)
    if kind in ('scalar', 'timestamp'):
        return float(values[0])
    if kind == 'masked':
        values = values.copy()
        values[MASKED] = np.nan
    return values


def _mismatched(name, kinds):
    # spa_python indexes its result by datetime64 time, naive UTC here,
    # which isn't the tz-aware index of the Series: pandas inputs with
    # different indexes raise
    return bool(FUNCTIONS[name][3] & set(kinds) and PANDAS & set(kinds)
                and not XARRAY & set(kinds))


def _expected_container(name, kinds):
    if XARRAY & set(kinds):
        return 'xarray'
    if (PANDAS | FUNCTIONS[name][3]) & set(kinds):
        return 'pandas'
    if 'masked' in kinds:
        return 'masked'
    if all(k in ('scalar', 'timestamp') for k in kinds):
        return 'scalar'
    return 'ndarray'


def _instants(index):
    # times as UTC instants; naive times are UTC
    return index.tz_localize('UTC') if index.tz is None \
        else index.tz_convert('UTC')


def _check_container(out, expected, several):
    import sys
    xr = sys.modules.get('xarray')
    if expected == 'xarray':
        assert isinstance(out, xr.Dataset if several else xr.DataArray), \
            type(out)
    elif expected == 'pandas':
        assert isinstance(out, pd.DataFrame if several else pd.Series), \
            type(out)
        assert _instants(out.index).equals(_instants(INDEX)), out.index
    else:
        if several:
            assert isinstance(out, dict), type(out)
        outputs = list(out.values()) if several else [out]
        if expected != 'scalar' and several:
            # as before the port, an output that depends on no array input
            # (poa_ground_diffuse for scalar ghi and tilt) stays a scalar
            outputs = [o for o in outputs if np.ndim(o) > 0]
            assert outputs, 'no array outputs'
        for o in outputs:
            if expected == 'masked':
                assert isinstance(o, np.ma.MaskedArray), type(o)
                assert np.ma.getmaskarray(o)[MASKED]
            elif expected == 'scalar':
                assert np.ndim(o) == 0 and not isinstance(o, np.ndarray), \
                    type(o)
            else:
                assert type(o) is np.ndarray, type(o)


def _as_numpy(x):
    if isinstance(x, np.ma.MaskedArray):
        return np.ma.filled(x.astype(float), np.nan)
    return np.asarray(getattr(x, 'values', x), dtype=float)


@pytest.mark.parametrize('name, kinds', list(_calls()))
def test_mixed_input_types(name, kinds):
    func, params, other, _ = FUNCTIONS[name]
    arguments = {p: _argument(p, kinds[p], values, category)
                 for p, (values, category) in params.items()}
    if _mismatched(name, kinds.values()):
        with pytest.raises(ValueError, match='different indexes'):
            func(**arguments, **other)
        return
    reference = func(**{p: _reference_argument(kinds[p], values, category)
                        for p, (values, category) in params.items()},
                     **other)
    out = func(**arguments, **other)

    several = isinstance(reference, dict)
    _check_container(out, _expected_container(name, kinds.values()), several)
    keys = list(reference) if several else [None]
    for key in keys:
        actual = _as_numpy(out[key] if several else out)
        expected = _as_numpy(reference[key] if several else reference)
        actual, expected = np.broadcast_arrays(actual, expected)
        if 'masked' in kinds.values() and actual.ndim:
            # the masked position's value is not part of the contract
            actual = np.delete(actual, MASKED, 0)
            expected = np.delete(expected, MASKED, 0)
        assert_allclose(actual, expected, rtol=1e-12, atol=1e-12,
                        equal_nan=True, err_msg=str(key))


@pytest.mark.parametrize('kind', ['ndarray', 'xarray'])
@pytest.mark.parametrize('name', list(FUNCTIONS))
def test_results_are_writable(name, kind):
    # results are new arrays, which can be changed in place
    func, params, other, _ = FUNCTIONS[name]
    time_kind = 'dt_xarray' if kind == 'xarray' else 'ndarray'
    out = func(**{p: _argument(p, time_kind if values is None else kind,
                               values, category)
                  for p, (values, category) in params.items()}, **other)
    if isinstance(out, dict):
        for k in out:
            out[k] += 0
    elif kind == 'xarray' and hasattr(out, 'data_vars'):
        for k in out.data_vars:
            out[k] += 0
    else:
        out += 0
