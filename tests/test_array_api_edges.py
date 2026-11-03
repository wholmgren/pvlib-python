"""
Parts of the array API port's contract that the other tests leave out:
dtypes other than float64, the numba path with numeric time, datetimes given
where numbers are expected, and pandas inputs with different indexes.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pvlib import irradiance, pvsystem, soiling, solarposition, temperature
from pvlib import tools

from .conftest import requires_numba
from .test_input_types import GHI, INDEX, TILT, ZEN

LAT, LON = 32.2, -110.9
TIMES = pd.date_range('2026-06-01', periods=48, freq='1h', tz='UTC')
UNIX = np.asarray((TIMES.tz_localize(None) - pd.Timestamp('1970-01-01'))
                  // pd.Timedelta('1s'), dtype=np.int64)


def _rain():
    rain = np.zeros(48)
    rain[[10, 11]] = 0.6  # 1.2 mm in two hours
    return rain


def test_hsu_integer_times_with_float32_data():
    # integer Unix seconds are exact, so float32 data must not round them to
    # float32, whose resolution near 1.8e9 s is 128 s
    rain, pm = _rain(), np.full(48, 1e-2)
    expected = soiling.hsu(rain, 1.0, 20., pm, 2 * pm, times=UNIX,
                           rain_accum_period=3601.)
    out = soiling.hsu(rain.astype(np.float32), 1.0, 20.,
                      pm.astype(np.float32), 2 * pm.astype(np.float32),
                      times=UNIX, rain_accum_period=3601.)
    # the same cleanings ...
    np.testing.assert_array_equal(np.diff(out) > 0, np.diff(expected) > 0)
    # ... and the same intervals between times
    assert_allclose(out, expected, rtol=1e-6)


def test_hsu_rejects_float32_times():
    # as spa_python and tools.unix_to_doy do
    with pytest.raises(TypeError, match='float64'):
        soiling.hsu(_rain(), 1.0, 20., 1e-2, 2e-2,
                    times=UNIX.astype(np.float32))


@pytest.mark.parametrize('call', [
    lambda f32, f64: irradiance.aoi(f32(20.), f64(180.), f32(30.), f64(150.)),
    lambda f32, f64: pvsystem.pvwatts_dc(f32(800.), f32(30.), f64(250.),
                                         f64(-0.004)),
    lambda f32, f64: temperature.faiman(f32(800.), f32(20.), f64(2.)),
    lambda f32, f64: irradiance.get_ground_diffuse(f32(20.), f32(800.),
                                                   f64(0.2)),
], ids=['aoi', 'pvwatts_dc', 'faiman', 'get_ground_diffuse'])
@pytest.mark.parametrize('scalar', [float, np.float32, np.float64])
def test_result_dtype_follows_numpy_promotion(call, scalar):
    # float32 arrays with Python floats stay float32; with NumPy float64
    # scalars, NumPy 2 promotes to float64. The result has the type NumPy
    # arithmetic on the same inputs has
    inputs = []

    def f32(v):
        inputs.append(np.full(3, v, dtype=np.float32))
        return inputs[-1]

    def f64(v):
        inputs.append(scalar(v))
        return inputs[-1]

    out = call(f32, f64)
    assert out.dtype == np.result_type(*inputs)


@requires_numba
@pytest.mark.parametrize('time', [UNIX[12].astype(float),
                                  UNIX[:3].astype(float)],
                         ids=['scalar', 'array'])
def test_spa_python_numba_numeric_time(time):
    expected = solarposition.spa_python(time, LAT, LON, how='numpy')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # reloading spa warns
        try:
            out = solarposition.spa_python(time, LAT, LON, how='numba')
        finally:
            solarposition._spa_python_import('numpy')
    for key in expected:
        assert_allclose(out[key], expected[key], rtol=1e-10, atol=1e-10,
                        err_msg=key)


@pytest.mark.parametrize('times', [
    TIMES[:3], TIMES[:3].tz_localize(None).values, pd.Series(TIMES[:3]),
    TIMES[0]], ids=['DatetimeIndex', 'datetime64', 'Series', 'Timestamp'])
def test_unix_to_doy_rejects_datetimes(times):
    # it takes numbers; datetimes would be read as nanoseconds
    with pytest.raises((TypeError, ValueError)):
        tools.unix_to_doy(times)


# pandas inputs are unwrapped to arrays, so their indexes must be the same:
# pvlib doesn't align them as pandas arithmetic would
MISMATCHED = {
    'shifted': (INDEX + pd.Timedelta('3h'), TILT),
    'reordered': (INDEX[[1, 0, 2, 3]], TILT),
    'subset': (INDEX[:3], TILT[:3]),
    # pandas aligns these by instant, but the indexes differ
    'other time zone': (INDEX.tz_convert('UTC'), TILT),
}


@pytest.mark.parametrize('index, values', MISMATCHED.values(),
                         ids=MISMATCHED.keys())
def test_series_with_different_indexes_raise(index, values):
    with pytest.raises(ValueError, match='index'):
        irradiance.aoi(pd.Series(values, index), 180., pd.Series(ZEN, INDEX),
                       180.)
    with pytest.raises(ValueError, match='index'):
        irradiance.get_total_irradiance(
            pd.Series(values, index), 180., pd.Series(ZEN, INDEX), 180.,
            pd.Series(GHI, INDEX), pd.Series(GHI, INDEX),
            pd.Series(GHI, INDEX))


def test_series_and_datetimeindex_with_different_times_raise():
    with pytest.raises(ValueError, match='index'):
        irradiance.erbs(pd.Series(GHI, INDEX), pd.Series(ZEN, INDEX),
                        INDEX + pd.Timedelta('3h'))


def test_series_with_equal_indexes_work():
    # equal indexes need not be the same object, as for Series from
    # different DataFrames
    out = irradiance.aoi(pd.Series(TILT, INDEX.copy()), 180.,
                         pd.Series(ZEN, INDEX.copy()), 180.)
    assert out.index.equals(INDEX)
