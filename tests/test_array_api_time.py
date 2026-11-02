"""
Time series modeling on array API libraries: time enters the array core as
float64 seconds since 1970-01-01 UTC (or day of year), and pandas and xarray
times are converted at the function boundary.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pvlib import (_array_api as _xa, iam, inverter, irradiance, pvsystem,
                   soiling, solarposition, temperature, tools)

from .test_array_api import (BACKENDS, _jax, _torch,  # noqa: F401
                             _restore_jax_x64)


LAT, LON = 32.2, -110.9


@pytest.fixture(params=list(BACKENDS))
def backend(request):
    mod, convert = BACKENDS[request.param]()
    if request.param == 'dask':
        # time series are long; tiny chunks make SPA's task graph huge
        def convert(xp, a):
            return xp.from_array(a, chunks=128)
    return request.param, mod, lambda a: convert(mod, np.asarray(a, float))


def _to_numpy(x):
    if hasattr(x, 'compute'):
        x = x.compute()
    return np.asarray(x)


@pytest.fixture(scope='module')
def weather():
    # irregular sampling: 15 minute steps with a gap, as in field data
    times = pd.date_range('2026-06-01', '2026-06-08', freq='15min',
                          tz='America/Phoenix', inclusive='left')
    times = times.delete(slice(200, 230))
    position = solarposition.spa_python(times, LAT, LON)
    cosz = np.cos(np.radians(position['apparent_zenith'].values))
    rng = np.random.default_rng(0)
    clear = np.where(cosz > 0, 1098 * cosz * np.exp(-0.057 / cosz), 0.)
    ghi = clear * rng.uniform(0.3, 1.0, len(times))
    rain = np.where(rng.uniform(size=len(times)) > 0.97, 1.0, 0.)
    return pd.DataFrame({'ghi': ghi, 'temp_air': 30.
                         + 8 * rng.uniform(size=len(times)),
                         'wind_speed': rng.uniform(0, 6, len(times)),
                         'rain': rain}, index=times)


def test_unix_to_civil(backend):
    name, mod, a = backend
    times = pd.date_range('1899-12-30', '2101-01-02', freq='37h')
    t = tools.datetime_to_unix(times)
    xp = _xa.array_namespace(a(t))
    fields = [_to_numpy(f) for f in _xa.unix_to_civil(a(t), xp)]
    expected = [times.year, times.month, times.day, times.day_of_year]
    for actual, exp in zip(fields, expected):
        np.testing.assert_array_equal(actual, exp)


def test_unix_to_doy(backend):
    name, mod, a = backend
    times = pd.date_range('1899-12-30', '2101-01-02', freq='37h', tz='UTC')
    out = tools.unix_to_doy(a(tools.datetime_to_unix(times)))
    assert type(out) is type(a([0.]))
    np.testing.assert_array_equal(_to_numpy(out), times.day_of_year)


def test_unix_to_doy_rejects_float32():
    torch = _torch()
    with pytest.raises(TypeError, match='float64'):
        tools.unix_to_doy(torch.asarray([1.78e9], dtype=torch.float32))


def test_spa_python_numeric_time(backend, weather):
    name, mod, a = backend
    times = weather.index
    expected = solarposition.spa_python(times, LAT, LON, delta_t=None)
    unix = a(tools.datetime_to_unix(times))
    out = solarposition.spa_python(unix, LAT, LON, delta_t=None)
    assert type(out['zenith']) is type(unix)
    for key in expected:
        assert_allclose(_to_numpy(out[key]), expected[key], atol=1e-9)


def test_spa_python_rejects_float32():
    torch = _torch()
    unix = torch.asarray([1.78e9], dtype=torch.float32)
    with pytest.raises(TypeError, match='float64'):
        solarposition.spa_python(unix, LAT, LON)


def test_spa_python_xarray_grid():
    xr = pytest.importorskip('xarray')
    times = pd.date_range('2026-03-20', periods=48, freq='1h')
    time = xr.DataArray(times, dims='time', coords={'time': times})
    lat = xr.DataArray(np.linspace(30., 40., 5), dims='lat')
    lon = xr.DataArray(np.linspace(-115., -105., 4), dims='lon')
    out = solarposition.spa_python(time, lat, lon)
    assert isinstance(out, xr.Dataset)
    assert out['zenith'].dims == ('time', 'lat', 'lon')
    assert out.indexes['time'].equals(times)
    expected = solarposition.spa_python(times.tz_localize('UTC'),
                                        float(lat[2]), float(lon[1]))
    assert_allclose(out['zenith'].isel(lat=2, lon=1), expected['zenith'])
    assert_allclose(out['equation_of_time'].isel(lat=0, lon=3),
                    expected['equation_of_time'])


def test_spa_python_xarray_grid_on_dask():
    # time is converted to a NumPy array at the boundary and joins the Dask
    # arrays of the grid
    xr = pytest.importorskip('xarray')
    pytest.importorskip('dask')
    import dask.array as da
    times = pd.date_range('2026-03-20', periods=48, freq='1h', tz='UTC')
    time = xr.DataArray(times, dims='time', coords={'time': times})
    lat = xr.DataArray(np.linspace(30., 40., 5), dims='lat')
    lon = xr.DataArray(np.linspace(-115., -105., 4), dims='lon')
    expected = solarposition.spa_python(time, lat, lon)
    out = solarposition.spa_python(time, lat.chunk(2), lon.chunk(2))
    assert isinstance(out['zenith'].data, da.Array)
    assert_allclose(out['zenith'], expected['zenith'])


def test_erbs_day_of_year(backend, weather):
    name, mod, a = backend
    times = weather.index
    zenith = solarposition.spa_python(times, LAT, LON)['zenith']
    expected = irradiance.erbs(weather['ghi'], zenith, times)
    unix = a(tools.datetime_to_unix(times))
    doy = tools.unix_to_doy(unix)
    out = irradiance.erbs(a(weather['ghi']), a(zenith), doy)
    for key in expected:
        assert_allclose(_to_numpy(out[key]), expected[key])


def test_hsu_irregular_times(backend, weather):
    name, mod, a = backend
    rain = weather['rain']
    expected = soiling.hsu(rain, 0.5, 20., 1e-2, 2e-2,
                           rain_accum_period=pd.Timedelta('2h'))
    unix = a(tools.datetime_to_unix(rain.index))
    out = soiling.hsu(a(rain), 0.5, 20., 1e-2, 2e-2, rain_accum_period=7200.,
                      times=unix)
    assert_allclose(_to_numpy(out), expected)


def _ac_power(times, doy, ghi, temp_air, wind_speed, rain, surface_tilt,
              rain_times=None):
    """Times to AC power, written once for pandas and array inputs."""
    position = solarposition.spa_python(times, LAT, LON)
    zenith, azimuth = position['apparent_zenith'], position['azimuth']
    dni_extra = irradiance.get_extra_radiation(doy)
    decomposed = irradiance.erbs(ghi, zenith, doy)
    poa = irradiance.get_total_irradiance(
        surface_tilt, 180., zenith, azimuth, decomposed['dni'], ghi,
        decomposed['dhi'], dni_extra=dni_extra, model='haydavies')
    aoi = irradiance.aoi(surface_tilt, 180., zenith, azimuth)
    soiling_ratio = soiling.hsu(rain, 0.5, surface_tilt, 1e-2, 2e-2,
                                times=rain_times)
    effective = soiling_ratio * (poa['poa_direct'] * iam.martin_ruiz(aoi)
                                 + poa['poa_diffuse'])
    temp_cell = temperature.faiman(poa['poa_global'], temp_air, wind_speed)
    pdc = pvsystem.pvwatts_dc(effective, temp_cell, 250., -0.004)
    return inverter.pvwatts(pdc, 240.)


def _ac_power_pandas(weather, surface_tilt):
    return _ac_power(weather.index, weather.index, weather['ghi'],
                     weather['temp_air'], weather['wind_speed'],
                     weather['rain'], surface_tilt)


def _ac_power_arrays(weather, surface_tilt, a):
    unix = a(tools.datetime_to_unix(weather.index))
    doy = tools.unix_to_doy(unix)
    return _ac_power(unix, doy, a(weather['ghi']), a(weather['temp_air']),
                     a(weather['wind_speed']), a(weather['rain']),
                     surface_tilt, rain_times=unix)


def test_time_series_chain(backend, weather):
    name, mod, a = backend
    expected = _ac_power_pandas(weather, 25.)
    assert isinstance(expected, pd.Series)
    assert expected.index.equals(weather.index)
    out = _ac_power_arrays(weather, 25., a)
    assert type(out) is type(a([0.]))
    assert_allclose(_to_numpy(out), expected, atol=1e-9)


def _energy_derivative_fd(weather):
    return (_ac_power_pandas(weather, 25.001).sum()
            - _ac_power_pandas(weather, 24.999).sum()) / 0.002


def test_time_series_chain_torch_gradient(weather):
    torch = _torch()

    def a(v):
        return torch.asarray(np.asarray(v, float), dtype=torch.float64)

    tilt = torch.tensor(25., dtype=torch.float64, requires_grad=True)
    _ac_power_arrays(weather, tilt, a).sum().backward()
    assert_allclose(tilt.grad.item(), _energy_derivative_fd(weather),
                    rtol=1e-5)


def test_time_series_chain_jax_jit_grad(weather):
    jnp = _jax()
    import jax

    def a(v):
        return jnp.asarray(np.asarray(v, float))

    energy = jax.jit(jax.grad(
        lambda tilt: jnp.sum(_ac_power_arrays(weather, tilt, a))))
    assert_allclose(energy(25.), _energy_derivative_fd(weather), rtol=1e-5)


def test_jax_jit_with_arrays_made_outside(weather):
    # inside jit, anything computed from arrays made outside it is traced
    # too, so input checks must skip rather than read the values
    jnp = _jax()
    import jax
    unix = jnp.asarray(tools.datetime_to_unix(weather.index))
    rain = jnp.asarray(weather['rain'].to_numpy())

    @jax.jit
    def f(latitude):
        zenith = solarposition.spa_python(unix, latitude, LON)['zenith']
        ratio = soiling.hsu(rain, 0.5, 20., 1e-2, 2e-2, times=unix)
        return zenith, ratio

    zenith, ratio = f(LAT)
    expected = solarposition.spa_python(weather.index, LAT, LON)['zenith']
    assert_allclose(np.asarray(zenith), expected, atol=1e-9)
    assert_allclose(np.asarray(ratio),
                    soiling.hsu(weather['rain'], 0.5, 20., 1e-2, 2e-2))


def test_spa_python_dask_stays_lazy(weather):
    da = pytest.importorskip('dask.array')
    unix = da.from_array(tools.datetime_to_unix(weather.index),
                         chunks=100)
    out = solarposition.spa_python(unix, LAT, LON)
    assert all(isinstance(v, da.Array) for v in out.values())


def test_hsu_ties_decided_the_same_on_every_library():
    # rain quantized to 0.1 mm gives window sums equal to the threshold;
    # whether such a tie cleans must not depend on summation rounding
    import math
    rng = np.random.default_rng(1)
    times = pd.date_range('2020-01-01', periods=20000, freq='1h')
    rain = pd.Series(np.round(rng.uniform(0, 1, len(times))
                              * (rng.uniform(size=len(times)) > 0.8), 1),
                     index=times)
    t = tools.datetime_to_unix(times)
    start = np.searchsorted(t, t - 3 * 3600., side='right')
    window_sums = [math.fsum(rain.values[s:i + 1])
                   for i, s in enumerate(start)]
    expected_cleaning = np.array(window_sums) >= 0.5 - 1e-9
    for name, backend_factory in BACKENDS.items():
        if name == 'dask':
            continue  # not needed to show the point, and slow here
        mod, convert = backend_factory()
        ratio = soiling.hsu(convert(mod, rain.to_numpy()), 0.5, 20., 1e-2,
                            2e-2, rain_accum_period=3 * 3600.,
                            times=convert(mod, t))
        cleaned = _to_numpy(ratio) == 1.0  # all mass removed at cleanings
        np.testing.assert_array_equal(cleaned, expected_cleaning,
                                      err_msg=name)


def test_hsu_rejects_unsorted_times(weather):
    with pytest.raises(ValueError, match='increasing'):
        soiling.hsu(weather['rain'].iloc[::-1], 0.5, 20., 1e-2, 2e-2)


def test_hsu_needs_times():
    rain = pd.Series([0., 1., 0.])  # RangeIndex: no times to take
    with pytest.raises(ValueError, match='times is required'):
        soiling.hsu(rain, 0.5, 20., 1e-2, 2e-2)
    with pytest.raises(ValueError, match='times is required'):
        soiling.hsu(rain.to_numpy(), 0.5, 20., 1e-2, 2e-2)


def test_tz_aware_datetime_series_as_time():
    times = pd.date_range('2026-12-31 20:00', periods=6, freq='1h',
                          tz='America/Phoenix')
    expected = irradiance.get_extra_radiation(times)
    out = irradiance.get_extra_radiation(pd.Series(times, name='t'))
    assert_allclose(out.to_numpy(), expected.to_numpy())


def test_spa_python_numeric_scalar_is_unix_seconds():
    t = pd.Timestamp('2026-06-01 19:00', tz='UTC')
    expected = solarposition.spa_python(t, LAT, LON)
    out = solarposition.spa_python(tools.datetime_to_unix(t), LAT,
                                   LON)
    assert_allclose(float(out['zenith']), expected['zenith'].iloc[0])


def test_spa_python_rejects_nanoseconds():
    times = pd.date_range('2026-06-01', periods=3, freq='1h')
    with pytest.raises(ValueError, match='seconds since 1970'):
        solarposition.spa_python(times.asi8, LAT, LON)
