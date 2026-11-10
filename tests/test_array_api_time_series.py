"""
Time series functions, which step through time (soiling.hsu and the NREL
snow model), on N-D data from any library.

Time comes from one place for each input type: the index of Series, the
dimension of a DataArray of times, or a 1-D array of times along the last
axis of other arrays. See "Time series" in the user guide's array libraries
page, and labeled_io's coordinate.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pvlib import snow, soiling, tools

from . import test_array_api_time
from .test_array_api import _jax, _torch, _restore_jax_x64  # noqa: F401
from .test_array_api_time import _to_numpy

backend = test_array_api_time.backend

SITES = 3

# name: (function, time series parameters, per-site parameters, others)
FUNCTIONS = {
    'hsu': (soiling.hsu, ['rainfall'], ['surface_tilt', 'pm2_5', 'pm10'],
            {'cleaning_threshold': 0.5}),
    'coverage_nrel': (snow.coverage_nrel,
                      ['snowfall', 'poa_irradiance', 'temp_air',
                       'snow_depth'], ['surface_tilt'], {}),
    'fully_covered_nrel': (snow.fully_covered_nrel,
                           ['snowfall', 'snow_depth'], [], {}),
}


@pytest.fixture(scope='module')
def winter():
    """Hourly data with a gap for three sites, as (sites, time) arrays, and
    per-site values as (sites, 1)."""
    times = pd.date_range('2026-01-01', periods=24 * 20, freq='1h',
                          tz='America/Denver')
    times = times.delete(slice(100, 106))
    rng = np.random.default_rng(2)
    shape = (SITES, len(times))
    data = {
        'snowfall': rng.choice([0., 0., 0., 0., 0.5, 1.5, 3.], shape),
        'poa_irradiance': np.clip(rng.normal(250, 250, shape), 0, None),
        'temp_air': rng.normal(-3, 5, shape),
        'snow_depth': rng.choice([0.5, 5., 20.], shape),
        'rainfall': np.where(rng.uniform(size=shape) > 0.95, 1., 0.),
        'surface_tilt': np.array([[15.], [30.], [45.]]),
        'pm2_5': np.array([[1e-2], [2e-2], [3e-2]]),
        'pm10': np.array([[2e-2], [3e-2], [5e-2]]),
    }
    return times, data


def _per_site(name, times, data, **kwargs):
    """The result for each site from Series, as before the port."""
    func, series, site, other = FUNCTIONS[name]
    rows = []
    for i in range(SITES):
        args = {k: pd.Series(data[k][i], index=times) for k in series}
        args.update({k: float(data[k][i, 0]) for k in site})
        rows.append(func(**args, **other, **kwargs).to_numpy())
    return np.stack(rows).astype(float)


@pytest.mark.parametrize('name', list(FUNCTIONS))
def test_nd_arrays_time_last(backend, winter, name):
    # sites along the first axis, time along the last, as for any library
    _, mod, a = backend
    func, series, site, other = FUNCTIONS[name]
    times, data = winter
    out = func(**{k: a(data[k]) for k in series + site}, **other,
               times=a(tools.datetime_to_unix(times)))
    assert isinstance(out, type(a([0.])))
    assert_allclose(np.asarray(_to_numpy(out), float),
                    _per_site(name, times, data), atol=1e-12)


@pytest.mark.parametrize('name', list(FUNCTIONS))
def test_nd_numpy_with_datetimes(winter, name):
    func, series, site, other = FUNCTIONS[name]
    times, data = winter
    out = func(**{k: data[k] for k in series + site}, **other, times=times)
    assert type(out) is np.ndarray and out.shape == (SITES, len(times))
    assert_allclose(out.astype(float), _per_site(name, times, data),
                    atol=1e-12)


def _labeled(winter, name):
    # time is the first dimension of the data; the times DataArray, not
    # the position, says it is time
    xr = pytest.importorskip('xarray')
    times, data = winter
    func, series, site, other = FUNCTIONS[name]
    utc = times.tz_convert('UTC').tz_localize(None)
    coords = {'time': utc, 'site': ['a', 'b', 'c']}
    args = {k: xr.DataArray(data[k].T, dims=('time', 'site'), coords=coords)
            for k in series}
    args.update({k: xr.DataArray(data[k][:, 0], dims='site',
                                 coords={'site': coords['site']})
                 for k in site})
    time = xr.DataArray(utc, dims='time', coords={'time': utc})
    return func, args, other, time


@pytest.mark.parametrize('name', list(FUNCTIONS))
def test_xarray_time_by_dimension(winter, name):
    times, data = winter
    func, args, other, time = _labeled(winter, name)
    out = func(**args, **other, times=time)
    # the time dimension comes last, as for the core
    assert out.dims == ('site', 'time')
    assert out.indexes['time'].equals(time.indexes['time'])
    assert_allclose(out.values.astype(float), _per_site(name, times, data),
                    atol=1e-12)


def test_xarray_on_dask_stays_lazy(winter):
    pytest.importorskip('dask')
    times, data = winter
    func, args, other, time = _labeled(winter, 'coverage_nrel')
    args = {k: v.chunk({'site': 1}) for k, v in args.items()}
    out = func(**args, **other, times=time)
    assert out.chunks is not None
    assert_allclose(out.values, _per_site('coverage_nrel', times, data),
                    atol=1e-12)


def _mean_coverage_fd(winter, step=1e-6, **kwargs):
    """Finite differences of the mean coverage with respect to
    slide_amount_coefficient, on NumPy."""
    times, data = winter
    args = {k: data[k] for k in FUNCTIONS['coverage_nrel'][1]}

    def mean(coef):
        return np.mean(snow.coverage_nrel(
            **args, surface_tilt=data['surface_tilt'], times=times,
            slide_amount_coefficient=coef, **kwargs))

    return (mean(0.197 + step) - mean(0.197 - step)) / (2 * step)


def test_coverage_nrel_jax_jit_grad(winter):
    # fitting the slide coefficient to data needs this derivative
    jnp = _jax()
    import jax
    times, data = winter
    args = {k: jnp.asarray(data[k]) for k in FUNCTIONS['coverage_nrel'][1]}
    unix = jnp.asarray(tools.datetime_to_unix(times))

    def mean_coverage(coef):
        return jnp.mean(snow.coverage_nrel(
            **args, surface_tilt=jnp.asarray(data['surface_tilt']),
            slide_amount_coefficient=coef, times=unix))

    gradient = jax.jit(jax.grad(mean_coverage))(0.197)
    assert_allclose(float(gradient), _mean_coverage_fd(winter), rtol=1e-6)


def test_coverage_nrel_torch_gradient_per_site(winter):
    torch = _torch()
    times, data = winter
    args = {k: torch.asarray(data[k], dtype=torch.float64)
            for k in FUNCTIONS['coverage_nrel'][1]}
    tilt = torch.tensor(data['surface_tilt'], dtype=torch.float64,
                        requires_grad=True)
    unix = torch.asarray(tools.datetime_to_unix(times), dtype=torch.float64)
    snow.coverage_nrel(**args, surface_tilt=tilt, times=unix).sum().backward()
    step = 1e-5
    expected = [(_per_site('coverage_nrel', times, _tilted(data, i, step))
                 - _per_site('coverage_nrel', times,
                             _tilted(data, i, -step)))[i].sum() / (2 * step)
                for i in range(SITES)]
    assert_allclose(tilt.grad[:, 0].numpy(), expected, rtol=1e-6)


def _tilted(data, site, change):
    tilt = data['surface_tilt'].copy()
    tilt[site] += change
    return {**data, 'surface_tilt': tilt}


# --- the rule ---------------------------------------------------------------

def _series(winter, k='rainfall'):
    times, data = winter
    return pd.Series(data[k][0], index=times)


def _hsu(rainfall, **kwargs):
    return soiling.hsu(rainfall, 0.5, 20., 1e-2, 2e-2, **kwargs)


@pytest.mark.parametrize('same', ['UTC', 'Unix seconds', 'datetime64'])
def test_series_times_may_repeat_the_index(winter, same):
    times, _ = winter
    rain = _series(winter)
    given = {'UTC': times.tz_convert('UTC'),
             'Unix seconds': tools.datetime_to_unix(times),
             'datetime64': times.tz_convert('UTC').tz_localize(None).values}
    out = _hsu(rain, times=given[same])
    assert isinstance(out, pd.Series) and out.index.equals(times)
    assert_allclose(out, _hsu(rain))


@pytest.mark.parametrize('other', ['local wall clock', 'shifted', 'shorter'])
def test_series_times_must_equal_the_index(winter, other):
    times, _ = winter
    given = {
        # naive local times are taken as UTC: 7 hours off
        'local wall clock': times.tz_localize(None),
        'shifted': times + pd.Timedelta('1h'),
        'shorter': times[:-1],
    }
    with pytest.raises(ValueError, match='differs from the index'):
        _hsu(_series(winter), times=given[other])


@pytest.mark.parametrize('name', list(FUNCTIONS))
def test_arrays_need_times(winter, name):
    func, series, site, other = FUNCTIONS[name]
    _, data = winter
    with pytest.raises(ValueError, match='times is required'):
        func(**{k: data[k] for k in series + site}, **other)


def test_series_without_datetimeindex_need_times(winter):
    times, _ = winter
    rain = _series(winter).reset_index(drop=True)
    with pytest.raises(ValueError, match='times is required'):
        _hsu(rain)
    # numbers, since pandas inputs must have equal indexes
    out = _hsu(rain, times=tools.datetime_to_unix(times))
    assert out.index.equals(rain.index)
    assert_allclose(out, _hsu(_series(winter)))


def test_dataarrays_need_a_dataarray_of_times(winter):
    func, args, other, time = _labeled(winter, 'coverage_nrel')
    for times in (None, time.indexes['time'], time.values):
        with pytest.raises(ValueError, match='must be a 1-D DataArray'):
            func(**args, **other, times=times)


def test_dataarray_times_label_unlabeled_data(winter):
    # plain arrays broadcast by position against the times DataArray
    times, data = winter
    _, _, _, time = _labeled(winter, 'hsu')
    out = _hsu(data['rainfall'][0], times=time)
    assert out.dims == ('time',)
    assert_allclose(out.values, _hsu(_series(winter)).to_numpy())


def test_times_are_1d(winter):
    times, data = winter
    unix = tools.datetime_to_unix(times)
    with pytest.raises(ValueError, match='must be 1-D'):
        _hsu(data['rainfall'], times=np.broadcast_to(unix, (SITES,
                                                            len(unix))))


def test_time_is_the_last_axis(winter):
    times, data = winter
    with pytest.raises(ValueError, match='last axis'):
        _hsu(data['rainfall'].T, times=times)
    # values constant in time have a last axis of size 1
    with pytest.raises(ValueError, match='last axis'):
        soiling.hsu(data['rainfall'], 0.5, data['surface_tilt'][:, 0], 1e-2,
                    2e-2, times=times)


def test_fully_covered_needs_two_times():
    with pytest.raises(ValueError, match='at least two times'):
        snow.fully_covered_nrel([2.], times=[0.])
