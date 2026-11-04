"""
Run array API ported functions on several array libraries and check that the
array type comes back unchanged and that results match NumPy.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import pvlib
from pvlib import (clearsky, iam, inverter, irradiance, pvsystem,
                   temperature)

from ._supported_functions import SUPPORTED, function, qualified_name


@pytest.fixture(autouse=True)
def _restore_jax_x64():
    # _jax enables float64, a process-wide JAX setting; put it back so that
    # later tests see JAX's default
    jax = sys.modules.get('jax')
    before = bool(jax.config.jax_enable_x64) if jax else False
    yield
    jax = sys.modules.get('jax')
    if jax is not None:
        jax.config.update('jax_enable_x64', before)


def _jax():
    jax = pytest.importorskip('jax')
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    return jnp


def _torch():
    return pytest.importorskip('torch')


def _strict():
    return pytest.importorskip('array_api_strict')


def _dask():
    pytest.importorskip('dask')
    import dask.array as da
    return da


BACKENDS = {
    'array_api_strict': lambda: (_strict(), lambda xp, a: xp.asarray(a)),
    'torch': lambda: (_torch(),
                      lambda xp, a: xp.asarray(a, dtype=xp.float64)),
    'jax': lambda: (_jax(), lambda xp, a: xp.asarray(a)),
    'dask': lambda: (_dask(), lambda xp, a: xp.from_array(a, chunks=2)),
}


@pytest.fixture(params=list(BACKENDS))
def backend(request):
    mod, convert = BACKENDS[request.param]()
    return request.param, mod, lambda a: convert(mod, np.asarray(a, float))


def _to_numpy(x):
    if hasattr(x, 'compute'):
        x = x.compute()
    return np.asarray(x)


ZEN = [10., 30., 50., 70., 89., 95.]
AZI = [90., 135., 180., 225., 270., 300.]
DNI = [900., 700., 500., 200., 10., 0.]
GHI = [800., 600., 400., 200., 50., 0.]
DHI = [100., 120., 150., 100., 40., 0.]
TAIR = [25., 30., 15., 5., 35., 20.]
WIND = [1., 2., 3., 4., 5., 0.]


@pytest.mark.parametrize('case', list(SUPPORTED))
def test_type_in_type_out(backend, case):
    # every supported function, with the inputs of its entry
    name, mod, a = backend
    entry, func = SUPPORTED[case], function(case)
    expected = func(**{k: np.asarray(v) for k, v in entry.arrays.items()},
                    **entry.params)
    out = func(**{k: a(v) for k, v in entry.arrays.items()}, **entry.params)
    if isinstance(expected, dict):
        assert out.keys() == expected.keys()
        pairs = [(out[k], expected[k]) for k in expected]
    else:
        pairs = [(out, expected)]
    for actual, values in pairs:
        assert isinstance(actual, type(a([1.])))
        assert_allclose(_to_numpy(actual), values, equal_nan=True)


def test_dask_stays_lazy():
    da = _dask()
    zen, azi, dni, ghi, dhi = (da.from_array(np.asarray(v), chunks=2)
                               for v in (ZEN, AZI, DNI, GHI, DHI))
    out = irradiance.get_total_irradiance(30, 180, zen, azi, dni, ghi, dhi)
    assert all(isinstance(v, da.Array) for v in out.values())


def test_dask_and_numpy_arrays_mix():
    # Dask arrays are NumPy arrays computed lazily, so NumPy arrays join in
    da = _dask()
    zen = da.from_array(np.asarray(ZEN), chunks=2)
    out = irradiance.aoi(30, 180, zen, np.asarray(AZI))
    assert isinstance(out, da.Array)
    assert_allclose(out.compute(), irradiance.aoi(30, 180, ZEN, AZI))


def test_mixed_libraries_raise():
    torch = _torch()
    with pytest.raises(TypeError, match='Multiple namespaces'):
        irradiance.aoi(30, 180, torch.asarray(ZEN, dtype=torch.float64),
                       np.asarray(AZI))


def test_pandas_unchanged():
    index = pd.date_range('2024-06-01 06:00', periods=len(ZEN), freq='2h')
    zen = pd.Series(ZEN, index=index)
    azi = pd.Series(AZI, index=index)
    out = irradiance.aoi(30, 180, zen, azi)
    assert isinstance(out, pd.Series) and out.name == 'aoi'
    out = irradiance.get_total_irradiance(30, 180, zen, azi, DNI, GHI, DHI)
    assert isinstance(out, pd.DataFrame)
    assert out.index.equals(index)


def test_xarray_broadcasts_by_dimension():
    xr = pytest.importorskip('xarray')
    zen = xr.DataArray(ZEN, dims='time')
    azi = xr.DataArray(AZI, dims='time')
    tilt = xr.DataArray([0., 20., 40.], dims='tilt')
    out = irradiance.get_total_irradiance(
        tilt, 180, zen, azi, xr.DataArray(DNI, dims='time'),
        xr.DataArray(GHI, dims='time'), xr.DataArray(DHI, dims='time'))
    assert isinstance(out, xr.Dataset)
    assert out['poa_global'].dims == ('tilt', 'time')
    assert_allclose(out['poa_global'].isel(tilt=2),
                    irradiance.get_total_irradiance(
                        40., 180, ZEN, AZI, DNI, GHI, DHI)['poa_global'])


def _ac_energy(xp, tilt, zen, azi, dni, ghi, dhi, tair, wind):
    poa = irradiance.get_total_irradiance(
        tilt, 180., zen, azi, dni, ghi, dhi, dni_extra=1360.,
        model='haydavies')
    aoi = irradiance.aoi(tilt, 180., zen, azi)
    eff = poa['poa_direct'] * iam.martin_ruiz(aoi) + poa['poa_diffuse']
    tcell = temperature.faiman(poa['poa_global'], tair, wind)
    pdc = pvsystem.pvwatts_dc(eff, tcell, 250., -0.004)
    pac = inverter.pvwatts(pdc, 240.)
    return xp.sum(pac)


def _ac_energy_numpy(tilt):
    return _ac_energy(np, tilt, *(np.asarray(v) for v in (
        ZEN[:-1], AZI[:-1], DNI[:-1], GHI[:-1], DHI[:-1], TAIR[:-1],
        WIND[:-1])))


def test_torch_gradient_wrt_tilt():
    torch = _torch()
    args = [torch.asarray(v[:-1], dtype=torch.float64)
            for v in (ZEN, AZI, DNI, GHI, DHI, TAIR, WIND)]
    tilt = torch.tensor(30., dtype=torch.float64, requires_grad=True)
    energy = _ac_energy(torch, tilt, *args)
    energy.backward()
    fd = (_ac_energy_numpy(30.001) - _ac_energy_numpy(29.999)) / 0.002
    assert_allclose(tilt.grad.item(), fd, rtol=1e-5)


def test_jax_jit_and_grad_wrt_tilt():
    jnp = _jax()
    import jax
    args = [jnp.asarray(v[:-1]) for v in (ZEN, AZI, DNI, GHI, DHI, TAIR,
                                          WIND)]
    f = jax.jit(jax.grad(lambda t: _ac_energy(jnp, t, *args)))
    fd = (_ac_energy_numpy(30.001) - _ac_energy_numpy(29.999)) / 0.002
    assert_allclose(f(30.), fd, rtol=1e-5)


def test_torch_inverter_gradient_finite_at_zero_power():
    # np.divide(..., where=) has no array API equivalent; apply_where must
    # keep the 1/0 branch out of the derivative as well as the value
    torch = _torch()
    pdc = torch.tensor([0., 50., 100.], dtype=torch.float64,
                       requires_grad=True)
    inverter.pvwatts(pdc, 120.).sum().backward()
    assert torch.isfinite(pdc.grad).all()


def test_torch_haurwitz_gradient_finite_below_horizon():
    torch = _torch()
    zenith = torch.tensor([30., 90., 120.], dtype=torch.float64,
                          requires_grad=True)
    clearsky.haurwitz(zenith)['ghi'].sum().backward()
    assert torch.isfinite(zenith.grad).all()
    assert zenith.grad[1:].eq(0).all()


def test_0d_arrays_move_to_the_device_of_the_data():
    # a 0-d CPU tensor (torch.tensor(30.)) must not pull data off its device;
    # PyTorch's meta device stands in for a GPU here
    torch = _torch()
    zen = torch.empty(3, dtype=torch.float64, device='meta')
    out = irradiance.aoi(torch.tensor(30., dtype=torch.float64), 180., zen,
                         torch.empty(3, dtype=torch.float64, device='meta'))
    assert out.device.type == 'meta'


def test_arrays_on_different_devices_raise():
    torch = _torch()
    zen = torch.empty(3, dtype=torch.float64, device='meta')
    azi = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match='different devices: meta, cpu'):
        irradiance.aoi(30., 180., zen, azi)


def test_erf_uses_scipy_special_when_scipy_array_api_is_on():
    # SciPy reads SCIPY_ARRAY_API when it is first imported, so run in a new
    # process, with the pvlib being tested
    torch = _torch()  # noqa: F841
    code = '\n'.join([
        'from unittest import mock',
        'import scipy.special, torch',
        'from pvlib import _array_api as _xa',
        'assert _xa.scipy_array_api()',
        'x = torch.tensor([0.1, 0.5], dtype=torch.float64,'
        ' requires_grad=True)',
        'with mock.patch.object(scipy.special, "erf",'
        ' wraps=scipy.special.erf) as erf:',
        '    y = _xa.erf(x, _xa.array_namespace(x))',
        'assert erf.call_count == 1',
        'y.sum().backward()',
        'assert torch.allclose(x.grad, 2 / torch.pi**0.5 * torch.exp(-x**2))',
    ])
    root = os.path.dirname(os.path.dirname(pvlib.__file__))
    env = {**os.environ, 'SCIPY_ARRAY_API': '1',
           'PYTHONPATH': os.pathsep.join(
               [root, os.environ.get('PYTHONPATH', '')])}
    subprocess.run([sys.executable, '-c', code], env=env, check=True)


def test_martin_ruiz_validates_eager_jax_but_not_traced():
    jnp = _jax()
    import jax
    with pytest.raises(ValueError, match='a_r'):
        iam.martin_ruiz(jnp.asarray([10.]), a_r=jnp.asarray(-1.))
    # inside jit the values are unknown, so the check is skipped
    jax.jit(lambda a_r: iam.martin_ruiz(jnp.asarray([10.]), a_r))(
        jnp.asarray(0.16))


def test_xarray_unlabeled_arrays_broadcast_by_position():
    xr = pytest.importorskip('xarray')
    zen = xr.DataArray(ZEN, dims='time')
    out = irradiance.aoi(30, 180, zen, np.asarray(AZI))
    assert out.dims == ('time',)
    assert_allclose(out, irradiance.aoi(30, 180, ZEN, AZI))
    # as in xarray arithmetic, an unlabeled array may not add a dimension,
    # which the result would have no name for
    with pytest.raises(ValueError, match='Pass a DataArray'):
        irradiance.aoi(np.array([[0.], [20.]]), 180, zen, 180)


def test_xarray_joins_like_xarray_arithmetic():
    xr = pytest.importorskip('xarray')
    zen = xr.DataArray([10., 20., 30.], dims='t', coords={'t': [0, 1, 2]})
    azi = xr.DataArray([90., 100., 110.], dims='t', coords={'t': [1, 2, 3]})
    out = irradiance.aoi(30., 180., zen, azi)
    assert list(out['t'].values) == list((zen + azi)['t'].values) == [1, 2]


@pytest.mark.parametrize('kind', ['scalar', 'series', 'torch'])
def test_nested_calls_find_the_namespace_once(monkeypatch, kind):
    # get_total_irradiance calls get_sky_diffuse, haydavies, aoi_projection,
    # get_ground_diffuse, aoi and poa_components through their cores, so the
    # boundary's work is done once per call
    from pvlib import _array_api
    if kind == 'torch':
        torch = _torch()

        def a(v):
            return torch.asarray(v, dtype=torch.float64)
    elif kind == 'series':
        def a(v):
            return pd.Series([v, v])
    else:
        def a(v):
            return v
    calls = []
    namespace = _array_api.array_namespace
    monkeypatch.setattr(_array_api, 'array_namespace',
                        lambda *xs: calls.append(xs) or namespace(*xs))
    irradiance.get_total_irradiance(
        a(30.), 180., a(20.), a(160.), a(800.), a(900.), a(100.),
        dni_extra=1360., model='haydavies')
    # calls with only numbers take a fast path that needs no lookup
    assert len(calls) == (0 if kind == 'scalar' else 1)


def test_scalar_fast_path_matches_full_boundary():
    # np.int64 isn't on the fast path, so the same values go through the
    # full boundary, in float64 since no input is floating point but float64
    kwargs = dict(surface_tilt=30., surface_azimuth=180., solar_zenith=40.,
                  solar_azimuth=150., dni=800., ghi=900., dhi=100.,
                  dni_extra=1360., model='haydavies')
    fast = irradiance.get_total_irradiance(**kwargs)
    full = irradiance.get_total_irradiance(
        **{**kwargs, 'surface_tilt': np.int64(30)})
    assert fast.keys() == full.keys()
    for k in fast:
        assert type(fast[k]) is np.float64
        assert fast[k] == full[k]


def test_public_signature_hides_xp():
    import inspect
    assert 'xp' not in inspect.signature(irradiance.aoi).parameters
    assert 'xp' in inspect.signature(irradiance.aoi.core).parameters
    with pytest.raises(TypeError, match='xp'):
        irradiance.aoi(30., 180., 20., 160., xp=np)


def test_spa_formula_numpy_names_have_array_api_equivalents():
    # spa formulas are written with np. for numba; every np name they use
    # must resolve on a strictly conforming namespace
    import ast
    import inspect
    from pvlib import spa
    xps = pytest.importorskip('array_api_strict')
    names = spa._NumpyNames(xps)
    for formula in spa._FORMULAS:
        func = getattr(getattr(spa, formula), 'py_func', getattr(spa, formula))
        for node in ast.walk(ast.parse(inspect.getsource(func))):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == 'np'):
                assert callable(getattr(names, node.attr)), (formula,
                                                             node.attr)


def _labeled_io_functions():
    """The public pvlib functions made with labeled_io."""
    import importlib
    import inspect
    import pkgutil
    found = set()
    for info in pkgutil.walk_packages(pvlib.__path__, 'pvlib.'):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # optional dependencies
            continue
        for name, obj in vars(module).items():
            if (inspect.isfunction(obj) and hasattr(obj, 'core')
                    and not name.startswith('_')
                    and obj.__module__ == module.__name__):
                found.add(f'{module.__name__}.{name}')
    return found


def test_every_labeled_io_function_has_an_entry():
    # an entry runs the function through the array library tests
    missing = _labeled_io_functions() - {qualified_name(k) for k in SUPPORTED}
    assert not missing, f'add entries to _supported_functions: {missing}'


def test_user_guide_lists_the_supported_functions():
    import pathlib
    import re
    guide = (pathlib.Path(__file__).parents[1] / 'docs' / 'sphinx' / 'source'
             / 'user_guide' / 'modeling_topics' / 'array_libraries.rst')
    if not guide.exists():
        pytest.skip('the docs are not here')
    section = guide.read_text().split('\nSupported functions\n')[1]
    section = section.split('\nOther functions take')[0]
    listed = set(re.findall(r':py:func:`~?(pvlib\.[\w.]+)`', section))
    assert listed == {qualified_name(k) for k in SUPPORTED}
