"""
Run array API ported functions on several array libraries and check that the
array type comes back unchanged and that results match NumPy.
"""
import sys

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pvlib import (atmosphere, iam, inverter, irradiance, pvsystem,
                   temperature, tools)


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

CASES = {
    'cosd': lambda a: tools.cosd(a(ZEN)),
    'acosd': lambda a: tools.acosd(a([0., 0.5, 1.])),
    'aoi': lambda a: irradiance.aoi(30, 180, a(ZEN), a(AZI)),
    'beam_component': lambda a: irradiance.beam_component(
        30, 180, a(ZEN), a(AZI), a(DNI)),
    'haydavies': lambda a: irradiance.haydavies(
        30, 180, a(DHI), a(DNI), 1360., a(ZEN), a(AZI)),
    'get_total_irradiance': lambda a: irradiance.get_total_irradiance(
        30, 180, a(ZEN), a(AZI), a(DNI), a(GHI), a(DHI))['poa_global'],
    'get_relative_airmass': lambda a: atmosphere.get_relative_airmass(
        a(ZEN)),
    'ashrae': lambda a: iam.ashrae(a(ZEN)),
    'martin_ruiz': lambda a: iam.martin_ruiz(a(ZEN)),
    'sapm_cell': lambda a: temperature.sapm_cell(
        a(GHI), a(TAIR), a(WIND), -3.56, -0.075, 3),
    'faiman': lambda a: temperature.faiman(a(GHI), a(TAIR), a(WIND)),
    'pvwatts_dc': lambda a: pvsystem.pvwatts_dc(
        a(GHI), a(TAIR), 250., -0.004, k=0.01),
    'inverter.pvwatts': lambda a: inverter.pvwatts(a(GHI) * 0.25, 120.),
}


@pytest.mark.parametrize('case', list(CASES))
def test_type_in_type_out(backend, case):
    name, mod, a = backend
    expected = CASES[case](np.asarray)
    out = CASES[case](a)
    assert type(out) is type(a([1.]))
    assert_allclose(_to_numpy(out), expected, equal_nan=True)


def test_dask_stays_lazy():
    da = _dask()
    zen, azi, dni, ghi, dhi = (da.from_array(np.asarray(v), chunks=2)
                               for v in (ZEN, AZI, DNI, GHI, DHI))
    out = irradiance.get_total_irradiance(30, 180, zen, azi, dni, ghi, dhi)
    assert all(isinstance(v, da.Array) for v in out.values())


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


def test_device_follows_largest_array():
    # a 0-d CPU tensor (torch.tensor(30.)) must not pull data off its device;
    # PyTorch's meta device stands in for a GPU here
    torch = _torch()
    zen = torch.empty(3, dtype=torch.float64, device='meta')
    out = irradiance.aoi(torch.tensor(30., dtype=torch.float64), 180., zen,
                         torch.empty(3, dtype=torch.float64, device='meta'))
    assert out.device.type == 'meta'


def test_martin_ruiz_validates_eager_jax_but_not_traced():
    jnp = _jax()
    import jax
    with pytest.raises(ValueError, match='a_r'):
        iam.martin_ruiz(jnp.asarray([10.]), a_r=jnp.asarray(-1.))
    # inside jit the values are unknown, so the check is skipped
    jax.jit(lambda a_r: iam.martin_ruiz(jnp.asarray([10.]), a_r))(
        jnp.asarray(0.16))


def test_xarray_joins_like_xarray_arithmetic():
    xr = pytest.importorskip('xarray')
    zen = xr.DataArray([10., 20., 30.], dims='t', coords={'t': [0, 1, 2]})
    azi = xr.DataArray([90., 100., 110.], dims='t', coords={'t': [1, 2, 3]})
    out = irradiance.aoi(30., 180., zen, azi)
    assert list(out['t'].values) == list((zen + azi)['t'].values) == [1, 2]


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
