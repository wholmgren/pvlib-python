"""
The single diode functions with arrays of any library.

i_from_v and v_from_i are closed-form with Lambert's W, and have entries in
_supported_functions, so the tests of input types, array libraries and
finite gradients run for them. The tests here are of Lambert's W itself, of
the functions that stay on NumPy because they call a SciPy solver, and of
the user guide's way to the maximum power point without them.
"""
import numpy as np
import pytest
import scipy.special
from numpy.testing import assert_allclose

from pvlib import _array_api as _xa
from pvlib import pvsystem
from pvlib.ivtools.utils import _lambertw_pvlib

from .test_array_api import (  # noqa: F401
    _dask, _jax, _torch, _restore_jax_x64)

MODULE = (7., 1e-9, 0.3, 300., 1.5)


# --- Lambert's W ------------------------------------------------------------

X = np.concatenate([[0.], np.logspace(-20, 300, 400)])


def test_lambertw_matches_scipy():
    assert_allclose(_lambertw_pvlib(X), scipy.special.lambertw(X).real,
                    rtol=1e-14)


def test_lambertw_from_the_log_where_x_overflowed():
    x = np.array([np.inf, 50., np.inf])
    logx = np.array([800., np.log(50.), 5000.])
    w = _lambertw_pvlib(x, np, logx=logx)
    assert_allclose(w + np.log(w), logx, rtol=1e-14)  # w exp(w) = x
    assert np.isnan(_lambertw_pvlib(x)[[0, 2]]).all()


def _dw_dx():
    w = scipy.special.lambertw(X).real
    with np.errstate(invalid='ignore'):
        return np.where(X > 0, w / (X * (1 + w)), 1.)


def test_lambertw_torch_gradient_is_exact():
    torch = _torch()
    x = torch.tensor(X, requires_grad=True)
    _lambertw_pvlib(x, _xa.array_namespace(x)).sum().backward()
    assert_allclose(x.grad.numpy(), _dw_dx(), rtol=1e-13)


def test_lambertw_jax_jit_gradient_is_exact():
    jnp = _jax()
    import jax
    x = jnp.asarray(X)
    xp = _xa.array_namespace(x)
    gradient = jax.jit(jax.grad(lambda x: _lambertw_pvlib(x, xp).sum()))(x)
    assert_allclose(np.asarray(gradient), _dw_dx(), rtol=1e-13)


# --- i_from_v and v_from_i --------------------------------------------------

def test_i_from_v_gradient_is_the_slope_of_the_curve():
    torch = _torch()
    v = torch.tensor(np.linspace(0., 30., 7), requires_grad=True)
    pvsystem.i_from_v(v, *MODULE).sum().backward()
    h = 1e-5
    slope = (pvsystem.i_from_v(v.detach().numpy() + h, *MODULE)
             - pvsystem.i_from_v(v.detach().numpy() - h, *MODULE)) / (2 * h)
    assert_allclose(v.grad.numpy(), slope, rtol=1e-7)


def test_v_from_i_gradient_is_finite_where_the_argument_overflows():
    # exp(800) overflows; W is calculated from the log
    torch = _torch()
    photocurrent = torch.tensor([8.], dtype=torch.float64,
                                requires_grad=True)
    v = pvsystem.v_from_i(0., photocurrent, 1e-10, 0.3, 1000., 0.01)
    v.sum().backward()
    assert np.isfinite(v.item())
    assert np.isfinite(photocurrent.grad.item())
    assert_allclose(v.item(), pvsystem.v_from_i(0., 8., 1e-10, 0.3, 1000.,
                                                0.01))


def test_nan_parameters_give_nan():
    for i in range(5):
        params = list(MODULE)
        params[i] = np.array([np.nan, MODULE[i]])
        for out in (pvsystem.i_from_v(10., *params),
                    pvsystem.v_from_i(2., *params)):
            assert np.isnan(out[0]) and np.isfinite(out[1])


# --- what stays on NumPy ----------------------------------------------------

@pytest.mark.parametrize('method', ['newton', 'brentq', 'chandrupatla'])
def test_scipy_methods_need_numpy(method):
    torch = _torch()
    x = torch.tensor([7., 5.], dtype=torch.float64)
    with pytest.raises(ValueError, match="Use method='lambertw'"):
        pvsystem.i_from_v(x, *MODULE, method=method)
    with pytest.raises(ValueError, match="Use method='lambertw'"):
        pvsystem.v_from_i(x, *MODULE, method=method)


@pytest.mark.parametrize('library', ['torch', 'jax', 'dask'])
@pytest.mark.parametrize('method', ['lambertw', 'newton'])
def test_singlediode_needs_numpy(library, method):
    values = np.array([7., 5.])
    if library == 'torch':
        photocurrent = _torch().tensor(values)
    elif library == 'jax':
        photocurrent = _jax().asarray(values)
    else:
        photocurrent = _dask().from_array(values)
    with pytest.raises(TypeError, match='Maximum power point'):
        pvsystem.singlediode(photocurrent, *MODULE[1:], method=method)


def test_singlediode_error_points_to_a_section_of_the_user_guide():
    from .conftest import TESTS_DATA_DIR
    guide = (TESTS_DATA_DIR.parents[1] / 'docs' / 'sphinx' / 'source'
             / 'user_guide' / 'modeling_topics' / 'array_libraries.rst')
    assert '\nMaximum power point\n-------------------\n' in guide.read_text()


# --- the user guide's maximum power point -----------------------------------

def test_user_guide_maximum_power_point_with_optimistix():
    # the example under "Maximum power point" in the user guide
    jnp = _jax()
    import jax
    optx = pytest.importorskip('optimistix')

    def power(v, params):
        return v * pvsystem.i_from_v(v, *params)

    def max_power_point(params):
        v_oc = pvsystem.v_from_i(0., *params)
        solution = optx.root_find(
            jax.grad(power), optx.Bisection(rtol=1e-12, atol=1e-12),
            0.8 * v_oc, args=params, options={'lower': 0., 'upper': v_oc})
        v_mp = solution.value
        return v_mp, power(v_mp, params)

    rng = np.random.default_rng(0)
    n = 50
    params = (rng.uniform(1, 10, n), np.full(n, 1e-9), np.full(n, 0.3),
              np.full(n, 300.), rng.uniform(1.4, 1.6, n))
    traced = tuple(jnp.asarray(p) for p in params)
    v_mp, p_mp = jax.jit(jax.vmap(max_power_point))(traced)
    expected = pvsystem.singlediode(*params)
    # SciPy finds v_mp to a relative tolerance of 1.5e-8
    assert_allclose(v_mp, expected['v_mp'], rtol=1e-7)
    assert_allclose(p_mp, expected['p_mp'], rtol=1e-12)

    gradient = jax.grad(
        lambda params: jnp.sum(jax.vmap(max_power_point)(params)[1]))(traced)
    h = 1e-4
    up, down = list(params), list(params)
    up[0] = params[0] * (1 + h)
    down[0] = params[0] * (1 - h)
    slope = ((pvsystem.singlediode(*up)['p_mp']
              - pvsystem.singlediode(*down)['p_mp']) / (2 * h * params[0]))
    assert_allclose(gradient[0], slope, rtol=1e-6)
    assert all(np.isfinite(np.asarray(g)).all() for g in gradient)


# --- helpers ----------------------------------------------------------------

def test_maybe_any():
    assert _xa.maybe_any(np.array([False, True]), np)
    assert not _xa.maybe_any(np.array([False, False]), np)
    da = _dask()
    lazy = da.from_array(np.array([False, False]))
    assert _xa.maybe_any(lazy, _xa.array_namespace(lazy))


def test_maybe_any_is_true_inside_jax_jit():
    jnp = _jax()
    import jax
    seen = []

    def f(x):
        seen.append(_xa.maybe_any(x > 0, _xa.array_namespace(x)))
        return x

    jax.jit(f)(jnp.asarray([-1., -2.]))
    assert seen == [True]
