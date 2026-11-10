"""
Gradients through the supported functions are finite, with PyTorch and JAX.

Gradients are promised function by function. An entry of
_supported_functions promises them unless its ``gradients`` gives a reason
not to. For each entry that does, the derivatives of the finite outputs with
respect to each array argument must be finite wherever that argument is. A
nan that a function masks, such as the airmass of night, stays out of its
values but not out of its derivatives (0 * nan) unless the function keeps it
out of the calculation, with _array_api.where_safe.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from pvlib import atmosphere, clearsky

from ._supported_functions import SUPPORTED, function
from .test_array_api import _jax, _torch, _restore_jax_x64  # noqa: F401


def _differentiated(entry):
    # times are data, not parameters to fit
    return [k for k in entry.arrays if k not in entry.time]


def _skip_unpromised(case):
    reason = SUPPORTED[case].gradients
    if reason is not True:
        pytest.skip(f'no gradients promised: {reason}')


HOW_TO_FIX = (
    'A branch that xp.where leaves out, or a nan that is masked later, '
    'still reaches the derivative. Give the excluded values a safe input '
    'with _array_api.where_safe; see "Gradients" in the contributing guide '
    '(docs/sphinx/source/contributing/array_libraries.rst). If the function '
    "can't have finite gradients, give the reason as gradients='...' in its "
    'entry in tests/_supported_functions.py.')


def _loss(out, xp):
    outputs = out.values() if isinstance(out, dict) else [out]
    return sum(xp.sum(xp.where(xp.isfinite(o), o, 0.)) for o in outputs)


def _check(case, values, gradients):
    for name, x, g in zip(_differentiated(SUPPORTED[case]), values,
                          gradients):
        if g is None:  # the outputs don't depend on it
            continue
        bad = ~np.isfinite(np.asarray(g)) & np.isfinite(np.asarray(x))
        at = np.flatnonzero(bad).tolist()
        assert not bad.any(), (
            f'{case}: non-finite derivative with respect to {name} at '
            f'positions {at}, where {name} is '
            f'{np.asarray(x)[bad].tolist()}. {HOW_TO_FIX}')


@pytest.mark.parametrize('case', list(SUPPORTED))
def test_torch_gradients_are_finite(case):
    torch = _torch()
    _skip_unpromised(case)
    entry = SUPPORTED[case]
    names = _differentiated(entry)
    inputs = {k: torch.tensor(v, dtype=torch.float64,
                              requires_grad=k in names)
              for k, v in entry.arrays.items()}
    loss = _loss(function(case)(**inputs, **entry.params), torch)
    gradients = torch.autograd.grad(loss, [inputs[k] for k in names],
                                    allow_unused=True)
    _check(case, [entry.arrays[k] for k in names], gradients)


@pytest.mark.parametrize('case', list(SUPPORTED))
def test_jax_gradients_are_finite(case):
    jnp = _jax()
    import jax
    _skip_unpromised(case)
    entry = SUPPORTED[case]
    names = _differentiated(entry)
    fixed = {k: jnp.asarray(v) for k, v in entry.arrays.items()
             if k not in names}

    def loss(*xs):
        out = function(case)(**fixed, **dict(zip(names, xs)),
                             **entry.params)
        return _loss(out, jnp)

    values = [jnp.asarray(entry.arrays[k]) for k in names]
    gradients = jax.grad(loss, argnums=tuple(range(len(names))))(*values)
    _check(case, values, gradients)


def test_user_guide_lists_the_functions_without_gradients():
    import pathlib
    import re
    from ._supported_functions import qualified_name
    guide = (pathlib.Path(__file__).parents[1] / 'docs' / 'sphinx' / 'source'
             / 'user_guide' / 'modeling_topics' / 'array_libraries.rst')
    if not guide.exists():
        pytest.skip('the docs are not here')
    section = guide.read_text().split('\nGradients\n')[1]
    sentence = section.split('except for')[1].split('\n\n')[0]
    listed = set(re.findall(r':py:func:`~?(pvlib\.[\w.]+)`', sentence))
    assert listed == {qualified_name(k) for k, entry in SUPPORTED.items()
                      if entry.gradients is not True}


# night (zenith 95°) makes the airmass nan
ZENITH = [20., 50., 80., 95.]


def _clear_sky_dni(xp, altitude, linke_turbidity):
    zenith = xp.asarray(ZENITH, dtype=xp.float64)
    airmass = atmosphere.get_absolute_airmass(
        atmosphere.get_relative_airmass(zenith),
        atmosphere.alt2pres(altitude))
    dni = clearsky.ineichen(zenith, airmass, linke_turbidity,
                            altitude=altitude)['dni']
    return xp.sum(dni)


def _finite_differences():
    step = 1e-4
    return [(_clear_sky_dni(np, 1500. + step, 3.)
             - _clear_sky_dni(np, 1500. - step, 3.)) / (2 * step),
            (_clear_sky_dni(np, 1500., 3. + step)
             - _clear_sky_dni(np, 1500., 3. - step)) / (2 * step)]


def test_torch_clear_sky_chain_through_night():
    # altitude reaches the airmass through pressure, so a nan airmass at
    # night must stay out of the derivatives of every function in the chain
    torch = _torch()
    altitude = torch.tensor(1500., dtype=torch.float64, requires_grad=True)
    turbidity = torch.tensor(3., dtype=torch.float64, requires_grad=True)
    gradients = torch.autograd.grad(
        _clear_sky_dni(torch, altitude, turbidity), [altitude, turbidity])
    assert_allclose([g.item() for g in gradients], _finite_differences(),
                    rtol=1e-6)


def test_jax_clear_sky_chain_through_night():
    jnp = _jax()
    import jax
    gradients = jax.jit(jax.grad(
        lambda altitude, turbidity: _clear_sky_dni(jnp, altitude, turbidity),
        argnums=(0, 1)))(1500., 3.)
    assert_allclose([float(g) for g in gradients], _finite_differences(),
                    rtol=1e-6)
