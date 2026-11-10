"""
Spectral functions, which integrate over wavelength, on N-D data from any
library.

Wavelength is a coordinate, like the time of time series functions (see
labeled_io's coordinate): it comes from the labels of pandas objects, from
the dimension of a DataArray of wavelengths, or from a 1-D array along the
last axis of other arrays. Unlike time, the result doesn't have it, and
DataFrames hold it in their columns.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pvlib import spectrum

from .test_array_api import backend, _jax, _torch, _to_numpy  # noqa: F401
from .test_array_api import _restore_jax_x64  # noqa: F401


@pytest.fixture(scope='module')
def spectra():
    """Three spectra as a DataFrame with wavelength in its columns, and a
    spectral response on the same wavelengths."""
    spectra = spectrum.get_reference_spectra().T.iloc[:, ::8]
    sr = spectrum.get_example_spectral_response()
    sr = pd.Series(np.interp(spectra.columns, sr.index, sr, left=0.,
                             right=0.), index=spectra.columns)
    return spectra, sr


def _functions(spectra, sr):
    """name: (function of the spectra and wavelength, result from pandas)."""
    return {
        'average_photon_energy': (
            lambda e, wavelength, convert=None:
                spectrum.average_photon_energy(e, wavelength),
            spectrum.average_photon_energy(spectra)),
        'calc_spectral_mismatch_field': (
            lambda e, wavelength, convert=np.asarray:
                spectrum.calc_spectral_mismatch_field(
                    convert(sr.to_numpy()), e,
                    convert(spectra.loc['global'].to_numpy()),
                    wavelength=wavelength),
            spectrum.calc_spectral_mismatch_field(
                sr, spectra, spectra.loc['global'])),
    }


NAMES = ['average_photon_energy', 'calc_spectral_mismatch_field']


@pytest.mark.parametrize('name', NAMES)
def test_pandas_labels_are_the_wavelengths(spectra, name):
    spectra, sr = spectra
    function, expected = _functions(spectra, sr)[name]
    assert isinstance(expected, pd.Series)
    assert expected.index.equals(spectra.index)
    # one spectrum, a Series, gives a number
    if name == 'average_photon_energy':
        one = spectrum.average_photon_energy(spectra.loc['direct'])
    else:
        one = spectrum.calc_spectral_mismatch_field(
            sr, spectra.loc['direct'], spectra.loc['global'])
    assert isinstance(one, np.float64)
    assert_allclose(one, expected['direct'], rtol=1e-12)


@pytest.mark.parametrize('name', NAMES)
def test_numpy_wavelength_last(spectra, name):
    spectra, sr = spectra
    function, expected = _functions(spectra, sr)[name]
    wavelength = spectra.columns.to_numpy()
    out = function(spectra.to_numpy(), wavelength)
    assert type(out) is np.ndarray and out.shape == (3,)
    assert_allclose(out, expected, rtol=1e-12)
    # more dimensions before the wavelength
    stacked = np.stack([spectra.to_numpy(), 2 * spectra.to_numpy()])
    out = function(stacked, wavelength)
    assert out.shape == (2, 3)
    assert_allclose(out[0], expected, rtol=1e-12)
    # one spectrum gives a NumPy scalar
    assert isinstance(function(spectra.to_numpy()[0], wavelength),
                      np.float64)


@pytest.mark.parametrize('name', NAMES)
def test_array_libraries(backend, spectra, name):  # noqa: F811
    _, xp, convert = backend
    spectra, sr = spectra
    function, expected = _functions(spectra, sr)[name]
    e = convert(spectra.to_numpy())
    out = function(e, convert(spectra.columns.to_numpy()), convert)
    assert type(out) is type(e)
    assert_allclose(_to_numpy(out), expected, rtol=1e-10)


@pytest.mark.parametrize('name', NAMES)
def test_xarray_wavelength_by_dimension(spectra, name):
    xr = pytest.importorskip('xarray')
    spectra, sr = spectra
    function, expected = _functions(spectra, sr)[name]
    e = xr.DataArray(spectra.to_numpy(), dims=('spectrum', 'wavelength'),
                     coords={'spectrum': list(spectra.index),
                             'wavelength': spectra.columns.to_numpy()})

    def labeled(values):
        return e.wavelength.copy(data=np.asarray(values))

    # wherever the wavelength dimension is
    for data in (e, e.transpose()):
        out = function(data, e.wavelength, labeled)
        assert isinstance(out, xr.DataArray)
        assert out.dims == ('spectrum',)
        assert_allclose(out.values, expected, rtol=1e-12)


def test_jax_jit_and_gradient(spectra):
    jnp = _jax()
    import jax
    spectra, _ = spectra
    wavelength = jnp.asarray(spectra.columns.to_numpy())
    e = jnp.asarray(spectra.to_numpy())
    ape = jax.jit(spectrum.average_photon_energy)
    assert_allclose(ape(e, wavelength),
                    spectrum.average_photon_energy(spectra), rtol=1e-10)
    gradient = jax.jit(jax.grad(
        lambda e: jnp.sum(spectrum.average_photon_energy(e, wavelength))))(e)
    assert gradient.shape == e.shape
    assert np.isfinite(np.asarray(gradient)).all()


def test_default_reference_spectrum_follows_the_data(spectra):
    # it is read from a file at the given wavelengths, at the boundary
    torch = _torch()
    spectra, sr = spectra
    wavelength = spectra.columns.to_numpy()
    expected = spectrum.calc_spectral_mismatch_field(
        sr.to_numpy(), spectra.to_numpy(), wavelength=wavelength)
    assert_allclose(expected, spectrum.calc_spectral_mismatch_field(
        sr, spectra), rtol=1e-12)
    out = spectrum.calc_spectral_mismatch_field(
        torch.tensor(sr.to_numpy()), torch.tensor(spectra.to_numpy()),
        wavelength=torch.tensor(wavelength))
    assert isinstance(out, torch.Tensor)
    assert_allclose(out.numpy(), expected, rtol=1e-10)


def test_default_reference_spectrum_needs_known_wavelengths(spectra):
    jnp = _jax()
    import jax
    spectra, sr = spectra
    with pytest.raises(ValueError, match='pass e_ref'):
        jax.jit(lambda e, w: spectrum.calc_spectral_mismatch_field(
            jnp.asarray(sr.to_numpy()), e, wavelength=w))(
                jnp.asarray(spectra.to_numpy()),
                jnp.asarray(spectra.columns.to_numpy()))


def test_arrays_need_wavelength(spectra):
    spectra, sr = spectra
    with pytest.raises(ValueError, match='wavelength is required'):
        spectrum.average_photon_energy(spectra.to_numpy())
    with pytest.raises(ValueError, match='wavelength is required'):
        spectrum.calc_spectral_mismatch_field(sr.to_numpy(),
                                              spectra.to_numpy())


def test_wavelength_is_the_last_axis(spectra):
    spectra, _ = spectra
    with pytest.raises(ValueError, match='last axis'):
        spectrum.average_photon_energy(spectra.to_numpy().T,
                                       spectra.columns.to_numpy())


def test_wavelength_may_repeat_the_labels_but_not_differ(spectra):
    spectra, _ = spectra
    wavelength = spectra.columns.to_numpy()
    assert_allclose(spectrum.average_photon_energy(spectra, wavelength),
                    spectrum.average_photon_energy(spectra), rtol=1e-12)
    with pytest.raises(ValueError, match='differs from the labels'):
        spectrum.average_photon_energy(spectra, wavelength + 1.)


def test_dataarrays_need_a_dataarray_of_wavelengths(spectra):
    xr = pytest.importorskip('xarray')
    spectra, _ = spectra
    e = xr.DataArray(spectra.to_numpy(), dims=('spectrum', 'wavelength'))
    with pytest.raises(ValueError, match='1-D DataArray'):
        spectrum.average_photon_energy(e)
    with pytest.raises(ValueError, match='1-D DataArray'):
        spectrum.average_photon_energy(e, spectra.columns.to_numpy())
