"""
Input type contract for model functions: the same values given as Python
scalars, lists, NumPy arrays, masked arrays, pandas Series and xarray
DataArrays give the same results, in the container type that came in.

There is one entry per function in CASES. A function whose handling of input
types changes shows up here, not in its own tests, which use one input type
each. Array API libraries (PyTorch, JAX, ...) are covered in test_array_api.
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from pvlib import (atmosphere, clearsky, iam, inverter, irradiance,
                   pvsystem, temperature, tools)

INDEX = pd.date_range('2026-06-01 06:00', periods=4, freq='3h',
                      tz='Etc/GMT+7')
MASKED = 1  # position masked in the masked array inputs

# Series inputs are named after their parameter. KEEP means a Series result
# keeps the name of the (single) Series argument.
KEEP = object()

ZEN = [20., 40., 60., 80.]
AZI = [100., 150., 200., 250.]
TILT = [10., 20., 30., 40.]
SAZI = [180., 170., 190., 200.]
DNI = [800., 700., 500., 200.]
GHI = [900., 700., 400., 100.]
DHI = [100., 120., 150., 60.]
TEMP = [20., 25., 30., 35.]
WIND = [1., 2., 3., 4.]

# name: (function, array arguments, other arguments, Series result name)
CASES = {
    'tools.cosd': (tools.cosd, {'angle': ZEN}, {}, KEEP),
    'tools.sind': (tools.sind, {'angle': ZEN}, {}, KEEP),
    'tools.tand': (tools.tand, {'angle': ZEN}, {}, KEEP),
    'tools.asind': (tools.asind, {'number': [0., .3, .6, .9]}, {}, KEEP),
    'tools.acosd': (tools.acosd, {'number': [0., .3, .6, .9]}, {}, KEEP),
    'tools.atand': (tools.atand, {'number': [0., .3, .6, .9]}, {}, KEEP),
    'irradiance.aoi_projection': (
        irradiance.aoi_projection,
        {'surface_tilt': TILT, 'surface_azimuth': SAZI,
         'solar_zenith': ZEN, 'solar_azimuth': AZI}, {}, 'aoi_projection'),
    'irradiance.aoi': (
        irradiance.aoi,
        {'surface_tilt': TILT, 'surface_azimuth': SAZI,
         'solar_zenith': ZEN, 'solar_azimuth': AZI}, {}, 'aoi'),
    'irradiance.beam_component': (
        irradiance.beam_component,
        {'surface_tilt': TILT, 'surface_azimuth': SAZI, 'solar_zenith': ZEN,
         'solar_azimuth': AZI, 'dni': DNI}, {}, None),
    'irradiance.get_ground_diffuse': (
        irradiance.get_ground_diffuse, {'surface_tilt': TILT, 'ghi': GHI},
        {}, 'poa_ground_diffuse'),
    'irradiance.isotropic': (
        irradiance.isotropic, {'surface_tilt': TILT, 'dhi': DHI}, {}, None),
    'irradiance.haydavies': (
        irradiance.haydavies,
        {'surface_tilt': TILT, 'surface_azimuth': SAZI, 'dhi': DHI,
         'dni': DNI, 'solar_zenith': ZEN, 'solar_azimuth': AZI},
        {'dni_extra': 1360.}, None),
    'irradiance.get_total_irradiance': (
        irradiance.get_total_irradiance,
        {'surface_tilt': TILT, 'surface_azimuth': SAZI, 'solar_zenith': ZEN,
         'solar_azimuth': AZI, 'dni': DNI, 'ghi': GHI, 'dhi': DHI},
        {'dni_extra': 1360., 'model': 'haydavies'}, None),
    'irradiance.clearness_index': (
        irradiance.clearness_index,
        {'ghi': GHI, 'solar_zenith': ZEN}, {'extra_radiation': 1360.}, None),
    'irradiance.get_extra_radiation': (
        irradiance.get_extra_radiation,
        {'datetime_or_doy': [1., 100., 200., 300.]}, {}, None),
    'irradiance.erbs': (
        irradiance.erbs,
        {'ghi': GHI, 'zenith': ZEN, 'datetime_or_doy': [152.] * 4}, {}, None),
    'temperature.sapm_cell': (
        temperature.sapm_cell,
        {'poa_global': GHI, 'temp_air': TEMP, 'wind_speed': WIND},
        {'a': -3.56, 'b': -0.075, 'deltaT': 3}, None),
    'temperature.faiman': (
        temperature.faiman,
        {'poa_global': GHI, 'temp_air': TEMP, 'wind_speed': WIND}, {}, None),
    'pvsystem.pvwatts_dc': (
        pvsystem.pvwatts_dc,
        {'effective_irradiance': GHI, 'temp_cell': TEMP},
        {'pdc0': 250., 'gamma_pdc': -0.004}, None),
    'pvsystem.pvwatts_dc(k)': (
        pvsystem.pvwatts_dc,
        {'effective_irradiance': GHI, 'temp_cell': TEMP},
        {'pdc0': 250., 'gamma_pdc': -0.004, 'k': 0.01}, None),
    'inverter.pvwatts': (
        inverter.pvwatts, {'pdc': [0., 50., 150., 260.]},
        {'pdc0': 240.}, KEEP),
    'iam.ashrae': (iam.ashrae, {'aoi': [0., 30., 60., 95.]}, {}, None),
    'iam.martin_ruiz': (
        iam.martin_ruiz, {'aoi': [0., 30., 60., 95.]}, {}, None),
    'clearsky.haurwitz': (
        clearsky.haurwitz, {'apparent_zenith': [20., 60., 89., 95.]}, {},
        None),
    'atmosphere.get_relative_airmass': (
        atmosphere.get_relative_airmass, {'zenith': [20., 60., 89., 95.]},
        {}, None),
    'atmosphere.get_absolute_airmass': (
        atmosphere.get_absolute_airmass,
        {'airmass_relative': [1.06, 2., 26., np.nan],
         'pressure': [101325., 95000., 90000., 60000.]}, {}, None),
    'atmosphere.alt2pres': (
        atmosphere.alt2pres, {'altitude': [0., 500., 1500., 4500.]}, {},
        KEEP),
    # nan airmass, as at night, gives 0
    'clearsky.ineichen': (
        clearsky.ineichen,
        {'apparent_zenith': [20., 60., 89., 95.],
         'airmass_absolute': [1.06, 2., 26., np.nan],
         'linke_turbidity': [2., 3., 4., 5.],
         'altitude': [0., 500., 1500., 4500.]}, {'dni_extra': 1360.}, None),
}

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
    if kind == 'series':
        return pd.Series(values, index=INDEX[:len(values)], name=param)
    if kind == 'xarray':
        xr = pytest.importorskip('xarray')
        return xr.DataArray(values, dims='time')


def _call(case, kind):
    func, arrays, params, _ = CASES[case]
    kwargs = {k: _convert(kind, k, v) for k, v in arrays.items()}
    return func(**kwargs, **params)


def _check_container(out, kind, case):
    expected_name = CASES[case][3]
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
            expected_name = next(iter(CASES[case][1]))
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
@pytest.mark.parametrize('case', list(CASES))
def test_input_types(case, kind):
    reference = _call(case, 'ndarray')
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
