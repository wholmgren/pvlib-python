"""
The functions that accept arrays from any library, one entry each, with
inputs to call them with.

The entries drive the array library tests: input types (test_input_types,
test_input_types_mixed), array libraries (test_array_api), gradients
(test_array_api_gradients) and the comparison with pvlib before the port
(test_array_api_vs_main). test_array_api checks that every function made
with labeled_io has an entry and that the user guide lists exactly these
functions.

Entries are keyed by the function's module and name in pvlib, with an
optional label in parentheses for a second entry, rather than holding the
function: _make_array_api_golden.py uses them with pvlib from before the
port, which lacks some functions.
"""
import collections
import importlib

import numpy as np
import pandas as pd

Supported = collections.namedtuple(
    'Supported', 'arrays params name time one_d gradients',
    defaults=({}, None, {}, (), True))
Supported.__doc__ = """
arrays : dict
    The array arguments, four values each. Time parameters hold numbers:
    day of year or Unix seconds.
params : dict
    Other arguments.
name : str, None or KEEP
    Name of a Series result. KEEP means the name of the single Series
    argument; Series arguments are named after their parameter.
time : dict
    Parameters that also take datetimes: 'doy' or 'unix'.
one_d : tuple of str
    Parameters that must be series along time, never scalars, as for
    soiling.hsu. For these functions, the parameter that is in both ``time``
    and ``one_d`` is the time of the data (labeled_io's index_time).
gradients : True or str
    True, the default, promises finite gradients with PyTorch and JAX, which
    test_array_api_gradients checks. A function that can't promise them
    gives the reason instead, and the user guide's Gradients section lists
    it.
"""

KEEP = object()

INDEX = pd.date_range('2026-06-01 06:00', periods=4, freq='3h',
                      tz='Etc/GMT+7')
UNIX = list((INDEX.tz_convert('UTC').tz_localize(None)
             - pd.Timestamp('1970-01-01')) / pd.Timedelta('1s'))

ZEN = [20., 40., 60., 80.]
AZI = [100., 150., 200., 250.]
TILT = [10., 20., 30., 40.]
SAZI = [180., 170., 190., 200.]
DNI = [800., 700., 500., 200.]
GHI = [900., 700., 400., 100.]
DHI = [100., 120., 150., 60.]
TEMP = [20., 25., 30., 35.]
WIND = [1., 2., 3., 4.]
ANGLES = {'surface_tilt': TILT, 'surface_azimuth': SAZI,
          'solar_zenith': ZEN, 'solar_azimuth': AZI}

SUPPORTED = {
    'tools.cosd': Supported({'angle': ZEN}, name=KEEP),
    'tools.sind': Supported({'angle': ZEN}, name=KEEP),
    'tools.tand': Supported({'angle': ZEN}, name=KEEP),
    'tools.asind': Supported({'number': [0., .3, .6, .9]}, name=KEEP),
    'tools.acosd': Supported({'number': [0., .3, .6, .9]}, name=KEEP),
    'tools.atand': Supported({'number': [0., .3, .6, .9]}, name=KEEP),
    'tools.unix_to_doy': Supported({'t': UNIX}),
    'solarposition.spa_python': Supported(
        {'time': UNIX, 'latitude': [30., 32., 34., 36.],
         'longitude': [-110., -111., -112., -113.]},
        time={'time': 'unix'}),
    'irradiance.aoi_projection': Supported(ANGLES, name='aoi_projection'),
    'irradiance.aoi': Supported(ANGLES, name='aoi'),
    'irradiance.beam_component': Supported({**ANGLES, 'dni': DNI}),
    'irradiance.get_ground_diffuse': Supported(
        {'surface_tilt': TILT, 'ghi': GHI}, name='poa_ground_diffuse'),
    'irradiance.isotropic': Supported({'surface_tilt': TILT, 'dhi': DHI}),
    'irradiance.haydavies': Supported(
        {'surface_tilt': TILT, 'surface_azimuth': SAZI, 'dhi': DHI,
         'dni': DNI, 'solar_zenith': ZEN, 'solar_azimuth': AZI},
        {'dni_extra': 1360.}),
    'irradiance.get_sky_diffuse': Supported(
        {**ANGLES, 'dni': DNI, 'ghi': GHI, 'dhi': DHI},
        {'dni_extra': 1360., 'model': 'haydavies'}),
    'irradiance.get_total_irradiance': Supported(
        {**ANGLES, 'dni': DNI, 'ghi': GHI, 'dhi': DHI},
        {'dni_extra': 1360., 'model': 'haydavies'}),
    'irradiance.poa_components': Supported(
        {'aoi': [10., 40., 70., 100.], 'dni': DNI,
         'poa_sky_diffuse': DHI, 'poa_ground_diffuse': [5., 6., 7., 8.]}),
    'irradiance.clearness_index': Supported(
        {'ghi': GHI, 'solar_zenith': ZEN}, {'extra_radiation': 1360.}),
    'irradiance.get_extra_radiation': Supported(
        {'datetime_or_doy': [1., 100., 200., 300.],
         'solar_constant': [1361., 1366.1, 1367., 1370.]},
        time={'datetime_or_doy': 'doy'}),
    'irradiance.erbs': Supported(
        {'ghi': GHI, 'zenith': ZEN, 'datetime_or_doy': [152.] * 4},
        time={'datetime_or_doy': 'doy'}),
    'temperature.sapm_cell': Supported(
        {'poa_global': GHI, 'temp_air': TEMP, 'wind_speed': WIND},
        {'a': -3.56, 'b': -0.075, 'deltaT': 3}),
    'temperature.sapm_module': Supported(
        {'poa_global': GHI, 'temp_air': TEMP, 'wind_speed': WIND},
        {'a': -3.56, 'b': -0.075}),
    'temperature.sapm_cell_from_module': Supported(
        {'module_temperature': TEMP, 'poa_global': GHI}, {'deltaT': 3}),
    'temperature.faiman': Supported(
        {'poa_global': GHI, 'temp_air': TEMP, 'wind_speed': WIND}),
    'pvsystem.pvwatts_dc': Supported(
        {'effective_irradiance': GHI, 'temp_cell': TEMP},
        {'pdc0': 250., 'gamma_pdc': -0.004}),
    'pvsystem.pvwatts_dc(k)': Supported(
        {'effective_irradiance': GHI, 'temp_cell': TEMP},
        {'pdc0': 250., 'gamma_pdc': -0.004, 'k': 0.01}),
    'inverter.pvwatts': Supported(
        {'pdc': [0., 50., 150., 260.]}, {'pdc0': 240.}, name=KEEP),
    'iam.ashrae': Supported({'aoi': [0., 30., 60., 95.]}),
    'iam.martin_ruiz': Supported({'aoi': [0., 30., 60., 95.]}),
    'clearsky.haurwitz': Supported(
        {'apparent_zenith': [20., 60., 89., 95.]}),
    # nan airmass, as at night, gives 0
    'clearsky.ineichen': Supported(
        {'apparent_zenith': [20., 60., 89., 95.],
         'airmass_absolute': [1.06, 2., 26., np.nan],
         'linke_turbidity': [2., 3., 4., 5.],
         'altitude': [0., 500., 1500., 4500.]}, {'dni_extra': 1360.}),
    'atmosphere.get_relative_airmass': Supported(
        {'zenith': [20., 60., 89., 95.]}),
    'atmosphere.get_absolute_airmass': Supported(
        {'airmass_relative': [1.06, 2., 26., np.nan],
         'pressure': [101325., 95000., 90000., 60000.]}),
    'atmosphere.alt2pres': Supported(
        {'altitude': [0., 500., 1500., 4500.]}, name=KEEP),
    'soiling.hsu': Supported(
        {'rainfall': [0., 1.6, 0., 1.2], 'times': UNIX,
         'pm2_5': [1e-2, 1.5e-2, 2e-2, 2.5e-2],
         'pm10': [2e-2, 2.5e-2, 3e-2, 3.5e-2]},
        {'cleaning_threshold': 1.0, 'surface_tilt': 20.},
        time={'times': 'unix'}, one_d=('rainfall', 'times')),
    # 3 hour steps: snowfall above 3 cm covers. Snow depth below 1 cm stops
    # coverage, and makes all the snow slide off
    'snow.fully_covered_nrel': Supported(
        {'snowfall': [0., 4., 6., 1.], 'snow_depth': [2., 0.5, 3., 2.],
         'times': UNIX},
        name=None, time={'times': 'unix'}, one_d=('snowfall', 'times'),
        gradients='its result is boolean'),
    'snow.coverage_nrel': Supported(
        {'snowfall': [0., 4., 0., 4.],
         'poa_irradiance': [100., 600., 800., 0.],
         'temp_air': [-5., 2., 3., -2.], 'surface_tilt': TILT,
         'snow_depth': [5., 5., 5., 0.5], 'times': UNIX},
        {'initial_coverage': 0.5}, name=None, time={'times': 'unix'},
        one_d=('snowfall', 'times')),
}


def function(key):
    """The pvlib function of the entry ``key``."""
    module, name = key.split('(')[0].rsplit('.', 1)
    return getattr(importlib.import_module(f'pvlib.{module}'), name)


def qualified_name(key):
    """``key`` as the function's full name, e.g. pvlib.irradiance.aoi."""
    return 'pvlib.' + key.split('(')[0]
