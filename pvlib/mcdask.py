import warnings

from functools import partial
from operator import truediv

import pandas as pd

from pvlib import (solarposition, pvsystem, clearsky, atmosphere, tools)
from pvlib.tracking import SingleAxisTracker
import pvlib.irradiance  # avoid name conflict with full import

from dask.compatibility import apply

def basic_chain(weather, latitude, longitude,
                surface_tilt, surface_azimuth,
                module_parameters, inverter_parameters,
                transposition_model='haydavies',
                solar_position_method='nrel_numpy',
                airmass_model='kastenyoung1989',
                altitude=None, pressure=None,
                dc_model=None, ac_model=None):

    """
    Parameters
    ----------
    see ModelChain documentation for now

    Returns
    -------
    dsk : dict
        Keys are dask task graph nodes, including:
            solar_position, poa_irrad, temps, effective_irradiance,
            dc, ac.
        Values are the instructions dask needs to compute the
        corresponding result.

    Examples
    --------
    times = pd.DatetimeIndex(start='20180601 0000-0700', freq='12H', periods=2)
    weather = pd.DataFrame({'ghi': [0, 1000], 'dni': [0, 950], 'dhi': [0, 100], 'temp_air': 25, 'wind_speed': 0}, index=times)
    dsk = pvlib.mcdask.basic_chain(weather, 32.2, -110.9, 30, 180, {'pdc0': 1000, 'gamma_pdc': -0.025}, {})
    dask.get(dsk, 'ac')
    2018-06-01 00:00:00-07:00           NaN
    2018-06-01 12:00:00-07:00    137.619168
    Freq: 12H, dtype: float64
    """

    # these parameters are used by several functions, so define
    # look-up methods once here.
    apparent_zenith = (getattr, 'solar_position', 'apparent_zenith')
    azimuth = (getattr, 'solar_position', 'azimuth')

    # define task graph
    # wind_speed-1 and similar prevent circular references
    dsk = {
        'times': weather.index,
        'wind_speed-1': (var_or_default, weather, 'wind_speed', 0),
        'temp_air-1': (var_or_default, weather, 'temp_air', 25),
        'pressure,altitude': (infer_pressure_altitude, altitude, pressure),
        'pressure': (lambda x: x[0], 'pressure,altitude'),
        'altitude': (lambda x: x[1], 'pressure,altitude'),
        'solar_position': (partial(solarposition.get_solarposition,
                                   method=solar_position_method),
                           'times', latitude, longitude),
        'airmass_relative': (partial(atmosphere.relativeairmass,
                                     model=airmass_model), apparent_zenith),
        'airmass_absolute': (atmosphere.absoluteairmass, 'airmass_relative',
                             'pressure'),
        'dni_extra-1': (pvlib.irradiance.extraradiation, 'times'),
        'aoi': (pvlib.irradiance.aoi, surface_tilt, surface_azimuth,
                apparent_zenith, azimuth),
        'poa_irrad': (apply, pvlib.irradiance.total_irrad,
                      [surface_tilt, surface_azimuth,
                       apparent_zenith, azimuth,
                       weather['dni'], weather['ghi'], weather['dhi']],
                      (dict, [['model', transposition_model],
                              ['dni_extra', 'dni_extra-1'],
                              ['airmass', 'airmass_absolute']])),
        'temps': (pvsystem.sapm_celltemp, (getattr, 'poa_irrad', 'poa_global'),
                  'wind_speed-1', 'temp_air-1'),
        'effective_irradiance': (getattr, 'poa_irrad', 'poa_global'),
        'losses': (1)
    }

    if dc_model is None:
        try:
            dc_model = infer_dc_model(module_parameters)
        except ValueError as e:
            warnings.warn(e)
    if dc_model is None:
        warnings.warn('no dc model registered')
    elif isinstance(dc_model, str):
        model = dc_model.lower()
        if model == 'pvwatts':
            dsk['dc'] = (pvsystem.pvwatts_dc, 'effective_irradiance',
                         (getattr, 'temps', 'temp_cell'),
                         module_parameters['pdc0'],
                         module_parameters['gamma_pdc'])
        elif model == 'sapm':
            dsk['dc'] = (pvsystem.sapm,
                         (truediv, 'effective_irradiance', 1000),
                         (getattr, 'temps', 'temp_cell'),
                         module_parameters)
        else:
            raise ValueError(model + ' is not a valid DC power model')
    else:
        dsk['dc'] = dc_model

    if ac_model is None:
        try:
            ac_model = infer_ac_model(module_parameters, inverter_parameters)
        except ValueError as e:
            warnings.warn(e)
    if ac_model is None:
        warnings.warn('no ac model registered')
    if isinstance(ac_model, str):
        model = ac_model.lower()
        if model == 'pvwatts':
            dsk['ac'] = (pvsystem.pvwatts_ac, 'dc', module_parameters['pdc0'])
        elif model == 'snlinverter':
            dsk['ac'] = (pvsystem.snlinverter, (getattr, 'dc', 'v_mp'),
                         (getattr, 'dc', 'p_mp'))
        else:
            raise ValueError(model + ' is not a valid AC power model')
    else:
        dsk['ac'] = ac_model

    return dsk


def var_or_default(weather, var, default):
    if var in weather.columns:
        return weather[var]
    else:
        return pd.Series(default, index=weather.index)


def infer_pressure_altitude(altitude, pressure):
    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = atmosphere.alt2pres(altitude)
    return altitude, pressure


def infer_dc_model(module_parameters):
    params = set(module_parameters.keys())
    if set(['A0', 'A1', 'C7']) <= params:
        return 'sapm'
    elif set(['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s']) <= params:
        return 'singlediode'
    elif set(['pdc0', 'gamma_pdc']) <= params:
        return 'pvwatts'
    else:
        raise ValueError('could not infer DC model from module_parameters')


def infer_ac_model(module_parameters, inverter_parameters):
    inverter_params = set(inverter_parameters.keys())
    module_params = set(module_parameters.keys())
    if set(['C0', 'C1', 'C2']) <= inverter_params:
        return 'snlinverter'
    elif set(['ADRCoefficients']) <= inverter_params:
        return 'adrinverter'
    elif set(['pdc0']) <= module_params:
        return 'pvwatts'
    else:
        raise ValueError('could not infer AC model from '
                         'module_parameters or inverter_parameters')
