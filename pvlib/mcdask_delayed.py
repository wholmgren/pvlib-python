import warnings

from functools import partial
from operator import truediv

import pandas as pd

from pvlib import (solarposition, pvsystem, clearsky, atmosphere, tools)
from pvlib.tracking import SingleAxisTracker
import pvlib.irradiance  # avoid name conflict with full import

from dask import delayed
from dask.compatibility import apply
from toolz import curry

def basic_chain(weather, latitude, longitude,
                surface_tilt, surface_azimuth,
                module_parameters, inverter_parameters,
                transposition_model='haydavies',
                solar_position_method='nrel_numpy',
                airmass_model='kastenyoung1989',
                altitude=None, pressure=None,
                dc_model=None, ac_model=None):

    # define task graph
    times = weather.index
    wind_speed = delayed(var_or_default)(weather, 'wind_speed', 0)
    temp_air = delayed(var_or_default)(weather, 'temp_air', 25)

    pressure_altitude = delayed(infer_pressure_altitude)(altitude, pressure)
    pressure = pressure_altitude[0]
    altitude = pressure_altitude[1]

    solar_position = delayed(solarposition.get_solarposition)(
        times, latitude, longitude, method=solar_position_method)
    apparent_zenith = solar_position['apparent_zenith']
    azimuth = solar_position['azimuth']

    airmass_relative = delayed(atmosphere.relativeairmass)(
        apparent_zenith, model=airmass_model)
    airmass_absolute = delayed(atmosphere.absoluteairmass)(airmass_relative,
                         pressure)
    dni_extra = delayed(pvlib.irradiance.extraradiation)(times)
    aoi = delayed(pvlib.irradiance.aoi)(surface_tilt, surface_azimuth,
            apparent_zenith, azimuth)
    poa_irrad = delayed(pvlib.irradiance.total_irrad)(
        surface_tilt, surface_azimuth,
        apparent_zenith, azimuth,
        weather['dni'], weather['ghi'], weather['dhi'],
        model=transposition_model,
        dni_extra=dni_extra,
        airmass=airmass_absolute)
    temps = delayed(pvsystem.sapm_celltemp)(poa_irrad['poa_global'],
              wind_speed, temp_air)
    effective_irradiance = poa_irrad['poa_global']
    losses = 1

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
            dc = delayed(pvsystem.pvwatts_dc)(effective_irradiance,
                         temps['temp_cell'],
                         module_parameters['pdc0'],
                         module_parameters['gamma_pdc'])
        elif model == 'sapm':
            dsk['dc'] = delayed(pvsystem.sapm)(
                         effective_irradiance,
                         temps['temp_cell'],
                         module_parameters)
        else:
            raise ValueError(model + ' is not a valid DC power model')
    else:
        dc = dc_model

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
            ac = delayed(pvsystem.pvwatts_ac)(dc, module_parameters['pdc0'])
        elif model == 'snlinverter':
            ac = delayed(pvsystem.snlinverter)(dc['v_mp'], dc['p_mp'])
        else:
            raise ValueError(model + ' is not a valid AC power model')
    else:
        ac = ac_model

    return ac


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
