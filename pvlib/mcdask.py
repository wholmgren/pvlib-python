import warnings

from functools import partial
from operator import truediv, mul, sub, add

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
                dc_model=None, ac_model=None,
                aoi_model=None, spectral_model=None, temp_model='sapm',
                losses_model='no_loss'):

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
    dsk = pvlib.mcdask.basic_chain(weather, 32.2, -110.9, 30, 180, {'pdc0': 1000, 'gamma_pdc': -0.025}, {}, losses_model='pvwatts', spectral_model='no_loss', aoi_model='no_loss')
    dask.get(dsk, 'ac')
    2018-06-01 00:00:00-07:00           NaN
    2018-06-01 12:00:00-07:00    118.248361
    Freq: 12H, dtype: float64
    """

    # these DataFrame columns are used by several functions and are difficult
    # to read if accessed inline, so define "look-up functions" here.
    apparent_zenith = (getattr, 'solar_position', 'apparent_zenith')
    azimuth = (getattr, 'solar_position', 'azimuth')
    poa_global = (getattr, 'poa_irrad', 'poa_global')
    poa_direct = (getattr, 'poa_irrad', 'poa_direct')
    poa_diffuse = (getattr, 'poa_irrad', 'poa_diffuse')

    # define initial dask task graph.
    # the more complicated model-dependent assignments are handled in
    # functions below.
    # wind_speed-1 and similar prevent circular references.
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
        'temps': (pvsystem.sapm_celltemp, poa_global,
                  'wind_speed-1', 'temp_air-1'),
        # effective_irradiance = spectral_modifier * (
        #     poa_direct * aoi_modifier + fd * poa_diffuse)
        # default diffuse fraction value of 1
        'effective_irradiance': (mul, 'spectral_modifier',
            (add,
                (mul, poa_direct, 'aoi_modifier'),
                (mul, (getattr, 'module_parameters', 'FD', 1), poa_diffuse)))
    }

    try:
        dsk['aoi_modifier'] = assign_aoi_model(aoi_model, module_parameters)
    except ValueError as e:
        warnings.warn(str(e))
        warnings.warn('no aoi model registered')

    try:
        dsk['spectral_modifier'] = assign_spectral_model(spectral_model,
                                                         module_parameters)
    except ValueError as e:
        warnings.warn(str(e))
        warnings.warn('no spectral model registered')

    try:
        dsk['dc'] = assign_dc_model(dc_model, module_parameters)
    except ValueError as e:
        warnings.warn(str(e))
        warnings.warn('no dc model registered')

    try:
        dsk['ac_no_loss'] = assign_ac_model(ac_model, module_parameters,
                                            inverter_parameters)
    except ValueError as e:
        warnings.warn(str(e))
        warnings.warn('no ac model registered')

    try:
        # losses logic may need refactoring if more complicated models
        # are implemented
        dsk['losses'] = assign_losses_model(losses_model)
        dsk['ac'] = (mul, 'ac_no_loss', 'losses')
    except ValueError as e:
        warnings.warn(str(e))
        warnings.warn('no losses model registered')
        try:
            dsk['ac'] = dsk['ac_no_loss']
        except KeyError:
            pass

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


def assign_aoi_model(aoi_model, module_parameters):
    if aoi_model is None:
        aoi_model = infer_aoi_model(module_parameters)
    if isinstance(aoi_model, str):
        model = aoi_model.lower()
        # no support for kwargs in pvlib function calls at present
        # investigate toolz for simple kwarg handling
        if model == 'ashrae':
            model_dsk = (pvsystem.ashraeiam, 'aoi')
        elif model == 'physical':
            model_dsk = (physicaliam, 'aoi')
        elif model == 'sapm':
            model_dsk = (pvsystem.sapm, 'aoi', module_parameters)
        elif model == 'no_loss':
            model_dsk = 1
        else:
            raise ValueError(model + ' is not a valid aoi loss model')
    else:
        model_dsk = aoi_model
    return model_dsk


def infer_aoi_model(module_parameters):
    params = set(module_parameters.keys())
    if set(['K', 'L', 'n']) <= params:
        return 'physical'
    elif set(['B5', 'B4', 'B3', 'B2', 'B1', 'B0']) <= params:
        return 'sapm'
    elif set(['b']) <= params:
        return 'ashrae'
    else:
        raise ValueError('could not infer AOI model from module_parameters')


def assign_spectral_model(spectral_model, module_parameters):
    if spectral_model is None:
        spectral_model = infer_spectral_model(module_parameters)
    if isinstance(spectral_model, str):
        model = spectral_model.lower()
        # no support for kwargs in pvlib function calls at present
        # investigate toolz for simple kwarg handling
        if model == 'first_solar':
            # need logic to infer module_type or coefficients
            raise NotImplementedError
            model_dsk = (pvlib.atmosphere.first_solar_spectral_loss,
                         (getattr, 'weather', 'precipitable_water'),
                         'airmass_absolute', None, None)
        elif model == 'sapm':
            model_dsk = (pvsystem.sapm, 'airmass_absolute', module_parameters)
        elif model == 'no_loss':
            model_dsk = 1
        else:
            raise ValueError(model + ' is not a valid spectral loss model')
    else:
        model_dsk = spectral_model
    return model_dsk


def infer_spectral_model(module_parameters):
    params = set(module_parameters.keys())
    if set(['A4', 'A3', 'A2', 'A1', 'A0']) <= params:
        return 'sapm'
    # removed check for pvsystem._infer_cell_type() is not None
    elif ('Technology' in params or 'Material' in params or
          'first_solar_spectral_coefficients' in params):
        return 'first_solar'
    else:
        raise ValueError('could not infer spectral model from '
                         'system.module_parameters. Check that the '
                         'parameters contain valid '
                         'first_solar_spectral_coefficients or a valid '
                         'Material or Technology value')


def assign_dc_model(dc_model, module_parameters):
    if dc_model is None:
        dc_model = infer_dc_model(module_parameters)
    if isinstance(dc_model, str):
        model = dc_model.lower()
        if model == 'pvwatts':
            model_dsk = (pvsystem.pvwatts_dc, 'effective_irradiance',
                         (getattr, 'temps', 'temp_cell'),
                         module_parameters['pdc0'],
                         module_parameters['gamma_pdc'])
        elif model == 'sapm':
            model_dsk = (pvsystem.sapm,
                         (truediv, 'effective_irradiance', 1000),
                         (getattr, 'temps', 'temp_cell'),
                         module_parameters)
        elif mode ==  'singlediode':
            raise NotImplementedError
        else:
            raise ValueError(model + ' is not a valid DC power model')
    else:
        model_dsk = dc_model
    return model_dsk


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


def assign_ac_model(ac_model, module_parameters, inverter_parameters):
    if ac_model is None:
        ac_model = infer_ac_model(module_parameters, inverter_parameters)
    if isinstance(ac_model, str):
        model = ac_model.lower()
        if model == 'pvwatts':
            # missing fillna
            model_dsk = (pvsystem.pvwatts_ac, 'dc', module_parameters['pdc0'])
        elif model == 'snlinverter':
            model_dsk = (pvsystem.snlinverter, (getattr, 'dc', 'v_mp'),
                         (getattr, 'dc', 'p_mp'))
        else:
            raise ValueError(model + ' is not a valid AC power model')
    else:
        model_dsk = ac_model
    return model_dsk


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


def assign_losses_model(losses_model):
    if losses_model is None:
        losses_model = 'None'
    if isinstance(losses_model, str):
        model = losses_model.lower()
        if model == 'pvwatts':
            # losses = (100 - pvsystem.pvwatts_losses()) / 100.
            model_dsk = (truediv, (sub, 100, (pvsystem.pvwatts_losses, )), 100)
        elif model == 'no_loss':
            model_dsk = 1
        else:
            raise ValueError(model + ' is not a valid losses model')
    else:
        model_dsk = losses_model
    return model_dsk
