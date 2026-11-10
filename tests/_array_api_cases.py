"""
Calls for comparing the ported functions with pvlib before the port.

Each case is a call that pvlib before the port also accepted, with Python,
NumPy and pandas inputs. ``_make_array_api_golden.py`` runs the cases with
that pvlib and saves what each returns, or that it raises, to
``data/array_api_main_results.json``; ``test_array_api_vs_main.py`` runs them
again and compares. So this module uses only APIs that existed before the
port.

A case's ``policy`` says what to compare:

- ``'match'``: the same result as before. If pvlib raised before, the case
  isn't compared, since the port may accept more input types.
- ``'raise'``: pvlib raised before and must still raise (``TypeError`` or
  ``ValueError``), rather than return a number.

Calls whose results before the port were never promised aren't here, such as
``soiling.hsu`` with a ``rain_accum_period`` that isn't a Timedelta.
"""
import collections
import datetime
import functools
import warnings

import numpy as np
import pandas as pd

from pvlib import (atmosphere, clearsky, iam, inverter, irradiance, location,
                   pvsystem, snow, soiling, solarposition, temperature)

from ._supported_functions import (SUPPORTED, INDEX, ZEN, AZI, TILT, SAZI,
                                   DNI, GHI, DHI, TEMP, WIND, function)
from .test_input_types import _convert

Case = collections.namedtuple('Case', 'call policy requires',
                              defaults=('match', ()))

# input kinds for the supported functions. xarray is left out, since pvlib
# didn't support it before the port, and so are masked arrays, which some
# functions got wrong then (inverter.pvwatts gave NaN at zero power);
# test_input_types compares masked arrays with NumPy arrays. Functions that
# take numbers as Unix seconds are left out too: before the port, numbers
# were nanoseconds. SPA_CASES, HSU_CASES and SNOW_CASES compare them with
# datetimes
KINDS = ['scalar', 'list', 'ndarray', 'series']

# documented changes from before the port, by case: the values must match,
# but the type of the result may differ
CONTAINER_CHANGES = {
    'irradiance.get_extra_radiation[series]':
        'a Series of day of year gives a Series, not an ndarray',
}

LAT, LON = 32.2, -110.9


def _series(values, index=INDEX, name=None):
    return pd.Series(values, index=index, name=name)


def _type_case(name, kind):
    # functions added by the port raise AttributeError before it
    entry = SUPPORTED[name]
    kwargs = {k: _convert(kind, k, v) for k, v in entry.arrays.items()}
    return function(name)(**kwargs, **entry.params)


# --- solar position ---------------------------------------------------------

def _spa(time, **kwargs):
    return solarposition.spa_python(time, kwargs.pop('latitude', LAT),
                                    kwargs.pop('longitude', LON), **kwargs)


def _spa_numba():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # reloading spa warns
        try:
            return _spa(INDEX, how='numba', numthreads=2)
        finally:
            solarposition._spa_python_import('numpy')


SPA_CASES = {
    'spa_python[DatetimeIndex]': lambda: _spa(INDEX),
    'spa_python[naive DatetimeIndex]': lambda: _spa(
        INDEX.tz_convert('UTC').tz_localize(None)),
    'spa_python[Timestamp]': lambda: _spa(INDEX[1]),
    'spa_python[datetime]': lambda: _spa(INDEX[1].to_pydatetime()),
    'spa_python[datetime64 array]': lambda: _spa(
        INDEX.tz_convert('UTC').tz_localize(None).values),
    'spa_python[list of Timestamps]': lambda: _spa(list(INDEX)),
    'spa_python[datetime Series]': lambda: _spa(pd.Series(INDEX)),
    'spa_python[delta_t=None]': lambda: _spa(INDEX, delta_t=None),
    'spa_python[delta_t array]': lambda: _spa(
        INDEX, delta_t=np.array([67., 68., 69., 70.])),
    'spa_python[options]': lambda: _spa(
        INDEX, altitude=1500., pressure=85000., temperature=25.,
        atmos_refract=0.5),
    'spa_python[latitude array]': lambda: _spa(
        INDEX, latitude=np.array([30., 32., 34., 36.])),
}

# --- soiling ----------------------------------------------------------------

HOURS = pd.date_range('2026-06-01', periods=48, freq='1h', tz='Etc/GMT+7')
# no window sum equals the threshold of 1 mm, so ties don't matter
RAIN = pd.Series(0., index=HOURS)
RAIN.iloc[[5, 6, 20, 30, 31, 32]] = [0.7, 0.6, 1.6, 0.4, 0.4, 0.45]


def _hsu(rain=RAIN, **kwargs):
    return soiling.hsu(rain, 1.0, 20., 1e-2, 2e-2, **kwargs)


def _rain_with_gap():
    return RAIN.drop(RAIN.index[10:14])


def _rain_with_nan():
    rain = RAIN.copy()
    rain.iloc[[6, 31]] = np.nan
    return rain


HSU_CASES = {
    'hsu[default]': lambda: _hsu(),
    'hsu[Timedelta 2h]': lambda: _hsu(rain_accum_period=pd.Timedelta('2h')),
    'hsu[Timedelta 3h]': lambda: _hsu(rain_accum_period=pd.Timedelta('3h')),
    'hsu[timedelta 2h]': lambda: _hsu(
        rain_accum_period=datetime.timedelta(hours=2)),
    'hsu[depo_veloc]': lambda: _hsu(depo_veloc={'2_5': 0.002, '10': 0.006}),
    'hsu[Series pm]': lambda: soiling.hsu(
        RAIN, 1.0, 20., _series(np.linspace(1e-2, 2e-2, 48), HOURS),
        _series(np.linspace(2e-2, 3e-2, 48), HOURS)),
    'hsu[irregular times]': lambda: _hsu(_rain_with_gap()),
    'hsu[NaN rain]': lambda: _hsu(_rain_with_nan()),
    'hsu[UTC]': lambda: _hsu(RAIN.tz_convert('UTC')),
}

# --- snow -------------------------------------------------------------------

SNOWFALL = pd.Series(0., index=HOURS)
SNOWFALL.iloc[[3, 4, 20, 21, 40]] = [2., 1.5, 3., 0.5, 2.5]
POA = pd.Series(np.clip(600 * np.sin(np.linspace(0, 6 * np.pi, 48)), 0, None),
                index=HOURS)
TEMP_AIR = pd.Series(np.linspace(-6., 4., 48), index=HOURS)
SNOW_DEPTH = pd.Series(10., index=HOURS)
SNOW_DEPTH.iloc[30:36] = 0.5


def _coverage(snowfall=SNOWFALL, surface_tilt=35., **kwargs):
    index = snowfall.index
    return snow.coverage_nrel(snowfall, POA.loc[index], TEMP_AIR.loc[index],
                              surface_tilt, **kwargs)


def _with_nan(series, positions):
    series = series.copy()
    series.iloc[positions] = np.nan
    return series


def _quarter_hours():
    index = pd.date_range('2026-01-01', periods=96, freq='15min',
                          tz='Etc/GMT+7')
    snowfall = pd.Series(0., index=index)
    snowfall.iloc[[10, 11, 50]] = [0.4, 0.3, 0.6]
    poa = pd.Series(np.linspace(0., 700., 96), index=index)
    return snow.coverage_nrel(snowfall, poa, -2., 25.)


SNOW_CASES = {
    'coverage_nrel[default]': lambda: _coverage(),
    'coverage_nrel[snow_depth]': lambda: _coverage(snow_depth=SNOW_DEPTH),
    'coverage_nrel[initial_coverage]': lambda: _coverage(
        initial_coverage=0.6),
    'coverage_nrel[tilt Series]': lambda: _coverage(
        surface_tilt=pd.Series(np.linspace(10., 50., 48), index=HOURS)),
    'coverage_nrel[irregular times]': lambda: _coverage(
        SNOWFALL.drop(SNOWFALL.index[10:14]), snow_depth=SNOW_DEPTH.drop(
            SNOW_DEPTH.index[10:14])),
    'coverage_nrel[NaN]': lambda: snow.coverage_nrel(
        _with_nan(SNOWFALL, [4]), _with_nan(POA, [8, 9]),
        _with_nan(TEMP_AIR, [25]), 35.,
        snow_depth=_with_nan(SNOW_DEPTH, [33])),
    'coverage_nrel[NaN tilt]': lambda: _coverage(
        surface_tilt=_with_nan(pd.Series(30., index=HOURS), [0, 7, 30])),
    'coverage_nrel[options]': lambda: _coverage(
        threshold_snowfall=0.8, threshold_depth=2., can_slide_coefficient=-60.,
        slide_amount_coefficient=0.25, snow_depth=SNOW_DEPTH),
    'coverage_nrel[15min]': _quarter_hours,
    'fully_covered_nrel[default]': lambda: snow.fully_covered_nrel(SNOWFALL),
    'fully_covered_nrel[snow_depth]': lambda: snow.fully_covered_nrel(
        SNOWFALL, SNOW_DEPTH, threshold_snowfall=1.5, threshold_depth=2.),
    'fully_covered_nrel[irregular times]': lambda: snow.fully_covered_nrel(
        SNOWFALL.drop(SNOWFALL.index[[1, 2]])),
}

# --- extraterrestrial radiation and decomposition ---------------------------

EXTRA_CASES = {
    'get_extra_radiation[DatetimeIndex]': lambda:
        irradiance.get_extra_radiation(INDEX),
    'get_extra_radiation[Timestamp]': lambda:
        irradiance.get_extra_radiation(INDEX[1]),
    'get_extra_radiation[date]': lambda:
        irradiance.get_extra_radiation(datetime.date(2026, 6, 1)),
    'get_extra_radiation[datetime]': lambda:
        irradiance.get_extra_radiation(INDEX[1].to_pydatetime()),
    'get_extra_radiation[datetime64]': lambda:
        irradiance.get_extra_radiation(np.datetime64('2026-06-01T12:00')),
    'get_extra_radiation[asce]': lambda:
        irradiance.get_extra_radiation(INDEX, method='asce'),
    'get_extra_radiation[nrel]': lambda:
        irradiance.get_extra_radiation(INDEX, method='nrel'),
    'get_extra_radiation[solar_constant]': lambda:
        irradiance.get_extra_radiation(INDEX, solar_constant=1361.),
    'get_extra_radiation[int doy]': lambda:
        irradiance.get_extra_radiation(152),
    'erbs[Series, DatetimeIndex]': lambda: irradiance.erbs(
        _series(GHI, name='ghi'), _series(ZEN, name='zenith'), INDEX),
    'erbs[ndarray, DatetimeIndex]': lambda: irradiance.erbs(
        np.array(GHI), np.array(ZEN), INDEX),
    'erbs[scalar, Timestamp]': lambda: irradiance.erbs(500., 30., INDEX[1]),
    'erbs[options]': lambda: irradiance.erbs(
        np.array(GHI), np.array(ZEN), INDEX, min_cos_zenith=0.1,
        max_zenith=80.),
}


# --- transposition, including the models that aren't ported -----------------

def _transpose(model, kind='series', **kwargs):
    a = _series if kind == 'series' else np.array
    return irradiance.get_total_irradiance(
        a(TILT), a(SAZI), a(ZEN), a(AZI), a(DNI), a(GHI), a(DHI),
        model=model, **kwargs)


def _poa_components_dataframe():
    sky = irradiance.haydavies(_series(TILT), _series(SAZI), _series(DHI),
                               _series(DNI), 1360., _series(ZEN),
                               _series(AZI), return_components=True)
    aoi = irradiance.aoi(_series(TILT), _series(SAZI), _series(ZEN),
                         _series(AZI))
    return irradiance.poa_components(aoi, _series(DNI), sky, _series(DHI))


MODELS = ['isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez',
          'perez-driesse']
TRANSPOSITION_CASES = {
    **{f'get_total_irradiance[{m}]':
       functools.partial(_transpose, m, dni_extra=1360.) for m in MODELS},
    **{f'get_total_irradiance[{m}, components]':
       functools.partial(_transpose, m, dni_extra=1360.,
                         diffuse_components=True)
       for m in ['isotropic', 'haydavies', 'reindl', 'perez',
                 'perez-driesse']},
    **{f'get_total_irradiance[{m}, ndarray]':
       functools.partial(_transpose, m, 'ndarray', dni_extra=1360.)
       for m in ['klucher', 'reindl', 'perez']},
    'get_total_irradiance[perez, airmass]': functools.partial(
        _transpose, 'perez', dni_extra=1360.,
        airmass=_series([1.1, 1.3, 2., 5.6])),
    'get_total_irradiance[albedo Series]': functools.partial(
        _transpose, 'isotropic', albedo=_series([0.1, 0.2, 0.3, 0.4])),
    'get_total_irradiance[surface_type]': functools.partial(
        _transpose, 'isotropic', surface_type='snow'),
    'get_sky_diffuse[perez]': lambda: irradiance.get_sky_diffuse(
        _series(TILT), _series(SAZI), _series(ZEN), _series(AZI),
        _series(DNI), _series(GHI), _series(DHI), dni_extra=1360.,
        model='perez'),
    'poa_components[DataFrame]': _poa_components_dataframe,
    'haydavies[projection_ratio]': lambda: irradiance.haydavies(
        _series(TILT), _series(SAZI), _series(DHI), _series(DNI), 1360.,
        projection_ratio=_series([0.9, 1.1, 1.4, 2.0])),
    'isotropic[components]': lambda: irradiance.isotropic(
        _series(TILT), _series(DHI), return_components=True),
    'beam_component[behind]': lambda: irradiance.beam_component(
        np.array(TILT), np.array(SAZI), np.array([100., 120., 60., 40.]),
        np.array(AZI), np.array(DNI)),
    'clearness_index[options]': lambda: irradiance.clearness_index(
        np.array(GHI), np.array(ZEN), 1360., min_cos_zenith=0.1,
        max_clearness_index=1.5),
}

# --- other models -----------------------------------------------------------

AIRMASS_MODELS = ['simple', 'kasten1966', 'youngirvine1967', 'kastenyoung1989',
                  'gueymard1993', 'young1994', 'pickering2002',
                  'gueymard2003']
OTHER_CASES = {
    **{f'get_relative_airmass[{m}]':
       functools.partial(atmosphere.get_relative_airmass,
                         _series([20., 60., 89., 95.]), m)
       for m in AIRMASS_MODELS},
    'ashrae[NaN and negative]': lambda: iam.ashrae(
        np.array([np.nan, -30., 89.9, 90.])),
    'ashrae[b array]': lambda: iam.ashrae(
        np.array([0., 30., 60., 80.]), b=np.array([0.04, 0.05, 0.06, 0.07])),
    'martin_ruiz[a_r array]': lambda: iam.martin_ruiz(
        np.array([0., 30., 60., 80.]), a_r=np.array([0.16, 0.2, 0.25, 0.3])),
    'martin_ruiz[NaN]': lambda: iam.martin_ruiz(np.array([np.nan, 30.])),
    'pvwatts[zero and negative]': lambda: inverter.pvwatts(
        np.array([0., -5., 50., 300.]), 240.),
    'pvwatts[pdc0 array]': lambda: inverter.pvwatts(
        np.array([10., 50., 150., 260.]),
        np.array([200., 240., 240., 280.])),
    'pvwatts_dc[cap_adjustment]': lambda: pvsystem.pvwatts_dc(
        _series([100., 400., 1000., 1200.]), _series(TEMP), 250., -0.004,
        k=0.01, cap_adjustment=True),
    'pvwatts_dc[temp_ref]': lambda: pvsystem.pvwatts_dc(
        np.array(GHI), np.array(TEMP), 250., -0.004, temp_ref=20.),
    'sapm_module': lambda: temperature.sapm_module(
        _series(GHI), _series(TEMP), _series(WIND), -3.56, -0.075),
    'sapm_cell_from_module': lambda: temperature.sapm_cell_from_module(
        _series(TEMP), _series(GHI), 3., irrad_ref=900.),
    'faiman[u0 array]': lambda: temperature.faiman(
        np.array(GHI), np.array(TEMP), np.array(WIND),
        u0=np.array([20., 25., 30., 35.])),
    'ineichen[perez_enhancement]': lambda: clearsky.ineichen(
        _series(ZEN), _series([1.06, 1.3, 2., 5.6]), 3., altitude=1000.,
        perez_enhancement=True),
    'ineichen[nan turbidity]': lambda: clearsky.ineichen(
        np.array(ZEN), np.array([1.06, 1.3, 2., 5.6]),
        np.array([2., np.nan, 3., 4.])),
    'Location.get_clearsky[ineichen]': lambda: location.Location(
        LAT, LON, altitude=700.).get_clearsky(HOURS[:24]),
}

# --- inputs that must not be accepted silently ------------------------------

REJECT_CASES = {
    'aoi[str]': Case(lambda: irradiance.aoi('30', 180., 30., 180.),
                     policy='raise'),
    'aoi[str with array]': Case(
        lambda: irradiance.aoi('30', 180., np.array(ZEN), 180.),
        policy='raise'),
    'ashrae[str]': Case(lambda: iam.ashrae('60'), policy='raise'),
    'faiman[str]': Case(lambda: temperature.faiman('800', 20.),
                        policy='raise'),
    'ashrae[DatetimeIndex]': Case(lambda: iam.ashrae(INDEX), policy='raise'),
    'aoi[DatetimeIndex]': Case(
        lambda: irradiance.aoi(30., 180., INDEX, 180.), policy='raise'),
    'faiman[datetime64 array]': Case(
        lambda: temperature.faiman(
            INDEX.tz_localize(None).values, 20.), policy='raise'),
    'get_relative_airmass[bad model]': Case(
        lambda: atmosphere.get_relative_airmass(30., 'nonsense'),
        policy='raise'),
    'martin_ruiz[a_r=0]': Case(lambda: iam.martin_ruiz(30., a_r=0.),
                               policy='raise'),
}


def _build():
    cases = {}
    for name, entry in SUPPORTED.items():
        if 'unix' in entry.time.values():
            continue
        for kind in KINDS:
            cases[f'{name}[{kind}]'] = Case(
                functools.partial(_type_case, name, kind))
    for group in (SPA_CASES, HSU_CASES, SNOW_CASES, EXTRA_CASES,
                  TRANSPOSITION_CASES, OTHER_CASES):
        cases.update({k: Case(v) for k, v in group.items()})
    cases['spa_python[how=numba]'] = Case(_spa_numba, requires=('numba',))
    cases.update(REJECT_CASES)
    return cases


CASES = _build()


# --- results as JSON --------------------------------------------------------

def _array(values):
    try:
        return np.asarray(values, dtype=float).tolist()
    except (TypeError, ValueError):
        return [str(v) for v in np.ravel(values)]


def summarize(out):
    """What a call returned, as JSON-compatible data, or the exception's
    class name if ``out`` is an exception."""
    if isinstance(out, BaseException):
        return {'type': 'error', 'error': type(out).__name__}
    if isinstance(out, pd.DataFrame):
        # the JSON file sorts keys, so the column order is kept separately
        return {'type': 'DataFrame', 'index': [str(i) for i in out.index],
                'order': [str(k) for k in out],
                'columns': {str(k): summarize(out[k]) for k in out}}
    if isinstance(out, pd.Series):
        return {'type': 'Series', 'index': [str(i) for i in out.index],
                'name': None if out.name is None else str(out.name),
                'dtype': str(out.dtype), 'values': _array(out)}
    if isinstance(out, dict):
        return {'type': 'dict',
                'items': {str(k): summarize(v) for k, v in out.items()}}
    if isinstance(out, tuple):
        return {'type': 'tuple', 'items': [summarize(o) for o in out]}
    if isinstance(out, np.ma.MaskedArray):
        return {'type': 'MaskedArray', 'shape': list(out.shape),
                'dtype': str(out.dtype),
                'values': _array(np.ma.filled(out.astype(float), np.nan))}
    scalars = (int, float, np.generic, np.ndarray)
    if np.ndim(out) == 0 and isinstance(out, scalars):
        # NumPy scalars and 0-d arrays alike
        return {'type': 'scalar', 'dtype': str(np.asarray(out).dtype),
                'values': _array(out)}
    if isinstance(out, np.ndarray):
        return {'type': 'ndarray', 'shape': list(out.shape),
                'dtype': str(out.dtype), 'values': _array(out)}
    return {'type': type(out).__name__}


def run(case):
    """Call ``case`` and summarize the result or the exception."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return summarize(case.call())
        except Exception as e:  # the error is the result
            return summarize(e)
