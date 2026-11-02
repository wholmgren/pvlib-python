"""
This module contains functions for soiling models
"""

import datetime
import numpy as np
import pandas as pd

from pvlib.tools import cosd
from pvlib import _array_api as _xa


@_xa.labeled_io(name=None, time={'times': 'unix'},
                timedelta=('rain_accum_period',), index_time='times')
def hsu(rainfall, cleaning_threshold, surface_tilt, pm2_5, pm10,
        depo_veloc=None, rain_accum_period=pd.Timedelta('1h'), times=None):
    """
    Calculates soiling ratio given particulate and rain data using the
    Fixed Velocity model from Humboldt State University (HSU).

    The HSU soiling model [1]_ returns the soiling ratio, a value between
    zero and one which is equivalent to (1 - transmission loss).
    Therefore a soiling ratio of 1.0 is equivalent to zero transmission loss.
    Due to the mathematical form of the HSU model, the soiling ratio has a
    minimum of approximately 0.6563. See ``Notes`` for details.

    Parameters
    ----------

    rainfall : Series or array
        Rain accumulated in each time period. [mm]

    cleaning_threshold : float
        Amount of rain in an accumulation period needed to clean the PV
        modules. [mm]

    surface_tilt : numeric
        Tilt of the PV panels from horizontal. [degree]

    pm2_5 : numeric
        Concentration of airborne particulate matter (PM) with
        aerodynamic diameter less than 2.5 microns. [g/m^3]

    pm10 : numeric
        Concentration of airborne particulate matter (PM) with
        aerodynamicdiameter less than 10 microns. [g/m^3]

    depo_veloc : dict, default {'2_5': 0.0009, '10': 0.004}
        Deposition or settling velocity of particulates. [m/s]

    rain_accum_period : Timedelta, default 1 hour
        Period for accumulating rainfall to check against `cleaning_threshold`
        It is recommended that `rain_accum_period` be between 1 hour and
        24 hours. A number is in seconds.

    times : DatetimeIndex or array, optional
        Times of ``rainfall``, increasing. Taken from the index of
        ``rainfall`` if it is a Series; required otherwise. A numeric array
        is float64 seconds since 1970-01-01 00:00 UTC.

    Returns
    -------
    soiling_ratio : Series or array
        Values between 0 and 1. Equal to 1 - transmission loss.

    Notes
    -------
    Due to the mathematical form of the HSU model
    (``SR = 1 - 0.3437 * erf(0.17 * ω^0.8473)``),
    the soiling ratio has a minimum value of approximately 0.6563
    (i.e., maximum transmission loss of ~34.37%), regardless of
    the accumulated particulate mass. The HSU model is developed
    (validated) for accumulated mass densities up to 10 g/m²,
    corresponding to a soiling ratio of approximately 0.6875.
    See [1]_ for details.

    As recommended by [1]_, the default values of ``depo_veloc``
    are the settling velocities of particulates, i.e., only the gravitational
    movement of particles.

    References
    -----------
    .. [1] M. Coello and L. Boyle, "Simple Model For Predicting Time Series
       Soiling of Photovoltaic Panels," in IEEE Journal of Photovoltaics.
       :doi:`10.1109/JPHOTOV.2019.2919628`
    .. [2] Atmospheric Chemistry and Physics: From Air Pollution to Climate
       Change. J. Seinfeld and S. Pandis. Wiley and Sons 2001.

    """
    # never use mutable input arguments
    if depo_veloc is None:
        depo_veloc = {'2_5': 0.0009, '10': 0.004}

    # times is seconds since 1970-01-01 UTC and rain_accum_period is
    # seconds, from labeled_io
    xp = _xa.array_namespace(rainfall, times, surface_tilt, pm2_5, pm10)
    rainfall, times, surface_tilt, pm2_5, pm10 = _xa.asarrays(
        rainfall, times, surface_tilt, pm2_5, pm10, xp=xp)

    # accumulate rainfall into periods for comparison with threshold.
    # rolling(closed='right') skips missing values
    rain = xp.where(xp.isnan(rainfall), 0., rainfall)
    accum_rain = _xa.rolling_sum(rain, times, rain_accum_period, xp=xp)
    # cleaning is True for intervals with rainfall greater than threshold
    cleaning = accum_rain >= cleaning_threshold

    # determine the time intervals in seconds (dt_sec), assuming that the
    # interval prior to the first value is equal in length to the first
    dt_diff = xp.diff(times)
    dt_sec = xp.concat([dt_diff[:1], dt_diff])

    horiz_mass_rate = (
        pm2_5 * depo_veloc['2_5'] + xp.maximum(pm10 - pm2_5, 0.)
        * depo_veloc['10']) * dt_sec
    tilted_mass_rate = horiz_mass_rate * cosd(surface_tilt)  # assuming no rain

    # tms -> tilt_mass_rate
    mass_no_cleaning = xp.cumulative_sum(
        tilted_mass_rate + xp.zeros_like(rainfall))
    # each cleaning removes all the mass accumulated up to that time
    mass_removed = _xa.take_last(mass_no_cleaning, cleaning, xp=xp)
    accum_mass = mass_no_cleaning - mass_removed

    # accum_mass**0.8473 has an infinite derivative at 0, e.g. after cleaning
    mass_term = _xa.where_safe(accum_mass > 0, accum_mass,
                               lambda m: m**0.8473, 0., xp)
    soiling_ratio = 1 - 0.3437 * _xa.erf(0.17 * mass_term, xp)

    return soiling_ratio


def kimber(rainfall, cleaning_threshold=6, soiling_loss_rate=0.0015,
           grace_period=14, max_soiling=0.3, manual_wash_dates=None,
           initial_soiling=0, rain_accum_period=24):
    """
    Calculates fraction of energy lost due to soiling given rainfall data and
    daily loss rate using the Kimber model.

    Kimber soiling model [1]_ assumes soiling builds up at a daily rate unless
    the daily rainfall is greater than a threshold. The model also assumes that
    if daily rainfall has exceeded the threshold within a grace period, then
    the ground is too damp to cause soiling build-up. The model also assumes
    there is a maximum soiling build-up. Scheduled manual washes and rain
    events are assumed to reset soiling to zero.

    Parameters
    ----------
    rainfall: pandas.Series
        Accumulated rainfall at the end of each time period. [mm]
    cleaning_threshold: float, default 6
        Amount of daily rainfall required to clean the panels. [mm]
    soiling_loss_rate: float, default 0.0015
        Fraction of energy lost due to one day of soiling. [unitless]
    grace_period : int, default 14
        Number of days after a rainfall event when it's assumed the ground is
        damp, and so it's assumed there is no soiling. [days]
    max_soiling : float, default 0.3
        Maximum fraction of energy lost due to soiling. Soiling will build up
        until this value. [unitless]
    manual_wash_dates : sequence, optional
        List or tuple of dates as Python ``datetime.date`` when the panels were
        washed manually. Note there is no grace period after a manual wash, so
        soiling begins to build up immediately.
    initial_soiling : float, default 0
        Initial fraction of energy lost due to soiling at time zero in the
        `rainfall` series input. [unitless]
    rain_accum_period : int, default 24
        Period for accumulating rainfall to check against `cleaning_threshold`.
        The Kimber model defines this period as one day. [hours]

    Returns
    -------
    pandas.Series
        fraction of energy lost due to soiling, has same intervals as input

    Notes
    -----
    The soiling loss rate depends on both the geographical region and the
    soiling environment type. Rates measured by Kimber [1]_ are summarized in
    the following table:

    ===================  =======  =========  ======================
    Region/Environment   Rural    Suburban   Urban/Highway/Airport
    ===================  =======  =========  ======================
    Central Valley       0.0011   0.0019     0.0020
    Northern CA          0.0011   0.0010     0.0016
    Southern CA          0        0.0016     0.0019
    Desert               0.0030   0.0030     0.0030
    ===================  =======  =========  ======================

    Rainfall thresholds and grace periods may also vary by region. Please
    consult [1]_ for more information.

    References
    ----------
    .. [1] "The Effect of Soiling on Large Grid-Connected Photovoltaic Systems
       in California and the Southwest Region of the United States," Adrianne
       Kimber, et al., IEEE 4th World Conference on Photovoltaic Energy
       Conference, 2006, :doi:`10.1109/WCPEC.2006.279690`
    """
    # convert rain_accum_period to timedelta
    rain_accum_period = datetime.timedelta(hours=rain_accum_period)

    # convert grace_period to timedelta
    grace_period = datetime.timedelta(days=grace_period)

    # get indices as np.datetime64 (UTC), calculate timestep as np.timedelta64,
    # and convert timestep to fraction of days
    rain_index_vals = rainfall.index.tz_convert(None).to_numpy()
    timestep_interval = (rain_index_vals[1] - rain_index_vals[0])
    day_fraction = timestep_interval / np.timedelta64(24, 'h')

    # accumulate rainfall
    accumulated_rainfall = rainfall.rolling(
        rain_accum_period, closed='right').sum()

    # soiling rate
    soiling = np.ones_like(rainfall.values) * soiling_loss_rate * day_fraction
    soiling[0] = initial_soiling
    soiling = np.cumsum(soiling)
    soiling = pd.Series(soiling, index=rainfall.index, name='soiling')

    # rainfall events that clean the panels
    rain_events = accumulated_rainfall > cleaning_threshold

    # grace periods windows during which ground is assumed damp, so no soiling
    grace_windows = rain_events.rolling(grace_period, closed='right').sum() > 0

    # clean panels by subtracting soiling for indices in grace period windows
    cleaning = pd.Series(float('NaN'), index=rainfall.index)
    cleaning.iloc[0] = 0.0
    cleaning[grace_windows] = soiling[grace_windows]

    # manual wash dates
    if manual_wash_dates is not None:
        rain_tz = rainfall.index.tz
        # convert manual wash dates to datetime index in the timezone of rain
        manual_wash_dates = pd.DatetimeIndex(manual_wash_dates, tz=rain_tz)
        cleaning[manual_wash_dates] = soiling[manual_wash_dates]

    # remove soiling by foward filling cleaning where NaN
    soiling -= cleaning.ffill()

    # check if soiling has reached the maximum
    return soiling.where(soiling < max_soiling, max_soiling)
