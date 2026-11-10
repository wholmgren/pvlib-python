"""
The ``snow`` module contains functions that model module snow cover and the
associated effects on PV module output
"""

import numpy as np

from pvlib import _array_api as _xa
from pvlib.tools import _sind, cosd, tand


@_xa.labeled_io(name=None,
                arrays=('snowfall', 'snow_depth', 'threshold_snowfall',
                        'threshold_depth', 'times'),
                time={'times': 'unix'}, index_time='times')
def fully_covered_nrel(snowfall, snow_depth=None, threshold_snowfall=1.,
                       threshold_depth=1., times=None, *, xp):
    '''
    Calculates the timesteps when modules are fully covered by snow.

    Parameters
    ----------
    snowfall: Series, array or DataArray
        Snowfall in each time period. [cm]
    snow_depth: Series, array or DataArray, optional
        Snow depth on the ground at the beginning of each time period.
        Must have the same index as ``snowfall``. [cm]
    threshold_snowfall: float, default 1.0
        Hourly snowfall above which the row is fully covered for that hour.
        [cm/hr]
    threshold_depth: float, default 1.0
        Snow depth on the ground, above which snow can affect the modules. [cm]
    times : DatetimeIndex, array or DataArray, optional
        Times of the data, increasing. Leave it out for Series: their index
        is the time. Otherwise it is required and 1-D, as datetimes or as
        float64 seconds since 1970-01-01 00:00 UTC: a DataArray, such as
        ``da.time``, for DataArray data, whose dimension is time, and for
        other arrays one along their last axis. See
        :ref:`array-libraries-time-series`.

    Returns
    ----------
    covered: Series, array or DataArray
        Boolean, True where the snowfall exceeds the defined threshold to
        fully cover the panel.

    Notes
    -----
    Implements the model described in [1]_ with minor improvements in [2]_.

    ``snow_depth`` is used to return `False` (not fully covered) when snow
    is less than ``threshold_depth``. This check is described in [2]_ as needed
    for systems with low tilt angle.

    The first snowfall value is used only if all time steps are equally
    long, since the length of the step before it is unknown.

    References
    ----------
    .. [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
       "Measured and modeled photovoltaic system energy losses from snow for
       Colorado and Wisconsin locations." Solar Energy 97; pp.112-121.
    .. [2] Ryberg, D; Freeman, J. "Integration, Validation, and Application
       of a PV Snow Coverage Model in SAM" (2017) NREL Technical Report
       NREL/TP-6A20-68705
    '''
    # times is Unix seconds along the last axis, from labeled_io
    if times.shape[0] < 2:
        raise ValueError('at least two times are needed for the length of '
                         'the time steps')
    hours = xp.diff(times) / 3600.
    # if all steps are equal, the step before the first is taken to be too,
    # as a pandas frequency would; otherwise the first snowfall is ignored
    regular = xp.all(hours == hours[:1])
    first = xp.where(regular, snowfall[..., :1] / hours[:1], 0.)
    hourly_snow_rate = xp.concat([first, snowfall[..., 1:] / hours], axis=-1)
    covered = hourly_snow_rate > threshold_snowfall
    # no coverage when no snow on the ground
    if snow_depth is not None:
        covered = covered & (snow_depth >= threshold_depth)
    return covered


@_xa.labeled_io(name=None,
                arrays=('snowfall', 'poa_irradiance', 'temp_air',
                        'surface_tilt', 'snow_depth', 'initial_coverage',
                        'threshold_snowfall', 'threshold_depth',
                        'can_slide_coefficient', 'slide_amount_coefficient',
                        'times'),
                time={'times': 'unix'}, index_time='times')
def coverage_nrel(snowfall, poa_irradiance, temp_air, surface_tilt,
                  snow_depth=None, initial_coverage=0, threshold_snowfall=1.,
                  threshold_depth=1., can_slide_coefficient=-80.,
                  slide_amount_coefficient=0.197, times=None, *, xp):
    '''
    Calculates the fraction of the slant height of a row of modules that is
    covered by snow at every time step.

    Implements the model described in [1]_ with minor improvements in [2]_,
    with the change that the output is in fraction of the row's slant height
    rather than in tenths of the row slant height. As described in [1]_, model
    validation focused on fixed tilt systems.

    Parameters
    ----------
    snowfall : Series, array or DataArray
        Snowfall within each time period. [cm]
    poa_irradiance : Series, array or DataArray
        Total in-plane irradiance [W/m^2]
    temp_air : Series, array or DataArray
        Ambient air temperature [C]
    surface_tilt : numeric
        Tilt of module's from horizontal, e.g. surface facing up = 0,
        surface facing horizon = 90. [degrees]
    snow_depth : Series, array or DataArray, optional
        Snow depth on the ground at the beginning of each time period.
        Must have the same index as ``snowfall``. [cm]
    initial_coverage : float, default 0
        Fraction of row's slant height that is covered with snow at the
        beginning of the simulation. [unitless]
    threshold_snowfall: float, default 1.0
        Hourly snowfall above which the row is fully covered for that hour.
        [cm/hr]
    threshold_depth: float, default 1.0
        Snow depth on the ground, above which snow can affect the modules. [cm]
    can_slide_coefficient : float, default -80.
        Coefficient to determine if snow can slide given irradiance and air
        temperature. [W/(m^2 C)]
    slide_amount_coefficient : float, default 0.197
        Coefficient to determine fraction of snow that slides off in one hour.
        [unitless]
    times : DatetimeIndex, array or DataArray, optional
        Times of the data, increasing. Leave it out for Series: their index
        is the time. Otherwise it is required and 1-D, as datetimes or as
        float64 seconds since 1970-01-01 00:00 UTC: a DataArray, such as
        ``da.time``, for DataArray data, whose dimension is time, and for
        other arrays one along their last axis. See
        :ref:`array-libraries-time-series`.

    Returns
    -------
    snow_coverage : Series, array or DataArray
        The fraction of the slant height of a row of modules that is covered
        by snow at each time step.

    Notes
    -----
    In [1]_, ``can_slide_coefficient`` is termed `m`, and the value of
    ``slide_amount_coefficient`` is given in tenths of a module's slant height.

    ``snow_depth`` is used to set ``snow_coverage`` to 0 (not fully covered)
    when snow is less than ``threshold_depth``. This check is described in
    [2]_ as needed for systems with low tilt angle.

    References
    ----------
    .. [1] Marion, B.; Schaefer, R.; Caine, H.; Sanchez, G. (2013).
       "Measured and modeled photovoltaic system energy losses from snow for
       Colorado and Wisconsin locations." Solar Energy 97; pp.112-121.
    .. [2] Ryberg, D; Freeman, J. (2017). "Integration, Validation, and
       Application of a PV Snow Coverage Model in SAM" NREL Technical Report
       NREL/TP-6A20-68705
    '''

    # find times with new snowfall
    new_snowfall = fully_covered_nrel.core(
        snowfall, snow_depth, threshold_snowfall, threshold_depth, times,
        xp=xp)

    # determine amount that snow can slide in each timestep
    can_slide = temp_air > poa_irradiance / can_slide_coefficient
    hours = xp.diff(times) / 3600.
    hours = xp.concat([xp.zeros_like(hours[:1]), hours])
    slide_amt = slide_amount_coefficient * _sind(surface_tilt, xp) * hours
    # no slide when snow can't slide, during snow events, or in the interval
    # preceding the snowfall data
    not_first = xp.concat([xp.zeros_like(times[:1]),
                           xp.ones_like(times[1:])]) > 0
    slide_amt = xp.where(can_slide & ~new_snowfall & not_first, slide_amt, 0.)

    if snow_depth is not None:
        # All slides off if snow on the ground is less than threshold_depth.
        # Described in [2] to avoid non-sliding snow for low-tilt systems.
        # Default threshold_depth of 1cm is from [2[ and SAM's implementation.
        # https://github.com/NatLabRockies/ssc/issues/1265
        slide_amt = xp.where(snow_depth < threshold_depth, 1., slide_amt)

    # cumulative slide amounts, restarting at each snowfall. slide_amt is 0
    # at snowfall steps, so that is the running total less its value at the
    # latest snowfall. Missing values are skipped, as pandas' cumsum does
    missing = xp.isnan(slide_amt)
    total = xp.cumulative_sum(xp.where(missing, 0., slide_amt), axis=-1)
    cumulative_sliding = total - _xa.take_last(total, new_snowfall, xp)

    # snow coverage without any sliding: 1 from the first snowfall on, and
    # initial_coverage before it
    snowed = xp.cumulative_sum(xp.astype(new_snowfall, total.dtype),
                               axis=-1) > 0
    snow_coverage = xp.where(snowed, 1., initial_coverage) - cumulative_sliding
    snow_coverage = xp.where(missing, xp.nan, snow_coverage)

    return xp.clip(snow_coverage, min=0.)


def dc_loss_nrel(snow_coverage, num_strings):
    '''
    Calculates the fraction of DC capacity lost due to snow coverage.

    DC capacity loss assumes that if a string is partially covered by snow,
    the string's capacity is lost; see [1]_, Eq. 11.8.

    Module orientation is accounted for by specifying the number of cell
    strings in parallel along the slant height.
    For example, a typical 60-cell module has 3 parallel strings, each
    comprising 20 cells in series, with the cells arranged in 6 columns of 10
    cells each. For a row consisting of single modules, if the module is
    mounted in portrait orientation, i.e., the row slant height is along a
    column of 10 cells, there is 1 string in parallel along the row slant
    height, so `num_strings=1`. In contrast, if the module is mounted in
    landscape orientation with the row slant height comprising 6 cells, there
    are 3 parallel strings along the row slant height, so `num_strings=3`.

    Parameters
    ----------
    snow_coverage : numeric
        The fraction of row slant height covered by snow at each time step.

    num_strings: int
        The number of parallel-connected strings along a row slant height.

    Returns
    -------
    loss : numeric
        fraction of DC capacity loss due to snow coverage at each time step.

    References
    ----------
    .. [1] Gilman, P. et al., (2018). "SAM Photovoltaic Model Technical
       Reference Update", NREL Technical Report NREL/TP-6A20-67399.
       Available at https://www.nlr.gov/docs/fy18osti/67399.pdf
    '''
    return np.ceil(snow_coverage * num_strings) / num_strings


def _townsend_effective_snow(snow_total, snow_events):
    '''
    Calculates effective snow using the total snowfall received each month and
    the number of snowfall events each month.

    Parameters
    ----------
    snow_total : array-like
        Snow received each month. Referred to as S in [1]_. [cm]

    snow_events : array-like
        Number of snowfall events each month. Referred to as N in [1]_. [-]

    Returns
    -------
    effective_snowfall : array-like
        Effective snowfall as defined in the Townsend model. [cm]

    References
    ----------
    .. [1] Townsend, Tim & Powers, Loren. (2011). Photovoltaics and snow: An
       update from two winters of measurements in the SIERRA. 37th IEEE
       Photovoltaic Specialists Conference, Seattle, WA, USA.
       :doi:`10.1109/PVSC.2011.6186627`
    '''
    snow_events_no_zeros = np.maximum(snow_events, 1)
    effective_snow = 0.5 * snow_total * (1 + 1 / snow_events_no_zeros)
    return np.where(snow_events > 0, effective_snow, 0)


def loss_townsend(snow_total, snow_events, surface_tilt, relative_humidity,
                  temp_air, poa_global, slant_height, lower_edge_height,
                  string_factor=1.0, angle_of_repose=40,
                  front_side_fraction=1.0):
    '''
    Calculates monthly snow loss based on the Townsend monthly snow loss
    model.

    This model is described in [1]_.

    Parameters
    ----------
    snow_total : array-like
        Snow received each month. Referred to as S in [1]_. [cm]

    snow_events : array-like
        Number of snowfall events each month. Snow events are defined as days
        in the month that have snowfall greater than 1 inch. May be int or
        float type for the average events in a typical month. Referred to as N
        in [1]_.

    surface_tilt : float
        Tilt angle of the array. [deg]

    relative_humidity : array-like
        Monthly average relative humidity. [%]

    temp_air : array-like
        Monthly average ambient temperature. [C]

    poa_global : array-like
        Monthly plane of array insolation. [Wh/m2]

    slant_height : float
        Row length in the slanted plane of array dimension. [m]

    lower_edge_height : float
        Distance from array lower edge to the ground. [m]

    string_factor : float, default 1.0
        Multiplier applied to monthly loss fraction. Use 1.0 if the DC array
        has only one string of modules in the slant direction, use 0.75
        otherwise. [-]

    angle_of_repose : float, default 40
        Piled snow angle, assumed to stabilize at 40°, the midpoint of
        25°-55° avalanching slope angles. [deg]

    front_side_fraction : numeric or array-like, default 1.0
        Fraction of monthly energy from front-side insolation. [unitless]
        Multiplies the calculated loss fraction. For example,
        use 0.9 when 90% of monthly energy is from the front side
        of a bifacial system and 10% is from the rear side.

    Returns
    -------
    loss : array-like
        Monthly average DC capacity loss fraction due to snow coverage.

    Notes
    -----
    This model has not been validated for tracking arrays; however, for
    tracking arrays [1]_ suggests using the maximum rotation angle in place
    of ``surface_tilt``. The author of [1]_ recommends using one-half the
    table width for ``slant_height``, i.e., the distance from the tracker
    axis to the module edge.

    The parameter `string_factor` is an enhancement added to the model after
    publication of [1]_, as described in [2]_.
    The definition for snow events documented above is based on [3]_.

    For bifacial systems, [2]_ recommends including both front-side and
    rear-side insolation in ``poa_global``. The resulting loss is
    scaled by the front-side energy fraction ``front_side_fraction``.

    References
    ----------
    .. [1] Townsend, Tim & Powers, Loren. (2011). Photovoltaics and snow: An
       update from two winters of measurements in the SIERRA. 37th IEEE
       Photovoltaic Specialists Conference, Seattle, WA, USA.
       :doi:`10.1109/PVSC.2011.6186627`
    .. [2] Townsend, T. and Previtali, J. (2023). A Fresh Dusting: Current
        Uses of the Townsend Snow Model. In "Photovoltaic Reliability
        Workshop (PVRW) 2023 Proceedings: Posters.", ed. Silverman,
        T. J. Dec. 2023. NREL/CP-5900-87918.
        Available at: https://www.osti.gov/biblio/2229734
    .. [3] Townsend, T. (2013). Predicting PV Energy Loss Caused by Snow.
        Solar Power International, Chicago IL.
        :doi:`10.13140/RG.2.2.14299.68647`
    '''

    # unit conversions from cm and m to in, from C to K, and from % to fraction
    # doing this early to facilitate comparison of this code with [1]
    snow_total_inches = snow_total / 2.54  # to inches
    relative_humidity_fraction = relative_humidity / 100.
    poa_global_kWh = poa_global / 1000.
    slant_height_inches = slant_height * 39.37
    lower_edge_height_inches = lower_edge_height * 39.37
    temp_air_kelvin = temp_air + 273.15

    C1 = 5.7e04
    C2 = 0.51

    snow_total_prev = np.roll(snow_total_inches, 1)
    snow_events_prev = np.roll(snow_events, 1)

    effective_snow = _townsend_effective_snow(snow_total_inches, snow_events)
    effective_snow_prev = _townsend_effective_snow(
        snow_total_prev,
        snow_events_prev
    )
    effective_snow_weighted = (
        1 / 3 * effective_snow_prev
        + 2 / 3 * effective_snow
    )

    # the lower limit of 0.1 in^2 is per private communication with the model's
    # author. CWH 1/30/2023
    lower_edge_distance = np.clip(
        lower_edge_height_inches**2 - effective_snow_weighted**2, a_min=0.1,
        a_max=None)
    gamma = (
        slant_height_inches
        * effective_snow_weighted
        * cosd(surface_tilt)
        / lower_edge_distance
        * 2
        * tand(angle_of_repose)
    )

    ground_interference_term = 1 - C2 * np.exp(-gamma)

    # Calculate Eqn. 3 in the reference.
    # Although the reference says Eqn. 3 calculates percentage loss, the y-axis
    # of Figure 7 indicates Eqn. 3 calculates fractional loss. Since the slope
    # of the line in Figure 7 is the same as C1 in Eqn. 3, it is assumed that
    # Eqn. 3 calculates fractional loss.

    loss_fraction = (
        C1
        * effective_snow_weighted
        * cosd(surface_tilt)**2
        * ground_interference_term
        * relative_humidity_fraction
        / temp_air_kelvin**2
        / poa_global_kWh**0.67
        * string_factor
    )

    loss_fraction = loss_fraction * front_side_fraction

    return np.clip(loss_fraction, 0, 1)
