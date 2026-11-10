.. _array_libraries:

Array libraries
===============

Most pvlib functions work with NumPy arrays and pandas objects. Some also
work with xarray DataArrays, and with arrays from any library that supports
the `Python array API standard <https://data-apis.org/array-api/latest/>`_,
such as PyTorch, JAX, CuPy and Dask. With them you can

- model a grid of locations or orientations with xarray, broadcasting by
  dimension name;
- differentiate a model with PyTorch or JAX, to fit it to data or to see how
  sensitive its output is to each input;
- compile a model with ``jax.jit``, or run it on a GPU;
- model data larger than memory with Dask.

Support for array libraries other than NumPy is experimental, and details
may change between releases.

Supported functions
-------------------

So far these functions accept arrays from other libraries:

- solar position: :py:func:`~pvlib.solarposition.spa_python`
- clear sky: :py:func:`~pvlib.clearsky.haurwitz`,
  :py:func:`~pvlib.clearsky.ineichen`
- irradiance: :py:func:`~pvlib.irradiance.aoi`,
  :py:func:`~pvlib.irradiance.aoi_projection`,
  :py:func:`~pvlib.irradiance.beam_component`,
  :py:func:`~pvlib.irradiance.get_total_irradiance` and
  :py:func:`~pvlib.irradiance.get_sky_diffuse` with the
  ``'isotropic'`` and ``'haydavies'`` models,
  :py:func:`~pvlib.irradiance.poa_components`,
  :py:func:`~pvlib.irradiance.get_ground_diffuse`,
  :py:func:`~pvlib.irradiance.isotropic`,
  :py:func:`~pvlib.irradiance.haydavies`,
  :py:func:`~pvlib.irradiance.get_extra_radiation` with the ``'asce'`` and
  ``'spencer'`` methods, :py:func:`~pvlib.irradiance.clearness_index`,
  :py:func:`~pvlib.irradiance.erbs`
- atmosphere: :py:func:`~pvlib.atmosphere.get_relative_airmass`,
  :py:func:`~pvlib.atmosphere.get_absolute_airmass`,
  :py:func:`~pvlib.atmosphere.alt2pres`
- incidence angle modifiers: :py:func:`~pvlib.iam.ashrae`,
  :py:func:`~pvlib.iam.martin_ruiz`
- temperature: :py:func:`~pvlib.temperature.sapm_cell`,
  :py:func:`~pvlib.temperature.sapm_module`,
  :py:func:`~pvlib.temperature.sapm_cell_from_module`,
  :py:func:`~pvlib.temperature.faiman`
- power: :py:func:`~pvlib.pvsystem.pvwatts_dc`,
  :py:func:`~pvlib.inverter.pvwatts`
- single diode model: :py:func:`~pvlib.pvsystem.i_from_v` and
  :py:func:`~pvlib.pvsystem.v_from_i`, with ``method='lambertw'``, their
  default. The other methods call SciPy's solvers and need NumPy arrays.
  See :ref:`array-libraries-mpp` for the maximum power point
- snow: :py:func:`~pvlib.snow.fully_covered_nrel`,
  :py:func:`~pvlib.snow.coverage_nrel`
- soiling: :py:func:`~pvlib.soiling.hsu`
- spectrum: :py:func:`~pvlib.spectrum.average_photon_energy`,
  :py:func:`~pvlib.spectrum.calc_spectral_mismatch_field`
- tools: :py:func:`~pvlib.tools.cosd`, :py:func:`~pvlib.tools.sind`,
  :py:func:`~pvlib.tools.tand`, :py:func:`~pvlib.tools.asind`,
  :py:func:`~pvlib.tools.acosd`, :py:func:`~pvlib.tools.atand`,
  :py:func:`~pvlib.tools.unix_to_doy`

Other functions take NumPy and pandas only. Given arrays from another
library, they may fail with an unclear error. To add a function to this
list, see :ref:`contributing-array-libraries`.

Type in, type out
-----------------

A function returns the type it was given:

==============================  ==============================================
Input                           Output
==============================  ==============================================
Python scalars                  NumPy scalar
lists, NumPy arrays             NumPy array
NumPy masked arrays             masked array, masked wherever an input was
pandas Series                   Series, or a DataFrame for several outputs
xarray DataArrays               DataArray, or a Dataset for several outputs
arrays from another library     arrays from that library
==============================  ==============================================

Scalars and datetimes mix with any of these. Arrays from two libraries don't
mix, except NumPy arrays with Dask arrays, so convert first. The result has
the common floating point type of the inputs, float64 if none is floating
point. Arrays on different devices raise ``ValueError``, except 0-d arrays,
such as a tilt given as ``torch.tensor(30.)``, which move to the device of
the others. Dask arrays stay lazy: the result is computed when you call
``.compute()``.

pandas inputs must have equal indexes. pvlib matches their values by
position and doesn't align them as pandas arithmetic does, so Series with
different indexes raise ``ValueError``; reindex data with gaps first. The
same instants in different time zones are equal, and the result has the
index of the first pandas input.
Strings and datetimes given where numbers are expected raise ``TypeError``.

xarray
------

DataArrays are aligned and broadcast by dimension name, as in xarray
arithmetic. Here solar position is calculated for every hour on a grid of
latitudes and longitudes:

.. ipython:: python

    import numpy as np
    import pandas as pd
    import xarray as xr
    from pvlib import clearsky, irradiance, solarposition, tools

    times = pd.date_range('2026-06-21', freq='1h', periods=24, tz='UTC')
    time = xr.DataArray(times, dims='time', coords={'time': times})
    lat = xr.DataArray(np.arange(25., 50., 5.), dims='lat')
    lon = xr.DataArray(np.arange(-120., -70., 10.), dims='lon')
    position = solarposition.spa_python(time, lat, lon)
    position['apparent_zenith'].sizes

    tilt = xr.DataArray([0., 20., 40.], dims='tilt')
    aoi = irradiance.aoi(tilt, 180., position['apparent_zenith'],
                         position['azimuth'])
    aoi.sizes

Inside the function each input keeps only its own dimensions, so a term that
depends only on time is calculated once per time, not once per grid cell.

Arrays without labels broadcast by position against the DataArrays'
dimensions, as in xarray arithmetic, and raise ``ValueError`` if they would
add a dimension.

Time
----

The array API standard has no datetime type, so time can also be given as
numbers:

- :py:func:`~pvlib.solarposition.spa_python` and the time series functions
  below take float64 seconds since 1970-01-01 00:00 UTC ("Unix seconds");
- :py:func:`~pvlib.irradiance.get_extra_radiation` and
  :py:func:`~pvlib.irradiance.erbs` take the day of year.

Datetimes from pandas, NumPy and xarray are converted to these numbers when
the function is called, as arrays of the same library as the other inputs
and on the same device. So code that passes datetimes works as before, with
data from any library. With other data the result has that library's type,
not a pandas type. To convert once for many calls, use
:py:func:`pvlib.tools.datetime_to_unix`, and get the day of year from Unix
seconds with :py:func:`pvlib.tools.unix_to_doy`, which works on any array
library and inside ``jax.jit``.

Unix seconds must be float64: in float32 they resolve only 128 seconds, so
the functions that take them raise ``TypeError`` for float32 seconds. Integer seconds are converted to
float64, also with float32 data. PyTorch and JAX create float32 arrays by
default. In JAX, enable float64 first with
``jax.config.update('jax_enable_x64', True)``. GPUs without float64, such as
Apple's, can't calculate solar position.

.. _array-libraries-time-series:

Time series
-----------

:py:func:`~pvlib.soiling.hsu`, :py:func:`~pvlib.snow.fully_covered_nrel` and
:py:func:`~pvlib.snow.coverage_nrel` step through time, so they need to know
which way time runs in their inputs. They take it from one place, chosen by
the type of the data:

==================  ===========================================================
Data                Time
==================  ===========================================================
pandas Series       the index, as before; leave out ``times``
xarray DataArrays   ``times``, a DataArray such as ``da.time``, whose dimension
                    is time wherever it is in the data
other arrays        ``times``, 1-D, along the last axis of the data
==================  ===========================================================

``times`` holds datetimes or Unix seconds. Given with Series, it must equal
their index, or ``ValueError`` is raised, so the two can't silently
disagree; naive local times, for example, would be taken as UTC. Without
Series, ``times`` is required. Results have time last: on the last axis of
an array, and as the last dimension of a DataArray.

So data for several sites or orientations run in one call. With arrays,
give values that are constant in time, such as a tilt for each row, a last
axis of size 1:

.. ipython:: python

    from pvlib import snow

    hours = pd.date_range('2026-01-01', freq='1h', periods=12, tz='Etc/GMT+7')
    snowfall = np.where(np.arange(12) == 2, 3., 0.)  # 3 cm at 02:00
    tilt = np.array([[20.], [40.]])  # two rows of modules
    coverage = snow.coverage_nrel(snowfall, 300., 1., tilt, times=hours)
    coverage.shape
    coverage[:, :6].round(2)

The snow slides off the steeper row faster.

Gradients
---------

PyTorch and JAX can differentiate a model built from the supported
functions. This model of a day's plane of array insolation uses JAX:

.. ipython:: python

    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    day = pd.date_range('2026-06-21', freq='1h', periods=24, tz='Etc/GMT+7')
    unix = jnp.asarray(tools.datetime_to_unix(day))
    doy = tools.unix_to_doy(unix)

    def insolation(surface_tilt):
        position = solarposition.spa_python(unix, 35., -106.)
        zenith = position['apparent_zenith']
        azimuth = position['azimuth']
        ghi = clearsky.haurwitz(zenith)['ghi']
        parts = irradiance.erbs(ghi, zenith, doy)
        poa = irradiance.get_total_irradiance(
            surface_tilt, 180., zenith, azimuth, parts['dni'], ghi,
            parts['dhi'], dni_extra=irradiance.get_extra_radiation(doy),
            model='haydavies')
        return jnp.sum(poa['poa_global']) / 1000  # kWh/m2

    insolation(20.)
    jax.jit(jax.grad(insolation))(20.)

On the summer solstice at 35° latitude, more tilt means less insolation.

Gradients are promised function by function. pvlib's tests check them, with
PyTorch and JAX, for every function in the list above except for
:py:func:`~pvlib.snow.fully_covered_nrel`, whose result is boolean.

Gradients are zero where a model is flat: where
:py:func:`~pvlib.inverter.pvwatts` clips, where the sun is behind the module
or below the horizon. They change abruptly at kinks such as the clearness
index thresholds in :py:func:`~pvlib.irradiance.erbs`, and are zero with
respect to thresholds such as ``cleaning_threshold`` in
:py:func:`~pvlib.soiling.hsu`. Night time steps can stay in: the supported
functions keep the NaN airmass of night out of the gradient, which would
otherwise make the gradient of the whole day NaN.

Input checks that need values, such as the check of ``a_r`` in
:py:func:`~pvlib.iam.martin_ruiz`, are skipped inside ``jax.jit`` and for
Dask arrays, whose values aren't known yet.


.. _array-libraries-mpp:

Maximum power point
-------------------

The current at a voltage and the voltage at a current of the single diode
model have closed forms, so :py:func:`~pvlib.pvsystem.i_from_v` and
:py:func:`~pvlib.pvsystem.v_from_i` work with every library. The maximum
power point has none. pvlib finds it with a SciPy solver, which needs NumPy
arrays, so :py:func:`~pvlib.pvsystem.singlediode` raises ``TypeError`` for
arrays of other libraries.

With another library, find it with that library's own solver. The power at
a voltage is ``v * i_from_v(v, ...)``, and the library differentiates it.
This example uses `Optimistix <https://docs.kidger.site/optimistix/>`_, a
solver library for JAX, to find where the slope of the power curve is zero
between short circuit and open circuit:

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import optimistix as optx
    from pvlib import pvsystem

    def power(v, params):
        return v * pvsystem.i_from_v(v, *params)

    def max_power_point(params):
        # for one IV curve; params are photocurrent, saturation_current,
        # resistance_series, resistance_shunt and nNsVth
        v_oc = pvsystem.v_from_i(0., *params)
        solution = optx.root_find(
            jax.grad(power), optx.Bisection(rtol=1e-12, atol=1e-12),
            0.8 * v_oc, args=params, options={'lower': 0., 'upper': v_oc})
        v_mp = solution.value
        return v_mp, power(v_mp, params)

    # for arrays of curves, compiled
    params = (jnp.asarray([7., 5., 3.]), 1e-9 * jnp.ones(3), 0.3 * jnp.ones(3),
              300. * jnp.ones(3), 1.5 * jnp.ones(3))
    v_mp, p_mp = jax.jit(jax.vmap(max_power_point))(params)

``jax.vmap`` applies the solver for one curve to each curve of the arrays.
Optimistix differentiates the solution, not the iterations that found it,
so ``jax.grad`` of a function of ``p_mp`` gives exact gradients with respect
to the five parameters, for fitting them to measured power.

SciPy
-----

The array API standard has no special functions.
:py:func:`~pvlib.soiling.hsu` needs one, the error function, and gets it in
one of two ways:

- If SciPy's array API support is on, pvlib calls
  :py:func:`scipy.special.erf`, and SciPy calls the error function of the
  array's library.
- Otherwise pvlib calls the error function of PyTorch, JAX or CuPy itself,
  and SciPy's for NumPy, Dask and other libraries.

The results are the same either way. SciPy's support is off unless the
environment variable ``SCIPY_ARRAY_API`` is set to ``1`` before SciPy is
first imported. pvlib imports SciPy, so set it before importing pvlib, or in
the shell:

.. code-block:: bash

    export SCIPY_ARRAY_API=1

Setting it after SciPy is imported has no effect. It also changes SciPy
itself, whose functions then reject NumPy masked arrays, for example. See
SciPy's `array API support
<https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html>`_.

Performance
-----------

Converting inputs and outputs costs a few microseconds per call, and calls
with only numbers take a shortcut that costs 1–3 µs. That only matters when
calling a function on scalars in a loop: pass arrays instead.
