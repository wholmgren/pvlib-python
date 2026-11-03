"""
Horizon Profile from Power Data
===============================

Inferring the horizon around a PV system from its AC power, using gradients
from JAX.
"""

# %%
# Hills, trees and buildings block the direct sun when it is low in the sky.
# Horizon profiles from terrain data, such as those from
# :py:func:`pvlib.iotools.get_pvgis_horizon`, miss nearby trees and
# buildings. A system's power data contains its horizon, though: power drops
# when the sun goes behind an obstruction. This example infers a horizon
# profile, 36 heights at 10° steps of azimuth, by fitting a model to AC
# power.
#
# Fitting 36 parameters needs the derivative of the fit error with respect to
# each of them. The model functions accept JAX arrays (see
# :ref:`array_libraries`), so ``jax.grad`` gives all 36 derivatives for
# about the cost of two model runs, where finite differences would need 37.

import pathlib
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import pvlib
from pvlib import (iam, inverter, irradiance, pvsystem, solarposition,
                   temperature)
from pvlib.iotools import read_tmy3

# JAX arrays are float32 unless float64 is enabled, and pvlib needs float64
jax.config.update('jax_enable_x64', True)

# %%
# Weather and the sun's position don't depend on the horizon, so they are
# calculated once, with pandas, and converted to JAX arrays. TMY3 data are
# right-labeled hourly intervals, so solar position is calculated 30 minutes
# before each label, in the middle of the interval.

DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'
tmy, metadata = read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=1990,
                          map_variables=True)
times = tmy.index - pd.Timedelta('30min')
position = solarposition.spa_python(times, metadata['latitude'],
                                    metadata['longitude'],
                                    altitude=metadata['altitude'])


def as_jax(series):
    return jnp.asarray(series.to_numpy(dtype=float))


zenith = as_jax(position['apparent_zenith'])
elevation = as_jax(position['apparent_elevation'])
azimuth = as_jax(position['azimuth'])
dni, ghi, dhi = as_jax(tmy['dni']), as_jax(tmy['ghi']), as_jax(tmy['dhi'])
temp_air, wind_speed = as_jax(tmy['temp_air']), as_jax(tmy['wind_speed'])
dni_extra = as_jax(irradiance.get_extra_radiation(times))

# %%
# The model blocks direct irradiance when the sun is below the horizon at its
# azimuth. A sharp step from blocked to clear would have a zero derivative
# almost everywhere, so the step is smoothed over a few degrees. The horizon
# blocks none of the diffuse irradiance here, which is a simplification.

horizon_azimuth = np.arange(0., 360., 10.)


def ac_power(horizon):
    # horizon height at the sun's azimuth, interpolated around the circle
    height = jnp.interp(azimuth, jnp.append(horizon_azimuth, 360.),
                        jnp.append(horizon, horizon[0]))
    clear = jax.nn.sigmoid(elevation - height)
    poa = irradiance.get_total_irradiance(
        25, 180, zenith, azimuth, dni * clear, dhi + (ghi - dhi) * clear,
        dhi, dni_extra=dni_extra, model='haydavies')
    aoi = irradiance.aoi(25, 180, zenith, azimuth)
    effective = poa['poa_direct'] * iam.martin_ruiz(aoi) + poa['poa_diffuse']
    temp_cell = temperature.faiman(poa['poa_global'], temp_air, wind_speed)
    pdc = pvsystem.pvwatts_dc(effective, temp_cell, pdc0=5000,
                              gamma_pdc=-0.004)
    return inverter.pvwatts(pdc, pdc0=5000)


# %%
# Instead of measurements, we simulate a year of hourly power behind a known
# horizon, with a line of trees to the east and a building to the southwest,
# and add 2% noise.

true_horizon = (3 + 15 * np.exp(-((horizon_azimuth - 100) / 25)**2)
                + np.where((horizon_azimuth >= 210)
                           & (horizon_azimuth <= 250), 20, 0))
rng = np.random.default_rng(0)
measured = ac_power(jnp.asarray(true_horizon)) * (
    1 + 0.02 * rng.standard_normal(len(times)))

# %%
# The fit minimizes the squared error, plus small penalties on differences
# between neighboring heights and, weaker, on the heights themselves. Where
# the data can't decide a height, the penalties make it follow its neighbors
# and stay low. The fit starts from an open horizon. ``jax.jit`` compiles the
# error and its gradient together, and SciPy's L-BFGS-B uses the gradient.


def fit_error(horizon):
    error = jnp.mean((ac_power(horizon) - measured)**2) / 100**2
    steps = jnp.diff(jnp.append(horizon, horizon[0]))
    return error + 1e-4 * jnp.sum(steps**2) + 1e-5 * jnp.sum(horizon**2)


error_and_gradient = jax.jit(jax.value_and_grad(fit_error))
error_only = jax.jit(fit_error)


def with_gradient(horizon):
    value, gradient = error_and_gradient(jnp.asarray(horizon))
    return float(value), np.asarray(gradient)


def without_gradient(horizon):
    return float(error_only(jnp.asarray(horizon)))


start_horizon = np.zeros(36)
bounds = [(0, 60)] * 36
with_gradient(start_horizon), without_gradient(start_horizon)  # compile

start = time.perf_counter()
fit = minimize(with_gradient, start_horizon, jac=True, method='L-BFGS-B',
               bounds=bounds)
seconds = format(time.perf_counter() - start, '.2f')
print(f'With gradients from JAX: {fit.nfev} model runs, {seconds} s')

start = time.perf_counter()
fit_fd = minimize(without_gradient, start_horizon, method='L-BFGS-B',
                  bounds=bounds)
seconds = format(time.perf_counter() - start, '.2f')
print(f'With finite differences: {fit_fd.nfev} model runs, {seconds} s')

# %%
# Both fits find the same horizon. The plot shows it with the true horizon
# and the sun's hourly positions through the year.

closed_azimuth = np.append(horizon_azimuth, 360)
fig, ax = plt.subplots(figsize=(7, 4))
up = np.asarray(elevation) > 0
ax.scatter(np.asarray(azimuth)[up], np.asarray(elevation)[up], s=1,
           color='gold', label='Sun, hourly')
ax.plot(closed_azimuth, np.append(true_horizon, true_horizon[0]), 'k',
        label='True horizon')
ax.plot(closed_azimuth, np.append(fit.x, fit.x[0]), 'C3--',
        label='Fitted horizon')
ax.set_xlim(0, 360)
ax.set_ylim(0, 50)
ax.set_xticks([0, 90, 180, 270, 360], ['N 0', 'E 90', 'S 180', 'W 270',
                                       'N 360'])
ax.set_xlabel('Azimuth [°]')
ax.set_ylabel('Elevation [°]')
ax.legend(loc='upper left')
fig.tight_layout()
plt.show()

# %%
# The fit recovers the horizon where the sun passes behind it: the trees
# that the sun rises over, and the building that it sets behind in winter.
# Elsewhere the power data can't tell the horizon's height. The sun never
# reaches the north. In the south it is always higher than the horizon, so
# any horizon below the sun's path fits equally well. On summer mornings and
# evenings, low in the northeast and northwest, the sun is behind the plane
# of these south-facing modules. There the penalties fill in the heights,
# which is why the fitted horizon bridges the gap between the trees and the
# building.
#
# With real data, anything the model gets wrong when the sun is low, such as
# diffuse shading, soiling or snow, would be blamed on the horizon. Fitting
# periods with clear mornings and evenings, or a better model of the system,
# reduces that.
#
# See also
# :ref:`sphx_glr_gallery_shading_plot_simple_irradiance_adjustment_for_horizon_shading.py`,
# which uses a known horizon profile.
