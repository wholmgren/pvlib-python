"""
Energy Loss by Orientation
==========================

Mapping annual energy over a grid of tilts and azimuths with xarray.
"""

# %%
# Roofs and sites don't always allow the best orientation, so a common
# question is how much energy a given tilt and azimuth gives up. This example
# answers it for every orientation on a grid at once, from a TMY weather file.
#
# Some pvlib functions accept xarray DataArrays and broadcast them by
# dimension name (see :ref:`array_libraries`). Here the weather and the sun's
# position vary with ``time``, and the orientation with ``tilt`` and
# ``azimuth``, so each step of the model returns results on the grid of all
# three, without a loop. Inside each function, every input keeps only its own
# dimensions, so the terms that depend only on time are calculated once per
# hour rather than once per orientation.

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import pvlib
from pvlib import (iam, inverter, irradiance, pvsystem, solarposition,
                   temperature)
from pvlib.iotools import read_tmy3

DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'
tmy, metadata = read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=1990,
                          map_variables=True)

# TMY3 datasets are right-labeled hourly intervals: 10:00 to 11:00 is labeled
# 11:00. Solar position should be calculated in the middle of each interval,
# so we label the data 30 minutes earlier. xarray times have no time zone, so
# we also convert to UTC.
times = tmy.index - pd.Timedelta('30min')
weather = (tmy[['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']]
           .set_axis(times.tz_convert('UTC').tz_localize(None))
           .rename_axis('time')
           .to_xarray())

# %%
# Solar position and extraterrestrial irradiance depend only on time, so
# they have only the ``time`` dimension.

position = solarposition.spa_python(weather['time'], metadata['latitude'],
                                    metadata['longitude'],
                                    altitude=metadata['altitude'])
dni_extra = irradiance.get_extra_radiation(weather['time'])
position['apparent_zenith'].dims

# %%
# The orientations are two more dimensions. Passing them to the
# transposition model gives plane of array irradiance for every hour and
# orientation. This uses the Hay-Davies model, because the Perez model
# doesn't support xarray yet.

tilt = xr.DataArray(np.arange(0., 91., 5.), dims='tilt')
tilt = tilt.assign_coords(tilt=tilt)
azimuth = xr.DataArray(np.arange(90., 271., 10.), dims='azimuth')
azimuth = azimuth.assign_coords(azimuth=azimuth)

poa = irradiance.get_total_irradiance(
    tilt, azimuth, position['apparent_zenith'], position['azimuth'],
    weather['dni'], weather['ghi'], weather['dhi'], dni_extra=dni_extra,
    model='haydavies')
poa['poa_global'].sizes

# %%
# The rest of the model follows the same way: reflection losses, cell
# temperature, DC power from a 1 kW array, and AC power from an inverter
# with the same DC rating. The annual AC energy then needs only a sum over
# time.

aoi = irradiance.aoi(tilt, azimuth, position['apparent_zenith'],
                     position['azimuth'])
effective = poa['poa_direct'] * iam.martin_ruiz(aoi) + poa['poa_diffuse']
temp_cell = temperature.faiman(poa['poa_global'], weather['temp_air'],
                               weather['wind_speed'])
pdc = pvsystem.pvwatts_dc(effective, temp_cell, pdc0=1000, gamma_pdc=-0.004)
pac = inverter.pvwatts(pdc, pdc0=1000)
energy = pac.sum('time') / 1000  # kWh per kW, from hourly power in W

best = energy.isel(energy.argmax(...))
print(f"Best: tilt {best['tilt'].item():.0f}°, "
      f"azimuth {best['azimuth'].item():.0f}°, "
      f"{best.item():.0f} kWh per kW")

# %%
# The map shows the energy lost relative to the best orientation.

loss = 100 * (1 - energy / energy.max())
levels = [0, 2, 5, 10, 15, 20, 30, 40, 50]
fig, ax = plt.subplots()
loss.plot.contourf(x='azimuth', y='tilt', levels=levels, cmap='viridis_r',
                   ax=ax, cbar_kwargs={'label': 'Energy loss [%]'})
lines = loss.plot.contour(x='azimuth', y='tilt', levels=levels[1:],
                          colors='k', linewidths=0.5, ax=ax)
ax.clabel(lines, fmt='%d%%')
ax.plot(best['azimuth'], best['tilt'], 'w*', markersize=12)
ax.set_xticks([90, 135, 180, 225, 270],
              ['E 90', 'SE 135', 'S 180', 'SW 225', 'W 270'])
ax.set_xlabel('Surface azimuth [°]')
ax.set_ylabel('Surface tilt [°]')
ax.set_title(f'Greensboro, NC, latitude {metadata["latitude"]:.0f}°')
plt.show()

# %%
# Near the best orientation the energy changes slowly, so a module can face
# well away from it and lose only a few percent. Steep tilts facing east or
# west lose the most. A flat module, along the bottom of the map, loses the
# same whatever its azimuth. Some orientations:

orientations = {
    'best tilt, 40° east of south': (30, 140),
    'best tilt, 40° west of south': (30, 220),
    'flat': (0, 180),
    '20° tilt, facing east': (20, 90),
    '20° tilt, facing west': (20, 270),
    'vertical, facing south': (90, 180),
}
for name, (t, a) in orientations.items():
    print(f'{name}: {loss.sel(tilt=t, azimuth=a).item():.0f}% loss')

# %%
# To compare orientations at another site, change the weather data. Any
# other input can become a dimension too: ``gamma_pdc`` given as a DataArray
# with dimension ``module`` would add that dimension to ``energy``, with no
# other change to the code.
