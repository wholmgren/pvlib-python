"""
Solar Position on a Grid
========================

Solar position and clear-sky irradiance for every cell of a latitude and
longitude grid, with xarray.
"""

# %%
# Weather models and satellite products give irradiance on grids of latitude
# and longitude. Comparing them with clear-sky irradiance, for example to
# find the clear-sky index, needs the sun's position in every grid cell at
# every time.
#
# :py:func:`pvlib.solarposition.spa_python` accepts xarray DataArrays and
# broadcasts them by dimension name (see :ref:`array_libraries`), so one call
# covers the whole grid, with no loop over cells. Inside the calculation,
# time keeps only its own dimension, so the parts of the solar position
# algorithm that depend only on time, such as the sun's declination, are
# calculated once per time step rather than once per cell.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from pvlib import clearsky, solarposition

# %%
# The grid covers the contiguous United States at 0.5° resolution, every 10
# minutes through the June and December solstices. xarray times have no time
# zone, so these are in UTC.

june = pd.date_range('2026-06-21', freq='10min', periods=144)
december = pd.date_range('2026-12-21', freq='10min', periods=144)
times = june.append(december)
time = xr.DataArray(times, dims='time', coords={'time': times})
lat = xr.DataArray(np.arange(25., 50.1, 0.5), dims='lat')
lat = lat.assign_coords(lat=lat)
lon = xr.DataArray(np.arange(-125., -66.9, 0.5), dims='lon')
lon = lon.assign_coords(lon=lon)

position = solarposition.spa_python(time, lat, lon)
position['apparent_zenith'].sizes

# %%
# That is the solar position at 288 times in each of the 5,967 grid cells,
# from a single call. The clear-sky model then works on the grid the same
# way.

ghi = clearsky.haurwitz(position['apparent_zenith'])['ghi']

# %%
# On the December solstice, morning reaches the east coast first. At 18:00
# UTC it is close to solar noon in the middle of the country, and by evening
# the sun has set in the east while it is still up in the west.

fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True)
for ax, hour in zip(axes, ['13:00', '18:00', '22:30']):
    ghi.sel(time=f'2026-12-21 {hour}').plot(
        ax=ax, vmin=0, vmax=800, cmap='inferno',
        cbar_kwargs={'label': 'Clear-sky GHI [W/m²]'})
    ax.set_title(f'{hour} UTC')
    ax.set_ylabel('Latitude [°]')
    ax.set_xlabel('')
axes[-1].set_xlabel('Longitude [°]')
fig.tight_layout()
plt.show()

# %%
# Summing each day over time gives every cell's clear-sky insolation. It
# hardly varies with longitude, so here it is averaged over longitude and
# shown against latitude. In December it falls steeply to the north, where
# the days are short and the sun is low. In June it hardly changes: the
# longer days in the north make up for the lower sun.

# kWh/m² from 10-minute irradiance in W/m²
insolation = ghi.groupby('time.month').sum() * 10 / 60 / 1000

fig, ax = plt.subplots(figsize=(6, 3.5))
for month, label in [(6, 'June 21'), (12, 'December 21')]:
    insolation.sel(month=month).mean('lon').plot(ax=ax, label=label)
ax.set_xlabel('Latitude [°]')
ax.set_ylabel('Clear-sky insolation [kWh/m²]')
ax.set_title('')
ax.set_ylim(bottom=0)
ax.legend()
fig.tight_layout()
plt.show()

print(insolation.sel(lat=[25, 35, 50]).mean('lon').round(1).to_pandas())

# %%
# With Dask installed, chunked DataArrays such as ``lat.chunk(10)`` give a
# lazy result, which suits grids too large for memory.
