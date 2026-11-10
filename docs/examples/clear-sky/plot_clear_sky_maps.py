"""
Clear-Sky Irradiance from Turbidity and Altitude Maps
=====================================================

Clear-sky irradiance over the Atacama Desert, from the maps of Linke
turbidity and altitude that come with pvlib, with xarray.
"""

# %%
# The Ineichen clear-sky model, :py:func:`pvlib.clearsky.ineichen`, needs the
# Linke turbidity of the atmosphere and the altitude. For one location,
# :py:func:`pvlib.clearsky.lookup_linke_turbidity` and
# :py:func:`pvlib.location.lookup_altitude` read them from global maps that
# come with pvlib, with cells of 1/12° (about 9 km). Read as grids, the maps
# show why the Atacama Desert in northern Chile has some of the clearest skies
# on Earth: much of it is a plateau 3 to 5 km high, above a large part of the
# atmosphere, and its air is clean and dry.
#
# The functions used here accept xarray DataArrays and broadcast them by
# dimension name (see :ref:`array_libraries`), so each call covers the whole
# grid, with no loop over cells.

import pathlib

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import pvlib
from pvlib import atmosphere, clearsky, irradiance, solarposition

# %%
# The maps are HDF5 files in pvlib's data folder. Each row of cells spans
# 1/12° of latitude, from 90° N southward, and each column 1/12° of
# longitude, from 180° W eastward. Here they are read for northern Chile and
# its neighbors.

DATA = pathlib.Path(pvlib.__file__).parent / 'data'
SOUTH, NORTH, WEST, EAST = -30, -18, -75, -64


def read_map(file, name):
    # Row i spans latitudes 90 - i/12 to 90 - (i + 1)/12, so its center is
    # at 90 - (i + 0.5)/12; column j's center is at -180 + (j + 0.5)/12.
    # Bounds in whole degrees fall on cell edges, so the slices take whole
    # cells, from the north and west bounds up to the south and east ones.
    rows = slice((90 - NORTH) * 12, (90 - SOUTH) * 12)
    cols = slice((WEST + 180) * 12, (EAST + 180) * 12)
    with h5py.File(DATA / file) as f:
        values = f[name][rows, cols]  # reads only these cells
    coords = {'lat': 90 - (np.arange(rows.start, rows.stop) + 0.5) / 12,
              'lon': -180 + (np.arange(cols.start, cols.stop) + 0.5) / 12}
    if values.ndim == 3:  # turbidity has a value for each month
        coords['month'] = np.arange(1, 13)
    return xr.DataArray(values, dims=list(coords), coords=coords)


# %%
# The maps store small integers: Linke turbidity times 20, and altitude in
# 28 m steps from -450 m, with 255 where there is no data, here over the
# ocean, which ``lookup_altitude`` takes as sea level.

# uint8, turbidity times 20, so in steps of 0.05
linke_turbidity = read_map('LinkeTurbidities.h5', 'LinkeTurbidity') / 20
# uint8, 28 m * value - 450 m, from -450 m to 6,662 m, and 255 for no data
stored = read_map('Altitude.h5', 'Altitude')
altitude = xr.where(stored == 255, 0., stored * 28. - 450.)

# %%
# Annual clear-sky insolation needs the sun's path through the year. One day
# in the middle of each month, hourly, stands for its month. xarray times
# have no time zone, so these are in UTC.

days = pd.date_range('2026-01-15', periods=12, freq=pd.DateOffset(months=1))
hours = pd.timedelta_range('30min', periods=24, freq='1h')
times = pd.DatetimeIndex((days.values[:, None] + hours.values).ravel())
time = xr.DataArray(times, dims='time', coords={'time': times})


def annual_clear_sky(altitude, linke_turbidity):
    """Annual clear-sky insolation [kWh/m²] at the cells of the inputs."""
    pressure = atmosphere.alt2pres(altitude)
    position = solarposition.spa_python(
        time, altitude['lat'], altitude['lon'], altitude=altitude,
        pressure=pressure)
    zenith = position['apparent_zenith']
    airmass = atmosphere.get_absolute_airmass(
        atmosphere.get_relative_airmass(zenith), pressure)
    # The 12 turbidity values are monthly averages, January first. Each day
    # takes its month's value, as lookup_linke_turbidity does with
    # interp_turbidity=False; by default it interpolates between them
    clear = clearsky.ineichen(
        zenith, airmass, linke_turbidity.sel(month=time.dt.month),
        altitude=altitude, dni_extra=irradiance.get_extra_radiation(time))
    # hourly irradiance [W/m²] on one day for each month
    return (clear * time.dt.days_in_month).sum('time') / 1000


# %%
# For the DNI map, cells of 1/4°, the averages of 3 by 3 cells, keep the
# calculation small: 288 times in each of 2,112 cells.

# The model isn't linear, so averaging its inputs differs a little from
# averaging its results at the finer cells; for a map that's fine
grid = annual_clear_sky(altitude.coarsen(lat=3, lon=3).mean(),
                        linke_turbidity.coarsen(lat=3, lon=3).mean())

# %%
# Clear-sky DNI follows altitude and turbidity. The plateau gets up to about
# 4,900 kWh/m² a year, the coast and the ocean around 3,000.

# Latitude of the line below, near the center of a cell. A latitude or
# longitude on the edge between two cells (a multiple of 1/12°, such as
# 67.75° W) is ambiguous: pvlib's lookups round it to the even-numbered cell,
# as numpy rounds halves to even, while xarray's sel(method='nearest') may
# take the other one
TRANSECT = -23.04

fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
panels = [
    (altitude / 1000, 'Altitude [km]', {'cmap': 'terrain', 'vmin': 0}),
    (linke_turbidity.mean('month'), 'Linke turbidity', {'cmap': 'viridis_r'}),
    (grid['dni'], 'Clear-sky DNI [kWh/m²]', {'cmap': 'inferno'}),
]
for ax, (data, label, style) in zip(axes, panels):
    data.plot(ax=ax, **style,
              cbar_kwargs={'label': label, 'orientation': 'horizontal'})
    ax.axhline(TRANSECT, color='white', linestyle='--', linewidth=1)
    ax.set_aspect('equal')
    ax.set_title('')
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('')
axes[0].set_ylabel('Latitude [°]')
fig.tight_layout()
plt.show()

# %%
# Along the dashed line, at the maps' full resolution, the land rises from
# the coast to 3 km within about 180 km, and to 5 km on the plateau, about
# 280 km inland.
# Calculating again with the altitude at sea level separates the two
# effects: lower turbidity accounts for part of the plateau's extra DNI, and
# the thinner air above it for the rest.

line_altitude = altitude.sel(lat=TRANSECT, method='nearest')
line_turbidity = linke_turbidity.sel(lat=TRANSECT, method='nearest')
line = annual_clear_sky(line_altitude, line_turbidity)
at_sea_level = annual_clear_sky(xr.zeros_like(line_altitude), line_turbidity)

fig, axes = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
(line_altitude / 1000).plot(ax=axes[0])
axes[0].set_ylabel('Altitude [km]')
line_turbidity.mean('month').plot(ax=axes[1])
axes[1].set_ylabel('Linke turbidity')
line['dni'].plot(ax=axes[2], label='DNI')
at_sea_level['dni'].plot(ax=axes[2], linestyle='--',
                         label='DNI with altitude at sea level')
line['ghi'].plot(ax=axes[2], label='GHI')
axes[2].set_ylabel('Clear-sky [kWh/m²]')
axes[2].legend()
for ax in axes:
    ax.set_title('')
    ax.set_xlabel('')
axes[-1].set_xlabel('Longitude [°]')
fig.tight_layout()
plt.show()

coast = int((stored.sel(lat=TRANSECT, method='nearest') != 255).argmax())
top = int(line_altitude.argmax())
for name, i in [('Coast', coast), ('Highest cell', top)]:
    cell_lon = format(line['lon'][i].item(), '.1f')
    cell_altitude = format(line_altitude[i].item(), '.0f')
    cell_turbidity = format(line_turbidity.mean('month')[i].item(), '.1f')
    cell_dni = format(line['dni'][i].item(), '.0f')
    sea_level_dni = format(at_sea_level['dni'][i].item(), '.0f')
    print(f'{name}: {cell_lon}°, {cell_altitude} m, '
          f'Linke turbidity {cell_turbidity}, '
          f'clear-sky DNI {cell_dni} kWh/m², {sea_level_dni} at sea level')

# %%
# These are clear-sky values, an upper bound: clouds, which pvlib's maps
# don't include, reduce real insolation. The same calculation works with
# gridded weather or satellite data in place of the maps.
