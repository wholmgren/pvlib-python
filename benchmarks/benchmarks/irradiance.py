"""
ASV benchmarks for irradiance.py
"""

import pandas as pd
from pvlib import irradiance, location


class Irradiance:

    def setup(self):
        self.times = pd.date_range(start='20180601', freq='1min',
                                   periods=14400)
        self.days = pd.date_range(start='20180601', freq='d', periods=30)
        self.location = location.Location(40, -80)
        self.solar_position = self.location.get_solarposition(self.times)
        self.clearsky_irradiance = self.location.get_clearsky(self.times)
        self.tilt = 20
        self.azimuth = 180
        self.aoi = irradiance.aoi(self.tilt, self.azimuth,
                                  self.solar_position.apparent_zenith,
                                  self.solar_position.azimuth)

    def time_get_extra_radiation(self):
        irradiance.get_extra_radiation(self.days)

    def time_aoi(self):
        irradiance.aoi(self.tilt, self.azimuth,
                       self.solar_position.apparent_zenith,
                       self.solar_position.azimuth)

    def time_aoi_projection(self):
        irradiance.aoi_projection(self.tilt, self.azimuth,
                                  self.solar_position.apparent_zenith,
                                  self.solar_position.azimuth)

    def time_get_ground_diffuse(self):
        irradiance.get_ground_diffuse(self.tilt, self.clearsky_irradiance.ghi)

    def time_get_total_irradiance(self):
        irradiance.get_total_irradiance(self.tilt, self.azimuth,
                                        self.solar_position.apparent_zenith,
                                        self.solar_position.azimuth,
                                        self.clearsky_irradiance.dni,
                                        self.clearsky_irradiance.ghi,
                                        self.clearsky_irradiance.dhi)

    def time_disc(self):
        irradiance.disc(self.clearsky_irradiance.ghi,
                        self.solar_position.apparent_zenith,
                        self.times)

    def time_dirint(self):
        irradiance.dirint(self.clearsky_irradiance.ghi,
                          self.solar_position.apparent_zenith,
                          self.times)

    def time_dirindex(self):
        irradiance.dirindex(self.clearsky_irradiance.ghi,
                            self.clearsky_irradiance.ghi,
                            self.clearsky_irradiance.dni,
                            self.solar_position.apparent_zenith,
                            self.times)

    def time_erbs(self):
        irradiance.erbs(self.clearsky_irradiance.ghi,
                        self.solar_position.apparent_zenith,
                        self.times)


def _as_input_type(series, input_type):
    # a scalar is the middle value: noon on July 2 for a year of hourly data
    if input_type == 'scalar':
        return float(series.iloc[len(series) // 2])
    if input_type == 'ndarray':
        return series.to_numpy()
    return series


class IrradianceInputTypes:
    params = ['scalar', 'ndarray', 'series']
    param_names = ['input_type']

    def setup(self, input_type):
        times = pd.date_range(start='20180101', freq='1h', periods=8760,
                              tz='Etc/GMT+5')
        loc = location.Location(40, -80)
        solar_position = loc.get_solarposition(times)
        clearsky = loc.get_clearsky(times, solar_position=solar_position)
        self.surface_tilt = 20
        self.surface_azimuth = 180
        self.zenith = _as_input_type(solar_position['apparent_zenith'],
                                     input_type)
        self.azimuth = _as_input_type(solar_position['azimuth'], input_type)
        self.dni = _as_input_type(clearsky['dni'], input_type)
        self.ghi = _as_input_type(clearsky['ghi'], input_type)
        self.dhi = _as_input_type(clearsky['dhi'], input_type)
        self.dni_extra = _as_input_type(
            irradiance.get_extra_radiation(times), input_type)
        # times for pandas, day of year otherwise
        if input_type == 'series':
            self.datetime_or_doy = times
        else:
            self.datetime_or_doy = _as_input_type(
                pd.Series(times.day_of_year, index=times), input_type)

    def time_aoi(self, input_type):
        irradiance.aoi(self.surface_tilt, self.surface_azimuth,
                       self.zenith, self.azimuth)

    def time_get_total_irradiance(self, input_type):
        irradiance.get_total_irradiance(
            self.surface_tilt, self.surface_azimuth, self.zenith,
            self.azimuth, self.dni, self.ghi, self.dhi,
            dni_extra=self.dni_extra, model='haydavies')

    def time_erbs(self, input_type):
        irradiance.erbs(self.ghi, self.zenith, self.datetime_or_doy)
