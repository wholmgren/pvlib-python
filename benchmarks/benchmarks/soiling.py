"""
ASV benchmarks for soiling.py
"""

import numpy as np
import pandas as pd
import pvlib
from pvlib import soiling
from packaging.version import Version


class Hsu:
    params = [1, 10, 100, 365]  # number of days
    param_names = ['ndays']

    def setup(self, ndays):
        if Version(pvlib.__version__) < Version('0.8.0'):
            raise NotImplementedError

        times = pd.date_range(start='20180601', freq='1min',
                              periods=1440*ndays)
        rng = np.random.default_rng(0)
        # rain in about 1% of minutes
        rain = rng.uniform(0, 0.5, len(times))
        rain[rng.uniform(size=len(times)) > 0.01] = 0
        self.rainfall = pd.Series(rain, index=times)
        self.cleaning_threshold = 1.
        self.surface_tilt = 20
        self.pm2_5 = 1e-5  # g/m^3
        self.pm10 = 2e-5  # g/m^3

    def time_hsu(self, ndays):
        soiling.hsu(self.rainfall, self.cleaning_threshold,
                    self.surface_tilt, self.pm2_5, self.pm10)

    # the cost of a rolling window can grow with the number of points in it
    def time_hsu_24h_rain_accum_period(self, ndays):
        soiling.hsu(self.rainfall, self.cleaning_threshold,
                    self.surface_tilt, self.pm2_5, self.pm10,
                    rain_accum_period=pd.Timedelta('24h'))
