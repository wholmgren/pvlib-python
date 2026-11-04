"""
ASV benchmarks for atmosphere.py
"""

import numpy as np
import pandas as pd
from pvlib import atmosphere


def _as_input_type(series, input_type):
    # a scalar is the middle value
    if input_type == 'scalar':
        return float(series.iloc[len(series) // 2])
    if input_type == 'ndarray':
        return series.to_numpy()
    return series


class AtmosphereInputTypes:
    params = ['scalar', 'ndarray', 'series']
    param_names = ['input_type']

    def setup(self, input_type):
        times = pd.date_range(start='20180101', freq='1h', periods=8760)
        # includes zenith angles beyond 90 degrees
        self.zenith = _as_input_type(
            pd.Series(np.linspace(0, 100, len(times)), index=times),
            input_type)

    def time_get_relative_airmass(self, input_type):
        atmosphere.get_relative_airmass(self.zenith)
