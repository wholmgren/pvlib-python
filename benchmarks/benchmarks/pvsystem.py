"""
ASV benchmarks for pvsystem.py
"""

import numpy as np
import pandas as pd
from pvlib import pvsystem


def _as_input_type(series, input_type):
    # a scalar is the middle value
    if input_type == 'scalar':
        return float(series.iloc[len(series) // 2])
    if input_type == 'ndarray':
        return series.to_numpy()
    return series


class PVSystemInputTypes:
    params = ['scalar', 'ndarray', 'series']
    param_names = ['input_type']

    def setup(self, input_type):
        times = pd.date_range(start='20180101', freq='1h', periods=8760)
        self.effective_irradiance = _as_input_type(
            pd.Series(np.linspace(0, 1100, len(times)), index=times),
            input_type)
        self.temp_cell = _as_input_type(
            pd.Series(np.linspace(0, 70, len(times)), index=times),
            input_type)

    def time_pvwatts_dc(self, input_type):
        pvsystem.pvwatts_dc(self.effective_irradiance, self.temp_cell,
                            250, -0.004)
