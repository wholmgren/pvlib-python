"""
ASV benchmarks for inverter.py
"""

import numpy as np
import pandas as pd
import pvlib
from packaging.version import Version


def _as_input_type(series, input_type):
    # a scalar is the middle value
    if input_type == 'scalar':
        return float(series.iloc[len(series) // 2])
    if input_type == 'ndarray':
        return series.to_numpy()
    return series


class InverterInputTypes:
    params = ['scalar', 'ndarray', 'series']
    param_names = ['input_type']

    def setup(self, input_type):
        if Version(pvlib.__version__) < Version('0.8.0'):
            raise NotImplementedError

        times = pd.date_range(start='20180101', freq='1h', periods=8760)
        # from zero to clipping
        self.pdc = _as_input_type(
            pd.Series(np.linspace(0, 275, len(times)), index=times),
            input_type)

    def time_pvwatts(self, input_type):
        pvlib.inverter.pvwatts(self.pdc, 250)
