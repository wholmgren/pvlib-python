"""
ASV benchmarks for iam.py
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


class IAMInputTypes:
    params = ['scalar', 'ndarray', 'series']
    param_names = ['input_type']

    def setup(self, input_type):
        if Version(pvlib.__version__) < Version('0.7.0'):
            raise NotImplementedError

        times = pd.date_range(start='20180101', freq='1h', periods=8760)
        # includes angles beyond 90 degrees
        self.aoi = _as_input_type(
            pd.Series(np.linspace(0, 100, len(times)), index=times),
            input_type)

    def time_ashrae(self, input_type):
        pvlib.iam.ashrae(self.aoi)

    def time_martin_ruiz(self, input_type):
        pvlib.iam.martin_ruiz(self.aoi)
