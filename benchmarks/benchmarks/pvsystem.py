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


def _diode_parameters(n=8760):
    # as from calcparams_desoto for a year: photocurrent from dark to full
    # sun, and a series resistance that doesn't vary
    times = pd.date_range(start='20180101', freq='1h', periods=n)
    return {
        'photocurrent': pd.Series(np.linspace(0., 10., n), index=times),
        'saturation_current': pd.Series(np.geomspace(1e-10, 1e-8, n),
                                        index=times),
        'resistance_series': 0.3,
        'resistance_shunt': pd.Series(np.linspace(1000., 200., n),
                                      index=times),
        'nNsVth': pd.Series(np.linspace(1.4, 1.7, n), index=times),
    }


class SingleDiodeInputTypes:
    # arguments are passed by name: their order changed in pvlib 0.10.0
    params = ['scalar', 'ndarray', 'series']
    param_names = ['input_type']

    def setup(self, input_type):
        parameters = _diode_parameters()
        times = parameters['photocurrent'].index
        self.parameters = {
            k: _as_input_type(v, input_type) if isinstance(v, pd.Series)
            else v for k, v in parameters.items()}
        self.voltage = _as_input_type(
            pd.Series(np.linspace(0., 30., len(times)), index=times),
            input_type)
        # below the photocurrent, except in the dark
        self.current = _as_input_type(
            0.9 * parameters['photocurrent'], input_type)

    def time_i_from_v(self, input_type):
        pvsystem.i_from_v(voltage=self.voltage, **self.parameters)

    def time_v_from_i(self, input_type):
        pvsystem.v_from_i(current=self.current, **self.parameters)

    def time_singlediode(self, input_type):
        pvsystem.singlediode(**self.parameters)


class SingleDiodeBranches:
    """
    The explicit solutions, for no series resistance (i_from_v) and for
    infinite shunt resistance (v_from_i), and the Lambert W solution are
    computed only if some element needs them. 'implicit' needs only Lambert
    W; 'mixed' needs both for a tenth of the elements each.
    """
    params = ['implicit', 'mixed']
    param_names = ['parameters']

    def setup(self, parameters):
        n = 8760
        self.parameters = {
            k: np.array(v) if isinstance(v, pd.Series) else np.full(n, v)
            for k, v in _diode_parameters(n).items()}
        if parameters == 'mixed':
            self.parameters['resistance_series'][::10] = 0.
            self.parameters['resistance_shunt'][5::10] = np.inf
        self.voltage = np.linspace(0., 30., n)
        self.current = 0.9 * self.parameters['photocurrent']

    def time_i_from_v(self, parameters):
        pvsystem.i_from_v(voltage=self.voltage, **self.parameters)

    def time_v_from_i(self, parameters):
        pvsystem.v_from_i(current=self.current, **self.parameters)
