import numpy as np
import pandas as pd
import functools


class AutocorrelationCalculator:
    def __init__(self, opts=None):
        self.opts = opts

    @functools.lru_cache(maxsize=None)
    def _autocorr_np(self, x, max_lag):
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:]

    @functools.lru_cache(maxsize=None)
    def _autocorr_pd(self, x, max_lag):
        return [pd.Series(x).autocorr(lag=lag) for lag in range(max_lag + 1)]

    def _calculate_autocorrelation(self, rates):
        if self.opts['method'] == 'np':
            return self._autocorr_np(rates, self.opts['max_lag'])
        else:
            return self._autocorr_pd(rates, self.opts['max_lag'])

    @property
    def children(self):
        if isinstance(self, Experiment):
            return self.groups
        elif isinstance(self, Group):
            return self.animals
        elif isinstance(self, Animal):
            return self.units['good']
        else:
            return None

    @property
    def rates(self):
        if isinstance(self, Unit):
            return np.mean(self.get_trials_rates(self.opts), axis=0)
        else:
            return np.mean([child.rates for child in self.children], axis=0)

    def calculate_all_autocorrelations(self, opts):
        self.opts = opts
        result = {}
        if isinstance(self, Unit):
            result['self_over_trials'] = [self._calculate_autocorrelation(rate) for rate in self.get_trials_rates(opts)]
        else:
            child_autocorr = {child.name: child.calculate_all_autocorrelations(opts) for child in self.children}
            for key, value in child_autocorr.items():
                result[key + '_over_trials'] = np.mean(value['self_over_trials'])
                result[key + '_over_rates'] = value['self_over_rates']
        result['self_over_rates'] = self._calculate_autocorrelation(self.rates)
        self.opts = None
        return result
