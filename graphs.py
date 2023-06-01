from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict as dd
from copy import deepcopy
import math
import os

import numpy as np
import pandas as pd

from graphs_utils import *


class Experiment:
    def __init__(self, conditions=None):
        if conditions is not None:
            self.conditions, self.groups = conditions.keys(), conditions.values()

    def plot_groups(self, opts, data_type=None):
        Plotter(opts, data_type=data_type).plot_groups(self.groups, ['PN', 'IN'])


class Group(MethodSelectorMixin, MathMixin):
    def __init__(self, name, animals=None):
        self.name = name
        self.animals = animals if animals else []

    @cache_method
    def get_average(self, data_type, opts, neuron_type=None):
        return np.mean(np.array([avg for avg in [animal.get_average(data_type, opts, neuron_type=neuron_type)
                                 for animal in self.animals] if not np.all(np.isnan(avg))]), axis=0)

    def plot_animals(self, opts, data_type=None):
        [Plotter(opts, data_type=data_type).plot_animals(self, neuron_type=neuron_type) for neuron_type in ['PN', 'IN']]

    @cache_method
    def get_avg_rates(self, opts):
        return np.mean([animal.get_average_rates(opts) for animal in self.animals], axis=0).flatten()


class Animal(AutocorrMixin, MathMixin):
    def __init__(self, name, condition, tone_period_onsets=None, tone_onsets_expanded=None, units=None):
        self.name = name
        self.condition = condition
        self.units = units if units is not None else dd(list)
        self.tone_onsets_expanded = tone_onsets_expanded if tone_onsets_expanded is not None else []
        self.tone_period_onsets = tone_period_onsets if tone_period_onsets is not None else []

    @cache_method
    def get_average(self, data_type, opts, neuron_type=None):
        unit_vals = [getattr(unit, f"get_{data_type}")(opts) for unit in self.units['good']
                     if neuron_type is None or unit.neuron_type == neuron_type]
        return np.mean(np.array(unit_vals), axis=0) if len(unit_vals) else [float('nan')]

    @cache_method
    def get_spectrum(self, method, opts):
        if method == 'autocorr':
            fft = np.fft.fft(self.get_average('autocorr'))
        elif method == 'xform':
            fft = self.get_average('fft', opts)
        return SignalProcessing.compute_one_sided_spectrum(fft, opts['lags'])

    def plot_units(self, opts):
        Plotter(opts).plot_units(self)

    @cache_method
    def get_avg_rates(self, opts):
        return np.mean([unit.get_avg_rates(opts) for unit in self.units['good']], axis=0).flatten()


class Unit(AutocorrMixin, MathMixin):
    def __init__(self, animal, category, spike_times, neuron_type=None):
        self.animal = animal
        self.category = category
        self.spike_times = spike_times
        self.animal.units[category].append(self)
        self.index = self.animal.units[category].index(self)
        self.neuron_type = neuron_type

    @cache_method
    def find_spikes(self, start, stop):
        return self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)]
    
    @cache_method
    def get_trials_spikes(self, opts):
        return [[(spike - start)/30000 for spike in
                 self.find_spikes(start - opts['pre_stim'] * 30000, start + opts['post_stim'] * 30000)]
                for start in self.animal.tone_onsets_expanded[slice(*opts['trials'])]]

    @cache_method
    def get_hist(self, spikes, opts, num_bins=None, spike_range=None):
        num_bins = num_bins if num_bins is not None else int((opts['post_stim'] + opts['pre_stim']) / opts['bin_size'])
        spike_range = spike_range if spike_range is not None else (-opts['pre_stim'], opts['post_stim'])
        hist = np.histogram(spikes, bins=num_bins, range=spike_range)
        return hist

    @cache_method
    def get_rates(self, spikes, opts, num_bins=None, spike_range=None):
        return self.get_hist(spikes, opts, num_bins=num_bins, spike_range=spike_range)[0] / opts['bin_size']

    @cache_method
    def get_pretone_means(self, opts):
        rate_set = []
        for onset in self.animal.tone_period_onsets:
            start = onset - 30 * 30000
            stop = onset - 1
            rate_set.append(self.get_rates(self.find_spikes(start, stop), opts, spike_range=(start, stop),
                                           num_bins=int(30/opts['bin_size'])))
        return np.mean(np.array(rate_set), axis=1)

    @cache_method
    def get_trials_rates(self, opts, num_bins=None, spike_range=None):
        return [self.get_rates(trial_spikes, opts, num_bins=num_bins, spike_range=spike_range)
                for trial_spikes in self.get_trials_spikes(opts)]

    @cachemethod
    def get_avg_rates(self, opts, num_bins=None, spike_range=spike_range):
        trials_rates = self.get_trials_rates(opts, num_bins=num_bins, spike_range=spike_range)
        return np.mean(np.array(trials_rates), axis=0).flatten()

    @cache_method
    def get_psth(self, opts):
        pretone_means = self.get_pretone_means(opts)
        trial_indices = list(range(150))[slice(*opts['trials'])]  # select only the trials indicated in opts
        trials_rates = self.get_trials_rates(opts)
        rate_set = [trials_rates[i] - pretone_means[trial_index // 30] for i, trial_index in enumerate(trial_indices)]
        return np.mean(np.array(rate_set), axis=0).flatten()

    @staticmethod
    def get_pandas_autocorr(x, max_lag):
        s = pd.Series(x)
        return [s.autocorr(lag=lag) for lag in range(max_lag + 1)]

    @cache_method
    def get_np_autocorr(self, opts):
        return Math.np_autocorr(self.get_trials_rates(opts))

    @cache_method
    def get_fft(self, opts):
        return np.fft.fft(self.get_autocorr(opts))

    @cache_method
    def get_spectrum(self, opts):
        return SignalProcessing.compute_one_sided_spectrum(self.get_fft(opts))

    @cache_method
    def get_og_autocorr(self, opts, method=None):
        method = self.method_selctor(opts)
        trials_rates = self.get_trials_rates(opts)


class AutocorrMixin:

    def get_autocorr(self, opts, method=None):
        method = self.method_selector(opts, method=method)
        return getattr(self, method)(self.get_avg_rates(opts))

    def method_selector(self, opts, method=None):
        if method is not None:
            return method
        else:
            return opts['method'] if method in opts else "np_autocorr"


class MathMixin:

    @staticmethod
    def compute_one_sided_spectrum(fft_values):
        """
        Computes the one-sided spectrum of a signal given its FFT.
        """
        N = len(fft_values)
        abs_values = np.abs(fft_values)
        one_sided_spectrum = abs_values[:N // 2]

        # multiply all frequency components by 2, except the DC component
        one_sided_spectrum[1:] *= 2

        return one_sided_spectrum

    @staticmethod
    def get_positive_frequencies(N, T):
        """
        Computes the positive frequencies for FFT of a dataset, given N, the length of the dataset, and T, the time
        spacing between samples.  Returns an array of positive frequencies (Hz) of length N/2.
        """
        frequencies = np.fft.fftfreq(N, T)  # Compute frequencies associated with FFT components
        positive_frequencies = frequencies[:N // 2]  # Get only positive frequencies
        return positive_frequencies

    @staticmethod
    def get_np_autocorr(x):
        full_result = np.correlate(series, series, mode='full')
        zero_lag_index = full_result.size // 2
        zero_lag_autocorr = full_result[full_result.size // 2]
        return full_result[zero_lag_index + 1: zero_lag_index + opts['lags'] + 1] / zero_lag_autocorr

    @staticmethod
    def get_pandas_autocorr(x, max_lag):
        return [pd.Series(x).autocorr(lag=lag) for lag in range(max_lag + 1)]



