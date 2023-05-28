import scipy.io as sio
from collections import defaultdict as dd
from bisect import bisect_left as bs_left, bisect_right as bs_right
import numpy as np
from statsmodels.tsa.stattools import acf
import functools
import os
import matplotlib.pyplot as plt
from matplotlib import patches
from copy import deepcopy


def cache_method(method):
    cache_name = "_cache_" + method.__name__

    def to_hashable(item):
        if isinstance(item, dict):
            return tuple(sorted(item.items()))
        elif isinstance(item, (list, set)):
            return tuple(to_hashable(i) for i in item)
        else:
            return item

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        cache = getattr(self, cache_name, {})
        cache_key = (tuple(to_hashable(arg) for arg in args), tuple(sorted(kwargs.items())))
        if cache_key not in cache:
            cache[cache_key] = method(self, *args, **kwargs)
            setattr(self, cache_name, cache)
        return cache[cache_key]

    return wrapper


def frequencies(Y, L):
    P2 = abs(Y / L)
    P1 = P2[0:int(L / 2) + 1]
    P1[1:-1] = 2 * P1[1:-1]
    return P1


class Group:
    def __init__(self, name, animals=None):
        self.name = name
        self.animals = animals if animals else []

    @cache_method
    def get_average(self, data_type, opts):
        return np.mean(np.array([animal.get_average(data_type, opts) for animal in self.animals]), axis=0)


class Animal:
    def __init__(self, name, condition, tone_period_onsets=None, tone_onsets_expanded=None, units=None):
        self.name = name
        self.condition = condition
        self.units = units if units is not None else dd(list)
        self.tone_onsets_expanded = tone_onsets_expanded if tone_onsets_expanded is not None else []
        self.tone_period_onsets = tone_period_onsets if tone_period_onsets is not None else []

    @cache_method
    def get_average(self, data_type, opts):
        return np.mean(np.array([getattr(unit, f"get_{data_type}")(opts) for unit in self.units['good']]), axis=0)

    @cache_method
    def get_frequencies(self, opts):
        if opts['method'] == 'autocorr':
            fft = np.fft.fft(self.get_average('autocorr'), opts)
        elif opts['method'] == 'xform':
            fft = self.get_average('fft', opts)
        return frequencies(fft, opts['lags'])

    def plot_units(self, opts):
        for i in range(0, len(self.units['good']), opts['units_in_fig']):
            fig = plt.figure(figsize=(10, 5*opts['units_in_fig']))
            for j in range(i, min(i + opts['units_in_fig'], len(self.units))):
                self.units['good'][j].plot_unit(fig, (j % opts['units_in_fig'])*2 + 1, opts)

            marker1 = i + 1
            marker2 = min(i + opts['units_in_fig'], len(self.units))

            fname = f"{self.name}_unit_{marker1}_to_{marker2}.png"
            fname = os.path.join(opts['graph_dir'], fname)
            fig.savefig(fname)
            plt.close(fig)


class Unit:
    def __init__(self, animal, category, spike_times):
        self.animal = animal
        self.category = category
        self.spike_times = spike_times
        self.animal.units[category].append(self)
        self.index = self.animal.units[category].index(self)

    def find_spikes(self, start, stop):
        return self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)]

    @cache_method
    def get_trials_spikes(self, opts):
        trials = opts['trials']
        if len(opts['trials']) == 2:
            starts = self.animal.tone_onsets_expanded[trials[0]:trials[1]]
        elif len(opts['trials']) == 3:
            starts = self.animal.tone_onsets_expanded[trials[0]:trials[1]:trials[2]]

        return [
            [(spike - start)/30000 for spike in
             self.find_spikes(start - opts['pre_stim_time'] * 30000, start + opts['post_stim_time'] * 30000)]
            for start in starts]

    @cache_method
    def get_hist(self, spikes, opts):
        pre_stim = opts['pre_stim_time']
        post_stim = opts['post_stim_time']
        bin_size = opts['bin_size_time']
        num_bins = int((post_stim + pre_stim) / bin_size)
        return np.histogram(spikes, bins=num_bins, range=(-pre_stim, post_stim))

    @cache_method
    def get_rates(self, spikes, opts):
        return self.get_hist(spikes, opts)[0] / opts['bin_size_time']

    def get_pretone_means(self, opts):
        return [np.mean(np.array(self.get_rates(self.find_spikes(onset-30*30000, onset-1), opts)))
                for onset in self.animal.tone_period_onsets]

    # TODO: implement subtracting pretone means
    @cache_method
    def get_psth(self, opts):
        pretone_means = self.get_pretone_means(opts)
        rates = [self.get_rates(spikes, opts) for spikes in self.get_trials_spikes(opts)]
        return np.mean(np.array(rates), axis=0)

    @cache_method
    def get_autocorr(self, opts):
        return acf(self.get_psth(opts), opts['lags'])

    @cache_method
    def get_fft(self, opts):
        return np.fft.fft(self.get_autocorr(opts))

    @cache_method
    def get_frequencies(self, opts):
        return frequencies(self.get_fft(opts), opts['lags'])

    def plot_unit(self, fig, subplot_position, opts):
        ax1 = fig.add_subplot(opts['units_in_fig']*2, 1, subplot_position)
        graph1 = Graph(ax1)
        graph1.plot_raster(self.get_trials_spikes(opts), opts)

        ax2 = fig.add_subplot(opts['units_in_fig']*2, 1, subplot_position + 1)
        graph2 = Graph(ax2)
        graph2.plot_psth(self.get_psth(opts), opts)


class Graph:
    def __init__(self, ax):
        self.ax = ax

    def plot_raster(self, data, opts):
        for i, spiketrain in enumerate(data):
            for spike in spiketrain:
                self.ax.vlines(spike, i + .5, i + 1.5)
        self.ax.set_ylim(.5, len(data) + .5)
        self.ax.set_xlim([-opts['pre_stim_time'], opts['post_stim_time']])
        self.ax.set_xticks(np.arange(0, opts['post_stim_time'] + opts['pre_stim_time'], step=opts['tick_step']))
        self.ax.add_patch(plt.Rectangle((0, self.ax.get_ylim()[0]), 0.05,
                                        self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                                        facecolor='gray', alpha=0.3))

    def plot_psth(self, data, opts):
        x = np.linspace(-opts['pre_stim_time'], opts['post_stim_time'], num=len(data))
        self.ax.bar(x, data, width=opts['bin_size_time'], color='k')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Relative Spike Rate (Hz)')
        self.ax.set_xlim(-opts['pre_stim_time'], opts['post_stim_time'])
        self.ax.set_xticks(np.arange(0, opts['post_stim_time'] + opts['pre_stim_time'], step=opts['tick_step']))
        self.ax.fill_betweenx([min(data), max(data)], 0, 0.05, color='k', alpha=0.2)


def init_animal(entry):
    name = entry[1][0]
    condition = entry[2][0]
    tone_period_onsets = entry[4][0]
    tone_onsets_expanded = entry[6][0]
    animal = Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded)
    categories = entry[3][0][0]
    category_names = [k for k in categories.dtype.fields.keys()]
    cat_units = dict(zip(category_names, [category[0] for category in categories]))
    units = {cat: [[spike_time[0] for spike_time in unit[0]] for unit in cat_units[cat]] for cat in category_names}
    {cat: [Unit(animal, cat, unit) for unit in units[cat]] for cat in units}
    return animal


mat_contents = sio.loadmat('/Users/katie/likhtik/data/single_cell_data.mat')
data = mat_contents['single_cell_data']

animals = [init_animal(entry) for entry in data[0]]

groups = [Group(name='name', animals=[animal for animal in animals if animal.condition == name])
          for name in ('control', 'stressed')]


base_opts = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4}
psth_opts = {**base_opts, **{'data_type': 'psth', 'pre_stim_time': 0.05, 'post_stim_time': 0.65,
                             'bin_size_time': 0.01, 'trials': (0, 150), 'tick_step': 0.1}}
autocorr_opts = {**base_opts, **{'data_type': 'autocorr', 'pre_stim_time': 0.0, 'post_stim_time': 30.0,
                                 'bin_size_time': 0.01, 'trials': (0, 150, 30), 'lags': 100}}
fft_opts = {**autocorr_opts, **{'data_type': 'fft', 'bin_size_time': 0.001}}


for animal in animals:
    animal.plot_units(psth_opts)










