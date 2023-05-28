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
from matplotlib.gridspec import GridSpec


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
        return Math.frequencies(fft, opts['lags'])

    from matplotlib.gridspec import GridSpec

    def plot_units(self, opts):
        multi = 2 if opts['data_type'] == 'psth' else 1

        for i in range(0, len(self.units['good']), opts['units_in_fig']):
            n_subplots = min(opts['units_in_fig'], len(self.units['good']) - i)
            fig = plt.figure(figsize=(10, 3 * multi * n_subplots))

            # Create a GridSpec for n_subplots * multi rows and 1 column
            gs = GridSpec(n_subplots * multi, 1, figure=fig)

            for j in range(i, i + n_subplots):
                if opts['data_type'] == 'psth':
                    # Add two subplots in the (2 * (j - i)) and (2 * (j - i) + 1)-th slots of the grid
                    ax1 = fig.add_subplot(gs[2 * (j - i), 0])
                    ax2 = fig.add_subplot(gs[2 * (j - i) + 1, 0])
                    self.units['good'][j].plot_unit([ax1, ax2], opts)
                elif opts['data_type'] == 'autocorr':
                    # Add one subplot in the (j - i)-th slot of the grid
                    ax = fig.add_subplot(gs[j - i, 0])
                    self.units['good'][j].plot_unit([ax], opts)

            marker1 = i + 1
            marker2 = i + n_subplots

            fname = f"{self.name}_unit_{marker1}_to_{marker2}.png"
            fig.suptitle(f"{self.name} Units {marker1} to {marker2}", weight='bold',
                         y=.95)

            if opts['data_type'] == 'psth':
                xlabel = 'Time (s)'
                ylabel = ''
            elif opts['data_type'] == 'autocorr':
                xlabel = 'Lags (s)'
                ylabel = 'Autocorrelation'

            # Add a big subplot without frame and set the x and y labels for this subplot
            big_subplot = fig.add_subplot(111, frame_on=False)
            big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            big_subplot.set_xlabel(xlabel, labelpad=30)  # change labelpad to adjust label position
            big_subplot.set_ylabel(ylabel, labelpad=30)  # change labelpad to adjust label position

            path = os.path.join(opts['graph_dir'], f"{opts['data_type']}_{'_'.join([str(t) for t in opts['trials']])}")
            os.makedirs(path, exist_ok=True)

            plt.subplots_adjust(hspace=0.5)  # Add space between subplots

            fig.savefig(os.path.join(path, fname))
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
        return acf(self.get_psth(opts), nlags=opts['lags'])

    @cache_method
    def get_fft(self, opts):
        return np.fft.fft(self.get_autocorr(opts))

    @cache_method
    def get_frequencies(self, opts):
        return Math.frequencies(self.get_fft(opts), opts['lags'])

    def plot_unit(self, axes, opts):

        if opts['data_type'] == 'psth':
            subplot = Subplot(axes[0])
            subplot.plot_raster(self.get_trials_spikes(opts), opts)

        subplot = Subplot(axes[-1])
        if opts['data_type'] == 'psth':
            subplot.plot_psth(self.get_psth(opts), opts)
        elif opts['data_type'] == 'autocorr':
            subplot.plot_autocorr(self.get_autocorr(opts)[1:], opts)

class Subplot:
    def __init__(self, ax):
        self.ax = ax

    def set_limits_and_ticks(self, x_min, x_max, x_tick_min, x_step, y_min=None, y_max=None):
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_xticks(np.arange(x_tick_min, x_max, step=x_step))
        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)

    def set_labels_and_titles(self, x_label='', y_label='', title=''):
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)

    def plot_raster(self, data, opts):
        for i, spiketrain in enumerate(data):
            for spike in spiketrain:
                self.ax.vlines(spike, i + .5, i + 1.5)
        self.set_labels_and_titles(y_label='Trial')
        self.set_limits_and_ticks(-opts['pre_stim_time'], opts['post_stim_time'], opts['tick_step'], .5, len(data) + .5)
        self.ax.add_patch(plt.Rectangle((0, self.ax.get_ylim()[0]), 0.05, self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                                        facecolor='gray', alpha=0.3))

    def plot_bar(self, data, width, x_min, x_max, num, x_tick_min, x_step, y_min=None, y_max=None, x_label='', y_label='',
                 color='k', title=''):
        x = np.linspace(x_min, x_max, num=num)
        self.ax.bar(x, data, width=width, color=color)
        self.set_limits_and_ticks(x_min, x_max, x_tick_min, x_step, y_min, y_max)
        self.set_labels_and_titles(x_label=x_label, y_label=y_label, title=title)

    def plot_psth(self, data, opts):
        self.plot_bar(data, width=opts['bin_size_time'], x_min=-opts['pre_stim_time'], x_max=opts['post_stim_time'],
                      num=len(data), x_tick_min=0, x_step=opts['tick_step'], y_label='Relative Spike Rate (Hz)')
        self.ax.fill_betweenx([min(data), max(data)], 0, 0.05, color='k', alpha=0.2)

    def plot_autocorr(self, data, opts):
        self.plot_bar(data, width=opts['bin_size_time'], x_min=opts['bin_size_time'],
                      x_max=opts['lags'] * opts['bin_size_time'], num=opts['lags'], x_tick_min=opts['tick_step'],
                      x_step=opts['tick_step'], y_min=0, y_max=max(data) + .05)


class Math:

    @staticmethod
    def frequencies(Y, L):
        P2 = abs(Y / L)
        P1 = P2[0:int(L / 2) + 1]
        P1[1:-1] = 2 * P1[1:-1]
        return P1


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
                                 'bin_size_time': 0.01, 'trials': (0, 150, 30), 'lags': 100, 'tick_step': 0.1}}
fft_opts = {**autocorr_opts, **{'data_type': 'fft', 'bin_size_time': 0.001}}


for animal in animals:
    animal.plot_units(psth_opts)
    animal.plot_units(autocorr_opts)










