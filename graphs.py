from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict as dd
from copy import deepcopy
import functools
import os

import numpy as np
import scipy.io as sio
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from statsmodels.tsa.stattools import acf


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
    def get_average(self, data_type, opts):  # sometimes you might want to average a data type other than what's in opts
        return np.mean(np.array([animal.get_average(data_type, opts) for animal in self.animals]), axis=0)

    def plot_animals(self, data_type):
        Plotter(opts, data_type=data_type).plot_units(self)


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
    def get_spectrum(self, method, opts):
        if method == 'autocorr':
            fft = np.fft.fft(self.get_average('autocorr'))
        elif method == 'xform':
            fft = self.get_average('fft', opts)
        return SignalProcessing.compute_one_sided_spectrum(fft, opts['lags'])

    def plot_units(self, opts):
        Plotter(opts).plot_units(self)


class Unit:
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
        rates = self.get_hist(spikes, opts, num_bins=num_bins, spike_range=spike_range)[0] / opts['bin_size']
        return rates

    @cache_method
    def get_pretone_means(self, opts):
        rate_set = []
        for onset in self.animal.tone_period_onsets:
            start = onset - 30 * 30000
            stop = onset - 1
            spikes = self.find_spikes(onset - 30 * 30000, onset - 1)
            rates = self.get_rates(spikes, opts, spike_range=(start, stop), num_bins=int(30/opts['bin_size']))
            rate_set.append(rates)
        return np.mean(np.array(rate_set), axis=1)


    @cache_method
    def get_psth(self, opts):
        pretone_means = self.get_pretone_means(opts)

        # Handle slice provided in opts['trials']
        trials_slice = slice(*opts['trials'])
        trials_indices = list(range(150))[trials_slice]

        # Calculate rates and subtract corresponding pretone mean
        trials = []
        for i, trial_index in enumerate(trials_indices):
            spikes = self.get_trials_spikes(opts)[trial_index]
            rates = self.get_rates(spikes, opts)
            # Determine which tone period the trial belongs to
            tone_period_index = trial_index // 30  # since every 30 trials are in the same tone period
            pretone_mean = pretone_means[tone_period_index]
            # Subtract pretone mean from rate
            adjusted_rates = rates - pretone_mean
            trials.append(adjusted_rates)

        return np.mean(np.array(trials), axis=0)

    @cache_method
    def get_autocorr(self, opts):
        return acf(self.get_psth(opts), nlags=opts['lags'])

    @cache_method
    def get_fft(self, opts):
        return np.fft.fft(self.get_autocorr(opts))

    @cache_method
    def get_spectrum(self, opts):
        return SignalProcessing.compute_one_sided_spectrum(self.get_fft(opts))

    @staticmethod
    def set_var(var, expression):
        return var if var is not None else expression


class Plotter:
    def __init__(self, opts, data_type=None):
        self.opts = opts
        self.dtype = data_type if data_type is not None else opts['data_type']
        self.labels = {'psth': ('Time (s)', 'Firing Rate (Hz'), 'autocorr': ('Lags (s)',  'Autocorrelation'),
                       'spectrum': ('Frequencies (Hz)',  'One-Sided Spectrum')}

    def plot_animals(self, group):
        n_animals = len(group.animals)
        fig = plt.figure(figsize=(10, 3 * n_animals))

        for i, animal in enumerate(group.animals):
            average_data = animal.get_average(data_type, self.opts)
            row = i // 2
            col = i % 2
            ax = fig.add_subplot(n_animals // 2 + n_animals % 2, 2, row * 2 + col + 1)
            subplotter = Subplotter(ax)
            getattr(subplotter, f"plot_{self.dtype}")(average_data, self.opts)
            ax.set_xlabel(self.labels[self.dtype])
            ax.set_ylabel(self.labels[self.dtype])
            ax.set_title(f"{animal.name} {self.dtype}")

        fig.tight_layout()

        path = os.path.join(opts['graph_dir'], f"{self.dtype}_{'_'.join([str(t) for t in opts['trials']])}")
        os.makedirs(path, exist_ok=True)

        fname = f"group_{data_type}.png"
        fig.savefig(os.path.join(path, fname))
        plt.close(fig)

    def plot_units(self, animal):
        multi = 2 if self.dtype == 'psth' else 1

        for i in range(0, len(animal.units['good']), self.opts['units_in_fig']):
            n_subplots = min(self.opts['units_in_fig'], len(animal.units['good']) - i)
            fig = plt.figure(figsize=(10, 3 * multi * n_subplots))

            # Create a GridSpec for n_subplots * multi rows and 1 column
            gs = GridSpec(n_subplots * multi, 1, figure=fig)

            for j in range(i, i + n_subplots):
                if self.dtype == 'psth':
                    # Add two subplots in the (2 * (j - i)) and (2 * (j - i) + 1)-th slots of the grid
                    axes = [fig.add_subplot(gs[2 * (j - i), 0]), fig.add_subplot(gs[2 * (j - i) + 1, 0])]
                elif self.dtype in ['autocorr', 'spectrum']:
                    # Add one subplot in the (j - i)-th slot of the grid
                    axes = [fig.add_subplot(gs[j - i, 0])]
                self.plot_unit(animal.units['good'][j], axes)

            marker1 = i + 1
            marker2 = i + n_subplots

            fname = f"{animal.name}_unit_{marker1}_to_{marker2}.png"
            fig.suptitle(f"{animal.name} Units {marker1} to {marker2}", weight='bold',
                         y=.95)

            # Add a big subplot without frame and set the x and y labels for this subplot
            big_subplot = fig.add_subplot(111, frame_on=False)
            big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            big_subplot.set_xlabel(self.labels[opts['data_type'][0]], labelpad=30)  # change labelpad to adjust position
            big_subplot.set_ylabel(self.labels[opts['data_type'][1]], labelpad=30)

            path = os.path.join(opts['graph_dir'], f"{opts['data_type']}_{'_'.join([str(t) for t in opts['trials']])}")
            os.makedirs(path, exist_ok=True)

            plt.subplots_adjust(hspace=0.5)  # Add space between subplots

            fig.savefig(os.path.join(path, fname))
            plt.close(fig)

    def plot_unit(self, unit, axes):
        if self.opts['data_type'] == 'psth':
            Subplotter(axes[0]).plot_raster(unit.get_trials_spikes(self.opts), self.opts)
        subplotter = Subplotter(axes[-1])
        getattr(subplotter, f"plot_{self.dtype}")(getattr(unit, f"get_{self.dtype}")(self.opts), self.opts)


class Subplotter:
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
        self.set_limits_and_ticks(-opts['pre_stim'], opts['post_stim'], opts['tick_step'], .5, len(data) + .5)
        self.ax.add_patch(plt.Rectangle((0, self.ax.get_ylim()[0]), 0.05, self.ax.get_ylim()[1] - self.ax.get_ylim()[0],
                                        facecolor='gray', alpha=0.3))

    def plot_bar(self, data, width, x_min, x_max, num, x_tick_min, x_step, y_min=None, y_max=None, x_label='', y_label='',
                 color='k', title=''):
        x = np.linspace(x_min, x_max, num=num)
        self.ax.bar(x, data, width=width, color=color)
        self.set_limits_and_ticks(x_min, x_max, x_tick_min, x_step, y_min, y_max)
        self.set_labels_and_titles(x_label=x_label, y_label=y_label, title=title)

    def plot_psth(self, data, opts):
        self.plot_bar(data, width=opts['bin_size'], x_min=-opts['pre_stim'], x_max=opts['post_stim'],
                      num=len(data), x_tick_min=0, x_step=opts['tick_step'], y_label='Relative Spike Rate (Hz)')
        self.ax.fill_betweenx([min(data), max(data)], 0, 0.05, color='k', alpha=0.2)

    def plot_autocorr(self, data, opts):
        self.plot_bar(data, width=opts['bin_size'], x_min=opts['bin_size'],
                      x_max=opts['lags'] * opts['bin_size'], num=opts['lags'], x_tick_min=opts['tick_step'],
                      x_step=opts['tick_step'], y_min=0, y_max=max(data) + .05)

    def plot_spectrum(self, data, opts):
        last_index = opts['up_to_hz'] + 1
        x = SignalProcessing.get_positive_frequencies(opts['lags'], opts['bin_size'])[:last_index]
        y = data[:last_index]
        self.ax.plot(x, y)


class SignalProcessing:

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


def init_animal(entry):
    name = entry[1][0]
    condition = entry[2][0]
    tone_period_onsets = entry[4][0]
    tone_onsets_expanded = entry[6][0]
    animal = Animal(name, condition, tone_period_onsets=tone_period_onsets, tone_onsets_expanded=tone_onsets_expanded)
    categories = entry[3][0][0]
    cat_names = [k for k in categories.dtype.fields.keys()]
    cat_units = dict(zip(cat_names, [category[0] for category in categories]))
    units = {cat: [{'spikes': [spike_time[0] for spike_time in unit[0]]} for unit in cat_units[cat]] for cat in cat_names}
    {cat: [Unit(animal, cat, unit['spikes']) for unit in units[cat]] for cat in units}
    for i, unit in enumerate(animal.units['good']):
        unit.neuron_type = 'PN' if cat_units['good'][i][8][0][0] < 2 else 'IN'
    return animal


mat_contents = sio.loadmat('/Users/katie/likhtik/data/single_cell_data.mat')
data = mat_contents['single_cell_data']

animals = [init_animal(entry) for entry in data[0]]

groups = [Group(name='name', animals=[animal for animal in animals if animal.condition == name])
          for name in ('control', 'stressed')]


base_opts = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4}
psth_opts = {**base_opts, **{'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01,
                             'trials': (0, 150), 'tick_step': 0.1}}
autocorr_opts = {**base_opts, **{'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.01,
                                 'trials': (0, 150, 30), 'lags': 100, 'tick_step': 0.1}}
fft_opts = {**autocorr_opts, **{'data_type': 'spectrum', 'bin_size': 0.001, 'lags': 1000, 'up_to_hz': 100}}


for animal in animals:
    animal.plot_units(psth_opts)
    animal.plot_units(autocorr_opts)
    animal.plot_units(fft_opts)










