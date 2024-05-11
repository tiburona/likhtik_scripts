import os
import math
import numpy as np
import seaborn as sns
import pandas as pd
from copy import deepcopy
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.ticker as ticker


from math_functions import get_positive_frequencies, get_spectrum_fenceposts
from plotting_helpers import smart_title_case, formatted_now, PlottingMixin
from utils import to_serializable
from data import Base
from stats import Stats
from phy_interface import PhyInterface


class Plotter(Base):
    """Makes plots, where a plot is a display of particular kind of data.  For displays of multiple plots of multiple
    kinds of data, see the figure module."""

    def __init__(self, experiment, graph_opts=None, lfp=None, behavior=None, plot_type='standalone'):
        self.experiment = experiment
        self.lfp = lfp
        self.graph_opts = graph_opts
        self.plot_type = plot_type
        self.fig = None
        self.axs = None
        self.y_min = float('inf')
        self.y_max = float('-inf')
        self.fname = ''
        self.title = ''
        self.dir_tags = None
        self.stats = None
        self.full_axes = None
        self.invisible_ax = None
        self.grid = None

    def initialize(self, data_opts, graph_opts, neuron_type='all'):
        """Both initializes values on self and sets values for the context."""
        self.graph_opts = graph_opts
        self.data_opts = data_opts  # Sets data_opts for all subscribers to context
        print("did I get here?")
        self.selected_neuron_type = neuron_type
        print("plotter initialize has run")

    def close_plot(self, basename):
        self.set_dir_and_filename(basename)
        if self.graph_opts.get('footer'):
            self.make_footer()
        self.save_and_close_fig()

    def save_and_close_fig(self):
        dirs = [self.graph_opts['graph_dir']]
        if self.dir_tags is not None:
            dirs += self.dir_tags
        path = os.path.join(*dirs)
        os.makedirs(path, exist_ok=True)
        self.fig.savefig(os.path.join(path, self.fname))
        
        opts_filename = self.fname.replace('png', 'txt')
        # Writing the dictionary to a file in JSON format
        with open(os.path.join(path, opts_filename), 'w') as file:
            json.dump(to_serializable(self.data_opts), file)

        plt.close(self.fig)


class PeriStimulusPlotter(Plotter, PlottingMixin):
    """Makes plots where the x-axis is time around the stimulus, and y can be a variety of types of data."""

    def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)
        self.multiplier = 1 if self.plot_type == 'standalone' else 0.5

    def plot(self, data_opts, graph_opts, neuron_type='all'):
        self.initialize(data_opts, graph_opts, neuron_type)
        level = self.data_opts['level']
        if level == 'group':
            self.plot_groups()
        elif level == 'animal':
            for group in self.experiment.groups:
                self.plot_animals(group)
        elif level == 'unit':
            [self.plot_units(animal) for group in self.experiment.groups for animal in group.children]
        elif level == 'unit_pair':
            [self.plot_unit_pairs(unit) for unit in self.experiment.all_units]
        else:
            print('unrecognized plot type')

    def plot_groups(self):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 15))
        self.fig.subplots_adjust(wspace=0.2, hspace=0.2)
        self.plot_groups_data()
        self.close_plot('groups')

    def plot_groups_data(self):
        self.iterate_through_group_subdivisions()
        self.set_y_scales()
        if self.data_type not in ['spontaneous_firing', 'cross_correlations']:
            self.set_pip_patches()
        if self.data_type in ['cross_correlations', 'correlogram']:
            n1, n2 = self.data_opts['unit_pairs'][0].split(',')
            self.set_labels(x_and_y_labels=('Lags (s)', f"{n1} to {n2}"))
        else:
            self.set_labels()

    def get_subdivisions(self):
        subdivision = 'period' if self.data_type in ['cross_correlations', 'correlogram'] else 'neuron'
        types_attribute = getattr(self.experiment, f"{subdivision}_types")
        if subdivision != 'period':
            self.selected_period_type = list(self.data_opts['periods'].keys())[0]  # TODO: this is really kludgy
        return subdivision, types_attribute

    def iterate_through_group_subdivisions(self):  # TODO: why do autocorr and spectrum graph these in different orders
        subdivision, types_attribute = self.get_subdivisions()
        for row, typ in enumerate(types_attribute):
            # Set the selected type based on the subdivision
            setattr(self, f"selected_{subdivision}_type", typ)
            for col, group in enumerate(self.experiment.groups):
                self.make_subplot(group, self.axs[row, col], title=f"{group.identifier.capitalize()} {typ}")
        # Reset the selected type to None after iteration
        setattr(self, f"selected_{subdivision}_type", None)

    def plot_animals(self, group):
        subdivision, types_attribute = self.get_subdivisions()
        for typ in types_attribute:
            setattr(self, f"selected_{subdivision}_type", typ)
            num_animals = len(group.children)
            nrows = math.ceil(num_animals / 3)
            self.fig, self.axs = plt.subplots(nrows, 3, figsize=(15, nrows * 6))
            self.fig.subplots_adjust(top=0.85, hspace=0.3, wspace=0.4)

            for i in range(nrows * 3):  # iterate over all subplots
                if i < num_animals:
                    row = i // 3  # index based on 3 columns
                    col = i % 3  # index based on 3 columns
                    ax = self.axs[row, col] if nrows > 1 else self.axs[i // 3]
                    animal = group.children[i]
                    self.make_subplot(animal, ax, f"{animal.identifier} {animal.selected_neuron_type}")
                else:
                    # Get the axes for the extra subplot and make it invisible
                    ax_ind = (i // 3, i % 3) if nrows > 1 else (i // 3,)
                    self.axs[ax_ind].set_visible(False)

            self.set_y_scales()
            self.set_pip_patches()
            self.close_plot(group.identifier)

    def make_subplot(self, data_source, ax, title=''):
        subplotter = PeriStimulusSubplotter(self, data_source, self.graph_opts, ax, self.plot_type,
                                            multiplier=self.multiplier)
        subplotter.plot_data()
        if self.graph_opts.get('sem'):
            subplotter.add_sem()
        self.prettify_subplot(ax, title=title, y_min=min(subplotter.y), y_max=max(subplotter.y))

    def plot_unit_pairs(self, unit):
        unit_pairs = unit.unit_pairs
        pair_categories = set([unit_pair.pair_category for unit_pair in unit_pairs])
        for pair_category in pair_categories:
            data_sources = [pair for pair in unit_pairs if pair.pair_category == pair_category]
            for i in range(0, len(data_sources), self.graph_opts['units_in_fig']):
                self.plot_units_level_data(data_sources, i)
                marker2 = min(i + self.graph_opts['units_in_fig'], len(data_sources))
                self.close_plot(
                    f"{unit.animal.identifier} {pair_category} unit {unit.identifier} pair {i + 1} to {marker2}")

    def plot_units(self, animal):
        for i in range(0, len(animal.children), self.graph_opts['units_in_fig']):
            self.plot_units_level_data(animal.children, i)
            marker2 = min(i + self.graph_opts['units_in_fig'], len(animal.children))
            self.close_plot(f"{animal.identifier} unit {i + 1} to {marker2}")

    def plot_units_level_data(self, data_sources, i):
        multi = 2 if self.data_type == 'psth' else 1
        n_subplots = min(self.graph_opts['units_in_fig'], len(data_sources) - i)
        self.fig = plt.figure(figsize=(15, 3 * multi * n_subplots))
        self.fig.subplots_adjust(top=0.8, bottom=.14)
        gs = GridSpec(n_subplots * multi, 1, figure=self.fig)
        for j in range(i, i + n_subplots):
            if self.data_type == 'psth':
                axes = [self.fig.add_subplot(gs[2 * (j - i), 0]), self.fig.add_subplot(gs[2 * (j - i) + 1, 0])]
            else:
                axes = [self.fig.add_subplot(gs[j - i, 0])]
            self.plot_unit_level_data(data_sources[j], axes)
        self.set_units_plot_frame_and_spacing()

    def plot_unit_level_data(self, data_source, axes):
        if self.data_type == 'psth':
            self.add_raster(data_source, axes)
        subplotter = PeriStimulusSubplotter(self, data_source, self.graph_opts, axes[-1])
        plotting_func = getattr(subplotter, f"plot_{self.data_type}")
        plotting_func()
        if self.graph_opts.get('sem'):
            subplotter.add_sem()

    def add_raster(self, unit, axes):
        subplotter = PeriStimulusSubplotter(self, unit, self.graph_opts, axes[0])
        subplotter.y = unit.get_spikes_by_events()  # overwrites subplotter.y defined by data_type, which is psth
        subplotter.plot_raster()

    def set_units_plot_frame_and_spacing(self):
        # Add a big subplot without frame and set the x and y labels for this subplot
        big_subplot = self.fig.add_subplot(111, frame_on=False)
        big_subplot.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        big_subplot.set_xlabel(self.get_labels()[self.data_type][0], labelpad=10, fontsize=14)
        if not self.data_type == 'psth':
            self.set_labels(x_and_y_labels=('', self.get_labels()[self.data_type][1]))
        self.fig.subplots_adjust(top=0.8, bottom=0.2, hspace=0.5)

    def get_ylim(self, ax, y_min, y_max):
        self.y_min = min(self.y_min, ax.get_ylim()[0], y_min)
        self.y_max = max(self.y_max, ax.get_ylim()[1], y_max)

    def set_y_scales(self):
        if self.graph_opts['equal_y_scales']:
            [ax.set_ylim(self.y_min, self.y_max) for ax in self.axs.flatten()]

    def set_pip_patches(self):
        [ax.fill_betweenx([self.y_min, self.y_max], 0, self.experiment.stimulus_duration, color='k', alpha=0.2)
         for ax in self.axs.flatten()]

    def prettify_subplot(self, ax, title, y_min, y_max):
        self.get_ylim(ax, y_min, y_max)
        ax.set_title(title, fontsize=17 * self.multiplier)

    def make_footer(self):
        text_vals = [('bin size', self.data_opts['bin_size']), ('selected events', self.join_events(' ')),
                     ('time generated', formatted_now())]
        [text_vals.append((k, self.data_opts[k])) for k in ['adjustment, average_method'] if k in self.data_opts]
        text = '  '.join([f"{k}: {v}" for k, v in text_vals])
        if self.data_type in ['autocorr', 'spectrum']:  # TODO update this with changes made to autocorrelation and spetrum
            ac_vals = [('method', self.data_opts['ac_key'])]
            ac_text = '  '.join([f"{k}: {v}" for k, v in ac_vals])
            text = f"{text}\n{ac_text}"
        self.fig.text(0.5, 0.02, text, ha='center', va='bottom', fontsize=15)

    def set_dir_and_filename(self, basename):
        tags = [self.data_type]
        self.dir_tags = tags + [self.join_events('_')]
        tags.insert(0, basename)
        if self.selected_neuron_type:
            tags += [self.selected_neuron_type]
        self.title = smart_title_case(' '.join([tag.replace('_', ' ') for tag in tags]))
        self.fig.suptitle(self.title, weight='bold', y=.95, fontsize=20)
        if self.data_opts.get('base'):
            tags += [self.data_opts.get('base')]
        self.fname = f"{'_'.join(tags)}.png"

    def join_events(self, s):
        tag = ''
        for key in self.data_opts:
            if 'event' in key:
                tag += key + '_' + s.join([str(t) for t in self.data_opts[key]])
        return tag

    def set_gridspec_axes(self, fig, gridspec, numrows, numcols, invisible_ax=None):
        self.fig = fig
        self.grid = gridspec

        if numrows > 1 and numcols > 1:
            self.axs = np.array([
                [plt.Subplot(fig, gridspec[row, col]) for col in range(numcols)]
                for row in range(numrows)
            ])
        elif numrows == 1 or numcols == 1:  # For case where only one row or one column
            self.axs = np.array([plt.Subplot(fig, gridspec[i]) for i in range(max(numrows, numcols))])
        else:
            raise ValueError("Number of rows or columns must be greater than zero.")

        for ax in np.ravel(self.axs):
            fig.add_subplot(ax)

        self.invisible_ax = invisible_ax


class PeriStimulusSubplotter(Plotter, PlottingMixin):
    """Constructs a subplot of a PeriStimulusPlot."""

    def __init__(self, plotter, data_source, graph_opts, ax, parent_type='standalone', multiplier=1):
        self.plotter = plotter
        self.data_source = data_source
        self.g_opts = graph_opts
        self.ax = ax
        self.x = None
        self.y = data_source.data
        self.parent_type = parent_type
        self.multiplier = multiplier
        self.plot_type = 'subplot'

    def set_limits_and_ticks(self, x_min, x_max, x_tick_min, x_step, y_min=None, y_max=None):
        self.ax.set_xlim(x_min, x_max)  # TODO: it would be nice if autocorrelation and cross-correlation took the same units of tick step
        xticks = np.arange(x_tick_min, x_max, step=x_step)
        self.ax.set_xticks(xticks)
        self.ax.tick_params(axis='both', which='major', labelsize=10 * self.multiplier, length=5 * self.multiplier,
                            width=2 * self.multiplier)
        if self.data_type in ['spontaneous_firing', 'cross_correlations']:
            self.ax.set_xticklabels(xticks * self.data_opts['bin_size'])

        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)

    def plot_raster(self):
        pre, post = [self.data_opts['events'][self.selected_period_type][opt] for opt in ['pre_stim', 'post_stim']]
        for i, spiketrain in enumerate(self.y):
            for spike in spiketrain:
                self.ax.vlines(spike, i + .5, i + 1.5)
        self.set_labels(x_and_y_labels=['', 'Event'])
        self.set_limits_and_ticks(pre, post, self.g_opts['tick_step'], .5, len(self.y) + .5)
        self.ax.add_patch(plt.Rectangle(
            (0, self.ax.get_ylim()[0]), 0.05, self.ax.get_ylim()[1] - self.ax.get_ylim()[0], facecolor='gray',
            alpha=0.3))

    def plot_bar(self, width, x_min, x_max, num, x_tick_min, x_step, y_min=None, y_max=None, x_label='', y_label='',
                 facecolor='white', zero_line=False):
        if self.data_source.name == 'group' and 'group_colors' in self.g_opts:
            color = self.g_opts['group_colors'][self.data_source.identifier]
        else:
            color = 'black'
        self.x = np.linspace(x_min, x_max, num=num)
        self.ax.bar(self.x, self.y, width=width, color=color)
        self.ax.set_facecolor(facecolor)
        self.ax.patch.set_alpha(0.2)
        self.set_limits_and_ticks(x_min, x_max, x_tick_min, x_step, y_min, y_max)

        if zero_line:
            self.ax.axhline(y=0, color='black', linewidth=1, linestyle='-')
            self.ax.axvline(x=0, color='black', linewidth=1, linestyle='-')

        if self.parent_type == 'standalone':
            self.set_labels(x_and_y_labels=(x_label, y_label))

    def plot_psth(self):
        pre, post = [self.data_opts['events'][self.selected_period_type][opt] for opt in ['pre_stim', 'post_stim']]
        xlabel, ylabel = self.get_labels()[self.data_opts['data_type']]
        self.plot_bar(width=self.data_opts['bin_size'], x_min=-pre, x_max=post, num=len(self.y), x_tick_min=0,
                      x_step=self.g_opts['tick_step'], y_label=ylabel)

    def plot_proportion(self):
        self.plot_psth()

    def plot_autocorr(self):
        opts = self.data_opts
        self.plot_bar(width=opts['bin_size'] * .95, x_min=opts['bin_size'], x_max=opts['max_lag'] * opts['bin_size'],
                      num=opts['max_lag'], x_tick_min=0, x_step=self.g_opts['tick_step'], y_min=min(self.y),
                      y_max=max(self.y))

    def plot_spectrum(self):
        freq_range, max_lag, bin_size = (self.data_opts.get(opt) for opt in ['freq_range', 'max_lag', 'bin_size'])
        first, last = get_spectrum_fenceposts(freq_range, max_lag, bin_size)
        self.x = get_positive_frequencies(max_lag, bin_size)[first:last]
        self.ax.plot(self.x, self.y)

    def plot_spontaneous_firing(self):  # TODO: don't hardcode period
        opts = self.data_opts
        self.plot_bar(width=opts['bin_size'], x_min=0, x_max=int(120 / self.data_opts['bin_size']), num=len(self.y),
                      x_tick_min=0, x_step=self.g_opts['tick_step'], y_label='Firing Rate (Spikes per Second')

    def plot_cross_correlations(self):
        opts = self.data_opts
        boundary = int(opts['max_lag'] / opts['bin_size'])
        tick_step = self.plotter.graph_opts['tick_step']
        if self.data_source.name == 'group' and 'group_colors' in self.g_opts:
            color = self.g_opts['group_colors'][self.data_source.identifier]
        else:
            color = 'black'
        self.ax.bar(np.linspace(-boundary, boundary, 2 * boundary + 1), self.y)
        self.ax.bar(self.x, self.y, color=color)
        tick_positions = np.arange(-boundary, boundary + 1, tick_step)
        tick_labels = np.arange(-opts['max_lag'], opts['max_lag'] + opts['bin_size'],
                                tick_step * self.data_opts['bin_size'])
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels([f'{label:.2f}' for label in tick_labels])

    def plot_correlogram(self):
        self.plot_cross_correlations()

    def plot_data(self):
        getattr(self, f"plot_{self.data_type}")()

    def add_sem(self):
        opts = self.data_opts
        if opts['data_type'] in ['autocorr', 'spectrum'] and opts['ac_key'] == self.data_source.name + '_by_rates':
            print("It doesn't make sense to add standard error to a graph of autocorr over rates.  Skipping.")
            return
        sem = self.data_source.sem_envelope
        self.ax.fill_between(self.x, self.y - sem, self.y + sem, color='blue', alpha=0.2)


class GroupStatsPlotter(PeriStimulusPlotter):

    def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)

    def plot_group_stats(self, data_opts, graph_opts=None, sig_markers=True):
        self.initialize(data_opts, graph_opts, neuron_type='all')
        self.fig, self.axs = plt.subplots(2, 1, figsize=(15, 15))
        self.current_ax = None
        self.plot_group_stats_data(sig_markers=sig_markers)
        self.close_plot('stats_plot')

    def plot_group_stats_data(self, sig_markers=True):
        self.stats = Stats(self.experiment, self.data_opts)
        if sig_markers:
            interaction_ps, neuron_type_specific_ps = self.stats.get_post_hoc_results()
        
        period_to_plot = self.graph_opts.get('period', 'tone')
        pre_stim, post_stim = (self.data_opts['events'][period_to_plot][opt] 
                               for opt in ['pre_stim', 'post_stim'])
        bin_size = self.data_opts.get('bin_size')
        for row, neuron_type in enumerate(self.neuron_types):
            self.selected_neuron_type = neuron_type
            for group in self.experiment.groups:
                color = self.graph_opts['group_colors'][group.identifier]
                x = np.arange(len(group.data)) * bin_size
                y = group.data
                if max(y) > self.y_max:
                    self.y_max = max(y)
                if min(y) < self.y_min:
                    self.y_min = min(y)
                self.axs[row].plot(x, y, label=group.identifier, color=color)
                sem = group.get_sem()
                self.axs[row].fill_between(x, y - sem, y + sem, color=color, alpha=0.2)

            self.axs[row].set_title(f"{neuron_type}", fontsize=17 * self.multiplier, loc='left')
            
            self.axs[row].set_xticks(np.arange(pre_stim, post_stim, 
                                               step=self.graph_opts['tick_step']))
            self.axs[row].tick_params(axis='both', which='major', labelsize=10 * self.multiplier,
                                      length=5 * self.multiplier, width=2 * self.multiplier)
            self.current_ax = self.axs[row]
            self.set_labels()

            if sig_markers:
                # Annotate significant points within conditions
                self.add_significance_markers(neuron_type_specific_ps[neuron_type], 'within_condition', row=row,
                                              y=self.y_max * 1.05)
        for row in range(len(self.neuron_types)):
            self.axs[row].set_ylim(self.y_min * 1.1, self.y_max * 1.1)
            self.axs[row].set_xlim(pre_stim, post_stim)
            [self.axs[row].spines[side].set_visible(False) for side in ['top', 'right']]

        self.selected_neuron_type = None
        if sig_markers:
            self.add_significance_markers(interaction_ps, 'interaction')
        self.place_legend()

    def add_significance_markers(self, p_values, p_type, row=None, y=None):
        bin_size = self.data_opts.get('bin_size')
        for time_bin, p_value in enumerate(p_values):
            if p_value < .05:
                self.get_significance_markers(row, p_type, time_bin, bin_size, y, p_values)

    def get_significance_markers(self, row, p_type, time_bin, bin_size, y, p_values):
        if p_type == 'within_condition':
            self.axs[row].annotate('*', (time_bin * bin_size, y * 0.85), fontsize=20 * self.multiplier,
                                   ha='center', color='black')
        else:
            self.get_interaction_text(time_bin, p_values)

    def get_interaction_text(self, time_bin, p_values):
        # calculate the x position as a fraction of the plot width
        x = time_bin / len(p_values)
        y = 0.485

        if self.plot_type == 'standalone':
            left = self.axs[0].get_position().xmin
            right = self.axs[0].get_position().xmax
            x = left + (right - left) * x

        else:
            gridspec_position = self.grid.get_subplot_params(self.fig)
            x = gridspec_position.left + (gridspec_position.right - gridspec_position.left) * x
            y = gridspec_position.bottom + (gridspec_position.top - gridspec_position.bottom) / 2 - .01

        self.fig.text(x, y, "\u2020", fontsize=15 * self.multiplier, ha='center', color='black')

    def place_legend(self):
        if self.plot_type == 'standalone':
            x, y = 1, 1
            lines = [mlines.Line2D([], [], color=color, label=condition, linewidth=2 * self.multiplier)
                     for color, condition in zip(['green', 'orange'], ['Control', 'Stressed'])]
            self.fig.legend(handles=lines, loc='upper left', bbox_to_anchor=(x, y), prop={'size': 14 * self.multiplier})


class PiePlotter(Plotter):
    """Constructs a pie chart of up- or down-regulation of individual neurons"""

    def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)

    def unit_upregulation_pie_chart(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts, neuron_type='all')
        labels = ['Up', 'Down', 'No Change']
        colors = ['yellow', 'blue', 'green']

        for nt in self.neuron_types + [None]:
            self.selected_neuron_type = nt
            for group in self.experiment.groups:
                if nt:
                    units = [unit for unit in self.experiment.all_units
                             if (unit.neuron_type == nt and unit.animal.condition == group.identifier)]
                else:
                    units = self.experiment.all_units
                sizes = [len([unit for unit in units if unit.upregulated() == num]) for num in [1, -1, 0]]

                self.fig = plt.figure()
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                plt.axis('equal')
                self.close_plot(f'{group.identifier}')


class NeuronTypePlotter(Plotter):

    def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)

    def scatterplot(self, _):

        x = [unit.fwhm_microseconds for unit in self.experiment.all_units]
        y = [unit.firing_rate for unit in self.experiment.all_units]
        self.fig = plt.figure()
        ax = self.fig.add_subplot(1, 1, 1)  # Arguments are (nrows, ncols, index)
        colors = [self.graph_opts['neuron_type_colors'][unit.neuron_type] for unit in self.experiment.all_units]
        ax.scatter(x, y, color=colors, alpha=0.5)
        ax.set_xlabel('FWHM (\u03BCs)', fontsize=10)
        ax.set_ylabel('Firing Rate (Spikes Per Second)', fontsize=10)
        self.close_plot('Neuron_Type_Scatterplot')

    def set_dir_and_filename(self, basename):

        self.title = 'Neuron Type Scatterplot'
        self.fig.suptitle(self.title, weight='bold', y=.95, fontsize=15)
        self.fname = f"{basename}.png"

    def plot_waveforms(self, _):
        self.fig = plt.figure()

        keys = ['data_path', 'animal_id', 'cluster_ids', 'electrodes_for_feature', 'electrodes_for_waveform',
                'el_inds', 'pc_inds', 'neuron_type_colors']
        (data_path, animal_id, cluster_ids, electrodes_for_feature, electrodes_for_waveform, el_inds, pc_inds, colors) = \
            (self.graph_opts[key] for key in keys)
        colors = list(colors.values())
        phy_interface = PhyInterface(data_path, animal_id)

        for i, cluster_id in enumerate(cluster_ids):
            ax = self.fig.add_subplot(1, 2, i + 1)
            waveform = phy_interface.get_mean_waveforms(cluster_id, electrodes_for_waveform[i])
            waveform = waveform[20:-20]
            if self.graph_opts.get('normalize_waveform'):
                waveform = waveform / abs((np.max(waveform) - np.min(waveform)))
                waveform -= np.mean(waveform)
                y_text = 'Normalized Amplitude'
            else:
                y_text = '\u03BCV'
            ax.plot(np.arange(len(waveform)), waveform, color=colors[i])
            if i == 0:  # We're only putting the FWHM markers on the second line
                min_y, max_y = np.min(waveform), np.max(waveform)
                max_x = np.argmax(waveform)  # find index of max
                half_min = (min_y + max_y) / 2
                # Draw lines indicating amplitude
                ax.hlines([min_y, max_y], xmin=[max_x - 2, max_x - 2], xmax=[max_x + 2, max_x + 2], color='.2', lw=.7)
                ax.vlines(max_x, ymin=min_y, ymax=max_y, color='.2', lw=.7)
                # Find indices where waveform is equal to half_min
                half_min_indices = np.where(np.isclose(waveform, half_min, rtol=3e-1))
                # Draw line connecting points at FWHM
                if half_min_indices[0].size > 0:
                    fwhm_start = half_min_indices[0][0]
                    fwhm_end = half_min_indices[0][-1]
                    # ax.hlines(half_min, xmin=fwhm_start, xmax=fwhm_end, color='.3', lw=.7)
                    ax.text(fwhm_start - 8, half_min, 'FWHM', fontsize=7, ha='right')

            self.label_phy_graph(ax, 'Samples (30k Hz)', y_text)
        self.close_plot('Waveforms')

    def label_phy_graph(self, ax, xlabel, ylabel):
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=5, length=1.25)


class MRLPlotter(Plotter):
    def __init__(self, experiment, lfp=None, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, lfp=lfp, graph_opts=graph_opts, plot_type=plot_type)

    def mrl_rose_plot(self, data_opts, graph_opts):
        self.make_plot(data_opts, graph_opts, 'rose_plot', self.make_rose_plot, 'polar')

    def mrl_heat_map(self, data_opts, graph_opts):
        self.make_plot(data_opts, graph_opts, 'heat_map', self.make_heat_map, None)

    def make_plot(self, data_opts, graph_opts, basename, plot_func, projection):
        self.initialize(data_opts, graph_opts, neuron_type='all')
        self.fig = plt.figure(figsize=(15, 15))

        ncols = 2 if self.data_opts.get('adjustment') == 'relative' else 4

        self.axs = [
            [self.fig.add_subplot(2, ncols, 1 + ncols * row + col, projection=projection) for col in range(ncols)]
            for row in range(2)
        ]

        for i, neuron_type in enumerate(self.experiment.neuron_types):
            self.selected_neuron_type = neuron_type
            for j, group in enumerate(self.lfp.groups): # TODO: think about how this should work now with multiple periods
                if self.data_opts.get('adjustment') == 'relative':
                    self.selected_period_type = 'tone'
                    plot_func(group, self.axs[i][j], title=f"{group.identifier.capitalize()} {neuron_type}")
                else:
                    for k, period_type in enumerate(self.experiment.period_types):
                        self.selected_period_type = period_type
                        plot_func(group, self.axs[i][j * 2 + k],
                                  title=f"{group.identifier.capitalize()} {neuron_type} {period_type}")
        self.selected_neuron_type = None
        self.close_plot(basename)

    def set_dir_and_filename(self, basename):
        frequency_band = str(self.lfp.current_frequency_band)
        tags = [frequency_band, self.current_brain_region]
        self.dir_tags = [self.data_type]
        self.title = smart_title_case(' '.join([tag.replace('_', ' ') for tag in tags]))
        self.fig.suptitle(self.title, weight='bold', y=.95, fontsize=20)
        self.fname = f"{basename}_{'_'.join(tags)}.png"

    def make_rose_plot(self, group, ax, title=""):
        n_bins = 36
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        color = self.graph_opts['group_colors'][group.identifier]
        width = 2 * np.pi / n_bins
        ax.bar(bin_edges[:-1], group.get_angle_counts(), width=width, align='edge', color=color, alpha=1)
        current_ticks = ax.get_yticks()
        ax.set_yticks(current_ticks[::2])  # every second tick
        ax.set_title(title)

    def make_heat_map(self, group, ax, title=""):
        data = group.data_by_period  # Todo put this back/figure out what it should be now
        im = ax.imshow(data.T, cmap='jet', interpolation='nearest', aspect='auto',
                       extent=[0.5, 5.5, self.current_frequency_band[0], self.current_frequency_band[1]],
                       origin='lower')
        cbar = ax.figure.colorbar(im, ax=ax, label='MRL')
        ax.set_title(title)

    def mrl_bar_plot(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts, neuron_type='all')

        data = []
        if data_opts.get('spontaneous'):
            for neuron_type in self.neuron_types:
                self.selected_neuron_type = neuron_type
                for group in self.lfp.groups:
                    data.append([neuron_type, group.identifier, group.data, group.sem, group.scatter,
                                 group.grandchildren_scatter])
            df = pd.DataFrame(data, columns=['Neuron Type', 'Group', 'Average MRL', 'sem', 'scatter', 'unit_scatter'])
        else: 
            for neuron_type in self.experiment.neuron_types:
                self.selected_neuron_type = neuron_type
                for group in self.lfp.groups:
                    for period_type in self.experiment.period_types:
                        self.selected_period_type = period_type
                        data.append([neuron_type, group.identifier, period_type, group.data, group.sem, group.scatter,
                                     group.grandchildren_scatter])
            df = pd.DataFrame(data, columns=['Neuron Type', 'Group', 'Period', 'Average MRL', 'sem', 'scatter',
                                             'unit_scatter'])

        group_order = df['Group'].unique()

        # Plot creation
        if data_opts.get('spontaneous'):
            g = sns.catplot(data=df, x='Group', y='Average MRL', row='Neuron Type', kind='bar',
                            height=4, aspect=1.5, dodge=False, legend=False, order=group_order)
        else:
            period_order = self.graph_opts['period_order']
            g = sns.catplot(data=df, x='Group', y='Average MRL', hue='Period', hue_order=period_order, row='Neuron Type', kind='bar',
                            height=4, aspect=1.5, dodge=True, legend=False, order=group_order)

        g.set_axis_labels("", "Average MRL")
        g.fig.subplots_adjust(top=0.85, hspace=0.4, right=0.85)
        g.despine()

        for ax, neuron_type in zip(g.axes.flat, self.experiment.neuron_types):
            bars = ax.patches
            num_groups = len(group_order)
            num_periods = len(period_order) if not data_opts.get('spontaneous') else 1
            total_bars = num_groups * num_periods

            for i in range(total_bars):
                group = group_order[i % num_groups]
                row_selector = {'Neuron Type': neuron_type, 'Group': group}
                bar = bars[i]
                bar.set_facecolor(self.graph_opts['group_colors'][group])
                if not data_opts.get('spontaneous'):
                    period = period_order[i // num_periods]
                    row_selector['Period'] = period
                    period_hatches = self.graph_opts.get('period_hatches', {'tone': '/', 'pretone': ''})
                    bar.set_hatch(period_hatches[period])
                row = df[(df[list(row_selector)] == pd.Series(row_selector)).all(axis=1)].iloc[0]

                # Plotting the error bars and scatter points
                bar_x = bar.get_x() + bar.get_width() / 2
                ax.errorbar(bar_x, row['Average MRL'], yerr=row['sem'], color='black', capsize=5)
                row_points = [point for point in row['scatter'] if not np.isnan(point)]
                jitter = np.random.rand(len(row_points)) * 0.1 - 0.05
                ax.scatter([bar_x + j for j in jitter], row_points, color='black', s=20)
                row_points = [point for point in row['unit_scatter'] if not np.isnan(point)]
                jitter = np.random.rand(len(row_points)) * 0.1 - 0.05
                ax.scatter([bar_x + j for j in jitter], row_points, color='gray', s=20)

                ax.set_title(neuron_type, fontsize=14)

        # Additional customizations for the non-spontaneous case
        if not data_opts.get('spontaneous'):
            legend_elements = [
                Patch(facecolor='white', hatch=period_hatches[period_type]*3, edgecolor='black', label=period_type.upper())
                for period_type in period_order]
            g.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, .9))

        # Saving the plot
        title = 'average_mrl_during_spontaneous_firing' if data_opts.get('spontaneous') else 'average_mrl'
        self.fig = g.fig
        self.close_plot(title)

    def make_footer(self):
        pass  # todo: make this do something

    def add_significance_markers(self):
        self.stats = Stats(self.experiment, self.context, self.data_opts, lfp=self.lfp)
        # TODO: implement this


class LFPPlotter(Plotter):
    def __init__(self, experiment, lfp=None, behavior=None, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, lfp=lfp, behavior=behavior, graph_opts=graph_opts, plot_type=plot_type)

    def plot_power(self, data_opts, graph_opts):
        self.line_plot_over_periods(data_opts, graph_opts)

    def plot_coherence(self, data_opts, graph_opts):
        self.line_plot_over_periods(data_opts, graph_opts)

    def plot_spectrogram(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts)
        if self.data_opts['level'] == 'group':
            self.plot_spectrogram_by_groups()
        elif self.data_opts['level'] == 'animal':
            self.plot_spectrogram_by_animals()
        elif self.data_opts['level'] == 'period':
            self.plot_spectrogram_periods()
        else:
            raise NotImplementedError

    def plot_spectrogram_by_groups(self):
        axes = self.create_figure_and_axes(self.lfp)
        group_info = {}

        for i, group in enumerate(self.lfp.groups):
            im_list, group_min, group_max = self.make_spectrogram_subplots(group, axes[i])
            group_info[group.identifier] = {'im': im_list, 'min': group_min, 'max': group_max}

        if self.graph_opts.get('equal_color_scales') == 'by_subplot':
            global_min = min([group_dict['min'] for group_dict in group_info.values()])
            global_max = min([group_dict['min'] for group_dict in group_info.values()])
            ims = [im for group_dict in group_info.values() for im in group_dict['im']]
            self.set_clim_and_make_colorbar(axes, ims, global_min, global_max)
        elif self.graph_opts.get('equal_color_scales') == 'within_group':
            for i, group in enumerate(group_info):
                self.set_clim_and_make_colorbar(axes[i, :], *(group_info[group][k] for k in ['im', 'min', 'max']))
        else:
            for im in group_info[group]['im']:
                self.set_clim_and_make_colorbar([im.axes], [im], im.get_array().min(), im.get_array().max())
        self.set_up_stimulus_patches(axes)
        self.close_plot('Spectrogram')

    def plot_spectrogram_by_animals(self):
        for i, group in enumerate(self.lfp.groups):
            axes = self.create_figure_and_axes(group)
            for j, animal in enumerate(group):
                ims, animal_min, animal_max = self.make_spectrogram_subplots(animal, axes[j])
                self.set_clim_and_make_colorbar(axes[j,:], ims, animal_min, animal_max)
            self.set_up_stimulus_patches(axes)
            self.close_plot(f"Spectrogram {group.identifier.capitalize()} Animals")

    def plot_spectrogram_periods(self):
        for group in self.lfp.groups:
            for animal in group:
                nrows = sum([len(self.data_opts['periods'][period_type]) for period_type in self.data_opts['periods']])
                if self.graph_opts.get('extend_periods'):
                    extend = True
                    width, ncols = 30, 1
                else:
                    extend = False
                    width, ncols = 10, 2
                self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width*ncols, 5*nrows), sharex=True)
                i = 0
                for period_type in self.data_opts['periods']:
                    self.selected_period_type = period_type
                    periods=[]

                    for period in animal:
                        data = period.data if not extend else period.extended_data
                        im = self.generate_image(axes[i], period)
                        repeat = period.event_duration if period.event_duration else period.target_period.event_duration
                        #self.set_up_stimulus_patches(np.array([axes[i]]), repeat=repeat)
                        self.set_clim_and_make_colorbar(np.array([axes[i]]), [im], data.min(), data.max())
                        axes[i].set_title(f"{animal.identifier} {period_type.capitalize()} {period.identifier+1}")
                        i += 1
                self.close_plot(f"Spectrogram {animal.identifier} Periods")

    def plot_coherence_over_frequencies(self, data_opts, graph_opts):
        self.line_plot_over_frequencies(data_opts, graph_opts)

    def line_plot_over_frequencies(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts)
        data = {}
        for group in self.lfp.groups:
            period_data = defaultdict(list)
            for period_type in self.data_opts['periods']:
                self.selected_period_type = period_type
                period_data[period_type] = group.data
            data[group.identifier] = period_data

        fig_x_dim = len(list(range(*self.lfp.freq_range)))
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(fig_x_dim, 10), sharex=True)

        for i, (ax, group) in enumerate(zip(axes, self.lfp.groups)):
            for period_type in data[group.identifier]:
                ax.plot(list(range(self.lfp.freq_range[0], self.lfp.freq_range[1] + 1)), 
                        data[group.identifier][period_type], '-o', label=period_type.capitalize(),
                color=self.graph_opts['period_colors'][period_type])

            ax.set_title(smart_title_case(f'{group.identifier} Group'))
            ax.set_ylabel(self.data_type.capitalize())
            if i == 1:
                ax.set_xlabel('Frequency')
            ax.legend(title='Period Type')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust this value as needed

        self.fig = fig
        self.close_plot(self.data_type)


    def line_plot_over_periods(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts)
        data = []
        periods = deepcopy(self.data_opts['periods'])
        for group in self.lfp.groups:
            for period_type in periods:
                self.selected_period_type = period_type
                for period in periods[period_type]:
                    self.update_data_opts([(['periods', period_type], [period])])
                    data.append([group.identifier, period + 1, period_type, group.mean_data, group.sem, group.scatter])
        self.update_data_opts([(['periods'], periods)])  # put this key, val pair back the way it was for further graphs

        df = pd.DataFrame(data, columns=['Group', 'Period', 'Period_Type', 'Power', 'SEM', 'Scatter'])


        # Plotting
        fig_x_dim = 2.5 * len(sorted(self.data_opts['periods'].values(), key=len)[-1])
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(fig_x_dim, 5), sharey=True)

        # Iterate over each group
        for i, (ax, (group, group_df)) in enumerate(zip(axes, df.groupby('Group'))):
            # Iterate over each period type within the group
            for period_type, period_df in group_df.groupby('Period_Type'):
                ax.errorbar(period_df['Period'], period_df['Power'], yerr=period_df['SEM'], label=period_type.capitalize(),
                            fmt='-o', color=self.graph_opts['period_colors'][period_type])  # Adjust fmt for desired line/marker style

            ax.set_title(smart_title_case(f'{group} Group'))
            ax.set_xlabel('Period')
            if i == 0:
                ax.set_ylabel(self.data_type.capitalize())
            ax.legend(title='Trial Type')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust this value as needed

        self.fig = fig
        self.close_plot(self.data_type)

    def plot_correlation(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts)
        data = {}
        for group in self.lfp.groups:
            period_data = defaultdict(list)
            for period_type in self.data_opts['periods']:
                self.selected_period_type = period_type
                period_data[period_type] = group.data
            data[group.identifier] = period_data

        fig_x_dim = self.data_opts['lags']/20
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(fig_x_dim, 10), sharex=True)

        total_samples = self.data_opts.get('lags', 100) * 2 + 1
        mid_index = total_samples // 2
        ms_per_sample = 1000 / self.lfp.sampling_rate  

        # Setting custom x-ticks and labels
        tick_spacing = 40  # Adjust this as needed for granularity
        ticks = list(range(0, total_samples, tick_spacing))
        tick_labels = [(t - mid_index) * ms_per_sample for t in ticks]

        for i, (ax, group) in enumerate(zip(axes, self.lfp.groups)):
            for period_type in data[group.identifier]:
                ax.plot(data[group.identifier][period_type], '-o', label=period_type.capitalize(),
                    color=self.graph_opts['period_colors'][period_type])

            ax.set_title(smart_title_case(f'{group.identifier} Group'))
            ax.set_ylabel(self.data_type.capitalize())
            ax.legend(title='Period Type')
            if i == 1:
                ax.set_xlabel('Lags (ms)')

            # Apply custom ticks
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  

        self.fig = fig
        self.close_plot(self.data_type)

    def plot_max_correlations(self, data_opts, graph_opts):
        self.initialize(data_opts, graph_opts)
        data = {}
        for group in self.lfp.groups:
            period_data = defaultdict(list)
            for period_type in self.data_opts['periods']:
                self.selected_period_type = period_type
                period_data[period_type] = group.get_sum('get_max_histogram', 
                                                        stop_at='correlation_calculator')
            data[group.identifier] = period_data

        fig_x_dim = self.data_opts['lags'] * self.data_opts['bin_size'] * 10
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(fig_x_dim, 10), sharex=True)

        num_lags = self.data_opts.get('lags', self.lfp.sampling_rate/10)  
        bin_size = self.data_opts.get('bin_size', .01) # in seconds
        lags_per_bin = bin_size * self.lfp.sampling_rate
        number_of_bins = int(num_lags * 2 / lags_per_bin)

        mid_index = number_of_bins // 2

        ms_per_bin = 1000 * bin_size
        bar_width = ms_per_bin # width of each bar in milliseconds

        # Adjust tick positions
        ticks = [t * ms_per_bin for t in range(0, number_of_bins + 1)]  # positions for the bin edges
        tick_labels = [(t - mid_index * ms_per_bin) for t in ticks]  # label each bin edge

        for i, (ax, group) in enumerate(zip(axes, self.lfp.groups)):
            for period_type in data[group.identifier]:
                # Calculate midpoints
                mid_points = [tick + bar_width / 2 for tick in ticks[:-1]]

                # Plot the line graph with midpoints
                ax.plot(mid_points, data[group.identifier][period_type], marker='o', linestyle='-', 
                        label=period_type.capitalize(), color=self.graph_opts['period_colors'][period_type], 
                        markerfacecolor='none', markeredgecolor=self.graph_opts['period_colors'][period_type])

            ax.set_title(smart_title_case(f'{group.identifier} Group'))
            ax.set_ylabel('Count of Max Correlations')
            ax.legend(title='Period Type')
            if i == 1:
                ax.set_xlabel('Lags (ms)')

            # Apply custom ticks
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)

            # Add a vertical dotted line at mid_index
            ax.axvline(mid_index * ms_per_bin, color='grey', linestyle='--', linewidth=2)  

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)

        self.fig = fig
        self.close_plot('max_correlation')




    def create_figure_and_axes(self, parent):
        ncols = len(self.data_opts['periods'])
        nrows = len(parent.children)
        #multipliers = (8, 3) if self.data_opts['level'] == 'animal' else (3, 5)
        self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows), sharex=True, sharey=True)
        if ncols == 1:
            axes = axes[:, np.newaxis]
        return axes

    def make_spectrogram_subplots(self, data_source, axes):
        data_source_min = float('inf')
        data_source_max = float('-inf')
        im_list = []
        for i, period_type in enumerate(self.data_opts['periods']):
            self.selected_period_type = period_type
            data = data_source.data
            im = self.generate_image(axes[i], data)
            period_str = period_type.capitalize() if len(self.data_opts['periods']) > 1 else ''
            data_id = data_source.identifier
            if self.data_opts['level'] == 'group':
                data_id = data_id.capitalize()
            axes[i].set_title(f"{data_id} {period_str}")
            im_list.append(im)
            data_source_min = min(data_source_min, data.min())
            data_source_max = max(data_source_max, data.max())
        return im_list, data_source_min, data_source_max

    def generate_image(self, ax, data):
        pre_stim, post_stim = (self.data_opts['events'][self.selected_period_type][opt]
                               for opt in ('pre_stim', 'post_stim'))
        im = ax.imshow(data, cmap='jet', interpolation='nearest', aspect='auto',
                            extent=[-pre_stim, post_stim, *self.current_frequency_band], origin='lower')
        return im

    def set_clim_and_make_colorbar(self, axes, im_list, minimum, maximum):
        [im.set_clim(minimum, maximum) for im in im_list]
        cbar = self.fig.colorbar(im_list[0], ax=axes.ravel().tolist(), shrink=0.7)

    def set_up_stimulus_patches(self, axes, repeat=None):
        if repeat:
            for start in np.arange(0, self.data_opts['events'][self.selected_period_type]['post_stim'], repeat):
                self.make_stimulus_patch(axes, start)
        else:
            self.make_stimulus_patch(axes, 0)

    def make_stimulus_patch(self, axes, start):
        [ax.fill_betweenx(self.lfp.freq_range, start, self.experiment.stimulus_duration, color='k', alpha=0.2)
         for ax in axes.ravel()]

    def set_dir_and_filename(self, basename):
        if 'coherence' not in self.data_type and 'correlation' not in self.data_type:
            brain_region = self.current_brain_region
        else:
            brain_region = self.data_opts.get('region_set')
        title_string = f"{'_'.join([brain_region, str(self.current_frequency_band), basename])}"
        
        self.title = smart_title_case(title_string.replace('_', ' '))
        self.fig.suptitle(self.title, weight='bold', y=.98, fontsize=14)
        self.fname = f"{title_string}.png"
        self.dir_tags = [self.data_type]

    def make_footer(self):
        pass



