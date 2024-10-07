from multiprocessing.dummy.connection import families
import os
import math
import numpy as np
import seaborn as sns
import pandas as pd
from copy import deepcopy
import json
from collections import defaultdict, deque

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch
import matplotlib.ticker as ticker



from math_functions import get_positive_frequencies, get_spectrum_fenceposts, nearest_power_of_10
from plotting_helpers import smart_title_case, PlottingMixin
from utils import to_serializable
from base_data import Base
from stats import Stats
from phy_interface import PhyInterface
from plotter_base import PlotterBase
from partition import Section, Segment, Subset
from subplotter import Figurer

class Plotter(PlotterBase, PlottingMixin):
    """Makes plots, where a plot is a display of particular kind of data.  For displays of multiple 
    plots of multiple kinds of data, see the figure module."""

    def __init__(self, experiment, graph_opts=None):
        self.experiment = experiment
        self.graph_opts = graph_opts
        
    def initialize(self, calc_opts, graph_opts):
        """Both initializes values on self and sets values for the context."""
        self.calc_opts = calc_opts  
        self.graph_opts = graph_opts
        self.experiment.initialize_data()

    def plot(self, calc_opts, graph_opts, parent_figure=None, index=None):
        self.initialize(calc_opts, graph_opts)
        plot_spec = graph_opts['plot_spec']
        self.parent_figure = parent_figure
        self.process_plot_spec(plot_spec, index=index)
        if not self.parent_figure:
            self.close_plot()

    def process_plot_spec(self, plot_spec, index=None):

        processor_classes = {
            'section': Section,
            'segment': Segment,
            'subset': Subset
        }

        self.active_spec_type, self.active_spec = list(plot_spec.items())[0]
        processor = processor_classes[self.active_spec_type](self, index=index)
        processor.start()

    def make_fig(self):
        self.fig = Figurer().make_fig()
        self.active_fig = self.fig
            
    def close_plot(self, basename='', fig=None, do_title=True):
        
        if not fig:
            fig = self.active_fig  
        fig.delaxes(fig.axes[0])
        self.set_dir_and_filename(fig, basename, do_title=do_title)
        self.save_and_close_fig(fig, basename)

    def set_dir_and_filename(self, fig, basename, do_title=True):
        tags = [basename] if basename else [self.calc_type]
        self.title = smart_title_case(' '.join([tag.replace('_', ' ') for tag in tags]))
        if do_title:
            bbox = fig.axes[0].get_position()
            fig.suptitle(self.title, fontsize=16, y=bbox.ymax + 0.1)
        self.fname = f"{'_'.join(tags)}.png"

    def save_and_close_fig(self, fig, basename):
        dirs = [self.graph_opts['graph_dir'], self.calc_type]
        path = os.path.join(*dirs)
        os.makedirs(path, exist_ok=True)
        fname = basename if basename else self.calc_type
        fig.savefig(os.path.join(path, fname), bbox_inches='tight')
        opts_filename = fname.replace('png', 'txt')

        with open(os.path.join(path, opts_filename), 'w') as file:
            json.dump(to_serializable(self.calc_opts), file)
        plt.close(fig)
    

class HistogramPlotter(Plotter):
    
    def plot_hist(self, x, y, width, ax):
        ax.bar(x, y, width=width) 


class CategoryPlotter(Plotter):
    
    def process_calc(self, info):

        self.bar_width = self.graph_opts.get('bar_width', .2)
        
        for row in info:
            aesthetic_args = self.get_aesthetics(row)
            position = self.find_position(row, self.active_spec)
            bar = self.active_ax.bar(position, getattr(row['data_source'], row['attr']), 
                                     self.bar_width, **aesthetic_args)
    
    # {'period_type': {'aesthetic': {'light_on': {'color': 'green'}}}}
    def get_aesthetics(self, row):

        aesthetics = {}
        for label in row.keys():
            for member, attrs_vals in self.active_spec.get(label, {}).get('aesthetic', {}).items():
                if row[label] == member or member == '___':
                    aesthetics.update(attrs_vals)
        return aesthetics

    def find_position(self, observation, division_types=None, start_position=0):
        segment_info = self.active_spec
        if division_types is None:
            division_types = deque(sorted(
                [k for k in segment_info.keys() if 'grouping' in segment_info[k]], 
                key=lambda x: segment_info[x]['grouping']))
        
        current_division_type = division_types.popleft()
        
        mult_factor = 1
        for dt in division_types:
            bars_width = len(segment_info[dt]['members']) * self.bar_width
            spacing = 2 * segment_info[dt].get('spacing', self.bar_width/.5)
            mult_factor *= bars_width + spacing
        
        value = observation[current_division_type]
        index = segment_info[current_division_type]['members'].index(value)
        
        position = index * mult_factor + start_position
        
        if len(division_types) == 0:
            return position
        else:
            return self.find_position(
                observation, segment_info, division_types=division_types, start_position=position)

    def compute_outer_label_positions(inner_positions, outer_labels, num_inner_per_outer):
        """
        Compute positions and labels for the outer grouping level.

        Parameters:
        - inner_positions (list of float): Positions of the inner level ticks.
        - outer_labels (list of str): Labels for the outer grouping level.
        - num_inner_per_outer (int or list of int): Number of inner positions per outer group.
            If an integer, it's assumed the same for all outer groups.
            If a list, it should have the same length as outer_labels.

        Returns:
        - outer_positions (list of float): Computed positions for outer group labels.
        - outer_labels (list of str): Labels for the outer grouping level.
        """
        outer_positions = []
        idx = 0  # Index to track position in inner_positions

        if isinstance(num_inner_per_outer, int):
            num_inner_per_outer = [num_inner_per_outer] * len(outer_labels)
        elif len(num_inner_per_outer) != len(outer_labels):
            raise ValueError("Length of num_inner_per_outer must match length of outer_labels.")

        for count in num_inner_per_outer:
            # Get the positions corresponding to the current outer group
            group_positions = inner_positions[idx:idx + count]
            # Compute the midpoint of these positions
            group_midpoint = sum(group_positions) / len(group_positions)
            outer_positions.append(group_midpoint)
            idx += count  # Move to the next set of positions

        return outer_positions


class CategoricalScatterPlotter(CategoryPlotter):

    # def plot(self, graph_opts, parent_figure=None):
    #     calc_opts, plot_spec = graph_opts
    #     self.initialize(calc_opts)
    #     # TODO: you might sometimes want different segments for different axes in a section
    #     plot_spec = {
    #         'section': 
    #         {   'aspect': 1,
    #             'unit': {
    #                 'members': self.experiment.all_units, 'dim': 1, 'attr': 'scatter'
    #                 },
    #             'segment': {
    #                 'period_type': {
    #                     'members': list(self.calc_opts['periods'].keys()),
    #                     'grouping': 0,
    #                     'spacing': .075,
    #                     'aesthetic': {
    #                         '___': {'color': 'black'},
    #                         'prelight': {'background_color': ('white', .2)},
    #                         'light': {'background_color': ('green', .2)},
    #                         'tone': {'background_color': ('green', .2)}
    #                         }}}}}
    #     self.process_plot_spec(plot_spec, parent_figure=parent_figure)
    #     if not self.parent_figure:
    #         self.close_plot()

    #      # section: {'data_source': {'members': ['1', '2', '3'], 'dim': 0}
    #      #           'segment': {'period_type': {'members': ['prelight', 'light', 'tone'], 'grouping':0},
    #      #                       'aesthetic': 'background_color'}}'

    def process_calc(self, info):
        ax = self.active_ax
        self.cat_width = self.active_spec.get('cat_width', .8)
        positions = []
        for row in info:
            if self.active_spec_type == 'segment':
                position = self.find_position(row)
                positions.append(position)
            else:
                # do something else
                pass
            val = row[row['attr']]
            jitter = np.random.rand(len(val)) * self.cat_width/2 - self.cat_width/4
            aesthetic_args = self.get_aesthetics(row)
            if 'background_color' in aesthetic_args:
                background_color, alpha = aesthetic_args.pop('background_color')
                ax.axvspan(
                    position - self.cat_width/2, position + self.cat_width/2, 
                    facecolor=background_color, alpha=alpha)
            ax.scatter([position + j for j in jitter], val, **aesthetic_args)
            for i, dim in enumerate(['x', 'y']):
                if ax.index[i] == 0:
                    getattr(ax, f"set_{dim}label")(
                        self.get_labels()[self.calc_type][i])
        
        tick_label_bbox = ax.get_xticklabels()[0].get_window_extent()
        bbox_in_ax = tick_label_bbox.transformed(ax.transAxes.inverted())
        tick_label_ymin = bbox_in_ax.ymin

        levels = reversed(
            sorted(self.active_spec.keys(), key=lambda k: self.active_spec[k]['grouping']))
        level_adjustment = 0
        for i, level in enumerate(levels):
            labels = self.active_spec[level]['members']
            if 'legend' not in self.active_spec[level]:
                level_adjustment -= .05
            if i == 0:
                ax.set_xticks(positions)
                ax.set_xticklabels(labels)
                inner_positions = ax.get_xticks
            else:
                num_inner_per_outer = int(len(labels)/self.active_spec[levels[i-1]]['members'])
                current_positions = self.compute_outer_label_positions(
                    inner_positions, labels, num_inner_per_outer)
                for lab, pos in zip(labels, current_positions):
                    ax.text(pos, tick_label_ymin - level_adjustment, lab)
                inner_positions = current_positions

class LinePlotter(Plotter):
    pass

class WaveformPlotter(LinePlotter):
  

    def process_calc(self, info):
        ax = self.active_ax
        for row in info:
            val = row[row['attr']]
            ax.plot(np.arange(len(val)), val)







class PeriStimulusHistogramPlotter(HistogramPlotter):

    def plot(self, calc_opts, graph_opts):
        self.initialize(calc_opts)
        if not graph_opts.get('plot_spec'):
            plot_spec = (
                'section', 
                {'neuron_type': {'dim': 0, 'members': self.experiment.neuron_types},
                 'data_source': {'dim': 1, 'members': self.experiment.all_groups},
                 'period_type': {'members': list(self.calc_opts['periods'].keys())}}
                 )
        self.process_plot_spec(plot_spec)
        self.close_plot()

    def process_calc(self, data_frame):
        #self.do_stuff_like_set_title(data_frame)
        num_bins = round((self.pre_stim + self.post_stim)/self.calc_opts['bin_size'])
        x = np.linspace(self.pre_stim, self.post_stim, num_bins)
        y = data_frame['calc'].iloc[0]
        self.plot_hist(x, y, self.calc_opts['bin_size'], self.active_ax)
    
    def add_stimulus_patch(self):
        pass




    


   



# class HistogramPlotter(Plotter, PlottingMixin):
#     """Makes plots where the x-axis is time around the stimulus, and y can be a variety of types of data."""

#     def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
#         super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)
#         self.multiplier = 5 # figure out what affects this

#     def process_calc(self, source, calcs, ax, **_):
#         _, ylabel = self.get_labels()[self.calc_opts['calc_type']]

#         if f"{source.name}_colors" in self.graph_opts:
#             color = self.graph_opts[f"{source.name}_colors"][source.identifier]
#         else:
#             color = 'black'

#         x = np.linspace(-self.pre_stim, self.post_stim, len(source.calc))
#         ax.bar(x, calcs[0], width=self.calc_opts['bin_size'], color=color)
    
#         ax.set_facecolor('white')
#         ax.patch.set_alpha(0.2)

#         try:
#             x_step, x_start = self.get_ticks() 
#         except AttributeError:
#             "executing class must have method get_ticks"

#         ax.set_xlim(x[0], x[-1])  
#         ax.set_xticks(np.arange(x_start, x[-1], step=x_step))
#         ax.tick_params(axis='both', which='major', labelsize=10 * self.multiplier, 
#                             length=5 * self.multiplier, width=2 * self.multiplier)
      

# class PeriStimulusHistogramPlotter(HistogramPlotter):

#     def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
#         super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)
    
#     def plot(self, calc_opts, graph_opts):
#         self.initialize(calc_opts, graph_opts)
#         self.selected_period_type = 'tone'

#         if self.calc_opts.get('level') == 'group':
#             partitions = {
#                 'neuron_type': {'dim': 0, 'members': self.experiment.neuron_types},
#                 'data_source': {'dim': 1, 'members': self.experiment.all_groups}}
#             self.create_figures('group', partitions=partitions, dim_maxes=(2, 2)) 

#     def get_ticks(self):
    
#         x_step = self.graph_opts.get(
#             'tick_step', nearest_power_of_10((self.pre_stim + self.post_stim)/10))
        
#         x_start = self.graph_opts.get(
#             'tick_start', x_step if self.pre_stim % x_step == 0 else self.pre_stim % x_step
#         )
#         return x_step, x_start 

class PeriStimulusSubplotter(Plotter, PlottingMixin):
    """Constructs a subplot of a PeriStimulusPlot."""

    def __init__(self, plotter, data_source, graph_opts, ax, parent_type='standalone', multiplier=1):
        self.plotter = plotter
        self.data_source = data_source
        self.g_opts = graph_opts
        self.ax = ax
        self.x = None
        self.y = data_source.calc
        self.parent_type = parent_type
        self.multiplier = multiplier
        self.plot_type = 'subplot'

    def set_limits_and_ticks(self, x_min, x_max, x_tick_min, x_step, y_min=None, y_max=None):
        self.ax.set_xlim(x_min, x_max)  # TODO: it would be nice if autocorrelation and cross-correlation took the same units of tick step
        xticks = np.arange(x_tick_min, x_max, step=x_step)
        self.ax.set_xticks(xticks)
        self.ax.tick_params(axis='both', which='major', labelsize=10 * self.multiplier, length=5 * self.multiplier,
                            width=2 * self.multiplier)
        if self.calc_type in ['spontaneous_firing', 'cross_correlations']:
            self.ax.set_xticklabels(xticks * self.calc_opts['bin_size'])

        if y_min is not None and y_max is not None:
            self.ax.set_ylim(y_min, y_max)

    def plot_raster(self):
        pre, post = [self.calc_opts['events'][self.selected_period_type][opt] for opt in ['pre_stim', 'post_stim']]
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

        # if self.parent_type == 'standalone':
        #     self.set_labels(x_and_y_labels=(x_label, y_label))

    def plot_psth(self):
        pre, post = [self.calc_opts['events'][self.selected_period_type][opt] for opt in ['pre_stim', 'post_stim']]
        _, ylabel = self.get_labels()[self.calc_opts['calc_type']]
        self.plot_bar(width=self.calc_opts['bin_size'], x_min=-pre, x_max=post, num=len(self.y), x_tick_min=0,
                      x_step=self.g_opts['tick_step'], y_label=ylabel)

    def plot_proportion(self):
        self.plot_psth()

    def plot_autocorr(self):
        opts = self.calc_opts
        self.plot_bar(width=opts['bin_size'] * .95, x_min=opts['bin_size'], x_max=opts['max_lag'] * opts['bin_size'],
                      num=opts['max_lag'], x_tick_min=0, x_step=self.g_opts['tick_step'], y_min=min(self.y),
                      y_max=max(self.y))

    def plot_spectrum(self):
        freq_range, max_lag, bin_size = (self.calc_opts.get(opt) for opt in ['freq_range', 'max_lag', 'bin_size'])
        first, last = get_spectrum_fenceposts(freq_range, max_lag, bin_size)
        self.x = get_positive_frequencies(max_lag, bin_size)[first:last]
        self.ax.plot(self.x, self.y)

    def plot_spontaneous_firing(self):  # TODO: don't hardcode period
        opts = self.calc_opts
        self.plot_bar(width=opts['bin_size'], x_min=0, x_max=int(120 / self.calc_opts['bin_size']), num=len(self.y),
                      x_tick_min=0, x_step=self.g_opts['tick_step'], y_label='Firing Rate (Spikes per Second')

    def plot_cross_correlations(self):
        opts = self.calc_opts
        boundary = round(opts['max_lag'] / opts['bin_size'])
        tick_step = self.plotter.graph_opts['tick_step']
        if self.data_source.name == 'group' and 'group_colors' in self.g_opts:
            color = self.g_opts['group_colors'][self.data_source.identifier]
        else:
            color = 'black'
        self.ax.bar(np.linspace(-boundary, boundary, 2 * boundary + 1), self.y)
        self.ax.bar(self.x, self.y, color=color)
        tick_positions = np.arange(-boundary, boundary + 1, tick_step)
        tick_labels = np.arange(-opts['max_lag'], opts['max_lag'] + opts['bin_size'],
                                tick_step * self.calc_opts['bin_size'])
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels([f'{label:.2f}' for label in tick_labels])

    def plot_correlogram(self):
        self.plot_cross_correlations()

    def plot_data(self):
        getattr(self, f"plot_{self.calc_type}")()

    def add_sem(self):
        opts = self.calc_opts
        if opts['calc_type'] in ['autocorr', 'spectrum'] and opts['ac_key'] == self.data_source.name + '_by_rates':
            print("It doesn't make sense to add standard error to a graph of autocorr over rates.  Skipping.")
            return
        sem = self.data_source.sem_envelope
        self.ax.fill_between(self.x, self.y - sem, self.y + sem, color='blue', alpha=0.2)


class PiePlotter(Plotter):
    """Constructs a pie chart of up- or down-regulation of individual neurons"""

    def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)

    def unit_upregulation_pie_chart(self, calc_opts, graph_opts):
        self.initialize(calc_opts, graph_opts, neuron_type='all')
        labels = ['Up', 'Down', 'No Change']
        colors = ['yellow', 'blue', 'green']

        for nt in self.experiment.neuron_types + [None]:
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


class MRLPlotter(Plotter, PlottingMixin):
    def __init__(self, experiment, lfp=None, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, lfp=lfp, graph_opts=graph_opts, plot_type=plot_type)

    def mrl_rose_plot(self, calc_opts, graph_opts):
        self.make_plot(calc_opts, graph_opts, 'rose_plot', self.make_rose_plot, 'polar')

    def mrl_heat_map(self, calc_opts, graph_opts):
        self.make_plot(calc_opts, graph_opts, 'heat_map', self.make_heat_map, None)

    def make_plot(self, calc_opts, graph_opts, basename, plot_func, projection):
        self.initialize(calc_opts, graph_opts, neuron_type='all')
        self.fig = plt.figure(figsize=(15, 15))

        ncols = 2 if self.calc_opts.get('adjustment') == 'relative' else 4

        self.axs = [
            [self.fig.add_subplot(2, ncols, 1 + ncols * row + col, projection=projection) for col in range(ncols)]
            for row in range(2)
        ]

        for i, neuron_type in enumerate(self.experiment.neuron_types):
            self.selected_neuron_type = neuron_type
            for j, group in enumerate(self.lfp.groups): # TODO: think about how this should work now with multiple periods
                if self.calc_opts.get('adjustment') == 'relative':
                    self.selected_period_type = 'tone'
                    plot_func(group, self.axs[i][j], title=f"{group.identifier.capitalize()} {neuron_type}")
                else:
                    for k, period_type in enumerate(self.experiment.period_types):
                        self.selected_period_type = period_type
                        plot_func(group, self.axs[i][j * 2 + k],
                                  title=f"{group.identifier.capitalize()} {neuron_type} {period_type}")
        self.selected_neuron_type = None
        self.close_plot(basename)

    def plot_phase_phase_over_frequencies(self, calc_opts, graph_opts):
        self.line_plot_over_frequencies(calc_opts, graph_opts)

    def iterate_through_groups_and_periods(self, calc_opts, graph_opts, method):
        self.initialize(calc_opts, graph_opts)
        period_groups = self.calc_opts.get('period_groups')
        period_groups = period_groups if period_groups else [None]
        original_periods = deepcopy(self.calc_opts['periods'])
        nrows = len(self.lfp.groups) * len(period_groups)
        ncols = len(self.calc_opts['periods'])
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*nrows, 5*ncols), sharex=True)
        axes = np.array(axes).reshape(2, -1)  # This makes sure axes is always 2D even if ncols or nrows is 1
        period_type = None
        for i, group in enumerate(self.lfp.groups):
            for j, pg in enumerate(period_groups):
                if len(period_groups) > 1:
                    new_periods = {p: original_periods[p][slice(*pg)] for p in original_periods}
                    self.update_calc_opts([(['periods'], new_periods)])
                for k, period_type in enumerate(original_periods):
                    ax = axes[i*len(self.lfp.groups) + j, k]
                    self.selected_period_type = period_type
                    method(group, ax, period_type=period_type, pg=j)
                    ax_title = ''
                    if len(self.lfp.groups) > 1:
                        ax_title += smart_title_case(f'{group.identifier} Group')
                    if len(period_groups) > 1:
                        ax_title += f' Periods {period_groups[j][0]+1}-{period_groups[j][1]}'
                    ax.set_title(ax_title)
                    ax.set_ylabel(self.calc_type.capitalize())
                    if i*len(self.lfp.groups) + j == len(self.lfp.groups) * len(period_groups) -1:  # Set xlabel on the last row
                        ax.set_xlabel('Time')
                    ax.legend(title='Period Type')

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  # Adjust this value as needed

        self.fig = fig
        self.close_plot(self.calc_type)

    def make_phase_phase_rose_plot(self, calc_opts, graph_opts):
        method = getattr(self, 'make_rose_plot')
        self.iterate_through_groups_and_periods(calc_opts, graph_opts, method)

    def make_phase_phase_trace_plot(self, calc_opts, graph_opts):
        method = getattr(self, 'make_trace_plot')
        self.iterate_through_groups_and_periods(calc_opts, graph_opts, method)

    def make_trace_plot(self, group, ax, period_type='', pg=None, title=''):
        data = group.calc
        # Plot the initial data with normal settings
        ax.plot(range(len(data[0])), data[0], '-o', label=period_type.capitalize(),
                color=self.get_color(group=group.identifier, period_type=period_type, period_group=pg))

        # Plot the event data with slightly thicker lines for better visibility
        for event in group.phase_relationship_events:
            data = event.calc
            ax.plot(range(len(data[0])), data[0], '-', label=period_type.capitalize(),
                    color=self.get_color(group=group.identifier, period_type=period_type, period_group=pg),
                    linewidth=0.1,  # Adjusted to a slightly thicker value for better rendering
                    alpha=0.2)
        event_info = self.calc_opts.get('events')
        if event_info:
            pre_stim = event_info[period_type]['pre_stim']
        else:
            pre_stim = 0

        total_samples = len(data)
        tick_spacing = 40  
        ticks = list(range(0, total_samples, tick_spacing))
        tick_labels = [t - pre_stim * 1000 for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)


    def make_rose_plot(self, group, ax, period_type='', pg=None, title="", **kwargs):
        n_bins = 36
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        width = 2 * np.pi / n_bins
        ax.bar(bin_edges[:-1], group.calc, width=width, align='edge', 
               color=self.get_color(group=group, period_type=period_type, period_group=pg), alpha=1)
        current_ticks = ax.get_yticks()
        ax.set_yticks(current_ticks[::2])  # every second tick
        ax.set_title(title)

    def set_dir_and_filename(self, basename):
        tags = [basename, str(self.lfp.current_frequency_band)]
        if self.current_region_set:
            tags.extend(self.current_region_set.split('_'))
        else:
            tags.append(self.current_brain_region)
        self.dir_tags = [self.calc_type]
        self.title = smart_title_case(' '.join([tag.replace('_', ' ') for tag in tags]))
        self.fig.suptitle(self.title, weight='bold', y=.95, fontsize=20)
        self.fname = f"{basename}_{'_'.join(tags)}.png"

    def make_heat_map(self, group, ax, title=""):
        data = group.data_by_period  # Todo put this back/figure out what it should be now
        im = ax.imshow(data.T, cmap='jet', interpolation='nearest', aspect='auto',
                       extent=[0.5, 5.5, self.current_frequency_band[0], self.current_frequency_band[1]],
                       origin='lower')
        cbar = ax.figure.colorbar(im, ax=ax, label='MRL')
        ax.set_title(title)

    def mrl_bar_plot(self, calc_opts, graph_opts):
        self.initialize(calc_opts, graph_opts, neuron_type='all')

        data = []
        if calc_opts.get('spontaneous'):
            for neuron_type in self.experiment.neuron_types:
                self.selected_neuron_type = neuron_type
                for group in self.lfp.groups:
                    data.append([neuron_type, group.identifier, group.calc, group.sem, group.scatter,
                                 group.grandchildren_scatter])
            df = pd.DataFrame(data, columns=['Neuron Type', 'Group', 'Average MRL', 'sem', 'scatter', 'unit_scatter'])
        else: 
            for neuron_type in self.experiment.neuron_types:
                self.selected_neuron_type = neuron_type
                for group in self.lfp.groups:
                    for period_type in self.experiment.period_types:
                        self.selected_period_type = period_type
                        data.append([neuron_type, group.identifier, period_type, group.calc, group.sem, group.scatter,
                                     group.grandchildren_scatter])
            df = pd.DataFrame(data, columns=['Neuron Type', 'Group', 'Period', 'Average MRL', 'sem', 'scatter',
                                             'unit_scatter'])

        group_order = df['Group'].unique()

        # Plot creation
        if calc_opts.get('spontaneous'):
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
            num_periods = len(period_order) if not calc_opts.get('spontaneous') else 1
            total_bars = num_groups * num_periods

            for i in range(total_bars):
                group = group_order[i % num_groups]
                row_selector = {'Neuron Type': neuron_type, 'Group': group}
                bar = bars[i]
                bar.set_facecolor(self.graph_opts['group_colors'][group])
                if not calc_opts.get('spontaneous'):
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
        if not calc_opts.get('spontaneous'):
            legend_elements = [
                Patch(facecolor='white', hatch=period_hatches[period_type]*3, edgecolor='black', label=period_type.upper())
                for period_type in period_order]
            g.fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, .9))

        # Saving the plot
        title = 'average_mrl_during_spontaneous_firing' if calc_opts.get('spontaneous') else 'average_mrl'
        self.fig = g.fig
        self.close_plot(title)

    def add_significance_markers(self):
        self.stats = Stats(self.experiment, self.context, self.calc_opts, lfp=self.lfp)
        # TODO: implement this


class LFPPlotter(Plotter):
    def __init__(self, experiment, graph_opts=None, plot_type='standalone'):
        super().__init__(experiment, graph_opts=graph_opts, plot_type=plot_type)

    def plot_power(self, calc_opts, graph_opts):
        self.line_plot_over_periods(calc_opts, graph_opts)

    def plot_coherence(self, calc_opts, graph_opts):
        self.line_plot_over_periods(calc_opts, graph_opts)

    def plot_spectrogram(self, calc_opts, graph_opts):
        self.initialize(calc_opts, graph_opts)
        if self.calc_opts['level'] == 'group':
            self.plot_spectrograms(data_source='group')
        elif self.calc_opts['level'] == 'animal':
            self.plot_spectrograms(data_source='animals')
        elif self.calc_opts['level'] == 'period':
            self.plot_spectrogram_periods()
        else:
            raise NotImplementedError

    def plot_spectrograms(self, data_source='group'):

        self.calc_opts['cache'] = -1 # caching doesn't speed this up 

        def get_ims(data_source, row, col, pt, **_):
            nonlocal mins_and_maxes
            ax = axes[row, col]
            ax.set_title(f"{data_source.identifier.capitalize()} {pt.capitalize()}")
            data = data_source.calc
            im = self.generate_image(axes[row, col], data)
            self.evaluate_mins_and_maxes(data_source, data, mins_and_maxes)
            return im

        ax_key = {'cols': 'data_source'}
        pts = self.calc_opts['periods'].keys()

        for ds_set in self.get_sets_of_data_sources(data_source):

            axes = self.create_figure_and_axes(ax_key, ds_set, pts)
            mins_and_maxes = self.initialize_mins_and_maxes(ds_set)
            ims = self.iterate_through_partitions(ds_set, process_calc=get_ims)

            if self.graph_opts.get('equal_color_scales') == 'by_subplot':
                self.set_clim_and_make_colorbar(axes, ims, *mins_and_maxes['global'].values())
            elif self.graph_opts.get('equal_color_scales') == 'within_data_source':
                for i, group in enumerate(ds_set):
                    self.set_clim_and_make_colorbar(axes[i, :], ims, 
                                                    *mins_and_maxes[group.identifier].values())
            else:
                for im in ims:
                    self.set_clim_and_make_colorbar([im.axes], [im], im.get_array().min(), 
                                                    im.get_array().max())
            self.set_up_stimulus_patches(axes)
            self.close_plot('Spectrogram')

    def plot_coherence_over_frequencies(self, calc_opts, graph_opts):
        self.line_plot_over_frequencies(calc_opts, graph_opts)

    def line_plot_over_frequencies(self, calc_opts, graph_opts):

        def frequency_plot(ax, data_source, pt, **_):
            x_vals = list(range(self.freq_range[0], self.freq_range[1] + 1))
            y_vals = data_source.get_mean(0)
            ax.plot(x_vals, y_vals, '-o', label=pt.capitalize(), 
                    color=self.graph_opts['period_colors'][pt])
            ax.set_xlabel('Frequency (Hz)')

        self.line_plot(calc_opts, graph_opts, frequency_plot, 
                       row_multiple=len(list(self.freq_range)))
      
    def plot_spectrogram_periods(self):
        for group in self.lfp.groups:
            for animal in group:
                nrows = sum([len(self.calc_opts['periods'][period_type]) for period_type in self.calc_opts['periods']])
                if self.graph_opts.get('extend_periods'):
                    extend = True
                    width, ncols = 30, 1
                else:
                    extend = False
                    width, ncols = 10, 2
                self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(width*ncols, 5*nrows), sharex=True)
                i = 0
                for period_type in self.calc_opts['periods']:
                    self.selected_period_type = period_type
                    periods=[]

                    for period in animal:
                        data = period.calc if not extend else period.extended_data
                        im = self.generate_image(axes[i], period)
                        repeat = period.event_duration if period.event_duration else period.target_period.event_duration
                        #self.set_up_stimulus_patches(np.array([axes[i]]), repeat=repeat)
                        self.set_clim_and_make_colorbar(np.array([axes[i]]), [im], data.min(), data.max())
                        axes[i].set_title(f"{animal.identifier} {period_type.capitalize()} {period.identifier+1}")
                        i += 1
                self.close_plot(f"Spectrogram {animal.identifier} Periods")

    def plot_granger(self, calc_opts, graph_opts):
        self.initialize(calc_opts, graph_opts)

        data = {}
        for group in self.lfp.groups:
            period_data = defaultdict(dict)
            for period_type in self.calc_opts['periods']:
                self.selected_period_type = period_type
                period_data[period_type]['data'] = group.data
                period_data[period_type]['sem'] = group.get_sem()
            data[group.identifier] = period_data

        nrows = len(self.lfp.groups)
        ncols = len(self.calc_opts['periods'])
        fig_x_dim = len(list(range(*self.lfp.freq_range))) 
        self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_x_dim, 5), sharex=True)

        for i, (group, row) in enumerate(zip(self.lfp.groups, axes)):
            for j, (period_type, ax) in enumerate(zip(self.calc_opts['periods'], row)):
                for k, (key, val) in enumerate(data[group.identifier][period_type]['data'].items()):
                    # Determine line style based on 'key'
                    line_style = '-' if key == 'forward' else '--'

                    sem_val = data[group.identifier][period_type]['sem'][key]
                    line_color = self.graph_opts['period_colors'][period_type]

                    # Plot the standard error envelope
                    ax.fill_between(
                        list(range(self.lfp.freq_range[0], self.lfp.freq_range[1] + 1)),
                        np.real(val) - sem_val,
                        np.real(val) + sem_val,
                        color=line_color,
                        alpha=0.3  # Control the transparency of the envelope
                    )

                    # Plot the main line using the determined line style
                    ax.plot(
                        list(range(self.lfp.freq_range[0], self.lfp.freq_range[1] + 1)),
                        np.real(val),
                        line_style + 'o',  # Combine line style with point marker
                        color=line_color,
                        label=smart_title_case(f"{self.current_region_set.split('_')[k]} Leads")
                    )

                    # Set title, labels, and other plot properties
                    ax.set_title(smart_title_case(f'{group.identifier} {period_type}'))
                    if i == 1:
                        ax.set_xlabel('Frequency')
                    if j == 0:
                        ax.set_ylabel(smart_title_case(self.calc_type))

                    ax.legend(title='Direction')
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        self.close_plot(self.calc_type)

    def plot_correlation(self, calc_opts, graph_opts):
        self.initialize(calc_opts, graph_opts)
        data = {}
        for group in self.lfp.groups:
            period_data = defaultdict(list)
            for period_type in self.calc_opts['periods']:
                self.selected_period_type = period_type
                period_data[period_type] = group.calc
            data[group.identifier] = period_data

        fig_x_dim = self.calc_opts['lags']/20
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(fig_x_dim, 10), sharex=True)

        total_samples = self.calc_opts.get('lags', 100) * 2 + 1
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
            ax.set_ylabel(self.calc_type.capitalize())
            ax.legend(title='Period Type')
            if i == 1:
                ax.set_xlabel('Lags (ms)')

            # Apply custom ticks
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)  

        self.fig = fig
        self.close_plot(self.calc_type)

    def plot_max_correlations(self, calc_opts, graph_opts):
        self.initialize(calc_opts, graph_opts)
        data = {}
        for group in self.lfp.groups:
            period_data = defaultdict(list)
            for period_type in self.calc_opts['periods']:
                self.selected_period_type = period_type
                period_data[period_type] = group.get_sum('get_max_histogram', 
                                                        stop_at='correlation_calculator')
            data[group.identifier] = period_data

        fig_x_dim = self.calc_opts['lags'] * self.calc_opts['bin_size'] * 10
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(fig_x_dim, 10), sharex=True)

        num_lags = self.calc_opts.get('lags', self.lfp.sampling_rate/10)  
        bin_size = self.calc_opts.get('bin_size', .01) # in seconds
        lags_per_bin = bin_size * self.lfp.sampling_rate
        number_of_bins = round(num_lags * 2 / lags_per_bin)

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

    def generate_image(self, ax, data):
        pre_stim, post_stim = (self.calc_opts['events'][self.selected_period_type][opt]
                               for opt in ('pre_stim', 'post_stim'))
        im = ax.imshow(data, cmap='jet', interpolation='nearest', aspect='auto',
                            extent=[-pre_stim, post_stim, *self.freq_range], origin='lower')
        return im

    def set_clim_and_make_colorbar(self, axes, im_list, minimum, maximum):
        [im.set_clim(minimum, maximum) for im in im_list]
        cbar = self.fig.colorbar(im_list[0], ax=axes.ravel().tolist(), shrink=0.7)

    def set_up_stimulus_patches(self, axes, repeat=None):
        if repeat:
            for start in np.arange(0, self.calc_opts['events'][self.selected_period_type]['post_stim'], repeat):
                self.make_stimulus_patch(axes, start)
        else:
            self.make_stimulus_patch(axes, 0)

    def make_stimulus_patch(self, axes, start):
        [ax.fill_betweenx(self.freq_range, start, self.experiment.stimulus_duration, color='k', alpha=0.2)
         for ax in axes.ravel()]

    def set_dir_and_filename(self, basename):
        if all([s not in self.calc_type for s in ('coherence', 'correlation', 'granger')]):
            brain_region = self.current_brain_region
        else:
            brain_region = self.calc_opts.get('region_set')
        title_string = f"{'_'.join([brain_region, str(self.current_frequency_band), basename])}"
        
        self.title = smart_title_case(title_string.replace('_', ' '))
        self.fig.suptitle(self.title, weight='bold', y=.98, fontsize=14)
        self.fname = f"{title_string}.png"
        self.dir_tags = [self.calc_type]

    def make_footer(self):
        pass



