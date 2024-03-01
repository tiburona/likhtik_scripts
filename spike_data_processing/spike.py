from bisect import bisect_left as bs_left, bisect_right as bs_right
from collections import defaultdict
import numpy as np


from data import Data, TimeBin
from period_constructor import PeriodConstructor
from context import Subscriber
from utils import cache_method, to_hashable
from plotting_helpers import formatted_now
from math_functions import calc_rates, spectrum, trim_and_normalize_ac, cross_correlation, correlogram

"""
This module defines SpikeData, Experiment, Group, Animal, Unit, Period, and Event, which comprise a hierarchical data 
model. (It also defines UnitPair, a bit of a special case.) SpikeData inherits from Data, which defines common 
properties and methods to data representations.  The rest of the classes inherit from SpikeData. Several of SpikeData's 
methods call `get_average`, a method that recurses down levels of the hierarchy to each object's children, and is 
overwritten by the base case, most frequently Event.  
"""


class SpikeData(Data):

    @property
    def last_event_vals(self):
        if hasattr(self, '_last_event_vals'):
            return self._last_event_vals
        else:
            return self.parent.last_event_vals

    @property
    def time_bins(self):
        return [TimeBin(i, data_point, self) for i, data_point in enumerate(self.data)]

    @property
    def mean(self):
        return np.mean(self.data)

    @property
    def unit_pairs(self):
        return [pair for pair in self.children.unit_pairs]

    @cache_method
    def get_demeaned_rates(self):
        rates = self.get_average('get_unadjusted_rates')
        return rates - np.mean(rates)

    def get_psth(self):
        return self.get_average('get_psth')

    def get_spontaneous_firing(self):
        return self.get_average('get_spontaneous_firing', stop_at='unit')

    def get_cross_correlations(self):
        return self.get_average('get_cross_correlations', stop_at='unit')

    def get_correlogram(self):
        return self.get_average('get_correlogram', stop_at='unit')

    @cache_method
    def proportion_score(self):
        return self.refer([1 if rate > 0 else 0 for rate in self.get_psth()])

    @cache_method
    def get_proportion(self):
        return self.get_average('proportion_score', stop_at=self.data_opts.get('base', 'event'))

    @cache_method
    def get_autocorrelation(self):
        stop = self.data_opts.get('base')
        return self.refer(self.get_average('_calculate_autocorrelation', stop_at=stop), stop_at=stop)

    def _calculate_autocorrelation(self):
        x = self.get_demeaned_rates()
        return trim_and_normalize_ac(np.correlate(x, x, mode='full'), self.data_opts['max_lag'])

    @cache_method
    def get_spectrum(self):
        # default: returns spectrum of data of current object; average over spectra with a different spectrum_base
        stop = self.data_opts.get('spectrum_base', self.name)
        return self.refer(self.get_average('_get_spectrum', stop_at=stop), stop_at=stop, is_spectrum=True)

    @cache_method
    def _get_spectrum(self):
        freq_range, max_lag, bin_size = (self.data_opts[opt] for opt in ['freq_range', 'max_lag', 'bin_size'])
        series = self.data_opts.get('spectrum_series', 'get_autocorrelation')
        return spectrum(getattr(self, series)(), freq_range, max_lag, bin_size)

    @cache_method
    def upregulated(self, duration, std_dev=.5):
        """
        Determines if the data is up- or down-regulated during a portion of its time series

        Returns:
            int: 1 if upregulated, -1 if downregulated, 0 otherwise.
        """
        first_ind, last_ind = (int(x / self.data_opts['bin_size']) for x in duration)
        activity = np.mean(self.data[first_ind:last_ind])
        if activity > std_dev * np.std(self.data):
            return 1
        elif activity < -std_dev * np.std(self.data):
            return -1
        else:
            return 0


class Experiment(SpikeData, Subscriber):
    """The experiment. Parent of groups. Subscribes to the context and notifies its descendants in the tree when changes
    in the context require them to update their children. Via the attributes and properties that begin with 'all'
    maintains a comprehensive record of entities and their associated values."""

    name = 'experiment'

    def __init__(self, info, groups):
        self.identifier = info['identifier'] + formatted_now()
        self.conditions = info['conditions']
        self._sampling_rate = info['sampling_rate']
        self.stimulus_duration = info.get('stimulus_duration')
        self.subscribe(self.context)
        self.groups = groups
        self.all_groups = self.groups
        self.all_animals = [animal for group in self.groups for animal in group]
        self._last_event_vals = None
        self.last_neuron_type = 'uninitialized'
        self.last_period_type = 'uninitialized'
        self.selected_animals = None  # None means all animals will be included; it's the default state
        self.children = self.groups
        for group in self.groups:
            group.parent = self
        self.period_types = set(period_type for animal in self.all_animals for period_type in animal.period_info)
        self._neuron_types = set([unit.neuron_type for unit in self.all_units])

    @property
    def all_units(self):
        return [unit for animal in self.all_animals for unit in animal.units['good']]

    @property
    def all_periods(self):
        return [period for unit in self.all_units for period in unit.all_periods]

    @property
    def all_events(self):
        return [event for period in self.all_periods for event in period]

    @property
    def all_unit_pairs(self):
        return [unit_pair for unit in self.all_units for unit_pair in unit.get_pairs()]

    def update(self, name):
        if name == 'data':
            event_vals = [to_hashable(self.data_opts.get('events'))] + [self.data_opts.get('bin_size')]
            if event_vals != self.last_event_vals:
                [unit.update_children() for unit in self.all_units]
                self._last_event_vals = event_vals
            if self.data_opts.get('selected_animals') != self.selected_animals:
                [group.update_children() for group in self.groups]
                self.selected_animals = self.data_opts.get('selected_animals')

        if name == 'neuron_type':
            if self.selected_neuron_type != self.last_neuron_type:
                [entity.update_children() for entity in self.all_groups + self.all_animals]
                self.last_neuron_type = self.selected_neuron_type  # TODO: think about whether `last_neuron_type` should be shared among all data

        if name == 'period_type':
            if self.selected_period_type != self.last_period_type:
                [unit.update_children() for unit in self.all_units]


class Group(SpikeData):
    """A group in the experiment, i.e., a collection of animals assigned to a condition, the child of an Experiment,
    parent of animals. Limits its children to the active neuron type."""

    name = 'group'

    def __init__(self, name, animals=None):
        self.identifier = name
        self.animals = animals if animals else []
        self.children = self.animals
        for animal in self.animals:
            animal.parent = self
        self.parent = None

    def update_children(self):
        if self.context.vals.get('neuron_type', 'all') == 'all':
            self.children = self.animals
        else:
            self.children = [animal for animal in self.animals
                             if len(getattr(animal, self.context.vals['neuron_type']))]
        if self.data_opts.get('selected_animals') is not None:
            self.children = [child for child in self.children
                             if child.identifier in self.data_opts.get('selected_animals')]


class Animal(SpikeData):
    """An animal in the experiment, the child of a group, parent of units. Updates its children, i.e., the active units
    for analysis, when the selected neuron type is altered in the context.

    Note that the `units` property is not a list, but rather a dictionary with keys for different categories in list.
    It would also be possible to implement context changes where self.children updates to self.units['MUA'].
    """

    name = 'animal'

    def __init__(self, identifier, condition, animal_info, neuron_types=None):
        self.identifier = identifier
        self.condition = condition
        self.animal_info = animal_info
        self.period_info = animal_info['period_info'] if 'period_info' in animal_info is not None else {}
        if neuron_types is not None:
            for nt in neuron_types:
                setattr(self, nt, [])
        self.units = defaultdict(list)
        self.raw_lfp = None
        self.children = None
        self.parent = None

    @property
    def unit_pairs(self):  # TODO: is this being used for anything? Can it be deleted?
        return [pair for unit in self.units['good'] for pair in unit.pairs]

    def update_children(self):
        if self.context.vals['neuron_type'] == 'all':
            self.children = self.units['good']
        else:
            self.children = getattr(self, self.context.vals['neuron_type'])


class Unit(SpikeData, PeriodConstructor):
    """A unit that was recorded from in the experiment, the child of an animal, parent of periods. Inherits from
    PeriodConstructorUpdates to build its children. Updates its children when the `selected_period_type` or the events
    structure (from data_opts) changes."""

    name = 'unit'

    def __init__(self, animal, category, spike_times, neuron_type=None):
        self.animal = animal
        self.parent = animal
        self.category = category
        self.animal.units[category].append(self)
        self.identifier = str(self.animal.units[category].index(self) + 1)
        self.neuron_type = neuron_type
        self.period_class = Period
        self.periods = defaultdict(list)
        self.spike_times = np.array(spike_times)

    @property
    def children(self):
        return self.filter_by_selected_periods(self.periods)

    @property
    def firing_rate(self):
        return self.animal.sampling_rate * len(self.spike_times) / float(self.spike_times[-1] - self.spike_times[0])

    @property
    def unit_pairs(self):
        all_unit_pairs = self.get_pairs()
        pairs_to_select = self.data_opts.get('unit_pair')
        if pairs_to_select is None:
            return all_unit_pairs
        else:
            return [unit_pair for unit_pair in all_unit_pairs
                    if ','.join([unit_pair.unit.neuron_type, unit_pair.pair.neuron_type]) == pairs_to_select]

    def get_pairs(self):
        return [UnitPair(self, other) for other in [unit for unit in self.animal if unit.identifier != self.identifier]]

    def update_children(self):
        self.prepare_periods()

    @cache_method
    def find_spikes(self, start, stop):
        return np.array(self.spike_times[bs_left(self.spike_times, start): bs_right(self.spike_times, stop)])

    @cache_method
    def get_spikes_by_events(self):
        return [event.spikes for period in self.children for event in period.children]

    @cache_method
    def get_spontaneous_firing(self):
        spontaneous_period = self.data_opts.get('spontaneous', 120)
        if not isinstance(spontaneous_period, tuple):
            start = self.earliest_period.onset - spontaneous_period * self.sampling_rate - 1
            stop = self.earliest_period.onset - 1
        else:
            start = spontaneous_period[0] * self.sampling_rate
            stop = spontaneous_period[1] * self.sampling_rate
        num_bins = int((stop-start) / (self.sampling_rate * self.data_opts['bin_size']))
        return calc_rates(self.find_spikes(start, stop), num_bins, (start, stop), self.data_opts['bin_size'])

    @cache_method
    def get_firing_std_dev(self, period_types=None):
        if period_types is None:  # default: take all period_types
            period_types = [period_type for period_type in self.periods]
        return np.std([rate for period in self.children for rate in period.get_unadjusted_rates()
                       if period.period_type in period_types])

    @cache_method
    def get_cross_correlations(self, axis=0):
        return np.mean([pair.get_cross_correlations(axis=axis, stop_at=self.data_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)

    @cache_method
    def get_correlogram(self, axis=0):
        return np.mean([pair.get_correlogram(axis=axis, stop_at=self.data_opts.get('base', 'period'))
                        for pair in self.unit_pairs], axis=axis)


class Period(SpikeData):
    """A period of time in the recording of a unit. The child of a unit, the parent of events."""

    name = 'period'

    def __init__(self, unit, index, period_type, period_info, onset, events=None, target_period=None, is_relative=False):
        self.unit = unit
        self.identifier = index
        self.period_type = period_type
        self.onset = onset
        self.event_starts = events if events is not None else []
        self._events = []
        self.shift = period_info.get('shift')
        self.duration = period_info.get('duration')
        self.reference_period_type = period_info.get('reference_period_type')
        self.target_period = target_period
        self._is_relative = is_relative
        self.animal = self.unit.animal
        self.parent = unit

    @property
    def children(self):
        return self.events

    @property
    def events(self):
        if not self._events:
            self.update_children()
        return self._events

    def update_children(self):
        events_settings = self.data_opts['events'].get(self.period_type, {'pre_stim': 0, 'post_stim': 1})
        pre_stim, post_stim = (events_settings[opt] * self.sampling_rate for opt in ['pre_stim', 'post_stim'])
        if self.is_relative:
            event_starts = self.target_period.event_starts - self.shift * self.sampling_rate
        else:
            event_starts = self.event_starts
        for i, start in enumerate(event_starts):
            spikes = self.unit.find_spikes(start - pre_stim, start + post_stim)
            self._events.append(Event(self, self.unit, [((spike - start) / self.sampling_rate) for spike in spikes],
                                      [(spike / self.sampling_rate) for spike in spikes], pre_stim, post_stim, i))

    @cache_method
    def get_unadjusted_rates(self):
        return [event.get_unadjusted_rates() for event in self.events]

    @cache_method
    def mean_firing_rate(self):
        return np.mean(self.get_unadjusted_rates())

    def find_equivalent(self, unit):
        return [period for period in unit.children][self.identifier]


class Event(SpikeData):
    """A single event in the experiment, the child of a unit. Aspects of an event, for instance, the start and end of
    relevant data, can change when the context is updated. Most methods on event are the base case of the recursive
    methods on Level."""

    name = 'event'

    def __init__(self, period, unit, spikes, spikes_original_times, pre_stim, post_stim, index):
        self.unit = unit
        self.spikes = spikes
        self.spikes_original_times = spikes_original_times
        self.identifier = index
        self.period = period
        self.period_type = self.period.period_type
        self.children = None
        self.parent = period
        self.pre_stim = pre_stim
        self.post_stim = post_stim
        self.duration = self.pre_stim + self.post_stim

    @cache_method
    def get_psth(self):  # default is to normalize
        rates = self.get_unadjusted_rates()
        if not self.reference or self.data_opts.get('adjustment') == 'none':
            return rates
        rates -= self.reference.mean_firing_rate() # TODO: does this make sense? Maybe this should be the period reference
        if self.data_opts.get('adjustment') == 'relative':
            return rates
        rates /= self.unit.get_firing_std_dev(period_types=self.period_type,)  # same as dividing unit psth by std dev
        return rates

    @cache_method
    def get_unadjusted_rates(self):
        bin_size = self.data_opts['bin_size']
        spike_range = (-self.pre_stim, self.post_stim)
        return calc_rates(self.spikes, self.num_bins_per_event, spike_range, bin_size)

    def get_cross_correlations(self, pair=None):
        other = pair.periods[self.period_type][self.period.identifier].events[self.identifier]
        cross_corr = cross_correlation(self.get_unadjusted_rates(), other.get_unadjusted_rates(), mode='full')
        boundary = int(self.data_opts['max_lag'] / self.data_opts['bin_size'])
        midpoint = cross_corr.size // 2
        return cross_corr[midpoint - boundary:midpoint + boundary + 1]

    def get_correlogram(self, pair=None, num_pairs=None):
        max_lag, bin_size = (self.data_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = int(max_lag/bin_size)
        return correlogram(lags, bin_size, self.spikes, pair.spikes, num_pairs)

    def get_autocorrelogram(self):
        max_lag, bin_size = (self.data_opts[opt] for opt in ['max_lag', 'bin_size'])
        lags = int(max_lag / bin_size)
        return self.refer(correlogram(lags, bin_size, self.spikes, self.spikes, 1))

    def find_equivalent(self, unit):
        return self.period.find_equivalent(unit).events[self.identifier]


class UnitPair(SpikeData):
    """A pair of two units for the purpose of calculating cross-correlations or correlograms."""

    name = 'unit_pair'

    def __init__(self, unit, pair):
        self.parent = unit.parent
        self.unit = unit
        self.pair = pair
        self.identifier = str((unit.identifier, pair.identifier))
        self.pair_category = ','.join([unit.neuron_type, pair.neuron_type])
        self.children = self.unit.children

    def get_cross_correlations(self, **kwargs):
        for kwarg, default in zip(['axis', 'stop_at'], [0, self.data_opts.get('base', 'period')]):
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs else default
        return self.get_average('get_cross_correlations', pair=self.pair, **kwargs)

    def get_correlogram(self, **kwargs):
        for kwarg, default in zip(['axis', 'stop_at'], [0, self.data_opts.get('base', 'period')]):
            kwargs[kwarg] = kwargs[kwarg] if kwarg in kwargs else default
        return self.get_average('get_correlogram', pair=self.pair, num_pairs=len(self.unit.unit_pairs), **kwargs)






