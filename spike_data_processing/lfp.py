import os
import json
import pickle

from scipy.signal import cwt, morlet
from collections import defaultdict
from copy import copy


from data import Data
from context import experiment_context
from block_constructor import BlockConstructor
from context import Subscriber
from matlab_interface import MatlabInterface
from math_functions import *
from utils import cache_method, get_ancestors, find_ancestor_attribute


class LFPData(Data):

    @property
    def frequency_bins(self):
        return self.get_frequency_bins(self.data)

    @property
    def time_bins(self):
        return self.get_time_bins(self.data)

    @property
    def freq_range(self):
        if find_ancestor_attribute(self, 'frozen_freq_range') is not None:
            return find_ancestor_attribute(self, 'frozen_freq_range')
        exp = self.find_experiment()
        if isinstance(self.current_frequency_band, type('str')):
            return exp.frequency_bands[self.current_frequency_band]
        else:
            return self.current_frequency_band

    @property
    def lfp_root(self):
        if hasattr(self, '_lfp_root'):
            return self._lfp_root
        elif hasattr(self, 'parent'):
            return self.parent.lfp_root
        else:
            return None

    @cache_method
    def get_mrl(self):
        axis = 0 if not self.data_opts.get('collapse_matrix') else None
        return self.get_average('get_mrl', stop_at='mrl_calculator', axis=axis)


    @cache_method
    def get_power(self):
        return self.get_average('get_power', stop_at='event')

    def get_time_bins(self, data):
        tbs = []
        if len(data.shape) > 1:
            for i, data_point in enumerate(range(data.shape[1])):
                column = data[:, i]
                tb = TimeBin(i, column, self)
                tbs.append(tb)
            return tbs
        else:
            return [TimeBin(i, data_point, self) for i, data_point in enumerate(data)]

    def get_frequency_bins(self, data):
        return [FrequencyBin(i, data_point, self) for i, data_point in enumerate(data)]

    def load(self, calc_name, other_identifiers):
        store = self.data_opts.get('store', 'pkl')
        lfp_dir = os.path.join(self.lfp_root, 'lfp')
        store_dir = os.path.join(lfp_dir, f"{calc_name}_{store}s")
        for p in [lfp_dir, store_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        store_path = os.path.join(store_dir, '_'.join(other_identifiers) + f".{store}")
        if os.path.exists(store_path) and not self.data_opts.get('force_recalc'):
            with open(store_path, 'rb') as f:
                if store == 'pkl':
                    return_val = pickle.load(f)
                else:
                    return_val = json.load(f)
                return True, return_val, store_path
        else:
            return False, None, store_path

    def save(self, result, store_path):
        store = self.data_opts.get('store', 'pkl')
        mode = 'wb' if store == 'pkl' else 'w'
        with open(store_path, mode) as f:
            if store == 'pkl':
                return pickle.dump(result, f)
            else:
                result_str = json.dumps([arr.tolist() for arr in result])
                f.write(result_str)


class LFPExperiment(LFPData, Subscriber):
    name = 'experiment'

    def __init__(self, experiment, info, raw_lfp):
        self.experiment = experiment
        self.subscribe(experiment_context)
        self._lfp_root = info['lfp_root']
        self._sampling_rate = info['lfp_sampling_rate']
        self.frequency_bands = info['frequency_bands']
        self.lost_signal = info['lost_signal']  # TODO: this is going to need to get more granular at some point but I haven't yet seen an analysis with multiple mtcsg args
        self.all_animals = [LFPAnimal(animal, raw_lfp[animal.identifier], self._sampling_rate)
                            for animal in self.experiment.all_animals]
        self.groups = [LFPGroup(group, self) for group in self.experiment.groups]
        self.children = self.groups
        self.all_groups = self.groups
        self.last_brain_region = None
        self.last_neuron_type = 'uninitialized'
        self.last_block_type = 'uninitialized'
        self.last_frequency_band = None
        self.selected_animals = None

    @property
    def all_blocks(self):
        blocks = []
        for animal in [animal for animal in self.all_animals if self.in_selected_animals(animal)]:
            [blocks.extend(animal_blocks) for animal_blocks in animal.blocks.values()]
        return blocks

    @property
    def all_mrl_calculators(self):
        mrl_calculators = []
        for animal in [animal for animal in self.all_animals if self.in_selected_animals(animal)]:
            [mrl_calculators.extend(animal_blocks) for animal_blocks in animal.mrl_calculators.values()]
        return mrl_calculators

    @property
    def all_events(self):
        if self.data_type == 'mrl':
            raise ValueError("You can't extract events from MRL data")
        return [event for block in self.all_blocks for event in block.events]

    def in_selected_animals(self, x):
        selected = self.data_opts.get('selected_animals')
        if not selected:
            return True
        else:
            return [ancestor for ancestor in get_ancestors(x) if ancestor.name == 'animal'][0].identifier in selected

    def update(self, _):

        if self.data_opts.get('selected_animals') != self.selected_animals:
            [group.update_children() for group in self.groups]
            self.selected_animals = self.data_opts.get('selected_animals')
        if self.data_class == 'lfp':
            [animal.update_if_necessary() for animal in self.all_animals]


class LFPGroup(LFPData):
    name = 'group'

    def __init__(self, group, lfp_experiment):
        self.spike_target = group
        self.experiment = lfp_experiment
        self.identifier = self.spike_target.identifier
        self.parent = lfp_experiment
        self.children = [animal for animal in self.experiment.all_animals if animal.condition == self.identifier]
        for animal in self.children:
            animal.parent = self
            animal.group = self

    @property
    def mrl_calculators(self):
        return [mrl_calc for animal in self.children for mrl_calc in animal.mrl_calculators if mrl_calc.is_valid]

    @property
    def grandchildren_scatter(self):
        if self.data_type != 'mrl':
            raise NotImplementedError("Grandchildren Scatter is currently only implemented for MRL")
        else:
            unit_points = []
            for animal in self.children:
                for unit in animal.spike_target.children:
                    unit_points.append(np.nanmean([mrl_calc.data for mrl_calc in animal.children
                                                   if mrl_calc.unit.identifier == unit.identifier]))
            return unit_points

    @property
    def data_by_block(self):
        if self.data_type != 'mrl':
            raise NotImplementedError("Data by block is currently only implemented for MRL")
        data_by_block = []
        for i in range(5):
            data_by_block.append(
                np.mean(
                    [mrl_calc.data for animal in self.children for mrl_calc in animal.children
                     if mrl_calc.block.identifier == i], axis=0)
            )
        return np.array(data_by_block)

    def update_children(self):
        self.children = [animal for animal in self.experiment.all_animals if animal.condition == self.identifier]
        if self.data_opts.get('selected_animals') is not None:
            self.children = [child for child in self.children if child.identifier in
                             self.data_opts.get('selected_animals')]

    def get_angle_counts(self):
        for calc in self.mrl_calculators:
            if any(np.isnan(calc.get_angle_counts())):
                print(f"{calc.parent.identifier} {calc.identifier}")
        counts = np.sum(np.array([calc.get_angle_counts() for calc in self.mrl_calculators]), axis=0)
        return counts

    def get_mrl(self):
        return self.get_average('get_mrl', stop_at='mrl_calculator', axis=None)


class LFPAnimal(LFPData, BlockConstructor):
    """An animal in the experiment. Processes the raw LFP data and divides it into blocks."""

    name = 'animal'

    def __init__(self, animal, raw_lfp, sampling_rate, is_mirror=False):
        self.spike_target = animal
        self.raw_lfp = raw_lfp
        self._sampling_rate = sampling_rate
        self.block_class = LFPBlock
        self.blocks = defaultdict(list)
        self.mrl_calculators = defaultdict(list)
        self.parent = None
        self.group = None
        self._processed_lfp = {}
        self.last_brain_region = None
        self.last_neuron_type = 'uninitialized'
        self.last_block_type = 'uninitialized'
        self.last_frequency_band = None
        self.is_mirror = is_mirror
        self.mirror = None
        self.frozen_freq_range = None
        self.frozen_blocks = None

    def __getattr__(self, name):
        prop = getattr(type(self), name, None)
        if isinstance(prop, property):
            return prop.fget(self)
        return getattr(self.spike_target, name)

    @property
    def children(self):
        children = self.mrl_calculators if self.data_type == 'mrl' else self.blocks
        if self.data_opts.get('spontaneous'):
            children = children['spontaneous']
        else:
            children = self.filter_by_selected_blocks(children)
        if self.data_type == 'mrl':
            children = [calc for calc in children if calc.unit.neuron_type == self.selected_neuron_type]
        return children

    def update_if_necessary(self):
        if self.data_opts.get('brain_region') not in self.raw_lfp:
            return
        old_and_new = [(self.last_brain_region, self.current_brain_region),
                       (self.last_frequency_band, self.current_frequency_band),
                       (self.last_neuron_type, self.selected_neuron_type),
                       (self.last_block_type, self.selected_block_type)]
        if any([old != new for old, new in old_and_new]):
            self.update_children()
            self.last_brain_region = self.current_brain_region
            self.last_frequency_band = self.current_frequency_band
            self.last_neuron_type = self.selected_neuron_type
            self.last_block_type = self.selected_block_type

    @property
    def processed_lfp(self):
        if self.current_brain_region not in self._processed_lfp:
            self.process_lfp()
        return self._processed_lfp

    def update_children(self):
        self.prepare_blocks()
        if 'mrl' in self.data_type:
            self.prepare_mrl_calculators()

    def process_lfp(self):
        data = self.raw_lfp[self.current_brain_region]/4
        filter = self.data_opts.get('filter', 'filtfilt')
        if filter == 'filtfilt':
            filtered = filter_60_hz(data, self.sampling_rate)
        elif filter == 'spectrum_estimation':
            ids = [self.identifier, self.data_opts['brain_region']]
            saved_calc_exists, filtered, pickle_path = self.load('filter', ids)
            if not saved_calc_exists:
                ml = MatlabInterface(self.data_opts['matlab_configuration'])
                filtered = ml.filter(data)
                self.save(filtered, pickle_path)
            filtered = np.squeeze(np.array(filtered))
        else:
            raise ValueError("Unknown filter")
        normed = divide_by_rms(filtered)
        self._processed_lfp[self.current_brain_region] = normed

    def prepare_mrl_calculators(self):
        if self.data_opts.get('spontaneous'):
            mrl_calculators = {'spontaneous': [SpontaneousMRLCalculator(unit, self)
                                               for unit in self.spike_target.units['good']]}
        else:
            mrl_calculators = {block_type: [BlockMRLCalculator(unit, block=block) for block in blocks
                                            for unit in self.spike_target.units['good']]
                               for block_type, blocks in self.blocks.items()}
        self.mrl_calculators = mrl_calculators

    def make_mirror(self):
        # This method permits calculating values to determine which data points might be noise with one set
        # frequencies when the actual calculation of interest uses another set. The mirror holds the memory of the first
        # calculation.
        self.mirror = LFPAnimal(self.spike_target, self.raw_lfp, self.sampling_rate, is_mirror=True)
        self.mirror.frozen_freq_range = self.data_opts['validate_events'].get('frequency', (0, 8))
        self.mirror.frozen_blocks = self.data_opts['validate_events'].get('blocks', self.data_opts.get('blocks'))
        self.mirror.parent = self.parent
        self.mirror.group = self.group
        self.mirror.update_children()


class LFPDataSelector:
    """A class with methods shared by LFPBlock and LFPEvent that are used to return portions of their data."""

    @property
    def mean_over_time_bins(self):
        return np.mean(self.data, axis=1)

    @property
    def mean_over_frequency(self):
        return np.mean(self.data, axis=0)

    @property
    def mean(self):
        return np.mean(self.data)

    @cache_method
    def slice_spectrogram(self):
        tolerance = .2  # todo: this might change with different mtcsg args
        indices = np.where(self.spectrogram[1] - tolerance <= self.freq_range[0])
        ind1 = indices[0][-1] if indices[0].size > 0 else None  # last index that's <= start of the freq range
        ind2 = np.argmax(self.spectrogram[1] > self.freq_range[1] + tolerance)  # first index > end of freq range
        return self.spectrogram[0][ind1:ind2, :]

    @cache_method
    def trimmed_spectrogram(self):
        return self.sliced_spectrogram[:, 75:-75]  # TODO: 75 should be a function of block length and mtscg args

    @property
    def sliced_spectrogram(self):
        return self.slice_spectrogram()


class LFPBlock(LFPData, BlockConstructor, LFPDataSelector):
    """A block in the experiment. Preprocesses data, initiates calls to Matlab to get the cross-spectrogram, and
    generates LFPEvents. Inherits from LFPSelector to be able to return portions of its data."""

    name = 'block'

    def __init__(self, lfp_animal, i, block_type, block_info, onset, events=None, target_block=None, is_relative=False):
        LFPDataSelector.__init__(self)
        self.animal = lfp_animal
        self.parent = lfp_animal
        self.identifier = i
        self.block_type = block_type
        self.onset = onset - 1
        self.target_block = target_block
        self._is_relative = is_relative
        if events is not None:
            self.event_starts = events - 1
        else:
            self.event_starts = np.array([])
        self.convolution_padding = block_info['lfp_padding']
        self.duration = block_info.get('duration')
        self.event_duration = block_info.get('event_duration')
        if self.event_duration is None:
            self.event_duration = target_block.event_duration
        self.reference_block_type = block_info.get('reference_block_type')
        start_pad_in_samples, end_pad_in_samples = np.array(self.convolution_padding) * self.sampling_rate
        duration_in_samples = self.duration * self.sampling_rate
        start = int(self.onset - start_pad_in_samples)
        stop = int(self.onset + duration_in_samples + end_pad_in_samples)
        self.raw_data = self.animal.raw_lfp[self.current_brain_region][start:stop]
        self.processed_data = self.animal.processed_lfp[self.current_brain_region][start:stop]
        self.unpadded_data = self.processed_data[start_pad_in_samples:-end_pad_in_samples]
        self._spectrogram = None
        self.last_brain_region = None
        self.experiment = self.animal.group.experiment

    @property
    def children(self):
        if self.data_opts.get('validate_events'):
            return [event for event in self.events if event.is_valid]
        else:
            return self.events

    @property
    def events(self):
        return self.get_events()

    @property
    def spectrogram(self):
        if self._spectrogram is None or self.current_brain_region != self.last_brain_region:
            self._spectrogram = self.calc_cross_spectrogram()
            self.last_brain_region = self.current_brain_region
        return self._spectrogram

    @property
    def extended_data(self):
        data = self.events[0].data
        for event in self:
            data = np.concatenate((data, event.data), axis=1)
        return data

    def lost_signal(self):
        return self.animal.parent.experiment.lost_signal

    def get_events(self):
        true_beginning = self.convolution_padding[0] - self.lost_signal()/2
        pre_stim, post_stim = (self.data_opts['events'][self.block_type][opt] for opt in ['pre_stim', 'post_stim'])
        time_bins = np.array(self.spectrogram[2])
        events = []
        epsilon = 1e-6  # a small offset to avoid floating-point inting issues

        for i, event_start in enumerate(self.event_starts):
            # get normed data for the event in samples
            start = int(event_start - self.onset)
            stop = int(start + self.event_duration*self.sampling_rate)
            normed_data = self.unpadded_data[start:stop]

            # get time points where the event will fall in the spectrogram in seconds
            spect_start = start/self.sampling_rate + true_beginning - pre_stim
            spect_end = spect_start + pre_stim + post_stim
            num_points = int(np.ceil((spect_end - spect_start) / .01 - epsilon))  # TODO: the .01s in here depend on the mtcsg args
            event_times = np.linspace(spect_start, spect_start + (num_points * .01), num_points, endpoint=False)
            event_times = event_times[event_times < spect_end]
            # a binary mask that is True when a time bin in the spectrogram belongs to this event
            mask = (np.abs(time_bins[:, None] - event_times) <= epsilon).any(axis=1)

            events.append(LFPEvent(i, event_times, normed_data, mask, self))
        return events

    def calc_cross_spectrogram(self):
        arg_set = self.data_opts['power_arg_set']
        pickle_args = [self.animal.identifier, self.data_opts['brain_region']] + [str(arg) for arg in arg_set] + \
                      [self.block_type, str(self.identifier)]
        saved_calc_exists, result, pickle_path = self.load('spectrogram', pickle_args)
        if not saved_calc_exists:
            ml = MatlabInterface(self.data_opts['matlab_configuration'])
            result = ml.mtcsg(self.processed_data, *arg_set)
            self.save(result, pickle_path)
        return [np.array(arr) for arr in result]

    @property
    def power_deviations(self):
        return self.get_power_deviations()

    @cache_method
    def get_power_deviations(self, moving_window=.15):
        parent_mean = self.parent.mean_data
        parent_sd = self.parent.sd_data
        moving_avgs = np.zeros(int(self.duration/.01))
        window_interval = int(moving_window/.01)  # TODO: bin size actually depends on args to mtcsg
        for i in range(self.trimmed_spectrogram().shape[1] - window_interval + 1):
            data = np.mean(self.trimmed_spectrogram()[:, i:i+window_interval])
            normalized_data = (data - parent_mean)/parent_sd
            for j in range(window_interval):
                if abs(normalized_data) > np.abs(moving_avgs[i + j]):
                    moving_avgs[i + j] = normalized_data
        return moving_avgs


class LFPEvent(LFPData, LFPDataSelector):
    name = 'event'

    def __init__(self, id, event_times, normed_data, mask, block):
        LFPDataSelector.__init__(self)
        self.identifier = id
        self.event_times = event_times
        self.mask = mask
        self.normed_data = normed_data
        self.block = block
        self.parent = block
        self.block_type = self.parent.block_type
        self.spectrogram = self.parent.spectrogram

    @cache_method
    def _get_power(self):
        return np.array(self.sliced_spectrogram)[:, self.mask]

    @cache_method
    def get_power(self):
        return self.refer(self._get_power())

    def get_original_data(self):
        return self._get_power()

    @property
    def is_valid(self):
        animal = self.block.animal
        if (not self.data_opts.get('validate_events')) or animal.is_mirror:
            return True

        if not animal.mirror:
            animal.make_mirror()

        mirror_event = animal.mirror.blocks[self.block_type][self.block.identifier].events[self.identifier]
        return mirror_event.validate()

    def validate(self):
        evoked = self.data_opts.get('evoked')
        frequency_bins = self.get_frequency_bins(self.get_original_data())
        for frequency in range(*self.data_opts['validate_events'].get('frequency', (0, 8))):
            self.update_data_opts(['evoked'], False)
            if not self.block.animal.mirror:
                self.block.animal.make_mirror()
            standard = self.block.animal.mirror.get_median(
                stop_at='event', extend_by=('frequency', 'time'),
                select_by=(('block', 'block_type', self.block_type), ('frequency_bin', 'identifier', frequency)))
            self.update_data_opts(['evoked'], evoked)
            for time_bin in frequency_bins[frequency].time_bins:
                if time_bin.data > self.data_opts['validate_events'].get('threshold', 20) * standard:
                    print(f"{self.current_brain_region} {self.block.animal.identifier} {self.block_type} "
                          f"{self.block.identifier} {self.identifier} invalid!")
                    return False
        return True


class FrequencyBin(LFPData):
    """A FrequencyBin contains a slice of cross-spectrogram or mrl calculation at the smallest available frequency
    resolution."""

    name = 'frequency_bin'

    def __init__(self, index, val, parent, unit=None):
        self.parent = parent
        self.val = val
        if self.parent.name == 'block':
            self.identifier = self.parent.spectrogram[1][index]
        elif hasattr(self.parent, 'block'):
            self.identifier = self.parent.block.spectrogram[1][index]
        self.block_type = self.parent.block_type
        self.unit = unit
        self.is_valid = self.parent.is_valid if hasattr(self.parent, 'is_valid') else True

    @property
    def data(self):
        return self.val

    @property
    def mean_data(self):
        return np.mean(self.val)


class TimeBin:
    name = 'time_bin'

    def __init__(self, i, data, parent):
        self.parent = parent
        self.block_type = self.parent.block_type
        self.identifier = i
        self.data = data
        self.mean_data = np.mean(self.data)
        self.ancestors = get_ancestors(self)
        self.position = self.get_position_in_block_time_series()
        self.block = [ancestor for ancestor in self.ancestors if ancestor.name == 'block'][0]

    @property
    def power_deviation(self):
        return self.block.power_deviations[self.position]

    def get_position_in_block_time_series(self):
        if self.parent.name == 'event':
            self.parent.num_bins_per_event * self.parent.identifier + self.identifier
        else:
            return self.identifier


class MRLCalculator(LFPData):
    """Calculates the Mean Resultant Length of the vector that represents the phase of a frequency in the LFP data on
    the occasion the firing of a neuron. MRL """

    name = 'mrl_calculator'

    def __init__(self, unit):
        self.unit = unit

    @property
    def mean_over_frequency(self):
        return np.mean(self.data, axis=0)

    @property
    def frequency_bins(self):
        return [FrequencyBin(i, data, self, unit=self.unit) for i, data in enumerate(self.data)]

    def get_wavelet_phases(self, scale):
        cwt_matrix = cwt(self.block.unpadded_data, morlet, [scale])
        # Since we're computing the CWT for only one scale, the result is at index 0.
        return np.angle(cwt_matrix[0, :])

    def get_phases(self):
        low = self.freq_range[0] + .05
        high = self.freq_range[1]
        if self.data_opts.get('phase') == 'wavelet':
            if 'gamma' in self.current_frequency_band: # TODO need to fix this since frequency band can currently be a range of numbers
                frequencies = np.logspace(np.log10(low), np.log10(high), int((high - low) / 5))
            else:
                frequencies = np.linspace(low, high, int(high - low) + 1)
            scales = [get_wavelet_scale(f, self.sampling_rate) for f in frequencies]
            return np.array([self.get_wavelet_phases(s) for s in scales])
        else:
            if isinstance(self.current_frequency_band, type('str')):
                return compute_phase(bandpass_filter(self.mrl_data, low, high, self.sampling_rate))
            else:
                frequency_bands = [(f + .05, f + 1) for f in range(*self.freq_range)]
                return np.array([compute_phase(bandpass_filter(self.block.unpadded_data, low, high, self.sampling_rate))
                                 for low, high in frequency_bands])

    @cache_method
    def get_angles(self):

        def adjust_angle(angle, weight):
            if np.isnan(weight):
                return np.nan
            return angle % (2 * np.pi)

        phases = self.get_phases().T
        weights = self.get_weights()

        # Apply the function to every element
        vfunc = np.vectorize(adjust_angle)

        if phases.ndim == 1:
            adjusted_phases = vfunc(phases, weights)
        else:
            # Expand the dimensions of weights to make it (60000, 1)
            weights_expanded = weights[:, np.newaxis]
            adjusted_phases = vfunc(phases, weights_expanded)

        # Filter out NaNs
        adjusted_phases = adjusted_phases[~np.isnan(adjusted_phases)]

        return adjusted_phases

    def get_angle_counts(self): # TODO: give this a data type and make it use data
        n_bins = 36
        bin_edges = np.linspace(0, 2 * np.pi, n_bins + 1)
        angles = self.get_angles()
        counts, _ = np.histogram(angles, bins=bin_edges)
        if self.data_opts.get('evoked'):
            counts = counts/len(angles)  # counts must be transformed to proportions for the subtraction to make sense
            if self.block_type == 'tone':  # TODO: generalize this
                counts -= self.equivalent_calculator.get_angle_counts()
        return counts

    @cache_method
    def get_mrl(self):
        w = self.get_weights()
        alpha = self.get_phases()
        dim = int(self.data_opts.get('phase') == 'wavelet' or not isinstance(self.current_frequency_band, type('str')))

        if w.ndim == 1 and alpha.ndim == 2:
            w = w[np.newaxis, :]

        # Handle NaN assignment based on the shape of alpha
        if alpha.ndim == 2:
            w[:, np.isnan(alpha).any(axis=0)] = np.nan
        else:
            w[np.isnan(alpha)] = np.nan

        if self.data_opts.get('mrl_func') == 'ppc':
            return self.refer(circ_r2_unbiased(alpha, w, dim=dim))
        else:
            return self.refer(compute_mrl(alpha, w, dim=dim))


class BlockMRLCalculator(MRLCalculator):

    def __init__(self, unit, block):
        super().__init__(unit)
        self.block = block
        self.block_type = block.block_type
        self.mrl_data = block.unpadded_data
        self.duration = block.duration
        self.identifier = f"{self.block.identifier}_{self.unit.identifier}"
        self.parent = self.block.parent
        self.spike_block = self.unit.blocks[self.block_type][self.block.identifier]
        self.spikes = [int((spike + i) * self.sampling_rate) for i, event in enumerate(self.spike_block.events)
                       for spike in event.spikes]
        self.num_events = len(self.spikes)

    @property
    def ancestors(self):
        return [self] + [self.unit] + [self.block] + self.parent.ancestors

    @property
    def equivalent_calculator(self):
        other_stage = self.spike_block.reference_block_type
        return [calc for calc in self.parent.mrl_calculators[other_stage] if calc.identifier == self.identifier][0]

    @property
    def is_valid(self):
        if self.data_opts.get('evoked'):
            return self.num_events > 4 and self.equivalent_calculator.num_events > 4
        else:
            return self.num_events > 4

    def get_weights(self):
        return np.array(
            [1 if weight in self.spikes else float('nan') for weight in range(self.duration * self.sampling_rate)])


class SpontaneousMRLCalculator(MRLCalculator):
    def __init__(self, unit, animal):
        super().__init__(unit)
        self.animal = animal
        self.parent = self.animal
        self.block_type = self.identifier = 'spontaneous'
        self.spikes = self.unit.get_spontaneous_firing()
        self.num_events = len(self.spikes)
        self.raw = self.animal.raw_lfp[self.brain_region]
        if isinstance(self.data_opts['spontaneous'], tuple):
            self.start, self.end = np.array(self.data_opts['spontaneous']) * self.sampling_rate
        else:
            self.end = self.animal.earliest_block.onset
            self.start = self.end - self.data_opts['spontaneous'] * self.sampling_rate
        self.mrl_data = self.raw[self.start:self.end]

    def get_weights(self):
        return np.array([1 if weight in self.spikes else float('nan') for weight in range(self.start, self.end)])

    @property
    def is_valid(self):
        return self.num_events > 4
