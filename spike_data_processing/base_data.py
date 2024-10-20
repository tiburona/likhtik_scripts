import numpy as np
from copy import deepcopy
from collections import defaultdict
import pickle
import json
import os

from utils import cache_method
from math_functions import sem


class Base:

    _calc_opts = {}
    _cache = defaultdict(dict)
    _filter = {}
    _selected_period_type = ''
    _selected_neuron_type = ''
    original_periods = None

    @property
    def calc_opts(self):
        return Base._calc_opts  
    
    @calc_opts.setter
    def calc_opts(self, value):
        Base._calc_opts = value
        self.set_filter_from_calc_opts()
        Base._cache = defaultdict(dict)

    @property
    def cache(self):
        return Base._cache
    
    def clear_cache(self):
        Base._cache = defaultdict(dict)

    @property
    def filter(self):
        return Base._filter
    
    @filter.setter
    def filter(self, filter):
        Base._filter = filter

    def set_filter_from_calc_opts(self):
        self.filter = defaultdict(lambda: defaultdict(tuple))
        filters = self.calc_opts.get('filter', {})
        if isinstance(filters, list):
            for filter in filters:
                self.add_to_filter(self.parse_natural_language_filter(filter))
        else:
            for object_type in filters:
                object_filters = self.calc_opts['filter'][object_type]
                for property in object_filters:
                    self.filter[object_type][property] = object_filters[property]   
            if self.calc_opts.get('validate_events'):
                self.filter['event']['is_valid'] = ('==', True)

    def add_to_filters(self, obj_name, attr, operator, target_val):
         self.filter[obj_name][attr] = (operator, target_val)

    def del_from_filters(self, obj_name, attr):
        del self.filter[obj_name][attr]
    
    @staticmethod
    def parse_natural_language_filter(filter):
        # natural language filter should be a tubple like:
        # ex1: ('for animals, identifier must be in', ['animal1', 'animal2'])
        # ex2L ('for units, quality must be !=', '3')] 
        condition, target_val = filter
        split_condition = condition.split(' ')
        obj_name = split_condition[1][:-2]
        attr = split_condition[3]
        be_index = condition.find('be')
        operator = condition[be_index + 3:]
        return obj_name, attr, operator, target_val
    
    @property
    def kind_of_data(self):
        return self.calc_opts.get('kind_of_data')

    @property
    def calc_type(self):
        return self.calc_opts['calc_type']

    @calc_type.setter
    def calc_type(self, calc_type):
        self.calc_opts['calc_type'] = calc_type

    @property
    def selected_neuron_type(self):
        return Base._selected_neuron_type

    @selected_neuron_type.setter
    def selected_neuron_type(self, neuron_type):
        Base._selected_neuron_type = neuron_type

    @property
    def selected_period_type(self):
        return Base._selected_period_type
        
    @selected_period_type.setter
    def selected_period_type(self, period_type):
        Base._selected_period_type = period_type

    @property
    def selected_period_group(self):
        return tuple(self.calc_opts['periods'][self.selected_period_type])
    
    @selected_period_group.setter
    def selected_period_group(self, period_group):
        self.calc_opts['periods'][self.selected_period_type] = period_group
    
    @property
    def current_frequency_band(self):
        return self.calc_opts['frequency_band']

    @current_frequency_band.setter
    def current_frequency_band(self, frequency_band):
        self.calc_opts['frequency_band'] = frequency_band

    @property
    def current_brain_region(self):
        return self.calc_opts.get('brain_region')
    
    @current_brain_region.setter
    def current_brain_region(self, brain_region):
        self.calc_opts['brain_region'] = brain_region

    @property
    def current_region_set(self):
        return self.calc_opts.get('region_set')

    @current_region_set.setter
    def current_region_set(self, region_set):
        self.calc_opts['region_set'] = region_set

    @property
    def freq_range(self):
        if isinstance(self.current_frequency_band, type('str')):
            return self.experiment.info['frequency_bands'][self.current_frequency_band]
        else:
            return self.current_frequency_band
        
    def get_data_sources(self, data_object_type=None, identifiers=None, identifier=None):
        if data_object_type is None:
            data_object_type = self.calc_opts['base']
            if data_object_type in ['period', 'event']:
                data_object_type = f"{self.kind_of_data}_{data_object_type}"
        data_sources = getattr(self.experiment, f"all_{data_object_type}s")
        if identifier and 'all' in identifier:
            return data_sources
        if identifiers:
            return [source for source in data_sources if source.identifier in identifiers]
        if identifier:
            return [source for source in data_sources if source.identifier == identifier][0]

    @property
    def pre_event(self):
        return self.get_pre_post('pre', 'event')
    
    @property
    def post_event(self):
        return self.get_pre_post('post', 'event')
    
    @property
    def pre_period(self):
        return self.get_pre_post('pre', 'period')
    
    @property
    def post_period(self):
        return self.get_pre_post('post', 'period')
    
    def get_pre_post(self, time, object_type):
        return self.calc_opts.get('periods', {}).get(
            self.selected_period_type, {}).get(f"{time}_{object_type}", 0)

    
    @property
    def bin_size(self):
        return self.calc_opts.get('bin_size', .01)


class Data(Base):

    def __init__(self):
        self.parent = None

    @property
    def name(self):
        return self._name

    def get_calc(self, calc_type=None):
        if calc_type is None:
            calc_type = self.calc_type
        if self.calc_opts.get('percent_change'):
            return self.percent_change
        return getattr(self, f"get_{calc_type}")()

    @property
    def calc(self):
        return self.get_calc()
    
    def fetch_opts(self, list_of_opts=None):
        if list_of_opts is not None:
            return (self.calc_opts.get(opt) for opt in list_of_opts)
        
    def include(self, check_ancestors=False):
        return self.select(self.filter, check_ancestors=check_ancestors)
    
    def active(self):
        return self.include() and self in self.parent.children
    
    @property
    def parent_identifier(self):
        try:
            return self.parent.identifier
        except AttributeError:
            return None
        
    @property
    def grandparent_identifier(self):
        try:
            return self.ancestors[-3].identifier
        except IndexError:
            return None
            
    def select(self, filters, check_ancestors=False):
        operations = {
            '==': lambda a, b: a == b,
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            'in': lambda a, b: a in b,
            '!=': lambda a, b: a != b,
            'not in': lambda a, b: a not in b
        }

        if not check_ancestors and self.name not in filters:
            return True
              
        for obj in (self.ancestors if check_ancestors else [self]):
            if obj.name not in filters:
                continue
            obj_filters = filters[obj.name]
            for attr in obj_filters:
                if hasattr(obj, attr):
                    object_value = getattr(obj, attr)
                    operation_symbol, target_value = obj_filters[attr]
                    function = operations[operation_symbol]
                    if not function(object_value, target_value):
                        return False
        return True
    
    @cache_method
    def get_average(self, base_method, stop_at='event', level=0, axis=0, *args, **kwargs):
        """
        Recursively calculates the average of the values of the computation in the base method on the object's
        descendants.

        Parameters:
        - base_method (str): Name of the method called when the recursion reaches the base case.
        - stop_at (str): `name` attribute of the base case object the `base_method` is called on.
        - level (int): a counter for how deep recursion has descended; limits the cache.
        - axis (int or None): Specifies the axis across which to compute the mean.
        - **kwargs: Additional keyword arguments to be passed to the base method.

        Returns:
        float or np.array: The mean of the data values from the object's descendants.
        """
        if not self.include():
            return float('nan')
                   
        if self.name == stop_at:  # we are at the base case and will call the base method
            if hasattr(self, base_method) and callable(getattr(self, base_method)):
                return getattr(self, base_method)(*args, **kwargs)
            else:
                raise ValueError(f"Invalid base method: {base_method}")

        if not len(self.children):
            return float('nan')
        
        child_vals = []

        for child in self.children:
            if not child.include():
                continue
            child_val = child.get_average(
                base_method, level=level+1, stop_at=stop_at, axis=axis, **kwargs)
            if not self.is_nan(child_val):
                child_vals.append(child_val)
            
        if len(child_vals) and isinstance(child_vals[0], dict):
            # Initialize defaultdict to automatically start lists for new keys
            result_dict = defaultdict(list)

            # Aggregate values from each dictionary under their corresponding keys
            for child_val in child_vals:
                for key, value in child_val.items():
                    result_dict[key].append(value)

            # Calculate average of the list of values for each key
            return_dict = {key: np.nanmean(values, axis) 
                            for key, values in result_dict.items()}
            return return_dict
                
        else:
            return np.nanmean(child_vals, axis)
        
    @property
    def mean(self):
        return np.mean(self.refer(self.calc))
    
    @property
    def sem(self):
        return self.get_sem(collapse_sem_data=True)
    
    @property
    def sem_envelope(self):
        return self.get_sem(collapse_sem_data=False)
    
    def get_mean(self, axis=0):
        return np.mean(self.calc, axis=axis)
        
    def get_sem(self, collapse_sem_data=False):
        """
        Calculates the standard error of an object's data. If object's data is a vector, it will always return a float.
        If object's data is a matrix, the `collapse_sem_data` argument will determine whether it returns the standard
        error of its children's average data points or whether it computes the standard error over children maintaining
        the original shape of children's data, as you would want, for instance, if graphing a standard error envelope
        around firing rate over time.
        """

        if self.calc_opts.get('sem_level'):
            sem_children = self.get_descendants(stop_at=self.calc_opts.get('sem_level'))
        else:
            sem_children = self.children

        if isinstance(sem_children[0].calc, dict):

            return_dict = {}

            for key in sem_children[0]:
                vals = [child.calc[key] for child in sem_children if not self.is_nan(child.calc)]
                if collapse_sem_data:
                    vals = [np.mean(val) for val in vals]
                return_dict[key] = sem(vals) 

            return return_dict

        else:
            vals = [child.calc for child in sem_children if not self.is_nan(child.calc)]

            if collapse_sem_data:
                vals = [np.mean(val) for val in vals]

            return sem(vals)
                
    @staticmethod
    def extend_into_bins(sources, extend_by):
        if 'frequency' in extend_by:
            sources = [freq_bin for src in sources for freq_bin in src.frequency_bins]
        if 'time' in extend_by:
            sources = [time_bin for src in sources for time_bin in src.time_bins]
        return sources
            
    def get_median(self, stop_at=None, extend_by=None):
        if not stop_at:
            stop_at = self.calc_opts.get('stop_at')
        vals_to_summarize = self.get_descendants(stop_at=stop_at)
        vals_to_summarize = self.extend_into_bins(vals_to_summarize, extend_by)
        return np.median([obj.calc for obj in vals_to_summarize])

    def is_nan(self, value):
        if isinstance(value, float) and np.isnan(value):
            return True
        elif isinstance(value, np.ndarray) and np.all(np.isnan(value)):
            return True
        elif isinstance(value, dict) and all(self.is_nan(val) for val in value.values()):
            return True                                
        else:
            return False
    
    @property
    def concatenation(self):
        return self.concatenate()
    
    def concatenate(self, method=None, max_depth=-1):   
        return np.concatenate(self.accumulate(method=method, max_depth=max_depth)[max_depth])
    
    @property
    def stack(self):
        return self.get_stack()
    
    def get_stack(self, depth=1, attr='calc', method=None, base=None):
        return np.vstack(
            self.accumulate(default_attr=attr, method=method, max_depth=depth, base=base)[depth])
   
    @property
    def scatter(self):
        return self.accumulate(default_attr='mean', max_depth=1)[1]
    
    @property
    def grandchildren_scatter(self):
        return self.accumulate(default_attr='mean', max_depth=2)[2]
    
    def accumulate(self, method=None, max_depth=1, depth=0, default_attr=None, accumulator=None, base=None):
        """Generalized recursive function to apply a method or default method to children."""

        
        
        f = lambda x: getattr(x, method)() if method else getattr(x, default_attr)
        
        if accumulator is None:
            accumulator = defaultdict(list)

        if depth == max_depth:
            accumulator[depth].append(f(self))
        else:
            if hasattr(self, 'children'):  
                for child in self.children:
                    child.accumulate(method, max_depth, depth + 1, default_attr, accumulator)

        return accumulator     

    @property
    def hierarchy(self):
        if self.name == 'experiment':
            return [self.name]
        if hasattr(self, 'parent'):
            return self.parent.hierarchy + [self.name]
 
    @property
    def sampling_rate(self):
        if self.name == 'experiment':
            return self._sampling_rate
        else:
            return self.experiment.sampling_rate
        
    @property
    def lfp_sampling_rate(self):
        if self.name == 'experiment':
            return self._lfp_sampling_rate
        else:
            return self.experiment.lfp_sampling_rate

    @property
    def ancestors(self):
        return [self] + self.parent.ancestors
        
    def get_descendants(self, stop_at=None, descendants=None, all=False):
   
        if descendants is None:
            descendants = []

        if self.name == stop_at or not hasattr(self, 'children'):
            descendants.append(self)
        else:
            if all:
                descendants.append(self)

            for child in self.children:
                child.get_descendants(descendants=descendants)

        return descendants
    
    def get_reference_calc(self, reference_period_type):
        orig_period_type = self.selected_period_type
        self.selected_period_type = reference_period_type
        reference_calc = getattr(self, f"get_{self.calc_type}")()
        self.selected_period_type = orig_period_type
        return reference_calc

    @property
    def has_reference(self):
        return hasattr(self, 'reference') and self.reference is not None
    
    @property
    def percent_change(self):
        return self.get_percent_change()
    
    def get_percent_change(self):
        # {'level': 'unit', 'reference': 'prelight'}
        percent_change = self.calc_opts.get('percent_change', {'level': 'period'})
        level = percent_change['level']
        # we are currently at a higher tree level than the % change ref level
        if level not in self.hierarchy or (
            self.hierarchy.index(self.name) < self.hierarchy.index(level)):
            return self.get_average('get_percent_change', stop_at=level)
        # we are currently at a lower tree level than the % change ref level
        elif self.hierarchy.index(self.name) > self.hierarchy.index(level):
            ref_obj = [anc for anc in self.ancestors if anc.name == level][0]
            ref = self.get_ref(ref_obj, percent_change['reference'])
        # we are currently at the % change ref level
        else: 
            ref = self.get_ref(self, percent_change['reference'])

        orig = getattr(self, f"get_{self.calc_type}")()
        return orig/np.mean(ref) * 100 - 100

    def get_ref(self, obj, reference_period_type):
        if obj.has_reference:
            return getattr(obj.reference, f"get_{self.calc_type}")
        else:
            return obj.get_reference_calc(reference_period_type)
        
    def refer(self, data, calc_type=None):
        if not calc_type:
            calc_type = self.calc_type
        if (self.calc_opts.get('evoked') 
            and hasattr(self, 'reference')
            and self.reference is not None):
            return data - self.reference.get_data(calc_type)
        else:
            return data
        
    def load(self, calc_name, other_identifiers):
        store = self.calc_opts.get('store', 'pkl')
        d = os.path.join(self.calc_opts['data_path'], self.kind_of_data)
        store_dir = os.path.join(d, f"{calc_name}_{store}s")
        for p in [d, store_dir]:
            if not os.path.exists(p):
                os.mkdir(p)
        store_path = os.path.join(store_dir, '_'.join(other_identifiers) + f".{store}")
        if os.path.exists(store_path) and not self.calc_opts.get('force_recalc'):
            with open(store_path, 'rb') as f:
                if store == 'pkl':
                    return_val = pickle.load(f)
                else:
                    return_val = json.load(f)
                return True, return_val, store_path
        else:
            return False, None, store_path

    def save(self, result, store_path):
        store = self.calc_opts.get('store', 'pkl')
        mode = 'wb' if store == 'pkl' else 'w'
        with open(store_path, mode) as f:
            if store == 'pkl':
                return pickle.dump(result, f)
            else:
                result_str = json.dumps([arr.tolist() for arr in result])
                f.write(result_str)
