import json
import os

from plotters import PeriStimulusPlotter, GroupStatsPlotter, PiePlotter, NeuronTypePlotter, MRLPlotter, LFPPlotter
from stats import Stats
from initialize_experiment import Initializer

peristimulus_plots = {
    f"plot_{data_type}": {'class': PeriStimulusPlotter, 'method': 'plot'}
    for data_type in [
        'psth', 'proportion', 'autocorrelation', 'spectrum', 'cross_correlation,' 'autocorrelogram'
    ]}

mrl_procs = {meth: {'class': MRLPlotter, 'method': meth} for meth in 
             ['mrl_bar_plot', 'mrl_rose_plot', 'mrl_heat_map']}

lfp_procs = {f'plot_{meth}': {'class': LFPPlotter, 'method': f'plot_{meth}'} for meth in 
             ['power', 'coherence', 'coherence_over_frequencies', 'spectrogram', 'correlation', 
              'max_correlations']}

other_procedures = {
    'plot_group_stats': {'class': GroupStatsPlotter, 'method': 'plot_group_stats'},
    'make_spreadsheet': {'class': Stats, 'method': 'make_df', 'follow_up': 'make_spreadsheet'},
    'unit_upregulation_pie_chart': {'class': PiePlotter, 'method': 'unit_upregulation_pie_chart'},
    'neuron_type_scatterplot': {'class': NeuronTypePlotter, 'method': 'scatterplot'},
    'plot_waveforms': {'class': NeuronTypePlotter, 'method': 'plot_waveforms'}
}

PROCEDURE_DICT = {**peristimulus_plots, **mrl_procs, **lfp_procs, **other_procedures}


class Runner:

    def __init__(self, config_file=None, lfp=False, behavior=False):
        self.config = config_file if config_file else os.getenv('INIT_CONFIG')
        self.initializer = Initializer(self.config)
        self.experiment = self.initializer.init_experiment()
        self.lfp = lfp
        self.behavior = behavior
        self.executing_class = None
        self.executing_instance = None
        self.executing_instances = {}
        self.executing_method = None
        self.loop_lists = {}
        self.follow_up_method = None
        self.data_opts = None
        self.graph_opts = None
        self.proc_name = None
        self.current_data_opts = None
        self.preparatory_method = None
        self.data_class_kwargs = {dc: getattr(self.initializer, f"init_{dc}_experiment")() 
                                  for dc in ['lfp', 'behavior'] if getattr(self, dc)}
        self.loop_lists = []
        self.modified_keys = set()

    def load_analysis_config(self, opts):
        if isinstance(opts, str):
            try:
                with open(opts, 'r', encoding='utf-8') as file:
                    data = file.read()
                    opts = json.loads(data)
            except FileNotFoundError:
                raise Exception(f"File not found: {opts}")
            except json.JSONDecodeError:
                raise Exception(f"Error decoding JSON from the file: {opts}")
        if isinstance(opts, list):
            self.data_opts = opts
        else:
            self.data_opts = opts.get('data_opts', {})
            self.graph_opts = opts.get('graph_opts', None)
            self.proc_name = opts.get('method')

    def prepare(self):
        self.executing_class = PROCEDURE_DICT[self.proc_name]['class']
        if self.executing_class.__name__ in self.executing_instances:
            self.executing_instance = self.executing_instances[self.executing_class.__name__]
        else:
            self.executing_instance = self.executing_class(self.experiment, 
                                                           **self.data_class_kwargs)
        method = PROCEDURE_DICT[self.proc_name].get('method')
        if method is None:
            method = self.proc_name
        self.executing_method = getattr(self.executing_instance, method)
        self.follow_up_method = PROCEDURE_DICT[self.proc_name].get('follow_up')

    def prepare_prep(self):
        if self.data_opts['data_class'] == 'spike':
            self.executing_instance = self.experiment
        else:
            self.executing_instance = self.data_class_kwargs[self.data_opts['data_class']]
        self.executing_method = getattr(self.executing_instance, self.proc_name)

    def get_loop_lists(self):
        for opt_list_key in ['brain_regions', 'frequency_bands', 'levels', 'unit_pairs', 
                             'neuron_qualities', 'inclusion_rules', 'region_sets']:
            opt_list = self.current_data_opts.get(opt_list_key)
            if opt_list is not None:
                self.loop_lists.append((opt_list_key, opt_list))

    def iterate_loop_lists(self, remaining_lists, current_index=0):
        if current_index >= len(self.loop_lists):
            self.execute_current_configuration()
            return  # Exit condition for recursion

        opt_list_key, opt_list = self.loop_lists[current_index]
        for opt in opt_list:
            key = opt_list_key[:-1] if opt_list_key != 'neuron_qualities' else 'neuron_quality'
            self.current_data_opts[key] = opt
            if self.current_data_opts.get('rules') and key in self.current_data_opts['rules']:
                self.apply_rules_based_on_key(key)
            
            if self.modified_keys:
                for key in self.modified_keys:
                    opt_list = self.current_data_opts.get(key)
                    remaining_lists.append((key, opt_list))
                      # Re-fetch loop lists if modifications have occurred
                self.modified_keys.clear()  # Reset modification tracker after handling
                self.iterate_loop_lists(remaining_lists, current_index=current_index+1)  # Restart this level with updated lists
                break  # Break the current for-loop to avoid redundant executions
            else:
                self.iterate_loop_lists(remaining_lists, current_index+1)  # Recurse to the next list

    def apply_rules(self):
        for key in self.current_data_opts['rules']:
            self.apply_rules_based_on_key(key)

    def apply_rules_based_on_key(self, key):
        for test, update_info in self.current_data_opts['rules'].get(key, {}).items():
            if self.current_data_opts[key] == test:
                for update_key, update_value in update_info:
                    self.current_data_opts[update_key] = update_value
                    if update_key in ['brain_regions', 'frequency_bands', 'levels', 'unit_pairs', 
                                      'neuron_qualities', 'inclusion_rules', 'region_sets']:
                        self.modified_keys.add(update_key)

    def validate(self):
        # TODO: update animal selection validation
        all_animal_ids = [animal.identifier for animal in self.experiment.all_animals]
        selected_animals = self.current_data_opts.get('selected_animals')
        if selected_animals is not None and not all([id in all_animal_ids for id in selected_animals]):
            raise ValueError("Missing animals")
        if self.current_data_opts['data_class'] == 'spike' and self.current_data_opts.get('adjustment') != 'none':
            if self.current_data_opts.get('evoked'):
                raise ValueError("It does not make sense to set 'evoked' to True and 'adjustment' to anything other "
                                 "than 'none'.  See Analysis Configuration Reference.")
            if not self.current_data_opts.get('periods'):
                raise ValueError("You picked a value for adjustment other than 'none' and haven't specified which "
                                 "periods to include.  This will result in a nonsensical result.  See the Analysis "
                                 "Configuration Reference.")

    def execute_current_configuration(self):
        self.validate()
        print(f"executing {self.executing_method} with options {self.current_data_opts}")
        if self.graph_opts is not None:
            self.executing_method(self.current_data_opts, self.graph_opts)
        else:
            self.executing_method(self.current_data_opts)

    def run_all(self):

        opts_list = self.data_opts if isinstance(self.data_opts, list) else [self.data_opts]
            
        for opts in opts_list:
            self.current_data_opts = opts
            self.get_loop_lists()
            if self.loop_lists:
                self.iterate_loop_lists(self.loop_lists)
            else:
                if self.current_data_opts.get('rules'):
                    self.apply_rules()
                self.execute_current_configuration()

    def run(self, proc_name, opts, *args, prep=None, **kwargs):
        
        if prep:
            self.load_analysis_config(prep)
            self.prepare_prep()
            self.run_all()
        self.load_analysis_config(opts)
        self.proc_name = proc_name
        self.prepare()
        self.run_all()
        if self.follow_up_method is not None:
            getattr(self.executing_instance, self.follow_up_method)(*args, **kwargs)