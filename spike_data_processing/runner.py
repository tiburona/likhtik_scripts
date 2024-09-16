import json
import os

from matplotlib.pylab import cond

from plotters import PeriStimulusHistogramPlotter, PiePlotter, NeuronTypePlotter, MRLPlotter, LFPPlotter
from stats import Stats
from initialize_experiment import Initializer

peristimulus_plots = {
    f"plot_{calc_type}": {'class': PeriStimulusHistogramPlotter, 'method': 'plot'}
    for calc_type in [
        'psth', 'proportion', 'autocorrelation', 'spectrum', 'cross_correlation', 'autocorrelogram'
    ]}

mrl_procs = {meth: {'class': MRLPlotter, 'method': meth} for meth in 
             ['mrl_bar_plot', 'mrl_rose_plot', 'mrl_heat_map', 'plot_phase_phase_over_frequencies',
              'make_phase_phase_rose_plot', 'make_phase_phase_trace_plot']}

lfp_procs = {f'plot_{meth}': {'class': LFPPlotter, 'method': f'plot_{meth}'} for meth in 
             ['power', 'coherence', 'coherence_over_frequencies', 'spectrogram', 'correlation', 
              'max_correlations', 'granger']}

other_procedures = {
    #'plot_group_stats': {'class': GroupStatsPlotter, 'method': 'plot_group_stats'},
    'make_spreadsheet': {'class': Stats, 'method': 'make_df', 'follow_up': 'make_spreadsheet'},
    'unit_upregulation_pie_chart': {'class': PiePlotter, 'method': 'unit_upregulation_pie_chart'},
    'neuron_type_scatterplot': {'class': NeuronTypePlotter, 'method': 'scatterplot'},
    'plot_waveforms': {'class': NeuronTypePlotter, 'method': 'plot_waveforms'}
}

PROCEDURE_DICT = {**peristimulus_plots, **mrl_procs, **lfp_procs, **other_procedures}


class Runner:

    def __init__(self, config_file=None):
        self.config = config_file if config_file else os.getenv('INIT_CONFIG')
        self.initializer = Initializer(self.config)
        self.experiment = self.initializer.init_experiment()
        self.executing_class = None
        self.executing_instance = None
        self.executing_instances = {}
        self.executing_method = None
        self.loop_lists = {}
        self.follow_up_method = None
        self.calc_opts = None
        self.graph_opts = None
        self.proc_name = None
        self.current_calc_opts = None
        self.preparatory_method = None

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
            self.calc_opts = opts
        else:
            self.calc_opts = opts.get('calc_opts', {})
            self.graph_opts = opts.get('graph_opts', None)
            self.proc_name = opts.get('method')

    def prepare(self):
        self.executing_class = PROCEDURE_DICT[self.proc_name]['class']
        if self.executing_class.__name__ in self.executing_instances:
            self.executing_instance = self.executing_instances[self.executing_class.__name__]
        else:
            self.executing_instance = self.executing_class(self.experiment)
        method = PROCEDURE_DICT[self.proc_name].get('method')
        if method is None:
            method = self.proc_name
        self.executing_method = getattr(self.executing_instance, method)
        self.follow_up_method = PROCEDURE_DICT[self.proc_name].get('follow_up')

    def get_loop_lists(self):
        for opt_list_key in ['brain_regions', 'frequency_bands', 'levels', 'unit_pairs', 
                             'neuron_qualities', 'inclusion_rules', 'region_sets']:
            opt_list = self.current_calc_opts.get(opt_list_key)
            if opt_list is not None:
                self.loop_lists[opt_list_key] = opt_list

    def iterate_loop_lists(self, remaining_loop_lists, current_index=0):
        if current_index >= len(remaining_loop_lists):
            self.execute()
            return
        opt_list_key, opt_list = remaining_loop_lists[current_index]
        for opt in opt_list:
            key = opt_list_key[:-1] if opt_list_key != 'neuron_qualities' else 'neuron_quality'
            self.current_calc_opts[key] = opt
            self.iterate_loop_lists(remaining_loop_lists, current_index + 1)

    def apply_rules(self):
        rules = self.current_calc_opts['rules']
        if isinstance(rules, dict):
            for rule in rules:
                self.assign_per_rule(*self.parse_natural_language_rule(rule))
        else:          
            # Assuming rules is a dictionary like: {'calc_type': {'mrl': [('time_type', 'block')]}}
            for trigger_k, conditions in rules.items():
                for trigger_v, target_vals in conditions.items():
                    for target_k, target_v in target_vals:
                        self.assign_per_rule(trigger_k, trigger_v, target_k, target_v)
    
    def assign_per_rule(self, trigger_k, trigger_v, target_k, target_v):
        if trigger_k not in self.current_calc_opts:
            raise ValueError(f"Key '{trigger_k}' not found in calc_opts")
        if self.current_calc_opts[trigger_k] == trigger_v:
            self.current_calc_opts[target_k] = target_v

    def parse_natural_language_rule(self, rule):
        # assuming rule is a tuple like ('if brain_region is bla', frequency_band is', 'theta')
        # or ('if brain_region is bla, frequency_bands are',  ['theta_1', 'theta_2'])
        string, target_val = rule
        split_string = '_'.split(string)
        trigger_key = split_string[1]
        trigger_val = split_string[3][:-1]
        target_key = split_string[4]
        return trigger_key, trigger_val, target_key, target_val
        
    def validate(self):
        # TODO: update animal selection validation
        all_animal_ids = [animal.identifier for animal in self.experiment.all_animals]
        selected_animals = self.current_calc_opts.get('selected_animals')
        if selected_animals is not None and not all([id in all_animal_ids for id in selected_animals]):
            raise ValueError("Missing animals")
        if self.current_calc_opts['kind_of_data'] == 'spike' and self.current_calc_opts.get('adjustment') != 'none':
            if self.current_calc_opts.get('evoked'):
                raise ValueError("It does not make sense to set 'evoked' to True and 'adjustment' to anything other "
                                 "than 'none'.  See Analysis Configuration Reference.")
            if not self.current_calc_opts.get('periods'):
                raise ValueError("You picked a value for adjustment other than 'none' and haven't specified which "
                                 "periods to include.  This will result in a nonsensical result.  See the Analysis "
                                 "Configuration Reference.")

    def execute(self):
        if self.current_calc_opts.get('rules'):
            self.apply_rules()
        self.validate()
        print(f"executing {self.executing_method} with options {self.current_calc_opts}")
        if self.graph_opts is not None:
            self.executing_method(self.current_calc_opts, self.graph_opts)
        else:
            self.executing_method(self.current_calc_opts)

    def run_all(self):

        opts_list = self.calc_opts if isinstance(self.calc_opts, list) else [self.calc_opts]
            
        for opts in opts_list:
            self.current_calc_opts = opts
            self.get_loop_lists()
            if self.loop_lists:
                self.iterate_loop_lists(list(self.loop_lists.items()))
            else:
                self.execute()

    def run(self, proc_name, opts, *args, prep=None, **kwargs):
        
        if prep:
            self.load_analysis_config(prep)
            self.executing_method = getattr(self.experiment, self.proc_name)
            self.run_all()
            self.loop_lists = {}
        self.load_analysis_config(opts)
        self.proc_name = proc_name
        self.prepare()
        self.run_all()
        if self.follow_up_method is not None:
            getattr(self.executing_instance, self.follow_up_method)(*args, **kwargs)
