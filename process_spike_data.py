import scipy.io as sio

from graph_utils import init_animal
from spike_data import Experiment, Group, Animal

base_opts = {'graph_dir': '/Users/katie/likhtik/data/graphs', 'units_in_fig': 4}
psth_opts = {**base_opts, **{'data_type': 'psth', 'pre_stim': 0.05, 'post_stim': 0.65, 'bin_size': 0.01,
                             'trials': (0, 150), 'tick_step': 0.1}}
autocorr_opts = {**base_opts, **{'data_type': 'autocorr', 'pre_stim': 0.0, 'post_stim': 30.0, 'bin_size': 0.1,
                                 'trials': (0, 150, 30), 'lags': 99, 'tick_step': .1}}
frequency_opts = {**autocorr_opts, **{'data_type': 'spectrum', 'bin_size': 0.1, 'lags': 99, 'up_to_hz': 25,
                                      'tick_step': .1}}

mat_contents = sio.loadmat('/Users/katie/likhtik/data/single_cell_data.mat')['single_cell_data']
animals = [init_animal(entry, Animal, Unit) for entry in mat_contents[0]]
experiment = Experiment({name: Group(name=name, animals=[animal for animal in animals if animal.condition == name])
                         for name in ['control', 'stressed']})

for opts_type in [autocorr_opts, frequency_opts]:
    experiment.plot_groups(opts_type)
    for group in experiment.groups:
        group.plot_animals(opts_type)
        for animal in group.animals:
            animal.plot_units(opts_type)
            