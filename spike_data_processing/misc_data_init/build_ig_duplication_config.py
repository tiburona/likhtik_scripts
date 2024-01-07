import json
import os
import shutil

root = '/Users/katie/likhtik/IG_INED_Safety_Recall'


defeat_ig_st = ['IG154', 'IG155', 'IG156', 'IG158', 'IG175', 'IG179']
control_ig_st = ['IG160', 'IG161', 'IG162', 'IG163', 'IG176', 'IG178', 'IG180']

animal_dirs = os.listdir(root)

single_cell_dir = '/Users/katie/likhtik/data/single_cell_data'

for animal in defeat_ig_st + control_ig_st:
    new_animal_dir = os.path.join(root, animal)
    if animal not in animal_dirs:
        os.mkdir(new_animal_dir)
    animal_files = os.listdir(new_animal_dir)
    for ext in ['.ns6', '.nev', '.mat', '.ns3']:
        if all([ext not in f for f in animal_files]):
            src = os.path.join(single_cell_dir, animal, 'Safety' + ext)
            dst = os.path.join(root, animal, animal + ext)
            shutil.copyfile(src, dst)

ined_lfp_electrodes = {'bla': 1, 'bf': 2, 'pl': 3}

control_ined = ['INED18', 'INED17', 'INED16', 'INED05', 'INED04']
defeat_ined = ['INED06', 'INED07', 'INED09', 'INED11', 'INED12']

control_ined_dict = {animal: {'lfp_electrodes': ined_lfp_electrodes, 'condition': 'control'} for animal in control_ined}
defeat_ined_dict = {animal: {'lfp_electrodes': ined_lfp_electrodes, 'condition': 'defeat'} for animal in control_ined}

pl_electrodes = {
    'IG154': (4, 6), 'IG155': (12, 14), 'IG156': (12, 14), 'IG158': (7, 14), 'IG160': (1, 8), 'IG161': (9, 11),
    'IG162': (13, 3), 'IG163': (14, 8), 'IG175': (15, 4), 'IG176': (11, 12), 'IG177': (15, 4), 'IG178': (6, 14),
    'IG179': (13, 15), 'IG180': (15, 4)
}


ig_electrodes = {'hpc': 0, 'bla': 2}

control_ig_dict, defeat_ig_dict = ({
    animal: {'condition': condition, 'lfp_electrodes': ig_electrodes,
             'lfp_from_stereotrodes': {'nsx_num': 6, 'electrodes': {'pl': pl_electrodes[animal]}}}
    for animal in animals} for condition, animals in [('control', control_ig_st), ('defeat', defeat_ig_st)])

no_st_electrodes = {'hpc': 0, 'bla': 1, 'pl': 3}
control_ig_no_st = ['IG171', 'IG173']
defeat_ig_no_st = ['IG172', 'IG174']
control_ig_no_st_dict = {animal: {'condition': 'control', 'lfp_electrodes': no_st_electrodes} for animal in control_ig_st}
defeat_ig_no_st_dict = {animal: {'condition': 'defeat', 'lfp_electrodes': no_st_electrodes} for animal in defeat_ig_st}

animal_info = {**control_ined_dict, **defeat_ined_dict, **control_ig_dict, **defeat_ig_dict, **control_ig_no_st_dict}


exp_info = {
    'conditions': ['defeat', 'control'],
    'identifier': 'IG_INED_Safety_Recall_Duplication',
    'neuron_types': [],
    'sampling_rate': 30000,
    'lfp_sampling_rate': 2000,
    'lfp_root': root,
    'lfp_path_constructor': ['identifier'],
    'lost_signal': .5
}

mice_with_no_light = ['IG175', 'IG176', 'IG177', 'IG178', 'IG179', 'IG180']


animals = []

for animal in animal_info:
    animal_file = os.path.join(root, animal, animal + '.json')
    tone_on_code = 65502 if animal not in mice_with_no_light else 65436
    with open(animal_file, 'r', encoding='utf-8') as file:
        data = file.read()
        json_data = json.loads(data)
        if animal == 'INED18':
            a = 'foo'
        time_stamps = json_data['NEV']['Data']['SerialDigitalIO']['TimeStamp']
        unparsed_data = json_data['NEV']['Data']['SerialDigitalIO']['UnparsedData']
        tone_onsets = [ts for i, ts in enumerate(time_stamps) if unparsed_data[i] == tone_on_code]
        events = [[onset + i * 30000 for i in range(30)] for onset in tone_onsets]
        pretone = {'reference': True, 'target': 'tone', 'shift': 30, 'duration': 30, 'lfp_padding': [1, 1]}
        tone = {'onsets': tone_onsets, 'events': events, 'duration': 30, 'lfp_padding': [1, 1],
                'event_duration': 1}
        animals.append({'identifier': animal, 'block_info': {'tone': tone, 'pretone': pretone}, **animal_info[animal]})

exp_info['animals'] = animals

with open(os.path.join(root, 'init_config.json'), 'w', encoding='utf-8') as file:
    json.dump(exp_info, file, ensure_ascii=False, indent=4)







