from pathlib import Path
import torch
import os
import numpy as np
import shutil
from kilosort import run_kilosort
from kilosort.io import save_probe, load_probe

root_dir = Path(r'D:\back_up_lenovo\data\Single_Cell_Data_No_Uv_KS4_Real_YC')

animals = ['160', '163', '176', '178', '180', '154', '156', '158', '175', '177', '179']

probe_dictionary = {
    'chanMap': np.array([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    'xc': np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]),
    'kcoords': np.zeros(14),
    'n_chan': 16,
    'yc': [163, 176, 48, 201, 61, 214, 86, 229, 99, 242, 124, 267, 137, 280]
}

save_probe(probe_dictionary, os.path.join(root_dir, 'probe.json'))
probe = load_probe(os.path.join(root_dir, 'probe.json'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


settings = {'n_chan_bin': 16, 'data_dtype': 'float64', 'device': device}

for animal in animals:
    data_binary = root_dir / f'IG{animal}' / 'data_binary.bin'
    settings['data_dir'] = data_binary.parent
    if os.path.isdir(root_dir / f'IG{animal}' / 'kilosort4'):
        shutil.rmtree(root_dir / f'IG{animal}' / 'kilosort4')
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = run_kilosort(settings=settings, probe=probe)

    

