from pathlib import Path
import torch
import os
import numpy as np
from kilosort import run_kilosort
from kilosort.io import save_probe, load_probe

root_dir = Path(r'D:\back_up_lenovo\data\Single_Cell_Data_No_Uv_KS4')

animals = ['160', '163', '176', '178', '180', '154', '156', '158', '175', '177', '179']

probe_dictionary = {
    'chanMap': np.array([1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
    'xc': np.ones(14),
    'kcoords': np.zeros(14),
    'n_chan': 16,
    'yc': np.array([.09, .1, .03, .11, .04, .12, .05, .13, .06, .14, .07, .15, .08, .16])
}
save_probe(probe_dictionary, os.path.join(root_dir, 'probe.json'))
probe = load_probe(os.path.join(root_dir, 'probe.json'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


settings = {'n_chan_bin': 16, 'data_dtype': 'float64', 'device': device}

for animal in animals:
    data_binary = root_dir / f'IG{animal}' / 'data_binary.bin'  # Corrected path representation
    settings['data_dir'] = data_binary.parent
    ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate = run_kilosort(settings=settings, probe=probe)

    

