addpath(genpath('/Users/katie/likhtik'))

CH084_nev.NEV =  openNEV('read', fullfile('/Users/katie/Downloads/CH084', 'EXTREC002.nev'));

CH087_nev.NEV = openNEV('read', fullfile('/Users/katie/Downloads/CH087', 'EXTREC001.nev'));


CH084_2_nev.NEV = openNEV('read', fullfile('/Users/katie/Downloads/CH084_2', 'EXT001.nev'));


CH087_2_nev.NEV = openNEV('read', fullfile('/Users/katie/Downloads/CH087_2', 'EXT003.nev'));

CH085_nev.Data = openNEV('read', fullfile('/Users/katie/Downloads/CH085', 'EXT002.nev'));

CH070_nev.Data = openNEV('read', fullfile('/Users/katie/Downloads/CH070', 'EXTREC.nev'));
