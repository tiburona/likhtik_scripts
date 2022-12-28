dbstop if error

addpath(genpath('C:\Users\Likhtik Lab\katie'));

datadir = 'C:\Users\Likhtik Lab\katie\data\155';
fname = 'IG 155 - Recall - CtxtA - Safety.ns6.ns6';
chanmapfile = 'Chan14.mat';

NS6 = openNSx('read', fullfile(datadir, fname), 'uv');
denoisedNS6 = NS6;
denoisedNSG.Data = removeLineNoise_SpectrumEstimation(NS6.Data, 30000);

fid = fopen(fullfile(datadir, 'data_binary.bin'), 'w');
fwrite(fid, int16(denoisedNS6.Data), 'int16');
fclose(fid);

run_kilosort_onfile_ks(datadir, chanmapfile)
