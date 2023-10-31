dbstop if error

addpath(genpath('/Users/katie/likhtik/software'));
addpath(genpath('/Users/katie/likhtik/likhtik_scripts'));


mice = {'155', '162', '171', '173', '174', '175'};

toneOffCode = 65535;
toneOnCode = 65502;

clear data

for i=1:length(mice)
   clear animal_data

    mousedir = ['/Users/Katie/likhtik/data/single_cell_data/IG' mice{i}];
    animal_data.NEV =  openNEV('read', fullfile(mousedir, 'Safety.nev'));
    save(['/Users/katie/likhtik/data/events/animal_data_' mice{i} '.mat'],  '-struct', 'animal_data', '-v7.3');
    if str2num(mice{i}) > 165; toneOnCode = 65436; end
    data(i) = populate_data_struct_no_spikes(mousedir, animal_data, toneOnCode, toneOffCode);
end

save('/Users/katie/likhtik/data/hpc_power_test_data.mat', 'data');






