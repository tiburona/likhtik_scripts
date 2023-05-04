dbstop if error

addpath(genpath('/Users/katie/likhtik'));
rmpath('/Users/katie/likhtik/likhtik_scripts/archive')
rmpath('/Users/katie/likhtik/software/archive')


mice = {'154',  '156', '158', '160', '161', '163', '175', '176', '177',...
    '178', '179', '180'};

toneOffCode = 65535;
toneOnCode = 65502;

clear data
for i=1:length(mice)
   clear animal_data
    mousedir = ['/Users/Katie/likhtik/data/single_cell_data/IG' mice{i}];
    animal_data.NS6 = openNSx('read', fullfile(mousedir, 'Safety.ns6'), 'uv');
    animal_data.NEV =  openNEV('read', fullfile(mousedir, 'Safety.nev'));
    save(['/Users/katie/likhtik/data/events/animal_data_' mice{i} '.mat'],  '-struct', 'animal_data', '-v7.3');
    if str2num(mice{i}) > 165; toneOnCode = 65436; end
    data(i) = process_post_phy(mousedir, animal_data, toneOnCode, toneOffCode);
end







