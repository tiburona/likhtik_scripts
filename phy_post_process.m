dbstop if error

addpath(genpath('C:\Users\Katie'));
rmpath('C:\Users\Katie\likhtik_scripts\archive\')
rmpath('C:\Users\Katie\software\archive\')


mice = {'154',  '156', '158', '160', '161', '163', '175', '176', '177',...
    '178', '179', '180'};

toneOffCode = 65535;
toneOnCode = 65502;

clear data
for i=1:length(mice)
   clear animal_data
    mousedir = ['C:\Users\Katie\data\archive\Single_Cell_Data\IG' mice{i}];
    animal_data.NS6 = openNSx('read', fullfile(mousedir, 'Safety.ns6'), 'uv');
    animal_data.NEV =  openNEV('read', fullfile(mousedir, 'Safety.nev'));
    save(['C:\Users\Katie\data\events\animal_data_' mice{i} '.mat'],  '-struct', 'animal_data', '-v7.3');
    if str2num(mice{i}) > 165; toneOnCode = 65436; end
    data(i) = process_post_phy(mousedir, animal_data, toneOnCode, toneOffCode);
end






