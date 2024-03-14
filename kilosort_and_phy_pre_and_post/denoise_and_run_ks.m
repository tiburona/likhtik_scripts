dbstop if error

addpath(genpath('C:\Users\Katie'));
rmpath('C:\Users\Katie\likhtik_scripts\archive\')
rmpath('C:\Users\Katie\software\archive\')

mice = {'160', '163', '176', '178', '180', '154', ...
    '156', '158', '175', '177', '179'};

%mice = {'179'};
rootdir = 'D:\back_up_lenovo\data\Single_Cell_Data_No_Uv_Diff_Scale';

name = 'Ch14';
chanMap = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
chanMap0ind = chanMap-1;
connected = ones(14, 1);
kcoords = ones(14, 1);
xcoords = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];
ycoords = [163, 176, 48, 201, 61, 214, 86, 229, 99, 242, 124, 267, 137, 280];
save(fullfile(rootdir, 'chan14.mat'), 'name', 'chanMap', 'chanMap0ind', ...
    'connected', 'kcoords', 'xcoords', 'ycoords');




for mouse = 1:length(mice)

    datadir = fullfile(rootdir, ['IG' mice{mouse}]);
    fpath = fullfile(datadir, 'Safety.ns6');
    NS6 = openFile(datadir, 'Safety.ns6');
    denoisedNS6 = NS6;
    denoisedNS6.Data = removeLineNoise_SpectrumEstimation(NS6.Data, 30000, 'NH=5, LF=60');
    writeBinary(datadir, denoisedNS6, []);
    main_kilosort2_ks(datadir, rootdir, 'Chan14.mat');
end

function writeBinary(datadir, Data, exclude_electrodes)
    fid = fopen(fullfile(datadir, 'data_binary.bin'), 'w+');
    electrodenumbers=vertcat(Data.ElectrodesInfo.ElectrodeID);
    fwrite(fid, Data.Data(~ismember(electrodenumbers,exclude_electrodes),:) ,'int16');
    fclose(fid);
end

function Data = openFile(datadir, fname)
    fpath = fullfile(datadir, fname);
    Data = openNSx('read', fpath);
end

