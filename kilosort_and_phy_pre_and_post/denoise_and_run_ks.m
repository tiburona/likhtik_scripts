dbstop if error

addpath(genpath('C:\Users\Katie'));
rmpath('C:\Users\Katie\likhtik_scripts\archive\')
rmpath('C:\Users\Katie\software\archive\')

mice = {'160', '163', '176', '178', '180', '154', ...
    '156', '158', '175', '177', '179'};

for mouse = 1:length(mice)

    datadir = ['C:\Users\Katie\data\Single_Cell_Data\IG' mice{mouse}];
    fpath = fullfile(datadir, 'Safety.ns6');
    NS6 = openFile(datadir, 'Safety.ns6');
    denoisedNS6 = NS6;
    denoisedNS6.Data = removeLineNoise_SpectrumEstimation(NS6.Data, 30000);
    writeBinary(datadir, denoisedNS6, []);
end

function writeBinary(datadir, Data, exclude_electrodes)
    fid = fopen(fullfile(datadir, 'data_binary.bin'), 'w+');
    electrodenumbers=vertcat(Data.ElectrodesInfo.ElectrodeID);
    fwrite(fid, Data.Data(~ismember(electrodenumbers,exclude_electrodes),:) ,'int16');
    fclose(fid);
end

function Data = openFile(datadir, fname)
    fpath = fullfile(datadir, fname);
    Data = openNSx('read', fpath, 'uv');
end

