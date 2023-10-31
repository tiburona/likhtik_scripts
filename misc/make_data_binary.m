dbstop if error

addpath(genpath('/Users/katie/likhtik'));

mice = {'IG162','IG171','IG173','IG176', 'IG155', 'IG174','IG175','IG179'};

for mouse = 1:length(mice)

    dir_path = ['/Users/katie/likhtik/data/single_cell_data' mice{mouse}];
    files = dir(fullfile(dir_path, '*.ns6')); 
    
    % Check if files is not empty
    if isempty(files)
        disp(['No .ns6 files found in directory: ', dir_path]);
        continue;
    end
    
    % Copy the .ns6 file to Safety.ns6
    selected_file = fullfile(dir_path, files(1).name);
    safety_file = fullfile(dir_path, 'Safety.ns6');
    copyfile(selected_file, safety_file);
    disp(['Copied ', files(1).name, ' to Safety.ns6 in directory: ', dir_path]);
    
%     NS6 = openFile(selected_file);
%     denoisedNS6 = NS6;
%     denoisedNS6.Data = removeLineNoise_SpectrumEstimation(NS6.Data, 30000);
%     
%     % Check if 'data_binary.bin' does not exist before writing
%     if ~isfile(fullfile(dir_path, 'data_binary.bin'))
%         writeBinary(dir_path, denoisedNS6, []);
%     else
%         disp(['data_binary.bin already exists in directory: ', dir_path]);
%    end
end

function writeBinary(datadir, Data, exclude_electrodes)
    fid = fopen(fullfile(datadir, 'data_binary.bin'), 'w+');
    electrodenumbers=vertcat(Data.ElectrodesInfo.ElectrodeID);
    fwrite(fid, Data.Data(~ismember(electrodenumbers,exclude_electrodes),:) ,'int16');
    fclose(fid);
end

function Data = openFile(fpath)
    Data = openNSx('read', fpath, 'uv');
end


