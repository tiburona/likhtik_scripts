dbstop if error

% Define the directory path
directoryPath = '/Users/katie/likhtik/IG_INED_Safety_Recall/lfp/spectrogram_jsons';


ctrl = {'IG162';'IG171';'IG173';'IG176'};


def = {'IG155'; 'IG174';'IG175';'IG179'};


% List all files in the directory with the specified format
files = dir(fullfile(directoryPath, 'IG*_hpc_*.json'));

% Initialize a struct to store the data_test_power
data_test_power = struct();

% Loop through the files
for i = 1:length(files)
    % Extract the file name
    [~, fileNameWithoutExt] = fileparts(files(i).name);
    
    % Split the file name on underscores
    parts = strsplit(fileNameWithoutExt, '_');
    
    % Extract the relevant variables
    animal_identifier = parts{1};
    brain_region = parts{2};
    period_type = parts{8};
    period = str2num(parts{9}) + 1;

    if (contains(ctrl, animal_identifier))
        data_test_power.(brain_region).(animal_identifier).condition = 'control';
    elseif (contains(def, animal_identifier))
        data_test_power.(brain_region).(animal_identifier).condition = 'defeat';
    end
    
    % Read the contents of the file
    filePath = fullfile(directoryPath, files(i).name);
    fileContents = jsondecode(fileread(filePath));
    
    % Extract the three lists: frequencies, time_bins, and power
    power = fileContents{1};
    frequencies = fileContents{2};
    time_bins = fileContents{3};
    
    % Perform data_test_power selection and manipulation
    % Remove 75 columns from the beginning and end of the power matrix
    power = power(:, 76:end-75);
    
    % Select rows where frequencies are between 0 and 15 with a tolerance of 0.2
    selected_rows = (frequencies >= 0) & (frequencies <= 15.2);
    selected_data_test_power = power(selected_rows, :);
  
    data_test_power.(brain_region).(animal_identifier).(period_type)(period).selected_data_test_power = selected_data_test_power;
    
    % If it's a "pretone" period_type, compute the average over time bins
    if strcmp(period_type, 'pretone')
        pretone_average = mean(selected_data_test_power, 2);
        data_test_power.(brain_region).(animal_identifier).(period_type)(period).pretone_average = pretone_average;
    end
end

column_indices = [];

num_time_points = 40;

for i = 1:100:3000
    column_indices = [column_indices, (i:i+num_time_points-1)];
end

animals = fieldnames(data_test_power.("hpc"));

for j = 1:length(animals)
    subtracted_data_test_power = data_test_power.("hpc").(animals{j}).("tone")(1).selected_data_test_power - ... 
        data_test_power.("hpc").(animals{j}).("pretone")(1).pretone_average;
    data_test_power.("hpc").(animals{j}).("tone")(1).subtracted_data_test_power = subtracted_data_test_power;

    column_sets_test_power = cell(1, 30);

% Extract the sets of columns and store them in the cell array
    for k = 1:30
        start_column = 1 + (k - 1) * 100; 
        end_column = start_column + num_time_points - 1; 
        column_sets_test_power{k} = subtracted_data_test_power(:, start_column:end_column);
    end
    column_sets_test_power_mat = cell2mat(column_sets_test_power);
    reshaped_column_sets_test_power = reshape(column_sets_test_power_mat, [16, num_time_points, 30]);
    mean_matrix = squeeze(mean(reshaped_column_sets_test_power, 3));
    data_test_power.("hpc").(animals{j}).("tone")(1).pip_mean =  mean_matrix;
end

groups = {ctrl; def};
group_means = cell(2, 1);

for k = 1:length(groups)
    group_data_test_power = cellfun(@(id) data_test_power.(brain_region).(id).("tone")(1).pip_mean, groups{k}, 'uni', 0);
    mat_group_data_test_power = cell2mat(group_data_test_power');
    reshaped_mat = reshape(mat_group_data_test_power, [16, num_time_points, 4]);
    group_means{k} = squeeze(mean(reshaped_mat, 3));
end
