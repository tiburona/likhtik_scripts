dbstop if error
% pyenv('Version', '/usr/bin/python3')

clear group_means

% Define the directory path
directoryPath = '/Users/katie/likhtik/IG_INED_Safety_Recall/lfp/spectrogram_jsons';


ctrl = {'IG162';'IG171';'IG173';'IG176'};


def = {'IG155'; 'IG174';'IG175';'IG179'};

mice_with_no_light = {'IG175'; 'IG176'; 'IG177'; 'IG178'; 'IG179'; 'IG180'};

all = vertcat(ctrl, def);

data_from_beginning = struct();

num_time_points = 30;

for i = 1:length(all) 
    data_from_beginningDir = fullfile('/Users/katie/likhtik/IG_INED_Safety_Recall', all{i});
    NS3 = openNSx('read', fullfile(data_from_beginningDir, [all{i} '.ns3']), 'uv');
    raw = NS3.Data(1,:);
    filtered = removeLineNoise_SpectrumEstimation(raw, 2000, ['NH=5','LF=60']);
    root_mean_square = (mean(arrayfun(@(x) x^2, filtered)))^.5;
    rms = arrayfun(@(x) x/root_mean_square, filtered);
    NEV = openNEV(fullfile(data_from_beginningDir, [all{i} '.nev']));
    timeStamps = NEV.Data.SerialDigitalIO.TimeStamp;
    unparsedData = NEV.Data.SerialDigitalIO.UnparsedData;
    if ismember(all{i}, mice_with_no_light)
        toneOnCode = 65436;
    else
        toneOnCode = 65502;
    end
    disp(toneOnCode)
    onTimeStamps = [];
    for j=1:length(timeStamps)
        if unparsedData(j) == toneOnCode
            onTimeStamps = [onTimeStamps, timeStamps(j)];
        end
    end

    onTimeStamps = arrayfun(@(x) int32(x * 1/15), onTimeStamps);
    
    
    data_from_beginning.("hpc").(all{i}).raw = raw;
    data_from_beginning.("hpc").(all{i}).filtered = filtered;
    data_from_beginning.("hpc").(all{i}).rms = rms;
    data_from_beginning.("hpc").(all{i}).timeStamps = timeStamps;
    data_from_beginning.("hpc").(all{i}).unparsedData = unparsedData; 
    data_from_beginning.("hpc").(all{i}).onTimeStamps = onTimeStamps; 

    for k=1:length(onTimeStamps)
        period_data_from_beginning = rms(onTimeStamps(k) - 2000:onTimeStamps(k)+62000-1);
        [power, frequencies, timeBins] = mtcsg(period_data_from_beginning, 2048, 2000, 1000, 980, 2);
        data_from_beginning.("hpc").(all{i}).("tone")(k).input = period_data_from_beginning;
        data_from_beginning.("hpc").(all{i}).("tone")(k).power = power; 
        data_from_beginning.("hpc").(all{i}).("tone")(k).frequencies = frequencies;
        data_from_beginning.("hpc").(all{i}).("tone")(k).timeBins = timeBins;
        selected_rows = (frequencies >= 0) & (frequencies <= 15.2);
        power = power(:, 76:end-75);
        selected_data_from_beginning = power(selected_rows, :);
        data_from_beginning.("hpc").(all{i}).("tone")(k).selected_data_from_beginning = selected_data_from_beginning;

        column_sets_from_beginning = cell(1, 30);

        for l = 1:30
            start_column = 1 + (l - 1) * 100;
            end_column = start_column + num_time_points - 1;
            column_sets_from_beginning{l} = selected_data_from_beginning(:, start_column:end_column);
        end
        data_from_beginning.("hpc").(all{i}).("tone")(k).pip_data = column_sets_from_beginning;
        column_sets_from_beginning_mat = cell2mat(column_sets_from_beginning);
        reshaped_column_sets_from_beginning = reshape(column_sets_from_beginning_mat, [16, num_time_points, 30]);
        mean_matrix = squeeze(mean(reshaped_column_sets_from_beginning, 3));
        data_from_beginning.("hpc").(all{i}).("tone")(k).pip_mean =  mean_matrix;
    end

end

groups = {ctrl; def};
group_means = cell(2, 1);

for k = 1:length(groups)
    group_data_from_beginning = cellfun(@(id) data_from_beginning.("hpc").(id).("tone")(1).pip_mean, groups{k}, 'uni', 0);
    mat_group_data_from_beginning = cell2mat(group_data_from_beginning');
    reshaped_mat = reshape(mat_group_data_from_beginning, [16, num_time_points, 4]);
    group_means{k} = squeeze(mean(reshaped_mat, 3));
end

% Display the matrix as a heatmap
imagesc(group_means{2}); % defeat

% Apply the 'jet' colormap
colormap('jet');

% Add a colorbar for reference
colorbar;

set(gca, 'YDir', 'normal');
