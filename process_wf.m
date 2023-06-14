dbstop if error
%addpath(genpath('/Users/katie/likhtik'));
%load('/Users/katie/likhtik/data/new_data.mat');
single_cell_data = process_waveforms(single_cell_data, ...
    '/Users/katie/likhtik/data/graphs/waveforms');
single_cell_data = fr_fwhm_kmeans(single_cell_data);

scatter_plot(single_cell_data, 'FWHM_microseconds', 'firing_rate', 'FWHM (microseconds)', ...
    'Firing Rate (Hz)', 'Scatterplot of Good Units', ...
    'FWHM_and_FR_Scatterplot_with_Adjusted_Histograms_Post_Spike_Max.fig');
%scatter_plot(single_cell_data, 'trough_to_peak_microseconds', 'firing_rate', 'Trough to Peak Distance (microseconds)', ...
%    'Firing Rate (Hz)', 'Scatterplot of Good Units', ...
%    'Trough_to_Peak_and_FR_Scatterplot_with_Adjusted_Histograms_Post_Spike_Max.fig');
%scatter_plot(single_cell_data, 'FWHM_microseconds', 'firing_rate', 'FWHM (microseconds) < 1000', ...
%    'Firing Rate (Hz)', 'Scatterplot of Good Units', ...
%    'FWHM_and_FR_Scatterplot_with_Adjusted_Histograms_FWHM_less_than_1000.fig', ...
%    '<1000');


%save('single_cell_data.mat', 'single_cell_data');


function data = process_waveforms(data, graph_path)

% Constants
sampling_rate = 30000; % 30 kHz sampling rate
wfWin = [-100 200]; % window of samples to take around a spike;

% Loop through animals
for i = 1:length(data)
    animal = data(i);

    chMap = readNPY(fullfile(animal.rootdir, 'channel_map.npy'))+1; 
    
    % Open binary file for the current animal
    file_path = fullfile(animal.rootdir, 'data_binary.bin');
    fid = fopen(file_path, 'r');
    
    % Read the waveform data from the binary file
    M = memmapfile(file_path);
    M1 = memmapfile(file_path, 'Format', {'int16', [16 length(M.Data) / 16 / 2], 'data'});
    all_data = M1.Data.data(chMap, :);

    % Apply a fifth-order median filter to the electrode data
    all_data = medfilt1(double(all_data), 5, [], 2);
    
    % Loop through units
    for j = 1:length(animal.units.good)

        spike_times = animal.units.good(j).spike_times;
        spike_times = spike_times(spike_times > 100 & spike_times < size(M1.Data.data, 2) - 200);
        electrodes = animal.units.good(j).electrodes;

        % Initialize arrays to store waveforms for each channel
        all_waveforms = zeros(length(spike_times), length(electrodes), wfWin(2) - wfWin(1) + 1);

        % transformation of phy electrode labels to index in raw data
        for el = 1:length(electrodes)
            if electrodes(el) > 1
                electrodes(el) = electrodes(el) - 1;
            end
        end
    
        for k = 1:length(spike_times)
            % Calculate the start and end sample indices for the waveform
            start_sample = spike_times(k) + wfWin(1);
            end_sample = spike_times(k) + wfWin(2);

            wf_data = all_data(electrodes', start_sample:end_sample); 
            
            % Store the waveforms for each channel
            all_waveforms(k, :, :) = wf_data;
        end
           
        mean_centered_waveforms = all_waveforms - mean(all_waveforms, 3);
        
        % Compute the mean waveform for each channel
        mean_waveforms = mean(mean_centered_waveforms, 1);

        plot_waveforms(animal, j, mean_waveforms, electrodes, graph_path);
        plot_waveforms(animal, j, mean_centered_waveforms, electrodes, graph_path, 'num_samples', 30);
        
        mean_of_electrodes = mean_waveforms;
        if length(electrodes) > 1
            mean_of_electrodes = squeeze(mean(mean_waveforms, 2)); 
        end
        
        % Calculate the Full Width Half Minimum (FWHM) of the mean waveform
        half_amplitude = abs(max(mean_of_electrodes(100:135)) - min(mean_of_electrodes(75:125))) / 2; % Sometimes drift is such that the waveform gets higher than the max of the waveform later in the series.
        below_half_min = mean_of_electrodes <= (max(mean_of_electrodes(100:135)) - half_amplitude);
        FWHM_samples = sum(below_half_min(50:150));
        FWHM_time = FWHM_samples / sampling_rate;

        % Calculate trough-to-peak distance 
        [min_value, min_index] = min(mean_of_electrodes(75:125));
        [max_value, max_index] = max(mean_of_electrodes(85:150));
        trough_to_peak = abs(max_index + 84 - (min_index + 74));
        trough_to_peak_time = trough_to_peak/sampling_rate;

        animal.units.good(j).FWHM_time = FWHM_time;
        animal.units.good(j).min = min(mean_of_electrodes);
        animal.units.good(j).trough_to_peak = trough_to_peak_time;
        animal.units.good(j).trough_to_peak_microseconds = trough_to_peak_time*1000000;
        animal.units.good(j).FWHM_microseconds = FWHM_time*1000000;
        
        % Select later spikes for global firing rate calculation to try to
        % reduce early noise
        spike_times_for_fr = spike_times(spike_times > 15000);
        fr = (length(spike_times_for_fr)/(double(spike_times_for_fr(end) - spike_times_for_fr(1))))*sampling_rate;
        animal.units.good(j).firing_rate = fr; 

    end

    % Update the data struct with the new fields

    data(i) = animal;

    % Close the binary file
    fclose(fid);

end


end

function plot_waveforms(animal, unit_idx, waveforms, electrodes, pth, varargin)
    num_samples = 0;
    for k = 1:length(varargin) - 1
        if varargin{1} == 'num_samples'
            num_samples = varargin{k + 1};
            waveforms = select_rows(num_samples, waveforms);
            
        end
    end  
    
    figure;
    if length(electrodes) == 1
        plot(squeeze(waveforms(:, 1, :))');
        xlabel('Samples');
        ylabel('microvolts')
    else
        for elec_idx = 1:length(electrodes)
            subplot(length(electrodes), 1, elec_idx);
           
            plot(squeeze(waveforms(:, elec_idx, :))');

            % Only add labels and title for the bottom subplot
            if elec_idx == length(electrodes)
                xlabel('Samples');
            else
                % Remove x and y axis labels for other subplots
                set(gca, 'XTickLabel', []);
             
            end
            ylabel('microvolts')
        end
    end
    
    if num_samples
        name_str = 'Sample Waveforms';
        ext = '.fig';
    else
        name_str = 'Mean Waveform';
        ext = '.png';
    end

    % Set the overall figure title
    sgtitle(sprintf('%s %s for Unit %d', animal.animal, name_str, unit_idx));
    
    % Save and close the figure
    saveas(gcf, fullfile(pth, sprintf('%s_%s_Unit_%d%s', animal.animal, ...
        strrep(name_str, ' ', '_'), unit_idx, ext)));
    close(gcf);

end


function sampled_waveforms = select_rows(num_rows, all_waveforms)
    
% Generate a random permutation of row indices
row_indices = randperm(size(all_waveforms, 1));

% Select the first 30 indices
selected_indices = row_indices(1:num_rows);

% Create the new matrix with the selected rows
sampled_waveforms = all_waveforms(selected_indices, :, :);

end

function plt(waveforms, elec_idx)
    plot(squeeze(waveforms(:, elec_idx, :))');
end

function data = fr_fwhm_kmeans(data)

    % Initialize arrays for k-means clustering
    all_FWHM = [];
    all_firing_rates = [];
    
    % Extract FWHM and firing rate data for all units across animals
    for i = 1:length(data)
        animal = data(i);
        FWHM = [animal.units.good.FWHM_time];
        firing_rate = [animal.units.good.firing_rate];
        all_FWHM = [all_FWHM, FWHM];
        all_firing_rates = [all_firing_rates, firing_rate];
    end
    
    % Perform k-means clustering on FWHM and firing rate data
    k = 2;
    X = [normalize(all_FWHM)', normalize(all_firing_rates)'];

    % Use K-means++ to initialize the centroids
    centroids = kmeans_plusplus(X, k);

    % Perform K-means clustering with the initial centroids
    [idx, ~] = kmeans(X, k, 'Start', centroids);
    
    % Store the cluster assignments in the data structure
    count = 1;
    for i = 1:length(data)
        for j = 1:length(data(i).units.good)
            data(i).units.good(j).cluster_assignment = idx(count);
            count = count + 1;
        end
    end

end

function scatter_plot(data, x_attr, y_attr, x_label, y_label, gtitle, gfname, condition)
    % Initialize x and y data for scatter plot
    x = [];
    y = [];
    idx = [];

    % Extract the desired attributes and cluster assignment data
    for i = 1:length(data)
        animal = data(i);
        x_attr_values = [animal.units.good.(x_attr)];
        y_attr_values = [animal.units.good.(y_attr)];
        cluster_assignments = [animal.units.good.cluster_assignment];
 
        if nargin > 7 % Check if a condition is provided
            valid_values = eval(['x_attr_values ' condition]); % Evaluate the condition
            x_attr_values = x_attr_values(valid_values);
            y_attr_values = y_attr_values(valid_values);
            cluster_assignments = cluster_assignments(valid_values);
        end

        x = [x, x_attr_values];
        y = [y, y_attr_values];
        idx = [idx, cluster_assignments];
    end


    % Create a scatter plot and histograms
    figure;
    h1 = subplot(3, 4, [2, 3, 6, 7]);

    % Color-code scatter plot by cluster assignment
    gscatter(h1, x, y, idx, 'rbg', '.', 20);

    % Set up data cursor
    dcm_obj = datacursormode(gcf);
    set(dcm_obj, 'UpdateFcn', @customdatatip, 'Enable', 'on', 'SnapToDataVertex', 'on');
    
    title(h1, gtitle);

    % Create histogram of FWHM
    h2 = subplot(3, 4, [10, 11]);
    histogram(h2, x, 'FaceColor', 'blue', 'BinWidth', 100); % Adjust 'BinWidth' to change the bin size
    xlabel(h2, x_label);
    ylabel(h2, 'Count');
    set(h2, 'XTickLabel', []);

    % Create histogram of firing rate
    h3 = subplot(3, 4, [1, 5]);
    histogram(h3, y, 'FaceColor', 'red', 'Orientation', 'horizontal', 'BinWidth', 1); % Adjust 'BinWidth' to change the bin size
    ylabel(h3, y_label);
    xlabel(h3, 'Count');
   
    set(gca, 'XDir', 'reverse');
    set(h3, 'YTickLabel', []);

    % Custom data cursor function
    function txt = customdatatip(~, event_obj)
        pos = get(event_obj, 'Position');
        idx = find(x == pos(1) & y == pos(2));
        % Find the animal and unit number corresponding to the clicked point
        cumulative_units = 0;
        for i = 1:length(data)
            animal = data(i);
            if idx <= cumulative_units + length(animal.units.good.(x_attr))
                unit_idx = idx - cumulative_units;
                txt = {['Animal: ', animal.animal], ['Unit: ', num2str(unit_idx)]};
                break;
            else
                cumulative_units = cumulative_units + length(animal.units.good.(x_attr));
            end
        end
    end

    saveas(gcf, fullfile('/Users/katie/likhtik/data/graphs', gfname));

end

function centroids = kmeans_plusplus(data, k)
    % Initialize the centroids list with a randomly chosen point from the data
    centroids = data(randi([1 size(data, 1)]), :);

    for i = 2:k
        % Compute the distance between each point and the nearest centroid
        D = min(pdist2(data, centroids), [], 2);

        % Compute the probabilities for each point
        P = D.^2 / sum(D.^2);

        % Choose the next centroid
        centroids(i, :) = data(randsample(size(data, 1), 1, true, P), :);
    end
end

function normalized_x = normalize(x)
    normalized_x = (x - mean(x))/std(x);
end


