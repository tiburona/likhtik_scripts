dbstop if error



process_waveforms(single_cell_data, true)

function data = process_waveforms(data, exclude_outliers)

if nargin < 2
    exclude_outliers = false;
end

% Constants
sampling_rate = 30000; % 30 kHz sampling rate


gwfparams.nCh = 14;                      % Number of channels that were streamed to disk in .dat file
gwfparams.wfWin = [-100 200];            % Number of samples before and after spiketime to include in waveform
gwfparams.nWf = 2000;                    % Number of waveforms per unit to pull out
% gwfparams.dataDir = '/path/to/data/';    % KiloSort/Phy output folder
% gwfparams.fileName = 'data.dat';         % .dat file containing the raw 
% gwfparams.dataType = 'int16';            % Data type of .dat file (this should be BP filtered)
% gwfparams.nCh = 32;                      % Number of channels that were streamed to disk in .dat file
% gwfparams.wfWin = [-40 41];              % Number of samples before and after spiketime to include in waveform
% gwfparams.nWf = 2000;                    % Number of waveforms per unit to pull out
% gwfparams.spikeTimes =    [2,3,5,7,8,9]; % Vector of cluster spike times (in samples) same length as .spikeClusters
% gwfparams.spikeClusters = [1,2,1,1,1,2]; % Vector of cluster IDs (Phy nomenclature)   same length as .spikeTimes

% Loop through animals
for i = 1:length(data)
    animal = data(i);
    if ~ismember(animal.animal, {'IG154', 'IG156', 'IG177', 'IG178', 'IG179', 'IG180'})
        continue
    end

    channel_indices = [2, 4:16];

    gwfparams.dataDir = animal.rootdir;
    gwfparams.fileName = 'data_binary.bin';
    gwfparams.dataType = 'int16';            % Data type of .dat file (this should be BP filtered)
    gwfparams.nCh = 16;                      % Number of channels that were streamed to disk in .dat file
    gwfparams.wfWin = [-100 201];             % Number of samples before and after spiketime to include in waveform
    gwfparams.nWf = 2000;                    % Number of waveforms per unit to pull out
    gwfparams.spikeTimes = readNPY(fullfile(gwfparams.dataDir, 'spike_times.npy'));

  
    [cids,cgs] = getclustermarkings(animal.rootdir);
    cl = readNPY(fullfile(animal.rootdir, 'spike_clusters.npy'));

    gwfparams.spikeClusters = cl;
    toremove = ismember(gwfparams.spikeClusters, cids(cgs==0));
    gwfparams.spikeTimes(toremove) = [];
    gwfparams.spikeClusters(toremove) = [];
    cids(cgs==0) = [];
    cgs(cgs==0) = [];
    wf = getWaveForms(gwfparams);

    chMap = readNPY(fullfile(gwfparams.dataDir, 'channel_map.npy'))+1; 
    
    % Create a new field in the data struct for the FWHM results
    animal.units_FWHM = cell(size(animal.units.good));
    
    % Open binary file for the current animal
    file_path = fullfile(animal.rootdir, 'data_binary.bin');
    fid = fopen(file_path, 'r');
    
    % Read the waveform data from the binary file
    M = memmapfile(file_path);
    M1 = memmapfile(file_path, 'Format', {'int16', [16 length(M.Data) / 16 / 2], 'data'});
    all_data = M1.Data.data(chMap, :);

    % Apply a fifth-order median filter to the electrode data
    all_data = medfilt1(double(all_data), 5, [], 2);

    % Calculate the range for each electrode's raw data
    ranges = range(all_data, 2);

    % Exclude electrodes with a range of values more than four times the range of the electrode with the smallest range of values
    if exclude_outliers
        min_range = min(ranges);
        valid_electrodes = find(ranges <= 4 * min_range);
        all_data = all_data(valid_electrodes, :);
        channel_indices = channel_indices(valid_electrodes);
    else
        valid_electrodes = 1:length(channel_indices);
    end
    
    % Loop through units
    for j = 1:length(animal.units.good)
        spike_times = animal.units.good{j};
        spike_times = spike_times(spike_times > 100 & spike_times < size(M1.Data.data, 2) - 200);
    
        % Initialize arrays to store waveforms for each channel
        all_waveforms = zeros(length(spike_times), gwfparams.wfWin(2) - gwfparams.wfWin(1) + 1, length(channel_indices));
        
        for k = 1:length(spike_times)
            % Calculate the start and end sample indices for the waveform
            start_sample = spike_times(k) + gwfparams.wfWin(1);
            end_sample = spike_times(k) + gwfparams.wfWin(2);
            
            wf_data = all_data(:, start_sample:end_sample)';
            
            % Store the waveforms for each channel
            all_waveforms(k, :, :) = wf_data;
        end
    
        % Compute the mean waveform for each channel
        mean_waveforms = squeeze(mean(all_waveforms, 1));
        
        % Calculate the Full Width Half Minimum (FWHM) of the mean waveform
        half_min = (min(mean_waveforms) + max(mean_waveforms)) / 2;
        below_half_min = mean_waveforms <= (max(mean_waveforms) + half_min);
        FWHM_samples = sum(below_half_min);
        FWHM_time = FWHM_samples / sampling_rate;


        % Save the FWHM result to the data struct
        animal.units_FWHM{j} = FWHM_time;

        % Plot and save the mean waveform for each valid electrode
        figure;
        for elec_idx = 1:length(valid_electrodes)
            subplot(length(valid_electrodes), 1, elec_idx);
            plot(mean_waveforms(:, elec_idx));
            
            % Only add labels and title for the bottom subplot
            if elec_idx == length(valid_electrodes)
                xlabel('Samples');
            else
                % Remove x and y axis labels for other subplots
                set(gca, 'XTickLabel', []);
             
            end
            set(gca, 'YTickLabel', []);
        end

% Set the overall figure title
sgtitle(sprintf('Mean Waveform for Unit %d', j));

% Save and close the figure
saveas(gcf, fullfile(animal.rootdir, sprintf('Mean_Waveform_Unit_%d.png', j)));
close(gcf);


    end

    % Update the data struct with the new FWHM field
    data(i).units_FWHM = animal.units_FWHM;

    % Close the binary file
    fclose(fid);
end

end
