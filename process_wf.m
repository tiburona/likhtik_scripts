dbstop if error

single_cell_data = process_waveforms(single_cell_data);
scatter_plot(single_cell_data);


function data = process_waveforms(data)

% Constants
sampling_rate = 30000; % 30 kHz sampling rate

gwfparams.nCh = 16;                      % Number of channels that were streamed to disk in .dat file
gwfparams.wfWin = [-100 200];            % Number of samples before and after spiketime to include in waveform
gwfparams.nWf = 2000;                    % Number of waveforms per unit to pull out
gwfparams.dataType = 'int16';            % Data type of .dat file
gwfparams.fileName = 'data_binary.bin';

% Loop through animals
for i = 1:length(data)
    animal = data(i);


    gwfparams.dataDir = animal.rootdir;
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
    animal.units_min = cell(size(animal.units.good));
    
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

        spike_times = animal.units.good{j};
        spike_times = spike_times(spike_times > 100 & spike_times < size(M1.Data.data, 2) - 200);
        electrodes = animal.electrodes.good{j};

        % Initialize arrays to store waveforms for each channel
        all_waveforms = zeros(length(spike_times), length(electrodes), gwfparams.wfWin(2) - gwfparams.wfWin(1) + 1);

        % transformation of phy electrode labels to index in raw data
        for el = 1:length(electrodes)
            if electrodes(el) > 1
                electrodes(el) = electrodes(el) - 1;
            end
        end
    
        for k = 1:length(spike_times)
            % Calculate the start and end sample indices for the waveform
            start_sample = spike_times(k) + gwfparams.wfWin(1);
            end_sample = spike_times(k) + gwfparams.wfWin(2);

            wf_data = all_data(electrodes', start_sample:end_sample); 
            
            % Store the waveforms for each channel
            all_waveforms(k, :, :) = wf_data;
        end
           
        mean_centered_waveforms = all_waveforms - mean(all_waveforms, 3);
        
%         if j ~= 2 || i~=2
%             continue
%         end

        % Compute the mean waveform for each channel
        mean_waveforms = mean(mean_centered_waveforms, 1);

        plot_waveforms(animal, j, mean_waveforms, electrodes);
        plot_waveforms(animal, j, mean_centered_waveforms, electrodes, 'num_samples', 30);
        
        mean_of_electrodes = mean_waveforms;
        if length(electrodes) > 1
            mean_of_electrodes = squeeze(mean(mean_waveforms, 2)); 
        end
        
        % Calculate the Full Width Half Minimum (FWHM) of the mean waveform
        half_min = (min(mean_of_electrodes) + max(mean_of_electrodes)) / 2;
        below_half_min = mean_of_electrodes <= (max(mean_of_electrodes) + half_min);
        FWHM_samples = sum(below_half_min);
        FWHM_time = FWHM_samples / sampling_rate;

       
        % Save the FWHM result to the data struct
        animal.units_FWHM{j} = FWHM_time;

        % Save the amplitude to the data struct
        animal.units_min{j} = min(mean_of_electrodes);


      

    end

    % Update the data struct with the new fields

    data(i).units_FWHM = animal.units_FWHM;
    data(i).units_min = animal.units_min;


    % Close the binary file
    fclose(fid);

 
end

end

function plot_waveforms(animal, unit_idx, waveforms, electrodes, varargin)
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
    saveas(gcf, fullfile('/Users/katie/likhtik/data/graphs/waveforms', ...
        sprintf('%s_%s_Unit_%d%s', animal.animal, strrep(name_str, ' ', '_'), unit_idx, ext)));
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

function scatter_plot(data)
    % Initialize x and y data for scatter plot
    x = [];
    y = [];

    % Extract FWHM and unit number data
    for i = 1:length(data)
        animal = data(i);
        FWHM = [animal.units_FWHM{:}];
        amplitude = [animal.units_min{:}];
        x = [x, amplitude];
        y = [y, FWHM];
    end

    % Create scatter plot
    h = scatter(x, y, 'filled');

    % Set up data cursor
    dcm_obj = datacursormode(gcf);
    set(dcm_obj, 'UpdateFcn', @customdatatip, 'Enable', 'on', 'SnapToDataVertex', 'on');

    % Set axis labels
    xlabel('Amplitude');
    ylabel('Full Width Half Minimum');
    title('Scatterplot of Good Units');

    % Custom data cursor function
    function txt = customdatatip(~, event_obj)
        pos = get(event_obj, 'Position');
        idx = find(x == pos(1) & y == pos(2));
        % Find the animal and unit number corresponding to the clicked point
        cumulative_units = 0;
        for i = 1:length(data)
            animal = data(i);
            if idx <= cumulative_units + length(animal.units_FWHM)
                unit_idx = idx - cumulative_units;
                txt = {['Animal: ', animal.animal], ['Unit: ', num2str(unit_idx)]};
                break;
            else
                cumulative_units = cumulative_units + length(animal.units_FWHM);
            end
        end
    end

    saveas(gcf, fullfile('/Users/katie/likhtik/data/graphs', ...
    'FWHM_and_Amplitude_Scatterplot.fig'));

    
end

