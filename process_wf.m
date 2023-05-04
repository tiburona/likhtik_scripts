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
        all_waveforms = zeros(length(spike_times), gwfparams.wfWin(2) - gwfparams.wfWin(1) + 1, length(electrodes));

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

            wf_data = all_data(electrodes', start_sample:end_sample)'; 
            
            % Store the waveforms for each channel
            all_waveforms(k, :, :) = wf_data;
        end
    
        % Compute the mean waveform for each channel
        mean_waveforms = squeeze(mean(all_waveforms, 1));
        
        dimension = 1;
        if length(electrodes) == 1; dimension = 2; end;
           
        mean_centered_waveforms = bsxfun(@minus, mean_waveforms, mean(mean_waveforms, dimension));
        
        if length(electrodes) > 1
            mean_of_electrodes = squeeze(mean(mean_centered_waveforms, 2)); 
        else
            mean_of_electrodes = mean_centered_waveforms;
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


        % Plot and save the mean waveform for each valid electrode
        figure;
        if length(electrodes) == 1
            plot(mean_waveforms(:));
            xlabel('Samples');
            ylabel('microvolts')
        else
            for elec_idx = 1:length(electrodes)
                subplot(length(electrodes), 1, elec_idx);
                plot(mean_waveforms(:, elec_idx));
                
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

% Set the overall figure title
sgtitle(sprintf('%s Mean Waveform for Unit %d', animal.animal, j));

% Save and close the figure
saveas(gcf, fullfile('/Users/katie/likhtik/data/graphs/waveforms', ...
    sprintf('%s_Mean_Waveform_Unit_%d.png', animal.animal, j)));
close(gcf);


    end

    % Update the data struct with the new fields

    data(i).units_FWHM = animal.units_FWHM;
    data(i).units_min = animal.units_min;


    % Close the binary file
    fclose(fid);
end

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
        x = [x, FWHM];
        y = [y, amplitude];
    end

    % Create scatter plot
    h = scatter(x, y, 'filled');

    % Set up data cursor
    dcm_obj = datacursormode(gcf);
    set(dcm_obj, 'UpdateFcn', @customdatatip, 'Enable', 'on', 'SnapToDataVertex', 'on');

    % Set axis labels
    xlabel('Full Width Half Minimum');
    ylabel('Amplitude');
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

    saveas(gc1, fullfile('/Users/katie/likhtik/data/graphs', ...
    'FWHM_and_Amplitude_Scatterplot.png'));

    
end

