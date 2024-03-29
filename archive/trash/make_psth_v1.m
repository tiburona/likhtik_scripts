function spike_data = make_psth_v1(data)
  
    dbstop if error
    % Define the time window for the PSTH
    pre_stim_time = 0.05;
    post_stim_time = 0.65;
    pre_stim_cycles = 30000*pre_stim_time; % 100 ms before stimulus onset
    post_stim_cycles = 30000*post_stim_time; % 100 ms after stimulus onset
    bin_size_cycles = 30000*.01; % 10 ms bin size
    
    % Set the number of units per figure
    
    num_units_per_fig = 4;
    
    % Iterate through each animal in the data struct
    for i_animal = 1:length(data)
        
        % Iterate through each unit for this animal
        for i_unit = 1:length(data(i_animal).units.good)
            tone_onset_times = data(i_animal).ToneOn_ts_expanded;
            spikes = data(i_animal).units.good{i_unit};
            
            % Extract spike times for each trial
            [data_for_raster, data_for_psth, found_spikes] = extract_spike_times(tone_onset_times, spikes, pre_stim_cycles, post_stim_cycles);
            
            % If spikes were found, plot a raster for the unit
            if found_spikes
                % Create a new figure for each unit
                if mod(i_unit, num_units_per_fig) == 1
                    fig = figure('Visible', 'off');
                    fig.Position = [0, 0, 800, 800];
                end
                
                % Plot the raster for the current unit on the current figure
                subplot(num_units_per_fig*2, 1, mod(i_unit-1, num_units_per_fig)*2+1);
                plotSpikeRaster(data_for_raster, 'AutoLabel', true, ...
                    'XLimForCell', [0 pre_stim_time + post_stim_time], ...
                    'EventShading', [.05, .1]);
                
                % Plot the PSTH for the current unit underneath the raster plot
               
                bins = 0:bin_size_cycles:pre_stim_cycles+post_stim_cycles;
                psth_data = mean(cell2mat(cellfun(@(x) get_counts(x, bins, bin_size_cycles), data_for_psth, 'uni', 0)));
                subplot(num_units_per_fig*2, 1, mod(i_unit-1, num_units_per_fig)*2+2);

                % Add the gray box in the background from 50 ms to 100 ms
                hold on;
                patch([0.05 0.1 0.10 0.05], [0 0 max(psth_data) max(psth_data)], [0.8 0.8 0.8], 'EdgeColor', 'none');
   

                bar(bins(2:end)/30000, psth_data, 'histc');
                xlim([0 bins(end)]/30000);
                xlabel('Time (s)');
                ylabel('Spike rate (spikes/s)');
                
                % Save the figure if we've plotted the last unit or if we've reached the
                % maximum number of units per figure
                if i_unit == length(data(i_animal).units.good) || mod(i_unit, num_units_per_fig) == 0
                    % Save the figure to memory
                    marker1 = idivide(int8(i_unit)-1, num_units_per_fig) * num_units_per_fig + 1;
                    marker2 = min((idivide(int8(i_unit), num_units_per_fig)+1)*num_units_per_fig, length(data(i_animal).units.good));
                    filename = sprintf('Unit_%d_to_%d_Animal_%d', marker1, marker2, i_animal);
                    saveas(fig, filename, 'fig');
                    close(fig);  
                end
            end
            % Store the spike data in a struct
            spike_data(i_animal).data(i_unit).spikes = data_for_raster;
        end
    end
end

function [raster, psth, found_spikes] = extract_spike_times(tone_onsets, spikes, pre_stim_cycles, post_stim_cycles)
    found_spikes = false;
    % Extract spike times for each unit
    num_trials = length(tone_onsets);
    raster = cell(num_trials, 1);
    psth = cell(num_trials, 1);
    for j = 1:num_trials

        tone_on = tone_onsets(j);
        start = tone_on - pre_stim_cycles;
        en = tone_on + post_stim_cycles;
        trial_spikes = spikes(spikes >= start & spikes <= en);
        
        if ~isempty(trial_spikes)
            trial_spikes = double(trial_spikes - start);
        end
        psth{j} = trial_spikes';
        raster{j} = (trial_spikes/30000)';
        if ~isempty(trial_spikes); found_spikes = true; end
      
    end

end

function counts = get_counts(spikes, bins, bin_size_cycles)
    counts = histcounts(spikes, bins) / (bin_size_cycles/30000); % spike counts per second
end


