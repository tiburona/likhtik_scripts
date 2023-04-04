function spike_data = make_psth(data)
  
    dbstop if error
    % Define the time window for the PSTH
    pre_stim_time = 0.05;
    post_stim_time = 0.65;
    pre_stim_cycles = 30000*pre_stim_time; % 100 ms before stimulus onset
    post_stim_cycles = 30000*post_stim_time; % 100 ms after stimulus onset
    bin_size_cycles = 30000*.01; % 10 ms bin size
    
    % Set the number of units per figure
    num_units_per_fig = 4;

    % Define control and stressed animal groups
    control = {'IG160', 'IG163', 'IG176', 'IG178', 'IG180'};
    stressed = {'IG154', 'IG156', 'IG158', 'IG177', 'IG179'};
    
    % Initialize figures for averaged PSTHs
    control_fig = initialize_figure('on');
    stressed_fig = initialize_figure('on');
    
    control_count = 0;
    stressed_count = 0;
    
    % Iterate through each animal in the data struct
    for i_animal = 1:length(data)
        
        % Determine if the current animal is control or stressed
        is_control = any(strcmp(data(i_animal).animal, control));
        is_stressed = any(strcmp(data(i_animal).animal, stressed));
        if is_control
            control_count = control_count + 1;
        elseif is_stressed
            stressed_count = stressed_count + 1;
        else
            continue
        end

        % Initialize the averaged PSTH data for the current animal
        averaged_psth_data = zeros(1, floor((pre_stim_cycles + post_stim_cycles) / bin_size_cycles));

        % Define trial bins
        bins = 0:bin_size_cycles:pre_stim_cycles+post_stim_cycles;
        tone_onset = .05; % the beginning of the shaded box in the graphs
        
        % Iterate through each unit for this animal
        for i_unit = 1:length(data(i_animal).units.good)
            tone_onsets = data(i_animal).ToneOn_ts_expanded;
            spikes = data(i_animal).units.good{i_unit};
            ToneOn_ts = data(i_animal).ToneOn_ts;
            
            [raster_data, computed_psth, found_spikes] = ...
                extract_spike_times(tone_onsets, spikes, pre_stim_cycles, ...
                post_stim_cycles, bin_size_cycles, ToneOn_ts);            
           
    
            averaged_psth_data = averaged_psth_data + computed_psth;
            
            % If spikes were found, plot a raster for the unit
            if found_spikes
                % Create a new figure for each unit
                if mod(i_unit, num_units_per_fig) == 1
                    fig = figure('Visible', 'off');
                    fig.Position = [0, 0, 800, 800];
                end
                
                % Plot the raster for the current unit on the current figure
                subplot(num_units_per_fig*2, 1, mod(i_unit-1, num_units_per_fig)*2+1);
                plotSpikeRaster(raster_data, 'AutoLabel', true, ...
                    'XLimForCell', [0 pre_stim_time + post_stim_time], ...
                    'EventShading', [.05, .1]);
                
                % Plot the PSTH for the current unit underneath the raster plot

                plot_psth(bins(1:end-1)/30000, computed_psth, '', tone_onset, 'subplot', [num_units_per_fig*2, 1, mod(i_unit-1, num_units_per_fig)*2+2])
              
                % Save the figure if we've plotted the last unit or if we've reached the
                % maximum number of units per figure
                if i_unit == length(data(i_animal).units.good) || mod(i_unit, num_units_per_fig) == 0
                    % Save the figure to memory
                    marker1 = idivide(int8(i_unit)-1, num_units_per_fig) * num_units_per_fig + 1;
                    marker2 = min((idivide(int8(i_unit), num_units_per_fig)+1)*num_units_per_fig, length(data(i_animal).units.good));
                    filename = sprintf('Unit_%d_to_%d_%s', marker1, marker2, data(i_animal).animal);
                    saveas(fig, filename, 'fig');
                    close(fig);  
                end
            end

        end
        
        % Calculate the averaged PSTH data for the current animal
        averaged_psth_data = averaged_psth_data / length(data(i_animal).units.good);
        
        % Plot the averaged PSTH data for the current animal on the control or stressed figure
        if is_control
            figure(control_fig);
            plot_psth(bins(1:end-1)/30000, averaged_psth_data, sprintf('Animal %s', data(i_animal).animal), tone_onset, 'subplot', [ceil(length(control) / 2), 2, control_count]);
        else
            figure(stressed_fig);
            plot_psth(bins(1:end-1)/30000, averaged_psth_data, sprintf('Animal %s', data(i_animal).animal), tone_onset, 'subplot', [ceil(length(stressed) / 2), 2, stressed_count]);
        end
   
    end

    % Save the control and stressed figures
    save_and_close_fig(control_fig, 'Control_Averaged_PSTHs')
    save_and_close_fig(stressed_fig, 'Stressed_Averaged_PSTHs')
    
    % Plot the first tone PSTH for control and stressed groups
    
    plot_and_save_first_tone_psth(data, control, 'Control', bins, tone_onset);
    plot_and_save_first_tone_psth(data, stressed, 'Stressed', bins, tone_onset);
end

function rates = get_rates(spikes, bins, bin_size_cycles)
    rates = histcounts(spikes, bins) / (bin_size_cycles/30000); % spike counts per second
end

function plot_psth(time, psth_data, title_text, tone_onset, varargin)
    % Check for optional arguments
    is_subplot = false;
    subplot_args = [];
    for i = 1:length(varargin)
        if ischar(varargin{i}) && strcmpi(varargin{i}, 'subplot')
            is_subplot = true;
            if i < length(varargin) && isnumeric(varargin{i+1})
                subplot_args = varargin{i+1};
            else
                error('No subplot arguments provided after ''subplot'' keyword.');
            end
        end
    end

    % Create subplot if specified
    if is_subplot
        if isempty(subplot_args) || numel(subplot_args) ~= 3
            error('Invalid subplot arguments. Three arguments required (rows, columns, index).');
        else
            subplot(subplot_args(1), subplot_args(2), subplot_args(3));
        end
    end

    % Plot the data
    bar(time, psth_data, 'k');
    hold on;
    ylabel('Normalized Spike Rate');
    xlabel('Time (s)');
    title(title_text);
    xlim([0 max(time)]);
    y_min = min(psth_data) - 0.1*abs(min(psth_data));
    y_max = max(psth_data) + 0.1*abs(max(psth_data));
    % Add the shaded translucent gray bar
    tone_duration = 0.05;
    tone_start = tone_onset;
    tone_end = tone_onset + tone_duration;
    patch([tone_start tone_end tone_end tone_start], [y_min y_min y_max y_max], 'k', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold off;
end

function [raster_data, computed_ptsh, found_spikes] = extract_spike_times(...
    tone_onsets, spikes, pre_stim_cycles, post_stim_cycles, ...
    bin_size_cycles, ToneOn_ts)
    found_spikes = false;
    % Extract spike times for each unit
    num_trials = length(tone_onsets);
    raster_data = cell(num_trials, 1);
    psth_data = cell(num_trials, 1);

    for j = 1:num_trials
        tone_on = tone_onsets(j);
        start = tone_on - pre_stim_cycles;
        en = tone_on + post_stim_cycles;
        trial_spikes = spikes(spikes >= start & spikes <= en);

        if ~isempty(trial_spikes)
            trial_spikes = double(trial_spikes) - double(start);
        end
        psth_data{j} = trial_spikes';
        raster_data{j} = (trial_spikes/30000)';
        if ~isempty(trial_spikes); found_spikes = true; end
    end

    % Compute the bins for the entire data stream 
    % lead in time is arbitrary
    bins = (tone_onsets(1) - 10*30000):bin_size_cycles:(tone_onsets(end) + 0.05*30000);

    % Compute the psth for the entire data stream
    stream_psth = get_rates(spikes, bins, bin_size_cycles);

    % Compute the standard deviation of the psth for the entire data stream
    std_dev_firing_rate = std(stream_psth);

    % Find the bins that fall within the tone on periods
    tone_on_bins = arrayfun(@(x) bins(bins >= x & bins <= (x + 29.05*30000)), ToneOn_ts, 'UniformOutput', false);
    tone_on_bins = unique([tone_on_bins{:}]);

    % Remove bins during the tone on period
    stream_psth_filtered = stream_psth(~ismember(bins, tone_on_bins));

    % Compute the mean firing rate for the inter-stimulus interval
    mean_firing_rate_ISI = mean(stream_psth_filtered);

    % Compute the computed_ptsh
    bins = 0:bin_size_cycles:pre_stim_cycles+post_stim_cycles;
    computed_ptsh = mean(cell2mat(cellfun(@(x) get_rates(x, bins, bin_size_cycles), psth_data, 'uni', 0)));

    % Normalize the computed_ptsh
    computed_ptsh = (computed_ptsh - mean_firing_rate_ISI) / std_dev_firing_rate;
end

function first_tone_psth_data = first_tone_psth(data, group)
    pre_stim_cycles = 30000 * 0.05; % 50 ms before stimulus onset
    post_stim_cycles = 30000 * 0.65; % 650 ms after stimulus onset
    bin_size_cycles = 30000 * 0.01; % 10 ms bin size
    
    averaged_psth_data = [];
    
    for i_animal = 1:length(data)
        if any(strcmp(data(i_animal).animal, group))
            tone_onsets = data(i_animal).ToneOn_ts; % Use every entry in ToneOn_ts
            spikes = data(i_animal).units.good{1};
            
            [~, computed_psth, ~] = extract_spike_times(tone_onsets, spikes, pre_stim_cycles, post_stim_cycles, bin_size_cycles, tone_onsets);
            if isempty(averaged_psth_data)
                averaged_psth_data = computed_psth;
            else
                averaged_psth_data = averaged_psth_data + computed_psth;
            end
        end
    end
    
    first_tone_psth_data = averaged_psth_data / length(group);
end

function fig = initialize_figure(visible)
    fig = figure('Visible', visible);
    fig.Position = [0, 0, 800, 800];
end

function save_and_close_fig(fig, name)
    saveas(fig, name, 'fig');
    close(fig);
end

function plot_first_tone_data(data, group, name)

   % Plot the first tone PSTH for control and stressed groups
    
    control_first_tone_psth_data = first_tone_psth(data, control);
    figure(control_first_tone_fig);
    plot_psth(bins(1:end-1)/30000, control_first_tone_psth_data, 'Control Group: First Tone PSTH', tone_onset);

    stressed_first_tone_psth_data = first_tone_psth(data, stressed);
    figure(stressed_first_tone_fig);
    plot_psth(bins(1:end-1)/30000, stressed_first_tone_psth_data, 'Stressed Group: First Tone PSTH', tone_onset);

    % Save the first tone PSTH figures
    save_and_close_fig(control_first_tone_fig, 'Control_First_Tone_PSTH');
    save_and_close_fig(stressed_first_tone_fig, 'Control_First_Tone_PSTH');

end

function plot_and_save_first_tone_psth(data, group, group_name, bins, tone_onset)
    group_first_tone_psth_data = first_tone_psth(data, group);
    
    group_first_tone_fig = initialize_figure('on');
    plot_psth(bins(1:end-1)/30000, group_first_tone_psth_data, [group_name, ' Group: First Tone PSTH'], tone_onset);
    
    save_and_close_fig(group_first_tone_fig, [group_name, '_First_Tone_PSTH']);
end


