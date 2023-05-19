function make_psth_edits(data, dir)

    dbstop if error   
    
    sps = 30000; % samples per second

    % Define the time window for the PSTH
    pre_stim_time = 0.05;
    post_stim_time = 0.65;
    bin_size_time = 0.01;

    C = struct( ...
        'sps', sps,'pre_stim_time', pre_stim_time, 'post_stim_time', post_stim_time, ...
        'pre_stim_samples', pre_stim_time*sps, 'post_stim_samples', post_stim_time*sps, ...
        'bin_size_samples', bin_size_time*sps, 'num_units_per_fig', 4 ...
        );

    C.bins = 0 : C.bin_size_samples : C.pre_stim_samples + C.post_stim_samples;

    % separate data into interneurons and pyramidal neurons
    data_keys = {'all_units', 'PN', 'IN'};
    [pn_data, in_data] = split_data(data);    
    data_sets = {data, pn_data, in_data};
    
    % Choose different subsets of trials in the experiment to graph
    trial_keys = {'30_trials', '60_trials', '150_trials', 'first_tones'}; 
    trial_selectors = {1:30, 1:60, 1:150, 1:30:150};

    psth_data = struct();
    
    for i = 1:length(trial_keys)
        new_dir = fullfile(dir, ['psth_' trial_keys{i}]);
        FunctionContainer.safeMakeDir(new_dir);
        for j = 1:length(data_keys)
            psth_data.(data_keys{j}) = collect_psth_data(data_sets{j}, trial_selectors{i}, C);
        end
        unit_graphs(psth_data.('all_units'), new_dir, trial_keys(i), C);
        avg_struct = average_graphs(psth_data, new_dir, trial_keys(i), C);
        four_panel_graph(avg_struct, new_dir, trial_keys(i), C);
    end
end

function data = collect_psth_data(data, trials, constants)
    for i_animal = 1:length(data)
        for i_unit = 1:length(data(i_animal).units.good)
            unit = data(i_animal).units.good(i_unit);
            unit = extract_spike_times(unit.spike_times, data(i_animal).tone_onsets_expanded(trials), ...
                data(i_animal).tone_period_onsets, constants);
            fields = fieldnames(unit);
            for i_field = 1:numel(fields)
                data(i_animal).units.good(i_unit).(fields{i_field}) = unit.(fields{i_field});
            end
        end
    end
end

function unit = extract_spike_times(spikes, tone_onsets, tone_period_onsets, C)

    found_spikes = false;

    num_trials = length(tone_onsets);
    raster_data = cell(num_trials, 1);
    psth_data = cell(num_trials, 1);
    normalized_ptsh = cell(num_trials, 1);
    
    % Compute normalization factors for each tone period
    num_periods = length(tone_period_onsets);
    norm_factors = cell(num_periods, 1);
    for i = 1:num_periods
        pre_tone_period_start = tone_period_onsets(i) - 30*C.sps;
        pre_tone_bins = pre_tone_period_start:C.bin_size_samples:(tone_period_onsets(i) - 1);
        pre_tone_rates = get_rates(spikes, pre_tone_bins, C); 
        norm_factors{i} = [mean(pre_tone_rates), std(pre_tone_rates)]; 
    end

    for j = 1:num_trials
        tone_on = tone_onsets(j);
        start = tone_on - C.pre_stim_samples;
        ned = tone_on + C.post_stim_samples;
        trial_spikes = spikes(spikes >= start & spikes <= ned);

        if ~isempty(trial_spikes)
            trial_spikes = double(trial_spikes) - double(start);
        end
        psth_data{j} = trial_spikes';
        raster_data{j} = (trial_spikes/C.sps)';
        if ~isempty(trial_spikes); found_spikes = true; end

        % Determine the tone period for the current trial
        period_index = find(tone_period_onsets <= tone_on, 1, 'last');

        % Calculate the rates and normalize the psth for the current trial
        rate = get_rates(psth_data{j}, C.bins, C);
        mean_rate = norm_factors{period_index}(1);
        std_dev = norm_factors{period_index}(2);
        normalized_ptsh{j} = (rate - mean_rate) / std_dev;
    end
    
    normalized_ptsh = mean(cell2mat(normalized_ptsh));

    unit.raster = raster_data;
    unit.psth = normalized_ptsh;
    unit.found_spikes = found_spikes;
end

function unit_graphs(data, graph_dir, name_tags, C)
    for i_animal = 1:length(data)
        for i_unit = 1:length(data(i_animal).units.good)
            unit = data(i_animal).units.good(i_unit);
            if unit.found_spikes
                % Create a new figure for each unit
                if mod(i_unit, C.num_units_per_fig) == 1
                    fig = figure('Visible', 'off');
                    fig.Position = [0, 0, 800, 800];
                end
    
                % Plot the raster for the current unit on the current figure
                subplot(C.num_units_per_fig*2, 1, mod(i_unit-1, C.num_units_per_fig)*2+1);
                plotSpikeRaster(unit.raster, 'AutoLabel', true, 'XLimForCell', ...
                    [0 C.pre_stim_time + C.post_stim_time], 'EventShading', [.05, .1]);
    
                % Plot the PSTH for the current unit underneath the raster plot
                plot_psth(C.bins(1:end-1)/30000, unit.psth, '', C.pre_stim_time, 'subplot', ...
                    [C.num_units_per_fig*2, 1, mod(i_unit-1, C.num_units_per_fig)*2+2])
    
                % Save the figure if we've plotted the last unit or if we've reached the
                % maximum number of units per figure
    
                if i_unit == length(data(i_animal).units.good) || mod(i_unit, C.num_units_per_fig) == 0
                    marker1 = idivide(int8(i_unit)-1, C.num_units_per_fig) * C.num_units_per_fig + 1;
                    marker2 = min((idivide(int8(i_unit)-1, C.num_units_per_fig) + 1) * C.num_units_per_fig, ...
                        length(data(i_animal).units.good));
                    fname_base = sprintf('unit_%d_to_%d_%s', marker1, marker2, data(i_animal).animal);
                    [title, fname] = title_and_fname(fname_base, name_tags);
                    save_and_close_fig(fig, graph_dir, fname, 'figure_title', title);
                end
            end
        end
    end
end

function average_struct = average_graphs(data, dir, name_tags, C)
    datasets = {'all_units', 'PN', 'IN'};
    groups = {'control', 'stressed'};

    % initialize a struct to collect the data to pass to the four panel graph function
    groups_struct1 = cell2struct({[]; []}, groups);
    groups_struct2 = cell2struct({[]; []}, groups);
    average_struct = cell2struct({groups_struct1; groups_struct2}, datasets(2:3));

    for i_dataset = 1:length(datasets)
        psth_data = data.(datasets{i_dataset});
        for i_group = 1:length(groups)
            % select only the animals within the specified group
            index = arrayfun(@(x) strcmp(x.group, groups{i_group}), psth_data);
            group = psth_data(index);
            group_fig = initialize_figure('on');
            % graph the subplot for each individual animal's average
            for i_animal=1:length(group)
                animal = group(i_animal);
                if ~isempty(animal.units.good)
                    avg_psth_data = nanmean(cell2mat({animal.units.good.psth}'), 1);
                    plot_psth(C.bins(1:end-1)/C.sps, avg_psth_data, sprintf('Animal %s', ...
                    group(i_animal).animal), C.pre_stim_time, 'subplot', ...
                    [ceil(length(group) / 2), 2, i_animal]);
                    % save the individual animal's average to the data struct
                    group(i_animal).avg_psth_data = avg_psth_data;
                end
            end
            [title, fname] = title_and_fname('animal_averages_PSTH', horzcat(name_tags, groups{i_group}));
            save_and_close_fig(group_fig, dir, fname, 'figure_title', title); 

            % graph individual figures with the averages over animals within group for each neuron type
            avg_fig = initialize_figure('on');
            [title, fname] = title_and_fname('group_average_PSTH', horzcat(name_tags, datasets{i_dataset}, groups{i_group}));
            avg_data = nanmean(cell2mat({group.avg_psth_data}'), 1);
            plot_psth(C.bins(1:end-1)/C.sps, avg_data, title, C.pre_stim_time, 'figure', avg_fig);
            save_and_close_fig(avg_fig, dir, fname); 

            % save the data to pass to the four panel function         
            average_struct.(datasets{i_dataset}).(groups{i_group}) = avg_data;
        end
    end    
end

function four_panel_graph(avg_data, dir, trial_num, C)
    y_min = inf;
    y_max = -inf;
    
    datasets = {'IN', 'PN'};
    groups = {'control', 'stressed'};

    % find y boundaries 
    for i = 1:length(datasets)
        for j = 1:length(groups)
            if max(avg_data.(datasets{i}).(groups{j})) > y_max 
                y_max = max(avg_data.(datasets{i}).(groups{j})); 
            end
            if min(avg_data.(datasets{i}).(groups{j})) < y_min 
                y_min = min(avg_data.(datasets{i}).(groups{j})); 
            end
        end
    end
    
    % graph four panels with averages within group and neuron type
    four_panel_fig = initialize_figure('on');
    for i_dataset = 1:length(datasets)
        data_set = datasets{i_dataset};
        for i_group = 1:length(groups)
            group = groups{i_group};
            [title, ~] = title_and_fname('', {data_set, group});
            plot_psth(C.bins(1:end-1)/C.sps, avg_data.(data_set).(group), title, C.pre_stim_time, 'subplot', ...
                [2, 2, (i_dataset-1)*2 + i_group], 'figure', four_panel_fig, 'y_dim', [y_min, y_max]);
        end
    end
    [title, fname] = title_and_fname('four_panel_PSTH', trial_num);
    save_and_close_fig(four_panel_fig, dir, fname, 'figure_title', title);
end
    
function [title, fname] = title_and_fname(base, name_tags)
    name = horzcat(base, name_tags);
    fname = strjoin(name, '_');
    title = title_case(strrep(fname, '_', ' '));
    strrep(title, 'To', 'to');
    fname = [fname '.fig'];
end

function rates = get_rates(spikes, bins, C)
    rates = histcounts(spikes, bins) / (C.bin_size_samples/C.sps); % spike counts per second
end

function plot_psth(time, psth_data, title_text, pre_stim_time, varargin)
    % Process optional arguments
    is_subplot = false;
    y_min = min(psth_data) - 0.1*abs(min(psth_data));
    y_max = max(psth_data) + 0.1*abs(max(psth_data));
    subplot_args = [];
    fig_handle = [];
    for i = 1:length(varargin)
        if ischar(varargin{i}) && strcmpi(varargin{i}, 'subplot')
            is_subplot = true;
            if i < length(varargin) && isnumeric(varargin{i+1})
                subplot_args = varargin{i+1};
            else
                error('No subplot arguments provided after ''subplot'' keyword.');
            end
        elseif ischar(varargin{i}) && strcmpi(varargin{i}, 'y_dim')
            if i < length(varargin) && isnumeric(varargin{i+1})
                y_min = varargin{i+1}(1);
                y_max = varargin{i+1}(2);
            else
                error('No y_max arguments provided after ''y_dim'' keyword.');
            end
        elseif ischar(varargin{i}) && strcmpi(varargin{i}, 'figure')
            if i < length(varargin) && ishandle(varargin{i+1}) && strcmp(get(varargin{i+1},'type'),'figure')
                fig_handle = varargin{i+1};
            else
                error('No valid figure handle provided after ''figure'' keyword.');
            end
        end
    end

    % Select the figure if a handle is provided
    if ~isempty(fig_handle)
        figure(fig_handle);
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
    time = time - pre_stim_time;
    bar(time, psth_data, 'k');
    hold on;
    ylabel('Normalized Spike Rate');
    xlabel('Time (s)');
    title(title_text);
    xlim([-pre_stim_time max(time)]);
    % Add the shaded translucent gray bar
    ylim = [y_min - 0.1*abs(y_min), y_max + 0.1*abs(y_max)];
    patch([0 pre_stim_time pre_stim_time 0], [ylim(1) ylim(1) ylim(2) ylim(2)], 'k', ...
        'FaceAlpha', 0.2, 'EdgeColor', 'none');
    hold off;
end

function fig = initialize_figure(visible)
    fig = figure('Visible', visible);
    fig.Position = [0, 0, 800, 800];
end

function save_and_close_fig(fig, graph_dir, name, varargin)
    % Check for optional arguments
    figure_title = '';
    for i = 1:length(varargin)
        if ischar(varargin{i}) && strcmpi(varargin{i}, 'figure_title')
            if i < length(varargin) && ischar(varargin{i+1})
                figure_title = varargin{i+1};
            else
                error('No title provided after ''figure_title'' keyword.');
            end
        end
    end

    % Set the figure title if specified
    if ~isempty(figure_title)
        axes('Position', [0, 0, 1, 1], 'Xlim', [0, 1], 'Ylim', [0, 1], 'Box', 'off', 'Visible', ...
            'off', 'Units', 'normalized', 'clipping', 'off');
        text(0.5, 1, figure_title, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
            'FontSize', 14, 'FontWeight', 'bold');
    end

    % Save and close the figure
    p = fullfile(graph_dir, name);
    saveas(fig, p, 'fig');
    close(fig);
end

function [data_below_cutoff, data_above_cutoff] = split_data(data)
    % Initialize the output structures
    data_below_cutoff = data;
    data_above_cutoff = data;
    
    % Iterate over each animal in the data
    for i = 1:length(data)
        % Initialize new "good" structures for this animal
        good_below_cutoff = [];
        good_above_cutoff = [];
        
        % Iterate over each "good" unit in this animal
        for j = 1:length(data(i).units.good)
            % Check if the FWHM_time is below or above the cutoff
            if data(i).units.good(j).cluster_assignment < 2
                good_below_cutoff = [good_below_cutoff, data(i).units.good(j)];
            else
                good_above_cutoff = [good_above_cutoff, data(i).units.good(j)];
            end
        end
        
        % Update the "good" units for this animal in the output structures
        data_below_cutoff(i).units.good = good_below_cutoff;
        data_above_cutoff(i).units.good = good_above_cutoff;
    end
end

function title_case_str = title_case(str)
    title_case_str = regexprep(str, '(?<=\s|^)([a-z])', '${upper($1)}');
end