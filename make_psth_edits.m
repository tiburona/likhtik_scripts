function make_psth_edits(data, data_types, trial_keys, dir)

    dbstop if error   
    
    C = populate_constants();
   
    % Choose different subsets of trials in the experiment to graph
    trials_map = containers.Map({'30_trials', '60_trials', '150_trials', 'first_tones'}, ...
        {1:30, 1:60, 1:150, 1:30:150});

    for i = 1:length(trial_keys)
        for j=1:length(data_types)
            safeMakeDir(fullfile(dir, [data_types{j} '_' trial_keys{i}]));
        end
        
        [average_data, data] = collect_data(data, trials_map(trial_keys{i}), data_types, C);
        unit_graphs(data, dir, data_types, trial_keys(i), C)
        average_graphs(average_data, data_types, dir, trial_keys(i), C); 
        four_panel_graph(average_data, data_types, dir, trial_keys(i), C);
    end
end

function C = populate_constants()

    C.sps = 30000;
    C.bin_size_time = 0.01;
    C.psth.pre_stim_time = 0.05;
    C.psth.post_stim_time = 0.65;

    C.autocorr.post_stim_time = 1.0;
    C.long_autocorr.post_stim_time = 30.0;
    C.autocorr_one_sided.post_stim_time = 30.0;
    C.autocorr.lags = 99;

    C.units_in_fig = 4;

    data_types = {'psth', 'autocorr', 'long_autocorr'};
    C.bin_size_samples = C.bin_size_time * C.sps;
 
    for i = 1:length(data_types)
        data_type = data_types{i};
        if contains(data_types{i}, 'autocorr')
            C.(data_type).pre_stim_time = 0;
        end
        fields = {'pre_stim_time', 'post_stim_time'};
        for j = 1:length(fields)
            field = fields{j};
            C.(data_type).(strrep(field, 'time', 'samples')) = C.(data_type).(field) * C.sps;
        end

        C.(data_type).bins = 0 : C.bin_size_samples : C.(data_type).pre_stim_samples + ...
            C.(data_type).post_stim_samples;
    end
    C.groups = {'control', 'stressed'};
end

function [averages, data] = collect_data(data, trials, data_types, C)

    % separate data into interneurons and pyramidal neurons
    data_keys = {'all_units', 'PN', 'IN'};

    for i_animal = 1:length(data)
        for i_unit = 1:length(data(i_animal).units.good)  
            if strcmp(data(i_animal).animal, 'IG160') 
                foo = 'a';
            end
            unit = data(i_animal).units.good(i_unit);
            unit = calc_spike_data(unit.spike_times, data(i_animal).tone_onsets_expanded(trials), ...
                data(i_animal).tone_period_onsets, data_types, C);
            fields = fieldnames(unit);
            for i_field = 1:numel(fields)
                data(i_animal).units.good(i_unit).(fields{i_field}) = unit.(fields{i_field});
            end
        end
    end

    [pn_data, in_data] = split_data(data); 
    datasets = {data, pn_data, in_data};
    averages = struct();
    for i_dataset = 1:length(datasets)
        data_set = datasets{i_dataset};
        data_key = data_keys{i_dataset};
        for i_group = 1:length(C.groups)
            averages = select_group_and_get_averages(...
                averages, C.groups{i_group}, data_set, data_key, data_types);       
        end
    end
end

function averages = select_group_and_get_averages(averages, condition, data_set, data_key, fields)
    index = arrayfun(@(x) strcmp(x.group, condition), data_set);
    group = data_set(index);
    for i_field = 1:length(fields)
        [group, group_average] = get_averages(group, fields{i_field});
        averages.(data_key).(condition).(fields{i_field}) = group_average;
    end
    averages.(data_key).(condition).animals = group; 
end

function [group, group_average] = get_averages(group, field)
    for i = 1:length(group)
        if isempty(group(i).units.good); continue; end
        group(i).averages.(field) = nanmean(cell2mat({group(i).units.good.(field)}'), 1);  
    end

    non_empty = arrayfun(@(x) ~isempty(x.averages), group);
    group_average = nanmean(cell2mat(arrayfun(@(x) x.averages.(field), group(non_empty), 'uni', 0)'), 1);
end

function unit = calc_spike_data(unit_spikes, tone_onsets, tone_period_onsets, data_types, C)
    
    found_spikes = false;
    num_trials = length(tone_onsets);
    norm_factors = get_norm_factors(unit_spikes, tone_period_onsets, C);
   
    raster_data = cell(num_trials, 1);

    for i = 1:length(data_types)
        data_type = data_types{i};
        trials_data = cell(num_trials, 1);
        for j = 1:length(trials_data)
            tone_on = tone_onsets(j);
            [trial_spikes, spike_data, found_trial_spikes] = calculate_data(unit_spikes, data_type, ...
                tone_on, norm_factors, tone_period_onsets, C);
            if found_trial_spikes; found_spikes = true; end
            trials_data{j} = spike_data;
            if strcmp(data_type, 'psth')
                raster_data{j} = (trial_spikes/C.sps)';
            end
        end
        unit.(data_type) = nanmean(cell2mat(trials_data));
    end

    unit.raster = raster_data;
    unit.found_spikes = found_spikes;
end

function [trial_spikes, spike_data, found_spikes] = calculate_data(unit_spikes, data_type, tone_on, ...
    norm_factors, tone_period_onsets, C)
    trial_spikes = find_trial_spikes(unit_spikes, tone_on - C.(data_type).pre_stim_samples, ...
        C.(data_type).pre_stim_samples + C.(data_type).post_stim_samples);
    found_spikes = ~isempty(trial_spikes);
    rates = histcounts(trial_spikes,  C.(data_type).bins) / C.bin_size_time; % spike counts per second
    if strcmp(data_type, 'psth')
        % Determine the tone period for the current trial
        period_index = find(tone_period_onsets <= tone_on, 1, 'last');
        mean_rate = norm_factors{period_index}(1);
        std_dev = norm_factors{period_index}(2);
        spike_data = rates - mean_rate;
    elseif contains(data_type, 'autocorr')
        rates = rates - mean(rates);
        spike_data = xcorr(rates, 99, 'coeff');
        spike_data = spike_data(end-C.autocorr.lags+1:end);
    end
end

function trial_spikes = find_trial_spikes(spikes, start, length_in_samples)
    ned = start + length_in_samples;
    trial_spikes = spikes(spikes >= start & spikes <= ned);
    if ~isempty(trial_spikes)
        trial_spikes = double(trial_spikes) - double(start);
    end
end

function norm_factors = get_norm_factors(spikes, tone_period_onsets, C)
    num_periods = length(tone_period_onsets);
    norm_factors = cell(num_periods, 1);
    for i = 1:num_periods
        pre_tone_period_start = tone_period_onsets(i) - 30*C.sps;
        pre_tone_bins = pre_tone_period_start:C.bin_size_samples:(tone_period_onsets(i) - 1);
        pre_tone_rates = get_rates(spikes, pre_tone_bins, C.bin_size_time); 
        norm_factors{i} = [mean(pre_tone_rates), std(pre_tone_rates)]; 
    end
    
    % if no spikes in a period make std. dev. the mean of the other periods' std. dev.
    for i = 1:num_periods
        if norm_factors{i}(2) == 0
            non_zero_indices = cellfun(@(x) x(2) ~= 0, norm_factors); 
            non_zero_values = cell2mat(norm_factors(non_zero_indices));
            norm_factors{i}(2) = mean(non_zero_values(2));  
        end
    end
end

function unit_graphs(data, graph_dir, data_types, trials_key, C)
    for i_animal = 1:length(data)
        animal = data(i_animal);
        for i_datatype = 1:length(data_types)
            create_units_figure(graph_dir, data_types{i_datatype}, animal, trials_key, C);
        end
    end
end

function create_units_figure(graph_dir, data_type, animal, trials_key, C)
    for i_unit = 1:length(animal.units.good)
        if ~animal.units.good(i_unit).found_spikes; continue; end
        if mod(i_unit, C.units_in_fig) == 1
            fig = initialize_figure('off');
        end
        fig = plot_unit(fig, graph_dir, animal, i_unit, data_type, trials_key, C); 
    end 
end

function fig = plot_unit(fig, graph_dir, animal, i_unit, data_type, trials_key, C)
    unit = animal.units.good(i_unit);
       
    plot_args = {data_type, unit.(data_type), C};

    if strcmp(data_type, 'psth')
        % first plot the raster
        subplot(C.units_in_fig*2, 1, mod(i_unit-1, C.units_in_fig)*2+1); % subplot args for raster 
        plotSpikeRaster(unit.raster, 'AutoLabel', true, 'XLimForCell', ...
             [0 C.psth.pre_stim_time + C.psth.post_stim_time], 'EventShading', [.05, .1])
        plot_args = [plot_args {'subplot', [C.units_in_fig*2, 1, mod(i_unit-1, C.units_in_fig)*2+2]}];
    elseif contains(data_type, 'autocorr')
        plot_args = [plot_args {'y_dim', [0, .3], 'subplot', [C.units_in_fig, 1, ...
        mod(i_unit, C.units_in_fig) + C.units_in_fig*double(mod(i_unit, C.units_in_fig) == 0)]}];
    end
    
    % plot PSTH or autocorr
    plot_data(plot_args{:});
       
    if i_unit == length(animal.units.good) || mod(i_unit, C.units_in_fig) == 0
        [marker1, marker2] = markers(animal, i_unit, C);
        fname_base = sprintf('unit_%d_to_%d_%s', marker1, marker2, animal.animal);
        [title, fname] = title_and_fname(fname_base, [data_type trials_key]);
        save_and_close_fig(fig, graph_dir, fname, data_type, trials_key{1}, 'figure_title', title);
    end

end

function [marker1, marker2] = markers(animal, i_unit, C)
    % generates the unit numbers in the graph name (as in, 1, 4 in Units 1 to 4)
    marker1 = idivide(int8(i_unit)-1, C.units_in_fig) * C.units_in_fig + 1;
    marker2 = min((idivide(int8(i_unit)-1, C.units_in_fig) + 1) * C.units_in_fig, ...
        length(animal.units.good));
end

function average_graphs(averages, data_types, graph_dir, trials_key, C)
    datasets = {'all_units', 'PN', 'IN'};

    for i_dataset = 1:length(datasets)
        for i_group = 1:length(C.groups)
            create_average_figures(averages.(datasets{i_dataset}), C.groups{i_group}, ...
                data_types, graph_dir, [trials_key, {datasets{i_dataset}, C.groups{i_group}}], C) 
        end
    end    
end

function create_average_figures(data, condition, data_types, graph_dir, name_tags, C)
    
    for i_type = 1:length(data_types)
        extra_args = {};
        if contains(data_types{i_type}, 'autocorr')
            extra_args = {'y_dim', [0, .2]};
        end
        average_figure(@plot_animal_averages, 'animal_averages', data_types{i_type}, condition, ...
            data, graph_dir, name_tags, C, extra_args);
        average_figure(@plot_group_average, 'group_averages', data_types{i_type}, condition, ...
            data,graph_dir, name_tags, C, extra_args);
    end
end

function average_figure(average_func, name, data_type, condition, data, graph_dir, name_tags, C, extra_args)
    average_fig = initialize_figure('on');
    average_func(data, condition, data_type, C, extra_args);
    [title, fname] = title_and_fname(name, horzcat(name_tags, data_type));
    save_and_close_fig(average_fig, graph_dir, fname, data_type, name_tags{1}, 'figure_title', title);
end

function plot_animal_averages(average_data, condition, data_type, C, extra_args)
    group = average_data.(condition).animals;
    for i_animal=1:length(group)
        animal = group(i_animal);
        if ~isempty(animal.units.good) 
            args = {'title_text', sprintf('Animal %s', animal.animal), 'subplot', ...
                [ceil(length(group) / 2), 2, i_animal]};
            args = horzcat(args, extra_args);
            plot_data(data_type, animal.averages.(data_type), C, args{:});
        end
   end
end

function plot_group_average(average_data, condition, data_type, C, extra_args)  
    plot_data(data_type, average_data.(condition).(data_type), C, extra_args{:});
end

function four_panel_graph(averages, data_types, dir, trial_num, C)
    datasets = {'IN', 'PN'};

    for i=1:length(data_types)
        [y_min, y_max] = find_y_boundaries(averages, datasets, C.groups, data_types{i});
        data_type = data_types{i};
        four_panel_fig = initialize_figure('on');

        for i_dataset = 1:length(datasets)
            data_set = datasets{i_dataset};
            for i_group = 1:length(C.groups)
                group = C.groups{i_group};
                [title, ~] = title_and_fname('', {data_set, group});
                args = {data_type, averages.(data_set).(group).(data_type), C, 'title_text', title,...
                    'subplot', [2, 2, (i_dataset-1)*2 + i_group], 'fig_handle', four_panel_fig};
                y_dim = [y_min, y_max];  
                args = [args, {'y_dim', y_dim}];
                plot_data(args{:}); 
            end
        end
        [title, fname] = title_and_fname('four_panel', [data_type trial_num]);
        save_and_close_fig(four_panel_fig, dir, fname, data_type, trial_num{1}, ...
            'figure_title', title);
    end   
end

function [y_min, y_max] = find_y_boundaries(averages, datasets, groups, data_type)
    y_min = inf;
    y_max = -inf;
    for i = 1:length(datasets)
        for j = 1:length(groups)
            if max(averages.(datasets{i}).(groups{j}).(data_type)) > y_max 
                y_max = max(averages.(datasets{i}).(groups{j}).(data_type)); 
            end
            if min(averages.(datasets{i}).(groups{j}).(data_type)) < y_min 
                y_min = min(averages.(datasets{i}).(groups{j}).(data_type)); 
            end
        end
    end 
end

function [title, fname] = title_and_fname(base, name_tags)
    for i = 1:length(name_tags)
        if strcmp(name_tags{i}, 'psth'); name_tags{i} = upper(name_tags{i}); end
    end
    name = horzcat(base, name_tags);
    fname = strjoin(name, '_');
    title = title_case(strrep(fname, '_', ' '));
    strrep(title, 'To', 'to');
    fname = [fname '.fig'];
end

function rates = get_rates(spikes, bins, time)
    rates = histcounts(spikes, bins) / time; % spike counts per second
end

function plot_data(data_type, y, C, varargin)
    p = inputParser;
    p.addParamValue('subplot_args', [], @isnumeric);
    p.addParamValue('fig_handle', gcf);
    p.addParamValue('title_text', '', @ischar);
    p.addParamValue('y_dim', [], @isnumeric);

    p.parse(varargin{:});

    subplot_args = p.Results.subplot_args;
    fig_handle = p.Results.fig_handle;
    title_text = p.Results.title_text;
    y_dim = p.Results.y_dim;
      

    % Select the figure if a handle is provided
    if ~isempty(fig_handle)
        figure(fig_handle);
    end

    % Create subplot if specified
    if ~isempty(subplot_args)
        if isempty(subplot_args) || numel(subplot_args) ~= 3
            error('Invalid subplot arguments. Three arguments required (rows, columns, index).');
        else
            subplot(subplot_args(1), subplot_args(2), subplot_args(3));
        end
    end
    
    if strcmp(data_type, 'psth')
        % Plot the data
        x = C.psth.bins(1:end-1)/C.sps - C.psth.pre_stim_time;
        bar(x, y, 'k');
        hold on;
        ylabel('Normalized Spike Rate');
        xlabel('Time (s)');
        xlim([-C.psth.pre_stim_time max(x)]);
        if ~isempty(y_dim)
            ylim(y_dim);
        end
        % Add the shaded translucent gray bar
        patch([0 C.psth.pre_stim_time C.psth.pre_stim_time 0], [min(y) min(y) max(y) max(y)], 'k', ...
            'FaceAlpha', 0.2, 'EdgeColor', 'none');
        hold off;
    elseif contains(data_type, 'autocorr')
         x = C.bin_size_time:C.bin_size_time:C.autocorr.lags*C.bin_size_time;
         bar(x, y, 'k');
         xlabel('Lag (s)');
         ylabel('Autocorrelation');
         y_min = min(y);
         ylim([y_min, y_dim(2)]);
    else
        error('Unknown graph type')
    end 
   
    title(title_text);
end

function fig = initialize_figure(visible)
    fig = figure('Visible', visible);
    fig.Position = [0, 0, 800, 800];
end

function save_and_close_fig(fig, graph_dir, name, data_type, trial_num, varargin)
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

    graph_dir = fullfile(graph_dir, [data_type '_' trial_num]);

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
