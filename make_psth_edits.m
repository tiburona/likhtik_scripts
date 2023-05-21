function make_psth_edits(data, graph_types, dir)

    dbstop if error   
    
    sps = 30000; % samples per second

    % Define the time window for the PSTH
    pre_stim_time = 0.05;
    post_stim_time = 0.65;
    bin_size_time = 0.01;

    C = struct( ...
        'sps', sps,'pre_stim_time', pre_stim_time, 'post_stim_time', post_stim_time, ...
        'pre_stim_samples', pre_stim_time*sps, 'post_stim_samples', post_stim_time*sps, ...
        'bin_size_samples', bin_size_time*sps, 'units_in_fig', 4, 'lags', -69:69 ... 
        );

    C.bins = 0 : C.bin_size_samples : C.pre_stim_samples + C.post_stim_samples;
    
    % Choose different subsets of trials in the experiment to graph
    trial_keys = {'30_trials', '60_trials', '150_trials', 'first_tones'}; 
    trial_selectors = {1:30, 1:60, 1:150, 1:30:150};

    trial_keys = {'150_trials'}; 
    trial_selectors = {1:150};

    for i = 1:length(trial_keys)
        new_dir = fullfile(dir, ['psth_' trial_keys{i}]);
        safeMakeDir(new_dir);
        [average_data, data] = collect_data(data, trial_selectors{i}, C);
        unit_graphs(data, new_dir, graph_types, trial_keys(i), C)
        average_graphs(average_data, graph_types, new_dir, trial_keys(i), C); 
        %four_panel_graph(avg_struct, new_dir, trial_keys(i), C);
    end
end

function [averages, data] = collect_data(data, trials, constants)

    % separate data into interneurons and pyramidal neurons
    data_keys = {'all_units', 'PN', 'IN'};
    groups = {'control', 'stressed'};
    graph_types = {'psth', 'autocorr'};

    for i_animal = 1:length(data)
        for i_unit = 1:length(data(i_animal).units.good)              
            unit = data(i_animal).units.good(i_unit);
            unit = calc_spike_data(unit.spike_times, data(i_animal).tone_onsets_expanded(trials), ...
                data(i_animal).tone_period_onsets, constants);
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
        for i_group = 1:length(groups)
            averages = select_group_and_get_averages(...
                averages, groups{i_group}, data_set, data_key, graph_types);       
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

function unit = calc_spike_data(spikes, tone_onsets, tone_period_onsets, C)

    found_spikes = false;

    num_trials = length(tone_onsets);
    raster_data = cell(num_trials, 1);
    psth_data = cell(num_trials, 1);
    all_rates = cell(num_trials, 1);
    normalized_psth = cell(num_trials, 1);
    autocorr = cell(num_trials, 1);
    
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
        rates = get_rates(psth_data{j}, C.bins, C);
        mean_rate = norm_factors{period_index}(1);
        std_dev = norm_factors{period_index}(2);
         
        if norm_factors{period_index}(2) == 0
            non_zero_indices = cellfun(@(x) x(2) ~= 0, norm_factors); 
            non_zero_values = cell2mat(norm_factors(non_zero_indices));
            std_dev = mean(non_zero_values(2));  
        end
 
        all_rates{j} = rates;
        normalized_psth{j} = (rates - mean_rate) / std_dev;

        % Collect autocorrelation data for each trial
        [autocorr{j}, lags] = xcorr(rates, 'coeff');
    end
    
    mean_rate = mean(cell2mat(all_rates));
    normalized_psth = mean(cell2mat(normalized_psth));

    unit.raster = raster_data;
    unit.psth = normalized_psth;
    unit.found_spikes = found_spikes;
    unit.lags = lags;
    unit.autocorr = xcorr(mean_rate, 'coeff');
end

function unit_graphs(data, graph_dir, graph_types, name_tags, C)
    for i_animal = 1:length(data)
        animal = data(i_animal);
        for i_graph_type = 1:length(graph_types)
            graph_type = graph_types{i_graph_type};
            fig = initialize_figure('on');
            for i_unit = 1:length(animal.units.good)
                if ~animal.units.good(i_unit).found_spikes; continue; end
                plot_unit(graph_dir, animal, i_unit, graph_type, name_tags, C); 
            end 
        end
    end
end

function plot_unit(graph_dir, animal, i_unit, graph_type, name_tags, C)
    unit = animal.units.good(i_unit);

    sp_args.psth = [C.units_in_fig*2, 1, mod(i_unit-1, C.units_in_fig)*2+2];
    sp_args.autocorr = [C.units_in_fig, 1, ...
        mod(i_unit, C.units_in_fig) + C.units_in_fig*double(mod(i_unit, C.units_in_fig) == 0)];
    
    if strcmp(graph_type, 'psth')
        subplot(C.units_in_fig*2, 1, mod(i_unit-1, C.units_in_fig)*2+1);
        plotSpikeRaster(unit.raster, 'AutoLabel', true, 'XLimForCell', ...
             [0 C.pre_stim_time + C.post_stim_time], 'EventShading', [.05, .1])
    end
    
    plot_data(graph_type, unit.(graph_type), C, 'subplot', sp_args.(graph_type));
       
    if i_unit == length(animal.units.good) || mod(i_unit, C.units_in_fig) == 0
        [marker1, marker2] = markers(animal, i_unit, C);
        fname_base = sprintf('unit_%d_to_%d_%s', marker1, marker2, animal.animal);
        [title, fname] = title_and_fname(fname_base, [graph_type name_tags]);
        save_and_close_fig(gcf, graph_dir, fname, 'figure_title', title);
    end

end

function [marker1, marker2] = markers(animal, i_unit, C)
    marker1 = idivide(int8(i_unit)-1, C.units_in_fig) * C.units_in_fig + 1;
    marker2 = min((idivide(int8(i_unit)-1, C.units_in_fig) + 1) * C.units_in_fig, ...
        length(animal.units.good));
end

function average_graphs(averages, graph_types, graph_dir, name_tags, C)
    datasets = {'all_units', 'PN', 'IN'};
    groups = {'control', 'stressed'};

    for i_dataset = 1:length(datasets)
        for i_group = 1:length(groups)
            create_animal_average_figures(averages.(datasets{i_dataset}), groups{i_group}, ...
                graph_types, graph_dir, [name_tags, {datasets{i_dataset}, groups{i_group}}], C) 
        end
    end    
end

function create_animal_average_figures(data, condition, graph_types, graph_dir, name_tags, C)
    group = data.(condition).animals;
    for i_type = 1:length(graph_types)
        group_fig = initialize_figure('on'); 
        for i_animal=1:length(group)
            animal = group(i_animal);
            plot_animal_average(group, animal, i_animal, graph_types, i_type, C) 
        end
        [title, fname] = title_and_fname('animal_averages', horzcat(name_tags, graph_types{i_type}, ...
            condition));
        save_and_close_fig(group_fig, graph_dir, fname, 'figure_title', title);
    end
end

function plot_animal_average(group, animal, i_animal, graph_types, i_type, C)
    if ~isempty(animal.units.good)
        data_type = graph_types{i_type};
        plot_data(data_type, animal.averages.(data_type), C, 'title_text', ...
            sprintf('Animal %s', animal.animal), 'subplot', [ceil(length(group) / 2), 2, i_animal]);
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

function rates = get_rates(spikes, bins, C)
    rates = histcounts(spikes, bins) / (C.bin_size_samples/C.sps); % spike counts per second
end

function plot_data(data_type, y, C, varargin)
    % Process optional arguments
    is_subplot = false;
    y_min = nan; y_max = nan;
    subplot_args = [];
    fig_handle = [];
    title_text = '';
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
        elseif ischar(varargin{i}) && strcmpi(varargin{i}, 'title_text')
            if i < length(varargin)
                title_text = varargin{i+1};
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
    
    title(title_text);

    if strcmp(data_type, 'psth')
        % Plot the data
        x = C.bins(1:end-1)/C.sps - C.pre_stim_time;
        bar(x, y, 'k');
        hold on;
        ylabel('Normalized Spike Rate');
        xlabel('Time (s)');
        xlim([-C.pre_stim_time max(x)]);
        if isnan(y_min)
            y_min = min(y); y_max = max(y);
        end
        ylim([y_min, y_max]);

        % Add the shaded translucent gray bar
        patch([0 C.pre_stim_time C.pre_stim_time 0], [min(y) min(y) max(y) max(y)], 'k', ...
            'FaceAlpha', 0.2, 'EdgeColor', 'none');
        hold off;
    elseif strcmp(data_type, 'autocorr')
        stem(C.lags, y);
        xlabel('Lag');
        ylabel('Autocorrelation');
    else
        error('Unknown graph type')
    end
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

