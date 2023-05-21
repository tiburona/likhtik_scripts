function autocorr(data)

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
        

    end
end
