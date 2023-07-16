function contam_rate = recalculate_contamination(directory, cluster_id)
    dbstop if error
    
    % assumes cluster_id is 0 indexed as in phy, changes it to Matlab
    % indexing
    cluster_id = cluster_id + 1;

    % Load the rez2.mat struct from the specified directory
    load(fullfile(directory, 'rez2.mat'), 'rez');

    % Create a copy of the rez struct called rez3
    rez3 = rez;

    cluster_ids = readNPY(fullfile(directory, 'spike_clusters.npy')) + 1;
    
    rez3.st3(:, 2) = cluster_ids;

    % Clear the field rez.est_contam_rate
    if isfield(rez3, 'est_contam_rate')
        rez3 = rmfield(rez3, 'est_contam_rate');
    end

    % Call the set_cutoff_ks function with rez3 as an argument
    rez3 = set_cutoff_ks(rez3);

    % Find the index of cluster_id in rez3.clusters and return the corresponding value in rez3.est_contam_rate
    cluster_idx = find(rez3.clusters == cluster_id);
    contam_rate = rez3.est_contam_rate(cluster_idx);
end
