function data = process_post_phy_ks(animal_dir, animal_data, ToneOnCode, ToneOffCode)

    spt = readNPY(fullfile(animal_dir,'spike_times.npy'));
    cl = readNPY(fullfile(animal_dir, 'spike_clusters.npy'));
    features = readNPY(fullfile(animal_dir,'pc_features.npy'));
    [cids,cgs] = getclustermarkings(animal_dir);
    
    result = regexp(animal_dir, 'IG(\w{3})', 'tokens');
    animal_num = result{1}{1};
    electrodes = read_electrode_tsv('/Users/katie/likhtik/data/single_cell_data/peak_electrodes.tsv', animal_num);
    
    data=struct();
    data.rootdir = animal_dir;
    fieldnames={'noise','MUA','good'};
    
    for a=max(cl)'+1;end
    if ~isempty(spt)
        data.units.good=cell(1,sum(cgs==2));
        data.units.MUA=cell(1,sum(cgs==1));
        data.units.noise=cell(1,sum(cgs==0));
        counts=[1 1 1];
        for a=1:length(cids)
            spike_times=spt(cl==cids(a));
            data.units.(fieldnames{cgs(a)+1}){counts(cgs(a)+1)} = spike_times;
            if cgs(a) > 0
                data.electrodes.(fieldnames{cgs(a)+1}){counts(cgs(a)+1)} = electrodes(cids(a));
            end
            [IsolDis,Lratio]=IsolationDistance(double(reshape(features,size(features,1),[])),find(cl==cids(a)),[],find(cgs==0));
            data.clustermetrics.(fieldnames{cgs(a)+1}).IsolationDistance(counts(cgs(a)+1))=IsolDis;
            data.clustermetrics.(fieldnames{cgs(a)+1}).Lratio(counts(cgs(a)+1))=Lratio;
            counts(cgs(a)+1)=counts(cgs(a)+1)+1;
        end
        else
            data.units.good=[];
            data.units.MUA=[];
            data.units.noise=[];
    end
    
    % Find the indices where ToneOn and ToneOff codes occur
    ToneOn = animal_data.NEV.Data.SerialDigitalIO.UnparsedData == ToneOnCode;
    ToneOff = animal_data.NEV.Data.SerialDigitalIO.UnparsedData == ToneOffCode;
    ToneOn_ts = animal_data.NEV.Data.SerialDigitalIO.TimeStamp(ToneOn);
    ToneOff_ts = animal_data.NEV.Data.SerialDigitalIO.TimeStamp(ToneOff);
    
    % Combine ToneOn and ToneOff timestamps and event IDs into a table and sort by time
    events = table([ToneOn_ts(:); ToneOff_ts(:)], [ones(sum(ToneOn), 1); -ones(sum(ToneOff), 1)], 'VariableNames', {'eventTime', 'eventID'});
    events = sortrows(events, {'eventTime'}, {'ascend'});
    
    % Iterate through events and determine whether each ToneOff is valid
    tone_on = false;
    valid_ToneOff = false(size(ToneOff));
    for i = 1:height(events)
        if events.eventID(i) == 1 % ToneOn event
            tone_on = true;
        else % ToneOff event
            if tone_on
                valid_ToneOff(i) = true;
                tone_on = false;
            end
        end
    end
    
    % Only keep the valid tone off codes
    ToneOff_ts = events.eventTime(valid_ToneOff);
    
    % Store relevant data in output struct
    data.ToneOn_ts = ToneOn_ts;
    data.ToneOff_ts = ToneOff_ts;

end

function cluster_electrode_map = read_electrode_tsv(filepath, animal)
    % Read TSV file
    opts = detectImportOptions(filepath, 'Delimiter', '\t', 'FileType', 'text');
    opts.VariableNamesLine = 1;
    opts.VariableTypes = {'string', 'double', 'string', 'string'};
    data = readtable(filepath, opts);
    
    % Filter rows based on the input animal
    filtered_data = data(strcmp(data.Animal, animal), :);
    
    % Create the containers.Map object
    cluster_electrode_map = containers.Map('KeyType', 'double', 'ValueType', 'any');
    
    % Iterate over the filtered data rows and populate the map
    for i = 1:height(filtered_data)
        cluster = double(filtered_data.Cluster(i));
        electrodes_str = filtered_data.Electrodes{i};
        electrodes = sort(str2double(split(electrodes_str, ',')));
        
        % Add the struct to the map with the cluster number as the key
        cluster_electrode_map(cluster) = electrodes;
    end
end

