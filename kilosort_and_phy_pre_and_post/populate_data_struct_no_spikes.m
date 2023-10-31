function data = populate_post_phy_data_structure(animal_dir, animal_data, ToneOnCode, ToneOffCode)

   
    
    data=struct();
    data.rootdir = animal_dir;
    animal = animal_dir(strfind(animal_dir, 'IG'):end);
    data.animal = animal;

    if ismember(animal, {'IG160', 'IG163', 'IG176', 'IG178', 'IG180'})
        data.group = 'control';
    elseif ismember(animal, {'IG154', 'IG156', 'IG158', 'IG177', 'IG179'})
        data.group = 'stressed';
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
    % This is necessary when there are spurious noise events that then trigger the
    % Tone Off code when they are turned off

    if strcmp(animal, 'IG155')
        a = 'foo';
    end

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
    data.tone_period_onsets = ToneOn_ts;
    data.tone_period_offsets = ToneOff_ts;

    % Define some constants
    sampling_rate = 30000; % Hz
    period_duration = 30; % seconds
    tones_per_period = period_duration; % there is one tone every second
    
    tone_onsets_expanded = zeros(1, length(data.tone_period_onsets) * tones_per_period);
    for j = 1:length(data.tone_period_onsets)
        start_of_period = data.tone_period_onsets(j);
        starts_of_tones = (start_of_period:1*sampling_rate:(tones_per_period-1)*sampling_rate+start_of_period);
        tone_onsets_expanded((j-1)*tones_per_period+1:(j-1)*tones_per_period+30) = starts_of_tones;
    end
    data.tone_onsets_expanded = tone_onsets_expanded;


end



