dbstop if error

write_csv(single_cell_data, 'firing_rates_by_unit.csv', 'by_unit');
write_csv(single_cell_data, 'firing_rates_by_stereotrode.csv', 'by_stereotrode');

function write_csv(data, output_file, method)

    % Open the output CSV file
    fid = fopen(output_file, 'w');

    % Write the header to the CSV file
    if strcmp(method, 'by_unit')
        fprintf(fid, 'animal,condition,unit_type,unit_num,period,spike_rate\n');
    elseif strcmp(method, 'by_stereotrode')
        fprintf(fid, 'animal,condition,unit_type,stereotrode,period,spike_rate\n');
    end

    % Process the data for each animal
    for animal_idx = 1:length(data)

        % Process good and MUA units
        process_units_or_stereotrodes(fid, data, animal_idx, 'good', method);
        if isfield(data(animal_idx).electrodes, 'MUA') % one animal doesn't have any MUA; this line avoids an unrecognized field name error
            process_units_or_stereotrodes(fid, data, animal_idx, 'MUA', method);
        end
    end

    % Close the output CSV file
    fclose(fid);
end

function process_units_or_stereotrodes(fid, data, animal_idx, unit_type, method)
    if strcmp(method, 'by_unit')
        elements = data(animal_idx).units.(unit_type);
    elseif strcmp(method, 'by_stereotrode')

        elements = group_units_by_stereotrode(data(animal_idx).units.(unit_type), data(animal_idx).electrodes.(unit_type));
    end
    
    for element_idx = 1:length(elements)
        
        if isempty(elements{element_idx}) % Some stereotrodes would not have spikes
            continue
        end
        % Calculate spike rates for the element (unit or stereotrode)
        spike_rates = calculate_spike_rates(data, animal_idx, elements{element_idx});

        % Write a row to the CSV file with the spike rate for this element in different periods
        write_spike_rates_to_csv(fid, data, animal_idx, unit_type, element_idx, spike_rates, method);
    end
end

function spike_rates = calculate_spike_rates(data, animal_idx, timestamps)
    % Constants
    CYCLES_PER_SECOND = 30000;
    SECONDS_BEFORE_FIRST_TONE = 60;
    
    % Initialize event counts for different conditions
    event_count_pre_tone = 0;
    event_count_inter_tone = 0;
    event_count_post_beep_0_300 = 0;
    event_count_post_beep_301_600 = 0;
    event_count_late_post_beep = 0;

    % Get ToneOn_ts_expanded
    tone_on_ts_expanded = data(animal_idx).ToneOn_ts_expanded;

    % Get the start of the recording (60 seconds before the first ToneOn)
    recording_start = data(animal_idx).ToneOn_ts(1) - SECONDS_BEFORE_FIRST_TONE * CYCLES_PER_SECOND;

    % Check the condition for each timestamp
    for i = 1:length(timestamps)
        timestamp = timestamps(i);
        % Pre-tone condition
        if timestamp >= recording_start && timestamp < data(animal_idx).ToneOn_ts(1)
            event_count_pre_tone = event_count_pre_tone + 1;
            continue;
        end
        
        % Inter-tone condition
        inter_tone = false;
        for j = 1:length(data(animal_idx).ToneOff_ts) - 1
            if data(animal_idx).ToneOff_ts(j) < timestamp && data(animal_idx).ToneOn_ts(j + 1) > timestamp
                event_count_inter_tone = event_count_inter_tone + 1;
                inter_tone = true;
                break;
            end
        end
        if inter_tone
            continue;
        end
     
        % Post-beep (0-600 ms) and Late post-beep (>600 ms) conditions
        for j = 1:length(tone_on_ts_expanded)
            time_diff = timestamp - tone_on_ts_expanded(j);
            if time_diff >= 0 && time_diff <= 300 * CYCLES_PER_SECOND / 1000
                event_count_post_beep_0_300 = event_count_post_beep_0_300 + 1;
                break;
            elseif time_diff >= 301 && time_diff <= 600 * CYCLES_PER_SECOND / 1000
                event_count_post_beep_301_600 = event_count_post_beep_301_600 + 1;
                break;
            elseif time_diff > 600 * CYCLES_PER_SECOND / 1000 && time_diff < 1000 * CYCLES_PER_SECOND / 1000
                event_count_late_post_beep = event_count_late_post_beep + 1;
                break;
            end
        end
    end
    
    % Initialize the elapsed time for different conditions
    elapsed_time_pre_tone = (data(animal_idx).ToneOn_ts(1) - recording_start) / CYCLES_PER_SECOND;
    elapsed_time_inter_tone = sum(data(animal_idx).ToneOn_ts(2:end) - data(animal_idx).ToneOff_ts(1:end-1)') / CYCLES_PER_SECOND;
    elapsed_time_post_beep_0_300 = (301 / 1000) * length(tone_on_ts_expanded);
    elapsed_time_post_beep_301_600 = (300 / 1000) * length(tone_on_ts_expanded);
    elapsed_time_late_post_beep = ((1000 - 600) / 1000) * length(tone_on_ts_expanded);
    
    % Calculate spike rates for different conditions
    spike_rates.pre_tone = event_count_pre_tone / elapsed_time_pre_tone;
    spike_rates.inter_tone = event_count_inter_tone / elapsed_time_inter_tone;
    spike_rates.post_beep_0_300 = event_count_post_beep_0_300 / elapsed_time_post_beep_0_300;
    spike_rates.post_beep_301_600 = event_count_post_beep_301_600 / elapsed_time_post_beep_301_600;
    spike_rates.late_post_beep = event_count_late_post_beep / elapsed_time_late_post_beep;

    end

function write_spike_rates_to_csv(fid, data, animal_idx, unit_type, element_idx, spike_rates, method)
    animal = data(animal_idx).animal;

    control = {'IG160', 'IG163', 'IG176', 'IG178', 'IG180'};
    stressed = {'IG154', 'IG156', 'IG158', 'IG177', 'IG179'};
    
    if ismember(animal, control)
        condition = 'control';
    elseif ismember(animal, stressed)
        condition = 'stressed';
    else
        condition = '';
    end

    % Write spike rate rows to the CSV file for each period
    periods = fieldnames(spike_rates);
    for i = 1:length(periods)
        period = periods{i};
        spike_rate = spike_rates.(period);
    
        if strcmp(method, 'by_unit')
            fprintf(fid, '%s,%s,%s,%d,%s,%.2f\n', animal, condition, unit_type, element_idx, period, spike_rate);
        elseif strcmp(method, 'by_stereotrode')
            fprintf(fid, '%s,%s,%s,%d,%s,%.2f\n', animal, condition, unit_type, element_idx, period, spike_rate);
        end
    end
end

function stereotrode_units = group_units_by_stereotrode(units, electrodes)
    stereotrode_units = repmat({[]}, 1, 7);
    for i = 1:length(units)
        stereotrode = find_stereotrode(electrodes{i});
        stereotrode_units{stereotrode} = vertcat(stereotrode_units{stereotrode}, units{i});
    end
end



function stereotrode = find_stereotrode(electrodes)
    % Function to find the stereotrode given the electrodes

    % Define the stereotrode mappings
    stereotrode_map = {[1, 3], [4, 6], [5, 7], [6, 8], [9, 11], [10, 12], [13, 15]};

    % Find the stereotrode
    for i = 1:length(stereotrode_map)
        if ~isempty(intersect(electrodes, stereotrode_map{i}))
            stereotrode = i;
            return;
        end
    end

    % If no matching stereotrode is found, return NaN
    stereotrode = NaN;
end
            