dbstop if error

% Define the file name and path for the output CSV file
csv_file = 'output.csv';

control = {'IG160', 'IG163', 'IG176', 'IG178', 'IG180'};
stressed = {'IG154', 'IG156', 'IG158', 'IG175', 'IG177', 'IG179'};

% Open the file for writing
fid = fopen(csv_file, 'w');

% Define the header row for the CSV file
header = {'animal', 'condition', 'unit_type', 'unit_num', 'count'};

% Write the header row to the CSV file
fprintf(fid, '%s,', header{1:end-1});
fprintf(fid, '%s\n', header{end});

% Loop over all animals in the 'data' struct
for animal_idx = 1:length(data)
    % Determine the condition of the current animal
    animal = data(animal_idx).rootdir;
    animal = animal(strfind(animal, 'IG'):end);
    if ismember(animal, control)
        condition = 'control';
    elseif ismember(animal, stressed)
        condition = 'stressed';
    else
        condition = '';
    end

    % Process the 'good' units for the current animal
    process_units(data, animal, animal_idx, 'good', fid, condition);

    % Process the 'mua' units for the current animal
    process_units(data, animal, animal_idx, 'MUA', fid, condition);
end

% Close the CSV file
fclose(fid);

% Define a function to process units of a given type
function process_units(data, animal, animal_idx, unit_type, fid, condition)
    units = data(animal_idx).units.(unit_type);
    for unit_num = 1:length(units)
        timestamps = units{unit_num};

        % Check whether the tone was on at each timestamp
        event_count = 0;
        for i = 1:length(timestamps)
            timestamp = timestamps(i);

            % Check whether the tone was on at the current timestamp
            tone_on = false;
            for j = 1:length(data(animal_idx).ToneOn_ts)
                if data(animal_idx).ToneOn_ts(j) <= timestamp && data(animal_idx).ToneOff_ts(j) > timestamp
                    tone_on = true;
                    break;
                end
            end

            % If the tone was on, increment the event count
            if tone_on
                event_count = event_count + 1;
            end
        end

        % Write a row to the CSV file with the event count for this unit
        row = {animal, condition, unit_type, unit_num, event_count};
        fprintf(fid, '%s,%s,%s,%d,%d\n', row{1:end-1}, row{end});
    end
end

