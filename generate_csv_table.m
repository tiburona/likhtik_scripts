function generate_csv_table(id_list, field_list, filepath, data_dir, remove_strings)
% Generate a table with headings from id_list and subheadings from field_list
% Remove remove_strings from the column names

% This function was used to generate the events spreadsheet that clarified
% the relationship between UnparsedData and TimeStamps and made it clear
% that there was a discontinuity in how data are recorded.

% Initialize an empty table
T = table();

% Loop over the IDs to calculate the maximum number of rows
max_num_rows = 0;
for i = 1:length(id_list)
    % Get the current ID
    current_id = id_list{i};
    
    % Extract the data for the current ID
    current_data = extract_data_for_id(current_id, field_list, data_dir);
    
    % Update the maximum number of rows
    max_num_rows = max(max_num_rows, size(current_data.data, 1));
end

% Loop over the IDs and extract the data
for i = 1:length(id_list)
    % Get the current ID
    current_id = id_list{i};
    
    % Extract the data for the current ID
    current_data = extract_data_for_id(current_id, field_list, data_dir);
    
    % Add the data to the table with prepended ID and field name
    col_names = strcat(current_id, '_', current_data.field_names);
    for j=1:length(remove_strings)
        col_names = strrep(col_names, remove_strings{j}, '');
    end
    current_table = array2table(current_data.data, 'VariableNames', col_names);
    
    % Pad the table if necessary
    num_rows_to_add = max_num_rows - size(current_table, 1);
    if num_rows_to_add > 0
        current_table(end+1:end+num_rows_to_add, :) = {''};
    end
    
    % Split the nested tables before adding them to the main table
    current_table = splitvars(current_table);
    
    % Add the data to the main table
    T = [T, current_table];
end

% Write the table to a CSV file
writetable(T, filepath);
end


function data = extract_data_for_id(id, field_list, data_dir)
% Extract row and column vectors for an ID and return as a struct

% Initialize the data struct
data = struct();

% Loop over the fields and extract the data
for i = 1:length(field_list)
    % Get the current field name
    current_field = field_list{i};
    
    % Extract the data for the current field
    current_data = extract_data_for_field(id, current_field, data_dir);
    
    % Transpose row vectors if necessary
    if isrow(current_data)
        current_data = current_data';
    end
    
    % Add the data to the struct
    data.data(:,i) = current_data;
    data.field_names{i} = current_field;
end

end

function data = extract_data_for_field(id, field, data_dir)
% Extract data for a field from a MAT file.

% Load the data from the MAT file.
filename = fullfile(data_dir, sprintf('%s.mat', id));
loaded_data = load(filename);

% Extract the data for the specified field.
data = eval(sprintf('loaded_data.%s', field));

% Convert numeric data to cell arrays
if isnumeric(data)
    data = num2cell(data);
end

% Transpose the row vectors.
if isrow(data)
    data = data.';
end
end

