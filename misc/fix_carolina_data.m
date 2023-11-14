

% New field values to be added
newFieldValues = ['CH272'; 'CH274'; 'CH275'];  % Example values

% New field name
newFieldName = 'animal';

originalStruct = single_cell_data;

% Add the new field to each entry in the structure array
for i = 1:length(originalStruct)
    % Get the field names of the current structure entry
    fieldNames = fieldnames(originalStruct(i));
    
    % Create a temporary structure with the first field
    tempStruct = struct;
    tempStruct.(fieldNames{1}) = originalStruct(i).(fieldNames{1});
    
    % Add the new field with its corresponding value
    tempStruct.(newFieldName) = newFieldValues(i);
    
    % Add the remaining fields from the original structure entry
    for j = 2:numel(fieldNames)
        tempStruct.(fieldNames{j}) = originalStruct(i).(fieldNames{j});
    end
    
    % Assign the temporary structure to the current entry
    newStruct(i) = tempStruct;
end

% Now newStruct is the updated structure array with the new field as the second field

