function updatedStruct = replaceStringInStruct(inputStruct, oldString, newString)
    updatedStruct = inputStruct;
    fieldNames = fieldnames(inputStruct);
    
    for i = 1:length(fieldNames)
        currentField = fieldNames{i};
        if isstruct(inputStruct.(currentField))
            % If the current field is a struct, call the function recursively            
            updatedStruct.(currentField) = replaceStringInStruct(inputStruct.(currentField), oldString, newString);
        elseif ischar(inputStruct.(currentField)) || isstring(inputStruct.(currentField))
            % If the current field is a char array or string, replace the old string with the new string
            updatedStruct.(currentField) = strrep(inputStruct.(currentField), oldString, newString);
        elseif iscell(inputStruct.(currentField))
            % If the current field is a cell array, iterate through the cell array
            cellArray = inputStruct.(currentField);
            for j = 1:numel(cellArray)
                if ischar(cellArray{j}) || isstring(cellArray{j})
                    cellArray{j} = strrep(cellArray{j}, oldString, newString);
                end
            end
            updatedStruct.(currentField) = cellArray;
        end
    end
end
