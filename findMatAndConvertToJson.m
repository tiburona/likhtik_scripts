function findMatAndConvertToJson(startPath)
    if nargin < 1
        startPath = pwd; % Current directory if no path is provided
    end

    % List all files and folders in the directory
    items = dir(startPath);

    for i = 1:length(items)
        if items(i).isdir
            % Skip the '.' and '..' directories
            if strcmp(items(i).name, '.') || strcmp(items(i).name, '..')
                continue;
            end
            % Recursive call for subdirectories
            findMatAndConvertToJson(fullfile(startPath, items(i).name));
        else
            [~, name, ext] = fileparts(items(i).name);
            if strcmp(ext, '.mat')
                % Load .mat file and convert to JSON
                data = load(fullfile(startPath, items(i).name));
                jsonStr = jsonencode(data);
                
                % Write JSON string to file
                jsonFileName = fullfile(startPath, [name '.json']);
                fid = fopen(jsonFileName, 'w');
                if fid == -1
                    error('Cannot open file for writing: %s', jsonFileName);
                end
                fwrite(fid, jsonStr, 'char');
                fclose(fid);
            end
        end
    end
end
