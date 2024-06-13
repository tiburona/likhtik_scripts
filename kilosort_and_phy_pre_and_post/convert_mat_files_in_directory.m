dbstop if error

% Define the directory path
directoryPath = '/Users/katie/likhtik/CH_EXT';

%animals = {'CH054'; 'CH069'; 'CH129'; 'CH130'; 'CH131'; 'CH134'; 'CH135'; 'CH151'; 'IG158'; 'CH152'; 'CH154'};

animals = {'CH069'};


for i = 1:length(animals)

    dirPath = fullfile(directoryPath, animals{i});
    findMatAndConvertToJson(dirPath);
end
