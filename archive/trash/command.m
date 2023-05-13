search_string = 'generate_csv_table';

history_file = 'command_history.txt';

% Redirect Command Window output to a file
diary(history_file);

% Search for the string using regular expressions
matches = regexp(history, search_string, 'match', 'ignorecase');

% Stop redirecting Command Window output
diary off;

% Read the command history from the file
history = fileread(history_file);


% Display the matched commands
if ~isempty(matches)
    disp('Matching commands:');
    for i = 1:numel(matches)
        disp(matches{i});
    end
else
    disp('No matches found.');
end
