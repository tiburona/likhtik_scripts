 % per animal and per group
    % center on firing rate during ISI + divide by standard deviation of
    % all firing rates
    % make plot with just the first pip. (by group)

    % just first two trials

    % make scatterplot of firing rate over all times by half maximum width
    % over all times see what patterns emerge

% Load the struct
load('./data.mat')

% Define some constants
sampling_rate = 30000; % Hz
period_duration = 30; % seconds
tone_duration = 0.05; % seconds
tones_per_period = period_duration; % there is one tone every second


for i=1:length(data)
    animal = data(i).rootdir;
    animal = animal(strfind(animal, 'IG'):end);
    data(i).animal = animal;

    tone_onsets_expanded = zeros(1, length(data(i).ToneOn_ts) * tones_per_period);
    for j = 1:length(data(i).ToneOn_ts)
        start_of_period = data(i).ToneOn_ts(j);
        starts_of_tones = (start_of_period:1*sampling_rate:(tones_per_period-1)*sampling_rate+start_of_period);
        tone_onsets_expanded((j-1)*tones_per_period+1:(j-1)*tones_per_period+30) = starts_of_tones;
    end
    data(i).ToneOn_ts_expanded = tone_onsets_expanded;
end

% Save the updated struct to a file
save('data.mat', 'data')