% 
% The struct is called data. It contains 11 entries for 11 animals. Information about the timing of the stimuli in the field called ToneOn_ts.   It contains an array with 5 elements.  Each of these elements is the time stamp that marks the beginning of the "tone on" period.  The units in this field are in cycles.  These data were recorded at 30000 Hz so one cycle is 1/30000 s.  This period actually contains 30 events.  At the beginning of the period, and every second there after, there is a 50 millisecond tone.  So there should be 150 events total.  
% 
% The data about when the neurons fired is found in the data.units.good field for each animal.  This field contains a cell array (in row orientation) that have the neurons from which we're going to draw our spike times.  If you iterate through this cell array, you find a column vector of firing times, in cycles.
% 
% For each of these units (there may be several units per animal), I would like to make both a raster plot and a peristimulus time histogram, ideally with the raster plot above the PSTH. The y-axis of both plots should be events, and the x-axis should be time.  I'd like the period from which to look at data to begin 100 milliseconds before the beginning of the tone and extend 100 milliseconds after.  I would like a vertical bar in the raster plot and the PSTH that marks the tone onset and offset.  The bin for the PSTH should be 15 milliseconds. 


% Define the time window for the PSTH
pre_stim_time = 0.1; % 100 milliseconds before the beginning of the tone
post_stim_time = 0.1; % 100 milliseconds after the beginning of the tone
bin_size = 0.015; % 15 milliseconds bin size

% Iterate through each animal in the data struct
for i = 1:length(data)
    
    % Extract the tone onset times for this animal
    tone_onset_times = data(i).ToneOn_ts(1):1/30e3:data(i).ToneOn_ts(5);
    
    % Iterate through each unit for this animal
    for j = 1:length(data(i).units.good)
        
        % Extract the spike times for this unit
        spike_times = data(i).units.good{j};
        
        % Create a PSTH for this unit
        edges = (tone_onset_times-pre_stim_time-bin_size/2):bin_size:(tone_onset_times+post_stim_time+bin_size/2);
        psth = histcounts(spike_times,edges)/(bin_size*length(tone_onset_times));
        
        % Create a raster plot for this unit
        raster = zeros(length(tone_onset_times),length(edges));
        for k = 1:length(tone_onset_times)
            spike_indices = find(spike_times>=tone_onset_times(k)-pre_stim_time & spike_times<tone_onset_times(k)+post_stim_time);
            raster(k,spike_indices-(tone_onset_times(k)-pre_stim_time)/bin_size) = j;
        end
        
        % Plot the PSTH and raster for this unit
        figure;
        subplot(2,1,1);
        imagesc(edges-tone_onset_times(1),1:length(tone_onset_times),raster); hold on;
        line([0 0],[0 length(tone_onset_times)],'Color','r','LineWidth',2);
        xlim([-pre_stim_time post_stim_time]);
        xlabel('Time (s)');
        ylabel('Events');
        title(sprintf('Raster Plot for Unit %d',j));
        subplot(2,1,2);
        bar(edges-tone_onset_times(1),psth,'histc');
        hold on;
        line([0 0],[0 max(psth)*1.1],'Color','r','LineWidth',2);
        xlim([-pre_stim_time post_stim_time]);
        xlabel('Time (s)');
        ylabel('Firing Rate (Hz)');
        title(sprintf('PSTH for Unit %d',j));
        
    end
    
end
