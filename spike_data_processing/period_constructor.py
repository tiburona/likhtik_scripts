import numpy as np


class PeriodConstructor:

    @property
    def earliest_period(self):
        return sorted([period for period in self.all_periods if not period.is_relative], 
                      key=lambda x: x.onset)[0]
    
    def get_all(self, attr):
        return [item for sublist in getattr(self, attr).values() for item in sublist]

    def prepare_periods(self):
        self.period_class = self.kind_of_data_to_period_type[self.kind_of_data]
        for boo, function in zip((False, True), (self.construct_periods, self.construct_relative_periods)):
            try:
                period_info = self.period_info
            except AttributeError:
                period_info = self.animal.period_info
            filtered_period_info = {
                k: v for k, v in period_info.items() if bool(v.get('relative')) == bool(boo)}
            for period_type in filtered_period_info:
                periods = getattr(self, f"{self.kind_of_data}_periods")
                periods[period_type] = function(period_type, filtered_period_info[period_type])

    def construct_periods(self, period_type, period_info):
        periods = []
        if not period_info:
            return []
        num_events = len([event for events_list in period_info['events'] for event in
                          events_list])  # all the events for this period type

        if self.calc_opts.get('events', {}).get(period_type, {}).get('selection') is not None:
            events = slice(*self.calc_opts['events'][period_type]['selection'])
        else:
            events = slice(0, num_events) # default is to take all events
        # indices of the events used in this data analysis
        selected_event_indices = list(range(num_events))[events]  
        # the time stamp of the beginning of a period
        period_onsets = period_info['onsets']  
        # the time stamps of things that happen within the period   
        period_events = period_info['events']          
        if self.kind_of_data == 'lfp': # type: ignore
            conversion_factor = self.lfp_sampling_rate/self.sampling_rate # type: ignore
            # For LFP, you need a subtraction for 0 indexing. For spikes, onsets are used to 
            # select time stamps, not elements of a Python iterable
            period_onsets = (np.array(period_onsets) * conversion_factor).astype(int) - 1
            period_events = (np.array(period_events) * conversion_factor).astype(int) - 1
        event_ind = 0
        # In this and the following method, periods are initialized with values in the sampling
        # rate with which their onsets were recorded.  Conversion to LFP sampling rates takes place
        # in the init function of those periods.
        for i, (onset, events) in enumerate(zip(period_onsets, period_events)):

            period_events = np.array([
                ev for j, ev in enumerate(events) if event_ind + j in selected_event_indices])
            event_ind += len(period_info['events'][i])
            periods.append(self.period_class(self, i, period_type, period_info, onset, 
                                             events=period_events, experiment=self.experiment)) # type: ignore
        return periods

    def construct_relative_periods(self, period_type, period_info):

        periods = []
        target_periods = getattr(self, f"{self.kind_of_data}_periods") # type: ignore
        paired_periods = target_periods[period_info['target']]
        exceptions = period_info.get('exceptions') 

        sampling_rate = self.lfp_sampling_rate if self.kind_of_data == 'lfp' else self.sampling_rate # type: ignore

        for i, paired_period in enumerate(paired_periods):
            i_key = str(i)
            if exceptions and i_key in exceptions:
                shift = exceptions[i_key]['shift']
                duration = exceptions[i_key]['duration']
            else:
                shift = period_info['shift']
                duration = period_info.get('duration')
            # if self is animal this is an lfp period
            if self.name == 'animal':  # type: ignore
                shift -= sum(self.calc_opts['lfp_padding']) # type: ignore
            shift_in_samples = shift * sampling_rate
            event_duration = paired_period.period_info['event_duration'] * sampling_rate
            onset = paired_period.onset + shift_in_samples

            event_starts = []
            for es in paired_period.event_starts:
                ref_es = es + shift_in_samples
                if ref_es + event_duration <= paired_period.onset:
                    event_starts.append(ref_es)
            event_starts = np.array(event_starts)
            duration = duration if duration else paired_period.duration
            reference_period = self.period_class(self, i, period_type, period_info, onset, 
                                                 events=event_starts, target_period=paired_period, 
                                                 is_relative=True, experiment=self.experiment)
            paired_period.paired_period = reference_period
            periods.append(reference_period)
        return periods
