function data=process_post_phy_ks(day_dir,animal_data, ToneOnCode, ToneOffCode)

spt=readNPY([day_dir,'\spike_times.npy']);
cl=readNPY([day_dir,'\spike_clusters.npy']);
features=readNPY([day_dir,'\pc_features.npy']);
[cids,cgs]=getclustermarkings(day_dir);

data=struct();
data.rootdir=day_dir;
fieldnames={'noise','MUA','good'};
for a=max(cl)'+1;end

if ~isempty(spt)
data.units.good=cell(1,sum(cgs==2));
data.units.MUA=cell(1,sum(cgs==1));
data.units.noise=cell(1,sum(cgs==0));
counts=[1 1 1];
for a=1:length(cids)
    spike_times=spt(cl==cids(a));
    data.units.(fieldnames{cgs(a)+1}){counts(cgs(a)+1)}=spike_times;
    data.electrodes.(fieldnames{cgs(a)+1}){counts(cgs(a)+1)}=electrodes(a);
    [IsolDis,Lratio]=IsolationDistance(double(reshape(features,size(features,1),[])),find(cl==cids(a)),[],find(cgs==0));
    data.clustermetrics.(fieldnames{cgs(a)+1}).IsolationDistance(counts(cgs(a)+1))=IsolDis;
    data.clustermetrics.(fieldnames{cgs(a)+1}).Lratio(counts(cgs(a)+1))=Lratio;
    counts(cgs(a)+1)=counts(cgs(a)+1)+1;
end
else
    data.units.good=[];
    data.units.MUA=[];
    data.units.noise=[];
end
try
data=addwaveforms(data);
end
try
electrodenumbers=vertcat(animal_data.NS6.ElectrodesInfo.ElectrodeID);
catch
    disp('Error. Entering debug mode.');
    keyboard
end
trunc_data=animal_data.NS6.Data;
    
   

trunc_data=nanmedian(trunc_data,1);
[trunc_data] = removeLineNoise_SpectrumEstimation(trunc_data, 30000, 'LF = 60 NH = 5');
trunc_data=lowpass(single(trunc_data),300,30000);
trunc_data=trunc_data(1:15:end);
% PL=animal_data.NS5.Data(electrodenumbers==32,:);
% 
% [PL] = removeLineNoise_SpectrumEstimation(PL, 30000, 'LF = 60 NH = 5');
% PL=lowpass(single(PL),300,30000);
% PL=PL(1:15:end);
% data.LFP_PL=int16(PL*4);
% mmf = memmapfile([day_dir,'\data_binary.bin'], 'Format', {'int16', size(animal_data.NS5.Data), 'x'});
% trunc_data=mmf.Data.x(:,1:15:end);




trunc_data=int16(trunc_data*4);%downsample to 2000 Hz
data.LFP_ts=1:15:size(animal_data.NS6.Data,2);
data.LFP_med=nanmedian(trunc_data);
data.LFP_full=trunc_data;
clear trunc_data;
data.Video_ts=animal_data.NEV.Data.VideoSync.TimeStamp;


% Find the indices where ToneOn and ToneOff codes occur
ToneOn = animal_data.NEV.Data.SerialDigitalIO.UnparsedData == ToneOnCode;
ToneOff = animal_data.NEV.Data.SerialDigitalIO.UnparsedData == ToneOffCode;
ToneOn_ts = animal_data.NEV.Data.SerialDigitalIO.TimeStamp(ToneOn);
ToneOff_ts = animal_data.NEV.Data.SerialDigitalIO.TimeStamp(ToneOff);

% Combine ToneOn and ToneOff timestamps and event IDs into a table and sort by time
events = table([ToneOn_ts(:); ToneOff_ts(:)], [ones(sum(ToneOn), 1); -ones(sum(ToneOff), 1)], 'VariableNames', {'eventTime', 'eventID'});
events = sortrows(events, {'eventTime'}, {'ascend'});

% Iterate through events and determine whether each ToneOff is valid
tone_on = false;
valid_ToneOff = false(size(ToneOff));
for i = 1:height(events)
    if events.eventID(i) == 1 % ToneOn event
        tone_on = true;
    else % ToneOff event
        if tone_on
            valid_ToneOff(i) = true;
            tone_on = false;
        end
    end
end

% Only keep the valid tone off codes
ToneOff_ts = events.eventTime(valid_ToneOff);

% Find the indices of the tone on and tone off events
ToneOn_ev = find(ToneOn);
ToneOff_ev = find(ToneOff & valid_ToneOff);

% Store relevant data in output struct
data.ToneOn_ts = ToneOn_ts;
data.ToneOff_ts = ToneOff_ts;
