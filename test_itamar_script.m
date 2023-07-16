dbstop if error

dir = '/Users/katie/likhtik/data/test_itamar_lfp_script/IG158/';

openNEV([dir,'safety.nev'])
load([dir,'safety.mat'])
openNSx('uv',[dir,'safety.ns3'])

% Itamar says "% downsampling to get the signal to 2kHz" but NS3 is already
% downsampled and ds_rate = 1.  Maybe this doesn't matter?
ds_rate = NS3.MetaTags.SamplingFreq/2000; 

names={'dlHPC', 'BLA'};

% I am assuming that the most recent version of the script of Itamar's I'm
% looking at used a different recording setup. It erases electrodes 3, 5,
% and 6.  I am going to erase all non-LFP electrodes.
index_to_keep = ismember(1:size(NS3.Data, 1), [1, 3]);  
NS3.ElectrodesInfo = NS3.ElectrodesInfo(index_to_keep);
NS3.Data = NS3.Data(index_to_keep, :);  


NS3.Data=double(NS3.Data);

% This is equivalent to 1:NS3.MetaTags.DataPoints
LFPs.ts = linspace(1, NS3.MetaTags.DataPoints, NS3.MetaTags.DataPoints); 
LFPs.dirname = dir; % called filename in Itamar's script; appears to be a dir.

for a=1:length(names)
    LFPs.dn60preRMS.(names{a}) = removeLineNoise_SpectrumEstimation(NS3.Data(a,:), 2000, ...
        ['NH=5','LF=60'])';
    LFPs.(names{a})=NS3.Data(a,:);
end


for a=1:length(names) % data needs to be RMS'ed
    LFPs.RMS.(names{a})=LFPs.dn60preRMS.(names{a})/rms(LFPs.dn60preRMS.(names{a}));
end


toneOnCode = 65502;

trial_start = double(NEV.Data.SerialDigitalIO.TimeStampSec(NEV.Data.SerialDigitalIO.UnparsedData == ...
    65502)); %trialstart points come from NEV file
% Katie: I don't think the following line does anything
trial_stop = NEV.Data.SerialDigitalIO.TimeStampSec(NEV.Data.SerialDigitalIO.UnparsedData == 65535);
LFPs.OnTimes=trial_start-1; % Katie wants to know why we're starting a full second before trial start?  
% Does it have to do with the light?  Probably
LFPs.OffTimes=trial_start + 31;  

for a=1:length(LFPs.OnTimes)
    start = LFPs.OnTimes(a)*2000;
    precsStart = (LFPs.OnTimes(a)-30)*2000;
    stop = LFPs.OffTimes(a)*2000;
    precsStop = (LFPs.OffTimes(a)-30)*2000;
    LFPs.CSts{a} = LFPs.ts(start:stop-1);
    LFPs.pCSts{a} = LFPs.ts(precsStart:precsStop-1);
end

% This code block repeats twice in I's script
for a = 1:length(names)
    for b = 1:length(LFPs.CSts)
        ind = cell2mat(LFPs.CSts(b));
        % deleted two lines that don't do anything
        [LFPs.Raw.(names{a}).CS{b}]=((LFPs.RMS.(names{a})(ind(1):ind(end))));
    end
end  


% after this Itamar just saves the file
% Here starts logic from 'PowerFromLFPs' script




freqs = {'TonePwrTheta', 'TonePwrGamma'};

for a=1:length(freqs)
    for b=1:length(names)
        for c = 1:5
            altogether = 'foo';
            crunched = 'bar';

            mean = 'baz'
        end
    end
end


function altogether = getAltogether()
    
end

animalsDef = {'IG158'};

% skipping lines 53-58 that define an empty struct

LFPsallDef.('IG158') = LFPs;




for a = 1:length(names)

    for b=1:5
         %Theta
         [LFPsallDef.(animalsDef{a}).TonePwrTheta.(names{b}).CSs.(CSname{c}),Freq_theta,Time_pw]=mtcsg(LFPsallDef.(animalsDef{a}).FiltContTheta.(names{b})(floor(LFPsallDef.(animalsDef{a}).CSts{1,c})),2048,2000,1000,980,2);
         [LFPsallDef.(animalsDef{a}).TonePwrTheta.(names{b}).pCSs.(pCSname{c}),Freq_theta,Time_pw]=mtcsg(LFPsallDef.(animalsDef{a}).FiltContTheta.(names{b})(floor(LFPsallDef.(animalsDef{a}).pCSts{1,c})),2048,2000,1000,980,2);
         %Delta
        % [LFPsallDef.(animalsDef{a}).TonePwrDelta.(names{b}).CSs.(CSname{c}),Freq_pw,Time_pw]=mtcsg(LFPsallDef.(animalsDef{a}).FiltContDelta.(names{b})(floor(LFPsallDef.(animalsDef{a}).CSts{1,c})),2048,2000,1000,980,2);
         %[LFPsallDef.(animalsDef{a}).TonePwrDelta.(names{b}).pCSs.(pCSname{c}),Freq_pw,Time_pw]=mtcsg(LFPsallDef.(animalsDef{a}).FiltContDelta.(names{b})(floor(LFPsallDef.(animalsDef{a}).pCSts{1,c})),2048,2000,1000,980,2);

        
     end
 end





