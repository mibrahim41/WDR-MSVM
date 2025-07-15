%% Extract and Write Features to Data
ens = simulationEnsembleDatastore(fullfile('.','Data'));

ens.DataVariables = [ens.DataVariables; ...
    "fPeak"; "pLow"; "pMid"; "pHigh"; "pKurtosis"; ...
    "qMean"; "qVar"; "qSkewness"; "qKurtosis"; ...
    "qPeak2Peak"; "qCrest"; "qRMS"; "qMAD"; "qCSRange"];
ens.ConditionVariables = ["LeakFault","BlockingFault","BearingFault"];

while hasdata(ens)

    % Read member data
    data = read(ens);

    % Preprocess and extract features from the member data
    [flow,flowP,flowF,faultValues] = preprocess(data);
    feat = extractCI(flow,flowP,flowF);

    % Add the extracted feature values to the member data
    dataToWrite = [faultValues, feat];
    writeToLastMemberRead(ens,dataToWrite{:})
end

%% Update Data Columns
ens = simulationEnsembleDatastore(fullfile('.','Data'));
reset(ens)
ens.SelectedVariables = [...
    "fPeak","pLow","pMid","pHigh","pKurtosis",...
    "qMean","qVar","qSkewness","qKurtosis",...
    "qPeak2Peak","qCrest","qRMS","qMAD","qCSRange",...
    "LeakFault","BlockingFault","BearingFault"];
idxLastFeature = 14;

% Load the condition indicator data into memory
data = gather(tall(ens));

pdmRecipPump_Parameters %Pump
CAT_Pump_1051_DataFile_imported %CAD

mdl = 'pdmRecipPump';
open_system(mdl)
leak_area_set_factor = [0,1e-3,1e-2];
leak_area_set = leak_area_set_factor*TRP_Par.Check_Valve.In.Max_Area;
leak_area_set = max(leak_area_set,1e-9); % Leakage area cannot be 0
blockingfactor_set = [0.8,0.75,0.7];
bearingfactor_set = [0,2e-4,4e-4];
leak_val_vec = leak_area_set;
block_val_vec = blockingfactor_set;
bear_val_vec = bearingfactor_set;

%%
mask_healthy = data.LeakFault == leak_val_vec(1) &...
    data.BlockingFault == block_val_vec(1) &...
    data.BearingFault == bear_val_vec(1);

mask_leak_min = data.LeakFault == leak_val_vec(2);
mask_leak_maj = data.LeakFault == leak_val_vec(3);

mask_block_min = data.BlockingFault == block_val_vec(2);
mask_block_maj = data.BlockingFault == block_val_vec(3);

mask_bear_min = data.BearingFault == bear_val_vec(2);
mask_bear_maj = data.BearingFault == bear_val_vec(3);

%%
data.FaultClass(mask_healthy) = 0;
data.FaultClass(mask_leak_min) = 1;
data.FaultClass(mask_block_min) = 2;
data.FaultClass(mask_bear_min) = 3;
data.FaultClass(mask_leak_maj) = 4;
data.FaultClass(mask_block_maj) = 5;
data.FaultClass(mask_bear_maj) = 6;


%% Write Data to Double and Save File
data_arr = table2array(data);
