[]% Load Parameters
pdmRecipPump_Parameters %Pump
CAT_Pump_1051_DataFile_imported %CAD

mdl = 'pdmRecipPump';
open_system(mdl)

% Define fault parameter variations
numParValues = 5;
leak_area_set_factor = [0,1e-3,1e-2];
leak_area_set = leak_area_set_factor*TRP_Par.Check_Valve.In.Max_Area;
leak_area_set = max(leak_area_set,1e-9); % Leakage area cannot be 0
blockingfactor_set = [0.8,0.75,0.7];
bearingfactor_set = [0,2e-4,4e-4];

% Number of Samples per Simulation
nPerGroup_h = 500;
nPerGroup_f = 1;
% Feed default seed to rng (Random number generator)
rng('default');

% No fault simulations
leakArea = repmat(leak_area_set(1),nPerGroup_h,1);
blockingFactor = repmat(blockingfactor_set(1),nPerGroup_h,1);
bearingFactor = repmat(bearingfactor_set(1),nPerGroup_h,1);

% Seal Leak Simulation
for i = 1:2
    leakArea = [leakArea; repmat(leak_area_set(i+1),nPerGroup_f,1)];
    blockingFactor = [blockingFactor;repmat(blockingfactor_set(1),nPerGroup_f,1)];
    bearingFactor = [bearingFactor;repmat(bearingfactor_set(1),nPerGroup_f,1)];
end

% % Blocked Inlet Simulation
for i = 1:2
leakArea = [leakArea; repmat(leak_area_set(1),nPerGroup_f,1)];
blockingFactor = [blockingFactor;repmat(blockingfactor_set(i+1),nPerGroup_f,1)];
bearingFactor = [bearingFactor;repmat(bearingfactor_set(1),nPerGroup_f,1)];
end
% 
% % Bearing Wear Simulation
for i = 1:2
leakArea = [leakArea; repmat(leak_area_set(1),nPerGroup_f,1)];
blockingFactor = [blockingFactor;repmat(blockingfactor_set(1),nPerGroup_f,1)];
bearingFactor = [bearingFactor;repmat(bearingfactor_set(i+1),nPerGroup_f,1)];
end

% Creating Simulation Input
for ct = numel(leakArea):-1:1
    simInput(ct) = Simulink.SimulationInput(mdl);
    simInput(ct) = setVariable(simInput(ct),'leak_cyl_area_WKSP',leakArea(ct));
    simInput(ct) = setVariable(simInput(ct),'block_in_factor_WKSP',blockingFactor(ct));
    simInput(ct) = setVariable(simInput(ct),'bearing_fault_frict_WKSP',bearingFactor(ct));
    simInput(ct) = setVariable(simInput(ct),'noise_seed_offset_WKSP',ct-1);
end

% Run the simulation and create an ensemble to manage the simulation
% results

if isfolder('./Data')
    % Delete existing mat files
    delete('./Data/*.mat')
end   

[ok,e] = generateSimulationEnsemble(simInput,fullfile('.','Data'),'UseParallel',false,'ShowProgress',false);    %Set the UseParallel flag to true to use parallel computing capabilities
ens = simulationEnsembleDatastore(fullfile('.','Data'));

