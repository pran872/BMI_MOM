load("monkeydata_training.mat")

% Process data with 30ms bins using gaussian filtering
processedData = smoothData(trial, 30, 'medianFilter', [5]);

% Access binned spike data
% processedData(trial,angle).binnedSpikes

binnedData = processedData(1,1).binnedSpikes;

% Access firing rates
ratesData = processedData(1,1).firingRates;

% Example script to plot firing rates for a specific trial
trialIdx = 1;  % Specify the trial index
angleIdx = 1;  % Specify the angle index

% Extract the firing rates for the chosen trial and angle
firingRates = processedData(trialIdx, angleIdx).firingRates;

% Determine the number of cells and bins
[numCells, numBins] = size(firingRates);

% Time axis for the plot
binSize = 30;  % Adjust this if different from your binSize
timeAxis = (0:numBins-1) * (binSize / 1000);  % Convert to seconds

% Plot firing rates
figure;
hold on;
for cellIdx = 1:numCells
    plot(timeAxis, firingRates(cellIdx, :), 'DisplayName', ['Cell ' num2str(cellIdx)]);
end
hold off;

% Add labels and legend
xlabel('Time (s)');
ylabel('Firing Rate (Hz)');
title(['Firing Rates for Trial ' num2str(trialIdx) ', Angle ' num2str(angleIdx)]);
legend('show');
grid on;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %%
% Test filtering of low firing neurons

[filteredData, cell_idxs] = filterLowFiringNeurons(processedData,80);
fprintf('%d Cells kept', size(cell_idxs,1))

% Extract the firing rates for the chosen trial and angle
firingRates = filteredData(trialIdx, angleIdx).firingRates;

% Determine the number of cells and bins
[numCells, numBins] = size(firingRates);

% Time axis for the plot
binSize = 30;  % Adjust this if different from your binSize
timeAxis = (0:numBins-1) * (binSize / 1000);  % Convert to seconds

% Plot firing rates
figure;
hold on;
for cellIdx = 1:numCells
    plot(timeAxis, firingRates(cellIdx, :), 'DisplayName', ['Cell ' num2str(cellIdx)]);
end
hold off;

% Add labels and legend
xlabel('Time (s)');
ylabel('Firing Rate (Hz)');
title(['Filtered Firing Rates for Trial ' num2str(trialIdx) ', Angle ' num2str(angleIdx)]);
legend('show');
grid on;


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%  Test filtering - keeping high variance neurons for each angle 
angle = 1;

[highVarNeurons, cell_idx] = selectVariableNeurons(processedData,angle,20);
fprintf('%d Cells kept', size(cell_idx,1))

% Extract the firing rates for the chosen trial and angle
firingRates = highVarNeurons(trialIdx, angle).firingRates;

% Determine the number of cells and bins
[numCells, numBins] = size(firingRates);

% Time axis for the plot
binSize = 30;  % Adjust this if different from your binSize
timeAxis = (0:numBins-1) * (binSize / 1000);  % Convert to seconds

% Plot firing rates
figure;
hold on;
for cellIdx = 1:numCells
    plot(timeAxis, firingRates(cellIdx, :), 'DisplayName', ['Cell ' num2str(cellIdx)]);
end
hold off;

% Add labels and legend
xlabel('Time (s)');
ylabel('Firing Rate (Hz)');
title(['High Var Neurons Firing Rates for Trial ' num2str(trialIdx) ', Angle ' num2str(angleIdx)]);
legend('show');
grid on;