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
binSize = 25;  % Adjust this if different from your binSize
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



