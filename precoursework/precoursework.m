%% Understanding the data=

% Load the data
load('monkeydata_training.mat');

% Inspect the 'trial' variable
disp('Size of trial array:');
disp(size(trial));

disp('Fields in each trial struct:');
disp(fieldnames(trial(1, 1)));

% Access the first trial for the first reaching angle
exampleTrial = trial(1, 1);

disp('Trial ID:');
disp(exampleTrial.trialId);

disp('Size of spikes matrix (neurons x time):');
disp(size(exampleTrial.spikes));

disp('Size of handPos matrix (3D position x time):');
disp(size(exampleTrial.handPos));

% Peek at the first few entries in spikes
disp('First 10 spike events for neuron 1:');
disp(exampleTrial.spikes(1, 1:10));

% Peek at the first few hand positions
disp('First 10 hand positions (x, y, z):');
disp(exampleTrial.handPos(:, 1:10));


%% Population raster plot over one trial

exampleTrial = trial(1, 1);
spikeMatrix = exampleTrial.spikes;

[numNeurons, numTimeSteps] = size(spikeMatrix);

figure;
hold on;
for neuronIdx = 1:numNeurons
    spikeTimes = find(spikeMatrix(neuronIdx, :));
    plot(spikeTimes, neuronIdx * ones(size(spikeTimes)), 'k.', 'MarkerSize', 5);
end
xlabel('Time (ms)');
ylabel('Neuron Index');
title('Population Raster Plot (Single Trial)');
hold off;




%% Population raster plot over all trials

angles_deg = [30, 70, 110, 150, 190, 230, 310, 350];
figure;
hold on;
for angleIdx = 1:8
    for trialNum = 1:100
        handPos = trial(trialNum, angleIdx).handPos;
        plot(handPos(1, :), handPos(2, :), 'Color', [0.8, 0.8, 0.8]); % Light gray for individual trials
    end
    % Plot mean trajectory for this angle
    avgTrajectory = zeros(2, 1);
    for trialNum = 1:100
        handPos = trial(trialNum, angleIdx).handPos;
        avgTrajectory = avgTrajectory + handPos(1:2, end); % End position
    end
    avgTrajectory = avgTrajectory / 100;
    plot(avgTrajectory(1), avgTrajectory(2), 'o', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', sprintf('%d°', angles_deg(angleIdx)));
end
xlabel('X Position (mm)');
ylabel('Y Position (mm)');
title('Hand Endpoints for Each Reaching Angle');
legend show;
hold off;


%% Single neuron raster plot

% Choose a neuron and reaching angle
neuronIdx = 1;  % Choose a neuron (1 to 98)
angleIdx = 1;   % Choose a reaching angle (1 to 8)

% Create figure for raster plot
figure;
hold on;

% Loop over all trials for this reaching angle
for trialNum = 1:100
    spikeTrain = trial(trialNum, angleIdx).spikes(neuronIdx, :);
    spikeTimes = find(spikeTrain); % Find time indices with spikes
    plot(spikeTimes, trialNum * ones(size(spikeTimes)), 'k.', 'MarkerSize', 5);
end

xlabel('Time (ms)');
ylabel('Trial Number');
title(['Raster Plot for Neuron ' num2str(neuronIdx) ' across Trials (Angle ' num2str((angleIdx - 1) * 40 + 30) '°)']);
hold off;






%% PSTH 

% Choose neuron and angle
neuronIdx = 1;
angleIdx = 1;

% Number of trials
numTrials = 100;

% Initialize spike count vector
maxTime = 0;
for trialNum = 1:numTrials
    maxTime = max(maxTime, size(trial(trialNum, angleIdx).spikes, 2));
end
spikeCount = zeros(1, maxTime);

% Aggregate spikes across trials
for trialNum = 1:numTrials
    spikeTrain = trial(trialNum, angleIdx).spikes(neuronIdx, :);
    spikeCount(1:length(spikeTrain)) = spikeCount(1:length(spikeTrain)) + spikeTrain;
end

% Compute firing rate (Hz)
firingRate = spikeCount / numTrials / 0.001;

% Plot PSTH
figure;
bar(firingRate, 'k');
xlabel('Time (ms)');
ylabel('Firing Rate (Hz)');
title(['PSTH - Neuron ' num2str(neuronIdx) ' - Angle ' num2str((angleIdx - 1) * 40 + 30) '°']);




% apply a guassian kernel to the PSTH
kernel = gausswin(10);
psth_smoothed = conv(firingRate, kernel, 'same');

% plot the PSTH and the smoothed PSTH
figure;
subplot(2, 1, 1);





%%

% Choose neurons to plot (e.g., neurons 1 to 5)
neuronsToPlot = [1, 5, 10, 20, 30];
angles_deg = [30, 70, 110, 150, 190, 230, 310, 350]; % Reaching angles
angles_rad = angles_deg * pi / 180;

% Initialize storage for mean and std firing rates
numAngles = length(angles_deg);
numNeurons = length(neuronsToPlot);
meanRates = zeros(numNeurons, numAngles);
stdRates = zeros(numNeurons, numAngles);

% Loop over chosen neurons
for neuronIdx = 1:numNeurons
    neuron = neuronsToPlot(neuronIdx);

    for angleIdx = 1:numAngles
        firingRates = zeros(100, 1); % Store rates across trials

        for trialNum = 1:100
            spikes = trial(trialNum, angleIdx).spikes(neuron, :);
            firingRates(trialNum) = sum(spikes) / length(spikes) / 0.001; % Convert to Hz
        end

        % Mean and standard deviation across trials
        meanRates(neuronIdx, angleIdx) = mean(firingRates);
        stdRates(neuronIdx, angleIdx) = std(firingRates);
    end
end

% Plot Tuning Curves with Error Bars
figure;
hold on;
for neuronIdx = 1:numNeurons
    errorbar(angles_deg, meanRates(neuronIdx, :), stdRates(neuronIdx, :), '-o', 'LineWidth', 1.5);
end
xlabel('Reaching Angle (degrees)');
ylabel('Firing Rate (Hz)');
title('Tuning Curves for Selected Neurons');
legend(arrayfun(@(x) ['Neuron ' num2str(x)], neuronsToPlot, 'UniformOutput', false));
hold off;

