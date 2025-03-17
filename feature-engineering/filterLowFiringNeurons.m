function [filteredData, selectedIndices] = filterLowFiringNeurons(processedData, percentToKeep, angleIndices)
%----------------------------------------------------------------------
% FILTERLOWFIRINGNEURONS - Filters out neurons with low mean firing rates
%
% Syntax:  [filteredData, selectedIndices] = filterLowFiringNeurons(processedData, percentToKeep, angleIndices)
%
% Inputs:
%   processedData - Structure with firing rates from smoothData function
%   percentToKeep - Percentage of top firing neurons to retain (0-100)
%   angleIndices  - (Optional) Specific angles to consider when calculating mean firing rates
%                   Default: all angles
%
% Outputs:
%   filteredData   - Processed data with only the top firing neurons retained
%   selectedIndices - Indices of neurons that were retained
%
% Example:
%   [filteredData, selectedNeurons] = filterLowFiringNeurons(processedData, 75);
%   [filteredData, selectedNeurons] = filterLowFiringNeurons(processedData, 60, [1 3 5]);
%----------------------------------------------------------------------

% Get dimensions
numTrials = size(processedData, 1);
numAngles = size(processedData, 2);
numCells = size(processedData(1, 1).firingRates, 1);

% If angleIndices not provided, use all angles
if nargin < 3 || isempty(angleIndices)
    angleIndices = 1:numAngles;
end

% Validate angleIndices
if any(angleIndices > numAngles) || any(angleIndices < 1)
    error('Invalid angle indices provided');
end

% Calculate mean firing rate for each neuron across specified angles
meanFiringRates = zeros(numCells, 1);

% Count number of samples for averaging
totalSamples = 0;

% Sum firing rates across all specified trials and angles
for trialIdx = 1:numTrials
    for angleIdx = angleIndices
        meanFiringRates = meanFiringRates + sum(processedData(trialIdx, angleIdx).firingRates, 2);
        totalSamples = totalSamples + size(processedData(trialIdx, angleIdx).firingRates, 2);
    end
end

% Compute mean firing rate (average across all bins, trials, and specified angles)
meanFiringRates = meanFiringRates / totalSamples;

% Sort neurons by mean firing rate (descending)
[~, sortedIndices] = sort(meanFiringRates, 'descend');

% Determine number of neurons to keep
numToKeep = round(numCells * percentToKeep / 100);
numToKeep = max(1, min(numToKeep, numCells)); % Ensure at least 1 and at most numCells

% Select top firing neurons
selectedIndices = sortedIndices(1:numToKeep);
selectedIndices = sort(selectedIndices); % Sort indices in ascending order

% Create filtered data structure with only selected neurons
filteredData = processedData;

% Update each trial and angle with only the selected neurons
for trialIdx = 1:numTrials
    for angleIdx = 1:numAngles
        filteredData(trialIdx, angleIdx).binnedSpikes = processedData(trialIdx, angleIdx).binnedSpikes(selectedIndices, :);
        filteredData(trialIdx, angleIdx).firingRates = processedData(trialIdx, angleIdx).firingRates(selectedIndices, :);
    end
end

end