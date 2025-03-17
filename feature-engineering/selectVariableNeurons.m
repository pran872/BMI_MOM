function [filteredData, selectedIndices] = selectVariableNeurons(processedData, angleIdx, percentToKeep)
%----------------------------------------------------------------------
% SELECTVARIABLENEURONS - Selects neurons with highest firing rate variation for a specific angle
%
% Syntax:  [filteredData, selectedIndices] = selectVariableNeurons(processedData, angleIdx, percentToKeep)
%
% Inputs:
%   processedData - Structure with firing rates from smoothData function
%   angleIdx      - Index of the angle to analyze for variation
%   percentToKeep - Percentage of most variable neurons to retain (0-100)
%
% Outputs:
%   filteredData   - Processed data with only the most variable neurons retained
%   selectedIndices - Indices of neurons that were retained
%
% Example:
%   [filteredData, selectedNeurons] = selectVariableNeurons(processedData, 2, 30);
%----------------------------------------------------------------------

% Get dimensions
numTrials = size(processedData, 1);
numAngles = size(processedData, 2);
numCells = size(processedData(1, 1).firingRates, 1);

% Validate angleIdx
if angleIdx > numAngles || angleIdx < 1
    error('Invalid angle index provided');
end

% Calculate firing rate variance for each neuron at the specified angle
% We'll concatenate data across trials to get more robust estimates
firingRateVariance = zeros(numCells, 1);

% First, calculate mean firing rate profile for each neuron at this angle
allFiringRates = cell(numCells, 1);
for cellIdx = 1:numCells
    % Initialize with empty array for concatenation
    allFiringRates{cellIdx} = [];
end

% Concatenate firing rates across trials
for trialIdx = 1:numTrials
    for cellIdx = 1:numCells
        allFiringRates{cellIdx} = [allFiringRates{cellIdx}, processedData(trialIdx, angleIdx).firingRates(cellIdx, :)];
    end
end

% Calculate variance for each cell
for cellIdx = 1:numCells
    % Calculate variance over time for this cell at this angle
    firingRateVariance(cellIdx) = var(allFiringRates{cellIdx});
end

% Sort neurons by variance (descending)
[~, sortedIndices] = sort(firingRateVariance, 'descend');

% Determine number of neurons to keep

numToKeep = round(numCells * percentToKeep / 100);
fprintf('%d cells in total, %d percent to keep, %d cells kept', numCells, percentToKeep, numToKeep)
numToKeep = max(1, min(numToKeep, numCells)); % Ensure at least 1 and at most numCells

% Select most variable neurons
selectedIndices = sortedIndices(1:numToKeep);
selectedIndices = sort(selectedIndices); % Sort indices in ascending order

% Create filtered data structure with only selected neurons
filteredData = processedData;

% Update each trial and angle with only the selected neurons
for trialIdx = 1:numTrials
    for angIdx = 1:numAngles
        filteredData(trialIdx, angIdx).binnedSpikes = processedData(trialIdx, angIdx).binnedSpikes(selectedIndices, :);
        filteredData(trialIdx, angIdx).firingRates = processedData(trialIdx, angIdx).firingRates(selectedIndices, :);
    end
end

end