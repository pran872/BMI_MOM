function [processedOutput] = smoothData(data, binSize, filterType, filterConfig)
%----------------------------------------------------------------------
% SMOOTHDATA - Transforms neural spike trains into processed firing rates
%
% Syntax:  [processedOutput] = smoothData(data, binSize, filterType, filterConfig)
%
% Inputs:
%   data          - Matrix of neural spike data arranged by trials and angles
%   binSize       - Duration of each bin in milliseconds
%   filterType    - Type of temporal filter to apply:
%                   'gaussianFilter', 'movingAverage', 'expDecay', 'medianFilter', 
%                   'polyFit', 'noFilter'
%   filterConfig  - Parameters for the selected filter:
%                   gaussianFilter: [kernelSize, stdDev]
%                   movingAverage: [kernelSize]
%                   expDecay: [decayFactor]
%                   medianFilter: [filterSpan]
%                   polyFit: [windowLength, polyDegree]
%                   noFilter: []
%
% Outputs:
%   processedOutput - Structure with binned and filtered neural activity
%
% Example:
%   processedData = smoothData(spikeData, 25, 'gaussianFilter', [9, 2]);
%----------------------------------------------------------------------

% Initialize output structure
processedOutput = struct();

% Determine dimensions
numCells = size(data(1,1).spikes, 1);
numAngles = size(data, 2);
numTrials = size(data, 1);

% Process each trial and angle combination
for angleIdx = 1:numAngles
    for trialIdx = 1:numTrials
        % Extract current trial data
        currentSpikes = data(trialIdx, angleIdx).spikes;
        totalDuration = size(currentSpikes, 2);
        
        % Create bin divisions
        binStarts = 1:binSize:totalDuration+1;
        numBins = length(binStarts) - 1;
        
        % Initialize aggregated spike matrix
        aggregatedSpikes = zeros(numCells, numBins);
        
        % Aggregate spikes within each bin
        for binIdx = 1:numBins
            binStart = binStarts(binIdx);
            binEnd = binStarts(binIdx+1) - 1;
            aggregatedSpikes(:, binIdx) = sum(currentSpikes(:, binStart:binEnd), 2);
        end
        
        % Apply variance stabilizing transform
        transformedSpikes = sqrt(aggregatedSpikes);
        
        % Store aggregated spike counts
        processedOutput(trialIdx, angleIdx).binnedSpikes = transformedSpikes;
        
        % Initialize activity rates matrix
        activityRates = zeros(numCells, numBins);
        
        % Apply temporal filtering to each cell's data
        for cellIdx = 1:numCells
            % Get spike profile for current cell
            cellProfile = transformedSpikes(cellIdx, :);
            
            % Apply selected temporal filter
            switch lower(filterType)
                case 'gaussianfilter'
                    % Extract parameters
                    kernelSize = filterConfig(1);
                    stdDev = filterConfig(2);
                    
                    % Create kernel
                    halfKernel = floor(kernelSize/2);
                    kernelPoints = -halfKernel:halfKernel;
                    gaussKernel = exp(-kernelPoints.^2/(2*stdDev^2));
                    gaussKernel = gaussKernel / sum(gaussKernel);
                    
                    % Apply filter
                    filteredProfile = conv(cellProfile, gaussKernel, 'same');
                    
                case 'movingaverage'
                    % Simple rectangular window average
                    kernelSize = filterConfig(1);
                    rectKernel = ones(1, kernelSize) / kernelSize;
                    filteredProfile = conv(cellProfile, rectKernel, 'same');
                    
                case 'expdecay'
                    % Exponential decay filter (recursive implementation)
                    decayFactor = filterConfig(1);
                    filteredProfile = zeros(1, numBins);
                    filteredProfile(1) = cellProfile(1);
                    
                    for i = 2:numBins
                        filteredProfile(i) = decayFactor * cellProfile(i) + ...
                                          (1-decayFactor) * filteredProfile(i-1);
                    end
                    
                case 'medianfilter'
                    % Median filter implementation without using medfilt1
                    filterSpan = filterConfig(1);
                    if mod(filterSpan, 2) == 0
                        filterSpan = filterSpan + 1; % Ensure odd window size
                    end
                    
                    halfSpan = floor(filterSpan/2);
                    paddedProfile = [repmat(cellProfile(1), [1, halfSpan]), ...
                                     cellProfile, ...
                                     repmat(cellProfile(end), [1, halfSpan])];
                    
                    filteredProfile = zeros(1, numBins);
                    for i = 1:numBins
                        window = paddedProfile(i:i+filterSpan-1);
                        filteredProfile(i) = median(window);
                    end
                    
                case 'polyfit'
                    % Custom polynomial smoothing implementation without sgolayfilt
                    windowLength = filterConfig(1);
                    polyDegree = filterConfig(2);
                    
                    if mod(windowLength, 2) == 0
                        windowLength = windowLength + 1; % Ensure odd window size
                    end
                    
                    halfWindow = floor(windowLength/2);
                    filteredProfile = zeros(1, numBins);
                    
                    % Pad the signal for edge handling
                    paddedProfile = [repmat(cellProfile(1), [1, halfWindow]), ...
                                    cellProfile, ...
                                    repmat(cellProfile(end), [1, halfWindow])];
                    
                    % Apply polynomial fitting for each point
                    for i = 1:numBins
                        % Extract window around the current point
                        windowIndices = (i:i+windowLength-1);
                        y = paddedProfile(windowIndices);
                        x = (1:windowLength)';
                        
                        % Fit polynomial of specified degree
                        p = polyfit(x, y, polyDegree);
                        
                        % Evaluate at center point
                        centerIdx = halfWindow + 1;
                        filteredProfile(i) = polyval(p, centerIdx);
                    end
                    
                case 'nofilter'
                    % No filtering applied
                    filteredProfile = cellProfile;
                    
                otherwise
                    error(['Unrecognized filter type: ', filterType]);
            end
            
            % Convert to firing rate (Hz)
            activityRates(cellIdx, :) = filteredProfile / (binSize/1000);
        end
        
        % Store firing rates
        processedOutput(trialIdx, angleIdx).firingRates = activityRates;
    end
end

end