function regressorModel = trainRegressor(trainingData)
    % Parameters
    binSize = 20;            % Spike binning window (ms)
    windowSize = 300;        % History window (ms)
    [numTrials, numAngles] = size(trainingData);
    numNeurons = size(trainingData(1,1).spikes, 1);
    
    % Initialize model struct
    regressorModel = struct();
    
    % Train separate regressors for each angle
    for angle = 1:numAngles
        allFeat = []; 
        allPos = [];
        
        % Collect features and positions
        for trialNum = 1:numTrials
            spikes = double(trainingData(trialNum, angle).spikes);
            handPos = trainingData(trialNum, angle).handPos(1:2, :);  
            T = size(spikes, 2); 
            
            % Bin spikes into 20ms windows
            binCount = floor(T / binSize);
            spikesBinned = sum(reshape(spikes(:, 1:binCount * binSize), numNeurons, binSize, []), 2);
            spikesBinned = permute(spikesBinned, [1, 3, 2]);  % Fix dimension order
            
            % Compute firing rate in Hz
            fr = (1000 / binSize) * spikesBinned;
            
            % Create feature vectors with 300ms history
            windowBins = windowSize / binSize;
            for t = windowBins:binCount-1
                windowData = fr(:, t-windowBins+1:t);
                featVec = windowData(:)';  
                allFeat(end+1, :) = featVec;
                pos = handPos(:, t * binSize)'; 
                allPos(end+1, :) = pos;
            end
        end
        
        % PCA with mean capture
        [coeff, score, ~, ~, explained, mu] = pca(allFeat);  % Critical: Get feature mean

        % keep top 50 components
        numComponents = 30;
        
        % Store model parameters
        regressorModel(angle).binSize = binSize;
        regressorModel(angle).windowSize = windowSize;
        regressorModel(angle).projMatrix = coeff(:, 1:numComponents);
        regressorModel(angle).mu = mu;  % Save feature mean for centering
        regressorModel(angle).Beta = [score(:, 1:numComponents), ones(size(score,1),1)] \ allPos;
    end
end