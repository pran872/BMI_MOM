function [decodedPosX, decodedPosY, newParams] = positionEstimator(past_current_trial, modelParams)
    persistent trialStates lastPosition
    
    %% 1. Initialize persistent variables
    if isempty(trialStates)
        trialStates = struct('trialId', {}, 'predCount', {}, 'lastDir', {});
    end
    if isempty(lastPosition)
        lastPosition = past_current_trial.startHandPos(1:2)';
    end
    
    %% 2. Trial State Management
    currentTrialId = past_current_trial.trialId;
    trialIdx = find(strcmp({trialStates.trialId}, currentTrialId), 1);
    
    % Initialize new trial
    if isempty(trialIdx)
        trialStates(end+1) = struct(...
            'trialId', currentTrialId, ...
            'predCount', 0, ...
            'lastDir', []);
        trialIdx = length(trialStates);
    end
    
    % Increment prediction counter
    trialStates(trialIdx).predCount = trialStates(trialIdx).predCount + 1;
    
    %% 3. Regressor Selection Logic
    if trialStates(trialIdx).predCount <= 100
        % Use initial general regressor
        regressor = modelParams.initRegressor;
        useInitRegressor = true;
    else
        % Use direction-specific regressor
        useInitRegressor = false;
        if isempty(trialStates(trialIdx).lastDir)
            %% 4. Direction Classification (First time after 100)
            [fr, ~] = preprocessSpikes(past_current_trial.spikes, modelParams.classifierParams.binSize);
            rawFeatures = fr(:)';
            
            % Feature alignment
            paddedFeatures = zeros(1, length(modelParams.featureMask));
            paddedFeatures(1:length(rawFeatures)) = rawFeatures;
            finalFeatures = paddedFeatures(modelParams.featureMask);
            
            % Classify
            try
                trialStates(trialIdx).lastDir = predict(modelParams.classifier, finalFeatures);
            catch
                trialStates(trialIdx).lastDir = 1; % Fallback
            end
        end
        regressor = modelParams.regressors{trialStates(trialIdx).lastDir};
    end

    %% 5. Position Regression
    try
        % Bin spikes
        [fr, binCount] = preprocessSpikes(past_current_trial.spikes, regressor.binSize);
        
        % Window selection
        if useInitRegressor
            % Use all available data for initial regressor
            featVec = fr(:, 1:min(binCount, regressor.windowSize/regressor.binSize));
        else
            % Use sliding window for specific regressor
            windowBins = regressor.windowSize / regressor.binSize;
            t = max(windowBins, binCount);
            featVec = fr(:, t-windowBins+1:t);
        end
        
        % Feature processing
        if useInitRegressor
            featCentered = featVec(:)' - regressor.mu;
        else
            featCentered = featVec(:)' - regressor.mu;
        end
        
        % Prediction
        decodedPos = [featCentered * regressor.projMatrix, 1] * regressor.Beta;
        newPos = decodedPos';
    catch
        newPos = lastPosition; % Fallback
    end

    %% 6. Maintain Smooth Trajectory
    lastPosition = 0.7*newPos + 0.3*lastPosition;
    decodedPosX = lastPosition(1);
    decodedPosY = lastPosition(2);
    
    %% 7. Update Output
    newParams = modelParams;
end

%% Helper Functions (Included for self-containment)
function [fr, bins] = preprocessSpikes(spikes, binSize)
    T = size(spikes, 2);
    bins = floor(T / binSize);
    spikesBinned = sum(reshape(spikes(:,1:bins*binSize), size(spikes,1), binSize, []), 2);
    fr = (1000 / binSize) * permute(spikesBinned, [1,3,2]);
end