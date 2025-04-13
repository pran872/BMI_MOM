function [decodedPosX, decodedPosY, newParams, pred_classes, testLDAProj, score] = positionEstimator(past_current_trial, modelParams, direc, pred_classes)
    persistent lastPosition;
    if isempty(past_current_trial.decodedHandPos)
        decodedPosX = past_current_trial.startHandPos(1);
        decodedPosY = past_current_trial.startHandPos(2);
        testLDAProj = [];
        score = [];
    else
        %% 1. Get Features

        mask = modelParams.featureMask;
        classifierParams = modelParams.classifierParams;
        pcaParams = modelParams.classifierPCA;
        
        currentSpikes = past_current_trial.spikes;
        [fr, ~] = preprocessSpikes(currentSpikes, classifierParams.binSize);
        features = fr(:)';

        if length(features) < length(mask)
            fprintf("\nFeatures is smaller than mask. Padding")
            features = [features, zeros(1, length(mask) - length(features))];  % pad
        end

        finalFeatures = features(mask);
        
        %% 2. Classification
        finalFeatures = (finalFeatures - pcaParams.X_mean)./ pcaParams.X_std;
        finalFeatures(isnan(finalFeatures)) = 0;
        finalFeaturesPCA = finalFeatures * pcaParams.projMatrix;

        testLDAProj = finalFeaturesPCA * modelParams.classifier.W(:, 1:2);

        [predictedDir, score] = predictLDA(modelParams.classifier, finalFeaturesPCA);

        % Perfect classification
        % predictedDir = direc;
        pred_classes(end+1) = predictedDir;
         
    
        %% 3. Position Regression
        try
            regressor = modelParams.regressors{predictedDir};
            [fr, binCount] = preprocessSpikes(currentSpikes, regressor.binSize);
            
            windowBins = regressor.windowSize / regressor.binSize;
            t = max(windowBins, binCount);
            featVec = fr(:, t-windowBins+1:t);
            featCentered = (featVec(:)' - regressor.mu)./ regressor.std;
            decodedPos = [featCentered * regressor.projMatrix, 1] * regressor.Beta;
            newPos = decodedPos';

        catch
            disp("Error in regression")
            newPos = lastPosition; % Fallback to last position
        end


        %% 4. Last State
        if isempty(lastPosition)
            % disp("lastPosiition is empty")
            decodedPosX = past_current_trial.startHandPos(1);
            decodedPosY = past_current_trial.startHandPos(2);
        else
            % disp("nope")
            lastPosition = newPos;
        end
        
        %% 4. Check Distance to Target Centroid (Stopping Condition)
        if isempty(past_current_trial.decodedHandPos)
            lastPosition = past_current_trial.startHandPos(1:2);
            
        else
            % decodedPosX = newPos(1);
            % decodedPosY = newPos(2);
            last_x = past_current_trial.decodedHandPos(1, end);
            last_y = past_current_trial.decodedHandPos(2, end);
            
        
            % Compute distances to all centroids
            distances = sqrt((modelParams.centroids_x - last_x).^2 + (modelParams.centroids_y - last_y).^2);
        
            % Find the nearest centroid (closest movement direction)
            [min_distance, closest_idx] = min(distances);
        
            
            % Define stopping radius threshold (e.g., 5 mm)
            % Compute centroid std-based radius
            stopping_radius = 20;
        
            % Check if the movement should stop
            if min_distance < stopping_radius
                alpha = 0.25;  % Convergence factor (adjustable for smoother/slower movement)
                beta = 0.1;   % Additional damping factor to reduce abrupt stops

                % Compute directional movement towards centroid
                dx = newPos(1) - modelParams.centroids_x(closest_idx);
                dy = newPos(2) - modelParams.centroids_y(closest_idx);

                decodedPosX = modelParams.centroids_x(closest_idx) + alpha * dx + beta * sign(dx) * min(abs(dx), stopping_radius);
                decodedPosY = modelParams.centroids_y(closest_idx) + alpha * dy + beta * sign(dy) * min(abs(dy), stopping_radius);
                % decodedPosY = modelParams.centroids_y(closest_idx);
                % newPos = lastPosition; % Keep position static
            else
                decodedPosX = newPos(1);
                decodedPosY = newPos(2);
            end
        end
    end
    
    newParams = modelParams;
end

function [fr, bins] = preprocessSpikes(spikes, binSize)
    T = size(spikes, 2);
    bins = floor(T / binSize);
    spikesBinned = sum(reshape(spikes(:,1:bins*binSize), size(spikes,1), binSize, []), 2);
    fr = (1000 / binSize) * permute(spikesBinned, [1,3,2]);
end

function [predictedDir, score] = predictLDA(ldaModel, Xtest)
    projectedData = Xtest * ldaModel.W;

    numClasses = size(ldaModel.classMeans, 1);
    numSamples = size(Xtest, 1);
    distances = zeros(numSamples, numClasses);

    %Mahalanobis distance
    for i = 1:numClasses
        classMeanProj = ldaModel.classMeans(i, :) * ldaModel.W;  
        distances(:, i) = sum((projectedData - classMeanProj).^2, 2);
    end

    % %Assign class with the smallest distance
    % fprintf("\n")
    % disp(distances)
    score = min(distances);
    [~, minIdx] = min(distances, [], 2);
    predictedDir = ldaModel.classes(minIdx);
end
