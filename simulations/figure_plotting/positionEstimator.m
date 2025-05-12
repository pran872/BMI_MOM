% NOTE: FINAL MODELS - LDA AND LINEAR REGRESSION - DO NOT USE ANY TOOLBOXES BUT OTHER MODELS EXPLORED MAY USE
%       THE STATISTICS AND MACHINE LEARNING MATLAB TOOLBOX 
function [decodedPosX, decodedPosY, newParams, predClasses, one_test_lda_proj] = positionEstimator(past_current_trial, modelParams, trueDir, predClasses, clsMethod, regMethod)
    persistent lastPosition;
    if isempty(past_current_trial.decodedHandPos)
        decodedPosX = past_current_trial.startHandPos(1);
        decodedPosY = past_current_trial.startHandPos(2);
        one_test_lda_proj = [];
    else
        %% 1. Get Features
        one_test_lda_proj = [];

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
        
        switch clsMethod
            case 'lda'
                [predictedDir, one_test_lda_proj] = predictLDA(modelParams.classifier, finalFeaturesPCA);
            case {'knn', 'logistic'}
                predictedDir = predict(modelParams.classifier, finalFeaturesPCA);
            otherwise
                error('Unknown classification method: %s', clsMethod);
        end

        % Perfect classification
        %predictedDir = trueDir;

        predClasses(end+1) = predictedDir;
         
    
        %% 3. Position Regression
        regressor = modelParams.regressors{predictedDir};
        [fr, binCount] = preprocessSpikes(currentSpikes, regressor.binSize);

        windowBins = regressor.windowSize / regressor.binSize;
        t = max(windowBins, binCount);
        featVec = fr(:, t-windowBins+1:t);
        
        switch regMethod
            case 'linear'
                newPos = runLinearRegression(featVec, regressor); % no matlab toolbox required
            case 'knn'
                newPos = runKNNRegression(featVec, regressor);
            case 'svr'
                newPos = runSVR(featVec, regressor);
            case 'rf'
                newPos = runRFRegression(featVec, regressor);
            otherwise
                error('Unknown regression method: %s', regressor.method);
        end


        %% 4. Last State
        if isempty(lastPosition)
            decodedPosX = past_current_trial.startHandPos(1);
            decodedPosY = past_current_trial.startHandPos(2);
        else
            lastPosition = newPos;
        end
        
        %% 4. Check Distance to Target Centroid (Stopping Condition)
        % No centroid
        % if isempty(past_current_trial.decodedHandPos)
        %     lastPosition = past_current_trial.startHandPos(1:2);
        % else
        %     decodedPosX = newPos(1);
        %     decodedPosY = newPos(2);
        % end
        if isempty(past_current_trial.decodedHandPos)
            lastPosition = past_current_trial.startHandPos(1:2);
            
        else
            last_x = past_current_trial.decodedHandPos(1, end);
            last_y = past_current_trial.decodedHandPos(2, end);
            
            distances = sqrt((modelParams.centroids_x - last_x).^2 + (modelParams.centroids_y - last_y).^2);
            [min_distance, closest_idx] = min(distances);
        
            stopping_radius = 20;
            if min_distance < stopping_radius
                alpha = 0.25;
                beta = 0.1;

                dx = newPos(1) - modelParams.centroids_x(closest_idx);
                dy = newPos(2) - modelParams.centroids_y(closest_idx);

                decodedPosX = modelParams.centroids_x(closest_idx) + alpha * dx + beta * sign(dx) * min(abs(dx), stopping_radius);
                decodedPosY = modelParams.centroids_y(closest_idx) + alpha * dy + beta * sign(dy) * min(abs(dy), stopping_radius);
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

function [predictedDir, projectedData2D] = predictLDA(ldaModel, Xtest)
    projectedData = Xtest * ldaModel.W;
    projectedData2D = Xtest * ldaModel.W(:, 1:2);

    numClasses = size(ldaModel.classMeans, 1);
    numSamples = size(Xtest, 1);
    distances = zeros(numSamples, numClasses);

    %Mahalanobis distance
    for i = 1:numClasses
        classMeanProj = ldaModel.classMeans(i, :) * ldaModel.W;  
        distances(:, i) = sum((projectedData - classMeanProj).^2, 2);
    end

    [~, minIdx] = min(distances, [], 2);
    predictedDir = ldaModel.classes(minIdx);
end


%% Other regression models
% Some use Stat and ML MATLAB toolbox. Linear regression (final model) doesn't

function newPos = runLinearRegression(featVec, regressor)
    featCentered = (featVec(:)' - regressor.mu) ./ regressor.std;
    decodedPos = [featCentered * regressor.projMatrix, 1] * regressor.Beta;
    newPos = decodedPos';
end

function newPos = runKNNRegression(featVec, regressor)
    Xtest = (featVec(:)' - regressor.mu) ./ regressor.std;
    Xtest = Xtest * regressor.projMatrix;

    dists = pdist2(Xtest, regressor.Xpca);
    [~, idx] = mink(dists, regressor.k, 2);
    x_pred = mean(regressor.Y(idx, 1));
    y_pred = mean(regressor.Y(idx, 2));
    newPos = [x_pred; y_pred];
end

function newPos = runSVR(featVec, regressor)
    Xtest = (featVec(:)' - regressor.mu) ./ regressor.std;
    Xtest = Xtest * regressor.projMatrix;

    x_pred = predict(regressor.modelX, Xtest);
    y_pred = predict(regressor.modelY, Xtest);
    newPos = [x_pred; y_pred];
end

function newPos = runRFRegression(featVec, regressor)
    Xtest = (featVec(:)' - regressor.mu) ./ regressor.std;
    Xtest = Xtest * regressor.projMatrix;

    x_pred = predict(regressor.modelX, Xtest);
    y_pred = predict(regressor.modelY, Xtest);
    newPos = [x_pred; y_pred];
end
