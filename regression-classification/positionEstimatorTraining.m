function modelParameters = positionEstimatorTraining(trainingData)
    modelParameters.binSize = 30;
    modelParameters.historyBins = 12;
    modelParameters.normalize = false;

    [Xtrain, Ytrain] = buildFeaturesPartialLDA(trainingData, modelParameters.binSize, ...
        modelParameters.historyBins, modelParameters.normalize, 'train');

    modelParameters.classificationModel = trainLDA(Xtrain(:, 2:end), Ytrain);

    % Evaluate on training set
    trainPreds = predictLDA(modelParameters.classificationModel, Xtrain(:, 2:end));
    finalTrainAcc = mean(trainPreds == Ytrain);
    disp(['Final train accuracy: ', num2str(finalTrainAcc * 100, '%.2f'), '%']);
    
    modelParameters.regressionModels = trainAngleSpecificSVRModels();
end

function ldaModel = trainLDA(X, Y)
    % trainLDA - Manually implements Linear Discriminant Analysis (LDA)
    %
    % Inputs:
    %   X - Feature matrix (samples x features)
    %   Y - Class labels (samples x 1)
    %
    % Outputs:
    %   ldaModel - Struct containing projection matrix and class statistics

    classes = unique(Y);
    numClasses = length(classes);
    numFeatures = size(X, 2);

    % Compute overall mean
    meanTotal = mean(X, 1);

    % Initialize scatter matrices
    Sw = zeros(numFeatures, numFeatures); % Within-class scatter
    Sb = zeros(numFeatures, numFeatures); % Between-class scatter

    % Store class means and priors
    classMeans = zeros(numClasses, numFeatures);
    classPriors = zeros(numClasses, 1);

    for i = 1:numClasses
        classData = X(Y == classes(i), :);
        classMean = mean(classData, 1);
        classMeans(i, :) = classMean;
        classPriors(i) = size(classData, 1) / size(X, 1);

        % Compute within-class scatter
        classScatter = (classData - classMean)' * (classData - classMean);
        Sw = Sw + classScatter;
        
        % Compute between-class scatter
        meanDiff = (classMean - meanTotal)';
        Sb = Sb + size(classData, 1) * (meanDiff * meanDiff');
    end

    % Solve the generalized eigenvalue problem
    [eigVecs, eigVals] = eig(Sb, Sw);
    [~, sortedIdx] = sort(diag(eigVals), 'descend');
    W = eigVecs(:, sortedIdx); % Projection matrix

    % Store trained LDA model
    ldaModel.W = W;
    ldaModel.classMeans = classMeans;
    ldaModel.classPriors = classPriors;
    ldaModel.classes = classes;
end

function [predictions, predScores] = predictLDA(ldaModel, Xtest)
    % predictLDA - Uses trained LDA model to classify new samples and return raw LDA scores
    %
    % Inputs:
    %   ldaModel - Struct containing LDA projection matrix & class statistics
    %   Xtest - New data (samples x features)
    %
    % Outputs:
    %   predictions - Predicted class labels
    %   predScores - Raw LDA scores (negative Mahalanobis distance to each class mean)

    % Project data into LDA space
    projectedData = Xtest * ldaModel.W;

    % Compute Mahalanobis distance to class means
    numClasses = size(ldaModel.classMeans, 1);
    numSamples = size(Xtest, 1);
    distances = zeros(numSamples, numClasses);

    for i = 1:numClasses
        classMeanProj = ldaModel.classMeans(i, :) * ldaModel.W;
        distances(:, i) = sum((projectedData - classMeanProj).^2, 2);
    end

    % Assign class with the smallest distance
    [~, minIdx] = min(distances, [], 2);
    predictions = ldaModel.classes(minIdx);
    
    % Convert distances to scores (negative distances so that higher means better match)
    predScores = -distances;
    
    % Apply softmax normalization to predScores to avoid overfitting to train data
    predScores = exp(predScores - max(predScores, [], 2));  % Prevent numerical instability
    predScores = predScores ./ sum(predScores, 2);
end

function [X, Y] = buildFeaturesPartialLDA(data, binSize, historyBins, leftoverNormalize, mode)
    % buildFeaturesPartialLDA:
    %  For each trial, we create partial bins, each bin includes spiking data => (numNeurons x 1).
    %  We scale partial leftover bin if leftoverNormalize==true. 
    %  For bin t in [1.. nb], we have "previousAngle" = 
    %    - if t=1 => 0 
    %    - if t>1 => the *label* from bin t-1 if mode='train'
    %
    % Inputs:
    %   data              - Trial data structure
    %   binSize           - Size of bins in ms
    %   historyBins       - Number of history bins to include
    %   leftoverNormalize - Whether to normalize leftover bins
    %   mode              - 'train' or 'test'
    %
    % Outputs:
    %   X - Feature matrix: [previousAngle, spikingFeatures]
    %   Y - Angle labels
        [numTrials, numAngles] = size(data); % 50 trials x 8 angles
        numNeurons = size(data(1, 1).spikes, 1); % 98 neurons
    
        bigX = {};
        bigY = {};
        for angle = 1:numAngles
            for trial = 1:numTrials
                rawSpikes = data(tr, a).spikes;
                Tms = size(rawSpikes, 2); % time is around 600ms (this example 632)
    
                % Calculate number of full and partial bins
                nFull = floor(Tms / binSize); % no of full bins is around 21 when binsize is 30ms
                leftover = Tms - nFull * binSize; % no of leftover time stamps is around 2ms
                nb = nFull + (leftover > 0); % total no of bins is around 22 

                % Create binned data
                binned = zeros(numNeurons, nb, 'single'); % (98neurons x 22bins)
                
                % For each bin size (time of 30ms), computes the mean no of spikes for each neuron
                % Mean Spike Rate
                for b = 1:nFull
                    bin_start = (b-1) * binSize + 1;
                    bin_end = b * binSize;
                    seg = rawSpikes(:, bin_start:bin_end); %seg of size 98 neurons x 30 timems
                    binned(:, b) = mean(seg, 2);
                end
                
                % Process leftover bin if exists
                if leftover > 0.5*binSize
                    bin_start = nFull * binSize + 1;
                    ed = Tms;
                    seg = rawSpikes(:, st:ed);
                    partialMean = mean(seg, 2);
                    
                    % Apply normalization if requested
                    if leftoverNormalize
                        ratio = binSize / leftover;
                        partialMean = partialMean * ratio;
                    end
                    
                    binned(:, nb) = partialMean;
                end
    
                locX = [];
                locY = [];
                
                % Build sliding window features
                for b = 1:nb
                    if b < historyBins
                        % Skip if we can't form a full window
                        continue;
                    end
                    
                    % Extract window of data
                    window = binned(:, b-historyBins+1 : b);
                    featRow = reshape(window, 1, []);
                    
                    % Determine previous angle feature
                    if b == 1
                        prevAngle = 0;  % No previous angle for first bin
                    else
                        if strcmpi(mode, 'train')
                            % Teacher forcing - use true label from bin t-1
                            prevAngle = a; 
                        else
                            % For test data building, placeholder (will be replaced in iterative decoding)
                            prevAngle = 0; 
                        end
                    end
    
                    % Store features and label
                    locX = [locX; [prevAngle, featRow]]; %#ok<AGROW>
                    locY = [locY; a]; %#ok<AGROW>
                end
                
                bigX{end+1} = locX;
                bigY{end+1} = locY;
            end
        end
        
        % Combine all trials into single matrices
        totalRows = sum(cellfun(@(x) size(x, 1), bigX));
        dimFeat = size(bigX{1}, 2);
        X = zeros(totalRows, dimFeat, 'single');
        Y = zeros(totalRows, 1, 'int32');
    
        rowPos = 1;
        for i = 1:length(bigX)
            blockX = bigX{i};
            blockY = bigY{i};
            nB = size(blockX, 1);
            X(rowPos:rowPos+nB-1, :) = blockX;
            Y(rowPos:rowPos+nB-1) = blockY;
            rowPos = rowPos + nB;
        end
    end

%%% REGRESSION FUNCTIONS

function models = trainAngleSpecificSVRModels()
    global trainingData testData bestParams
    
    [~, numAngles] = size(trainingData);
    
    % Store models per angle
    models = cell(numAngles, 1);
    
    for angle = 1:numAngles
        models{angle} = trainSingleAngleSVRModel(angle);
    end
    
    % Evaluate models
    rmse = evaluateAngleSpecificSVR(testData, models);
    fprintf('Final RMSE with SVR: %.4f\n', rmse);
end

%% ===================== Train SVR for a Single Angle =====================
function model = trainSingleAngleSVRModel(angle)
    global trainingData bestParams
    
    fprintf('Training SVR model for angle %d...\n', angle);
    
    % Build features for this angle
    [Xtrain, Ytrain] = buildFeatures(trainingData, bestParams.binSize, bestParams.historyBins, bestParams.normalize, angle);
    
    % Train SVR regressors for (X, Y) separately
    model.svrX = fitrsvm(Xtrain, Ytrain(:,1), 'KernelFunction', 'gaussian');
    model.svrY = fitrsvm(Xtrain, Ytrain(:,2), 'KernelFunction', 'gaussian');
    
    fprintf('SVR models for angle %d trained successfully.\n', angle);
end

function [X, Y] = buildFeatures(data, binSize, historyBins, normalize, angleFilter)
    global pcaCoeff pcaMean numPCs

    [numTrials, numAngles] = size(data);
    numNeurons = size(data(1,1).spikes, 1);
    
    if nargin < 5
        useFilter = false;
        targetAngles = 1:numAngles;
    else
        useFilter = true;
        targetAngles = angleFilter;
    end
    
    % Preallocate based on estimated size
    totalSamples = estimateSampleSize(data, binSize, historyBins, targetAngles);
    X = zeros(totalSamples, numPCs * historyBins, 'single');
    Y = zeros(totalSamples, 2, 'single');
    
    sampleIdx = 1;
    for angle = targetAngles
        for trial = 1:numTrials
            spikes = data(trial, angle).spikes;
            Tms = size(spikes, 2);
            nBins = ceil(Tms / binSize);
            
            if nBins < historyBins
                continue;
            end
            
            % Binning
            binned = binSpikes(spikes, binSize, nBins, normalize);
            
            % PCA Transform - corrected dimensions
            projectedBins = (binned - repmat(pcaMean, size(binned, 1), 1)) * pcaCoeff;
            
            % Sliding window features
            for binIdx = historyBins:nBins
                window = reshape(projectedBins(binIdx-historyBins+1:binIdx, :)', 1, []);
                X(sampleIdx, :) = window;
                
                % Get hand position at the correct time
                handPosIdx = min(binSize * binIdx, Tms);
                Y(sampleIdx, :) = data(trial, angle).handPos(1:2, handPosIdx)';
                
                sampleIdx = sampleIdx + 1;
            end
        end
    end
    
    % Trim excess
    X = X(1:sampleIdx-1, :);
    Y = Y(1:sampleIdx-1, :);
end

function nSamples = estimateSampleSize(data, binSize, historyBins, targetAngles)
    % Estimates the number of total samples
    nSamples = 0;
    [numTrials, ~] = size(data);
    
    for angle = targetAngles
        for trial = 1:numTrials
            Tms = size(data(trial, angle).spikes, 2);
            nBins = ceil(Tms / binSize);
            if nBins >= historyBins
                nSamples = nSamples + (nBins - historyBins + 1);
            end
        end
    end
end

function binned = binSpikes(spikes, binSize, nBins, normalize)
    % Bins the spikes into time windows
    numNeurons = size(spikes, 1);
    binned = zeros(nBins, numNeurons);
    
    for bin = 1:nBins
        idxStart = (bin-1)*binSize + 1;
        idxEnd = min(bin*binSize, size(spikes, 2));
        segment = spikes(:, idxStart:idxEnd);
        binMean = mean(segment, 2);
        
        if normalize && (idxEnd - idxStart + 1) < binSize
            binMean = binMean * (binSize / (idxEnd - idxStart + 1));
        end
        
        binned(bin, :) = binMean';
    end
end

function rmse = evaluateAngleSpecificSVR(data, models)
    global bestParams
    
    [~, numAngles] = size(data);
    allPreds = [];
    allTrueY = [];
    
    for angle = 1:numAngles
        [Xtest, Ytest] = buildFeatures(data, bestParams.binSize, bestParams.historyBins, bestParams.normalize, angle);
        if isempty(Xtest), continue; end
        
        predX = predict(models{angle}.svrX, Xtest);
        predY = predict(models{angle}.svrY, Xtest);
        
        allPreds = [allPreds; predX, predY];
        allTrueY = [allTrueY; Ytest];
    end
    
    rmse = sqrt(mean(sum((allTrueY - allPreds).^2, 2)));
end
