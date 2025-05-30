function ClassificationPipelineNoBuiltin()
    clc; close all;
    global trainingData testData bestParams pcaCoeff pcaMean numPCs
    global pcaCoeff pcaMean numPCs

    %% 1) Data loading
    if ~exist('trial','var')
        disp('Loading monkeydata_training.mat');
        load('monkeydata_training.mat', 'trial');
    end

    % Set random seed for reproducibility
    rng(2013);
    
    % Split data into training and test sets
    ix = randperm(length(trial));
    trainingData = trial(ix(1:50), :);
    testData = trial(ix(51:end), :);

    disp(['Training set: ', num2str(size(trainingData, 1)), ...
          ' trials, test set: ', num2str(size(testData, 1)), ' trials.']);
    
    %% 2) Classification Pipeline

    disp('Training final model with best parameters...');
    bestParams.binSize = 30;
    bestParams.historyBins = 12;
    bestParams.normalize = 0;

    % Build features with best parameters
    [Xtrain, Ytrain] = buildFeatures(trainingData, bestParams.binSize, ...
        bestParams.historyBins, bestParams.normalize);

    [Xtrain_red, V_reduced, X_mean, X_std] = pcaReduction(Xtrain, 0.95);
    
    % Train final LDA model
    finalModel = trainLDA(Xtrain_red, Ytrain);
    disp(size(Xtrain_red)) %(4369 x 933)
    
    % Evaluate on training set
    trainPreds = predictLDA(finalModel, Xtrain_red);
    finalTrainAcc = mean(trainPreds == Ytrain);
    disp(['Final train accuracy: ', num2str(finalTrainAcc * 100, '%.2f'), '%']);
    
    % Evaluate on test set
    [finalTestAcc, binAccVec] = iterativeTestLDA(testData, finalModel, V_reduced, X_mean, X_std, bestParams.binSize, ...
                                                bestParams.historyBins, bestParams.normalize);
    disp(['Final test accuracy: ', num2str(finalTestAcc * 100, '%.2f'), '%']);


    %% Helper Classification Functions

    function [Xpca, V_reduced, X_mean, X_std] = pcaReduction(X, varianceThreshold)
        X_mean = mean(X, 1);
        X_std = std(X, [], 1);
        X_std(X_std == 0) = 1; % Wherever std is 0, make it 1 so no zero error
        X_norm = (X - X_mean)./X_std;

        % U - eigenvectors (PCs)
        [U, S, V] = svd(X_norm, 'econ');
        singular_values = diag(S);
        explained_var = (singular_values .^ 2) / sum(singular_values .^ 2);
        cum_var = cumsum(explained_var);
    
        numPCs = find(cum_var >= varianceThreshold, 1, 'first');
        V_reduced = V(:, 1:numPCs);
    
        disp("Size of X_norm: "), disp(size(X_norm));
        disp("Size of V_reduced: "), disp(size(V_reduced));
    
        Xpca = X_norm * V_reduced;
    
        disp("Original Feature Size:"), disp(size(X))
        disp("Reduced Feature Size:"), disp(size(Xpca))
        disp("Number of PCs Retained:"), disp(numPCs)
    
    end

function [testAcc, binAccVec] = iterativeTestLDA(data, ldaModel, V_reduced, X_mean, X_std, binSize, historyBins, leftoverNormalize)
% Performs iterative classification over time using an accumulating angle probability distribution.
% Confidence over the correct angle increases steadily.

    [numTrials, numAngles] = size(data);
    totalCorrect = 0;
    totalCount = 0;
    maxBinIndex = 0;
    binCorrect = zeros(200, 1); % Assume max 200 bins
    binCount = zeros(200, 1);

    for a = 1:numAngles
        for tr = 1:numTrials
            rawSpikes = data(tr, a).spikes;
            Tms = size(rawSpikes, 2);

            % Compute bin counts
            nFull = floor(Tms / binSize);
            leftover = Tms - nFull * binSize;
            nb = nFull + (leftover > 0);
            binned = zeros(size(rawSpikes, 1), nb, 'single');

            % Process bins
            for b = 1:nFull
                st = (b-1) * binSize + 1;
                ed = b * binSize;
                seg = rawSpikes(:, st:ed);
                binned(:, b) = mean(seg, 2);
            end
            
            if leftover > 0
                st = nFull * binSize + 1;
                ed = Tms;
                seg = rawSpikes(:, st:ed);
                partialMean = mean(seg, 2);
                if leftoverNormalize
                    ratio = binSize / leftover;
                    partialMean = partialMean * ratio;
                end
                binned(:, nb) = partialMean;
            end

            % Initialize confidence vector (uniform prior)
            angleProbs = ones(1, 8) / 8;

            for b = 1:nb
                if b < historyBins
                    continue;
                end

                % Extract feature window
                window = binned(:, b-historyBins+1 : b);
                featRow = reshape(window, 1, []);


                featRow = (featRow - X_mean) ./ X_std;
                featRow(isnan(featRow)) = 0;

                featRow_pca = featRow * V_reduced;

                [predLabel, predScores] = predictLDA(ldaModel, featRow_pca); % Get raw LDA scores
                predScores = predScores - min(predScores);  % Shift scores to be non-negative
                predProbs = predScores ./ sum(predScores);  % Normalize scores instead of exp()

                % Accumulate probabilities across bins (confidence should only increase)
                angleProbs = angleProbs .* predProbs;  % Multiply previous confidence with new likelihoods
                angleProbs = angleProbs / sum(angleProbs);  % Normalize

                % Choose angle based on highest accumulated probability
                [~, predAngle] = max(angleProbs);

                % Track correctness
                actualAngle = a;
                isCorrect = (predAngle == actualAngle);
                totalCount = totalCount + 1;
                totalCorrect = totalCorrect + isCorrect;

                % Track bin-specific accuracy
                if b > maxBinIndex
                    maxBinIndex = b;
                end
                binCount(b) = binCount(b) + 1;
                if isCorrect
                    binCorrect(b) = binCorrect(b) + 1;
                end
            end
        end
    end

    % Compute overall accuracy
    testAcc = (totalCorrect / totalCount);
    
    % Compute bin-specific accuracy
    binAccVec = zeros(maxBinIndex, 1);
    for i = 1:maxBinIndex
        if binCount(i) > 0
            binAccVec(i) = binCorrect(i) / binCount(i);
        else
            binAccVec(i) = NaN;
        end
    end
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

    
function [X, Y] = buildFeatures(data, binSize, historyBins, leftoverNormalize)
    %  For each trial, we create partial bins, each bin includes spiking data => (numNeurons x 1).
    %
    % Inputs:
    %   data              - Trial data structure
    %   binSize           - Size of bins in ms
    %   historyBins       - Number of history bins to include
    %   leftoverNormalize - Whether to normalize leftover bins
    %
    % Outputs:
    %   X - Feature matrix: (totalBins, spikingFeatures) 
    %           totalBins is across all trials
    %           spikingFeatures is historyBins * numNeurons -> mean spike rate
    %   Y - Angle labels: (totalBins, 1)

        [numTrials, numAngles] = size(data); % 50 trials x 8 angles
        numNeurons = size(data(1, 1).spikes, 1); % 98 neurons
    
        bigX = {};
        bigY = {};
        for angle = 1:numAngles
            for trial = 1:numTrials
                rawSpikes = data(trial, angle).spikes;
                timeMs = size(rawSpikes, 2); % time is around 600ms (no cropping)
    
                nFull = floor(timeMs / binSize); % no of full bins
                leftover = timeMs - nFull * binSize; % no of leftover time stamps 
                numBins = nFull + (leftover > 0); % total no of bins

                % For each bin, computes the mean no of spikes for each neuron - Mean Spike Rate
                binned = zeros(numNeurons, numBins, 'single');
                for b = 1:nFull
                    bin_start = (b-1) * binSize + 1;
                    bin_end = b * binSize;
                    seg = rawSpikes(:, bin_start:bin_end);
                    binned(:, b) = mean(seg, 2);
                end
                
                % Process leftover bin if exists and if greater than half the binSize
                if leftover > 0.5*binSize
                    bin_start = nFull * binSize + 1;
                    bin_end = timeMs;
                    seg = rawSpikes(:, bin_start:bin_end);
                    partialMean = mean(seg, 2);
                    
                    % Apply normalization if needed
                    if leftoverNormalize
                        ratio = binSize / leftover;
                        partialMean = partialMean * ratio;
                    end
                    
                    binned(:, numBins) = partialMean;
                end
    
                locX = [];
                locY = [];
                
                % Build sliding window features
                for b = 1:numBins
                    if b < historyBins
                        % Skip if can't form a full window
                        continue;
                    end
                    
                    % Get window of data
                    window = binned(:, b-historyBins+1 : b); % (98 neurons x history_bin_size)
                    featRow = reshape(window, 1, []);
    
                    % Store features and label
                    locX = [locX; featRow]; % Data to train for classification
                    locY = [locY; angle]; % Ground truth label for classification - my angles 
                end
                % locX is of size (no_of_bins x feat row) wherein no_of_bins varies as each trial is of different lengths
                % feat row is of size (1, 1176) wherein 1176 = number of neurons (98) * history_bin_size (12)
                bigX{end+1} = locX; % size is (1, numTrials*numAngles)
                bigY{end+1} = locY; % size is (1, numTrials*numAngles)
            end
        end
        
        % Combine all trials into single matrices
        totalBins = sum(cellfun(@(x) size(x, 1), bigX)); % Total no of bins across all trials = 4369
        dimFeat = size(bigX{1}, 2); % this 1176 (numNeurons*history_bin_size)
        X = zeros(totalBins, dimFeat, 'single');
        Y = zeros(totalBins, 1, 'int32');
        rowPos = 1;
        for i = 1:length(bigX)
            blockX = bigX{i};
            blockY = bigY{i};
            numBins = size(blockX, 1);
            X(rowPos:rowPos+numBins-1, :) = blockX;
            Y(rowPos:rowPos+numBins-1) = blockY;
            rowPos = rowPos + numBins;
        end
    end


    %% Helper Regression Functions
    
    function rmse = evaluateRegressor(data, model, binSize, historyBins, normalize)
        [Xtest, Ytest] = buildFeatures(data, binSize, historyBins, normalize);
        predX = predict(model.x, Xtest);
        predY = predict(model.y, Xtest);
        pred = [predX predY];
        rmse = sqrt(mean(sum((Ytest-pred).^2,2)));
    end

%     function [coeffs, X_pca, explained, mu] = performPCA(X, varianceToRetain)
%         % Perform PCA and retain components that explain desired variance
        
%         % Standardize data
%         mu = mean(X, 1);
%         X_centered = X - mu;
        
%         % Compute covariance matrix
%         covMatrix = (X_centered' * X_centered) / (size(X_centered, 1) - 1);
        
%         % Compute eigenvectors and values
%         [eigVecs, eigVals] = eig(covMatrix, 'vector');
%         [eigVals, sortIdx] = sort(eigVals, 'descend');
%         eigVecs = eigVecs(:, sortIdx);
        
%         % Compute explained variance
%         totalVariance = sum(eigVals);
%         explainedVariance = eigVals / totalVariance * 100;
%         cumulativeVariance = cumsum(explainedVariance);
        
%         % Determine number of components to keep
%         numComponents = find(cumulativeVariance >= varianceToRetain, 1);
        
%         % Select principal components
%         coeffs = eigVecs(:, 1:numComponents);
        
%         % Project data onto principal components
%         X_pca = X_centered * coeffs;
        
%         % Return explained variance
%         explained = cumulativeVariance(1:numComponents);
        
%         disp(['PCA reduced dimensions from ', num2str(size(X, 2)), ' to ', num2str(numComponents), ' components']);
%         disp(['Retained variance: ', num2str(explained(end)), '%']);
%     end

    
    
end