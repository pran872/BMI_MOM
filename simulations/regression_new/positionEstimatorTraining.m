function modelParameters = positionEstimatorTraining(trainingData)
    % Parameters
    classifierBinSize = 20;        % 20ms bins for classifier
    classifierWindowSize = 300;    % 300ms analysis window
    regressorBinSize = 20;         % Matching regressor params
    regressorWindowSize = 300;
    
    [numTrials, numAngles] = size(trainingData);
    numNeurons = size(trainingData(1,1).spikes, 1);

    %% **1. Classifier Training with PCA**
    % Extract raw features
    [classifierFeatures, validFeatures, featureMask] = preprocessClassifierFeatures(trainingData, classifierBinSize, classifierWindowSize);
    classifierLabels = repelem(1:numAngles, numTrials)';

    % **Apply PCA**
    [classifierFeaturesPCA, V_reduced, X_mean, X_std] = pcaReduction(classifierFeatures, 0.95);

    % Train LDA with PCA-reduced features
    classifier = trainLDA(classifierFeaturesPCA, classifierLabels);
    % classifier = trainLDA(classifierFeatures, classifierLabels);
    % classifier = fitcdiscr(classifierFeatures, classifierLabels, ...
    %     'DiscrimType', 'diagLinear', ...
    %     'Gamma', 0.3, ...
    %     'Prior', 'uniform');

    %% **2. Regressor Training**
    regressors = cell(1, numAngles);
    for dir = 1:numAngles
        [allFeat, allPos] = preprocessForRegression(trainingData(:, dir), regressorBinSize, regressorWindowSize);
        
        % Apply PCA for regressor training
        [Xpca, V_reduced_reg, X_mean_reg, X_std_reg] = pcaReduction(allFeat, 0.95);
        Beta = Xpca \ allPos;
        
        regressors{dir} = struct(...
            'projMatrix', V_reduced_reg, ...
            'Beta', Beta, ...
            'mu', X_mean_reg, ...
            'binSize', regressorBinSize, ...
            'windowSize', regressorWindowSize);
    end

    %% **3. Store Model Parameters**
    modelParameters = struct(...
        'classifier', classifier, ...
        'classifierPCA', struct(...
            'projMatrix', V_reduced, ...
            'X_mean', X_mean, ...
            'X_std', X_std), ...
        'regressors', {regressors}, ...
        'featureMask', featureMask, ...
        'expectedFeatures', size(classifierFeaturesPCA, 2), ...
        'classifierParams', struct('binSize', classifierBinSize, 'windowSize', classifierWindowSize), ...
        'classificationAccPlus', 0, ...
        'classificationAccMinus', 0);
    % modelParameters = struct(...
    %     'classifier', classifier, ...
    %     'regressors', {regressors}, ...
    %     'featureMask', featureMask, ...
    %     'expectedFeatures', size(classifierFeatures, 2), ...
    %     'classifierParams', struct('binSize', classifierBinSize, 'windowSize', classifierWindowSize), ...
    %     'classificationAccPlus', 0, ...
    %     'classificationAccMinus', 0);
        
end

function [features, validFeatures, featureMask] = preprocessClassifierFeatures(data, binSize, windowSize)
    % Feature extraction and selection
    numNeurons = size(data(1,1).spikes, 1);
    features = [];
    
    % 1. Extract raw features
    for angle = 1:size(data,2)
        for trial = 1:size(data,1)
            spikes = data(trial,angle).spikes(:,1:windowSize);
            [fr, ~] = preprocessSpikes(spikes, binSize);
            features = [features; fr(:)']; 
        end
    end
    
    % 2. Feature selection
    featVars = var(features);
    validFeatures = featVars > 1e-6; % Threshold for minimum variance
    disp(['Before filtering: ', num2str(size(features, 1)), ' x ', num2str(size(features, 2))]);
    features = features(:, validFeatures);
    disp(['After filtering: ', num2str(size(features, 1)), ' x ', num2str(size(features, 2))]);
    
    % 3. Create full feature mask
    featureMask = false(1, numNeurons*(windowSize/binSize));
    featureMask(1:length(validFeatures)) = validFeatures;
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


function [Xpca, V_reduced, X_mean, X_std] = pcaReduction(X, varianceThreshold)
    % Normalize Data
    X_mean = mean(X, 1);
    X_std = std(X, [], 1);
    X_std(X_std == 0) = 1; % Avoid zero division
    X_norm = (X - X_mean) ./ X_std;

    % Perform Singular Value Decomposition (SVD)
    [U, S, V] = svd(X_norm, 'econ');

    % Compute explained variance
    singular_values = diag(S);
    explained_var = (singular_values .^ 2) / sum(singular_values .^ 2);
    cum_var = cumsum(explained_var);

    % Determine number of components to keep
    numPCs = find(cum_var >= varianceThreshold, 1, 'first');

    % Select principal components
    V_reduced = V(:, 1:numPCs);

    % Project data onto PCA space and add bias term
    % Xpca = [X_norm * V_reduced, ones(size(X,1),1)];
    Xpca = X_norm * V_reduced;

    % Display results
    disp("Original Feature Size:"), disp(size(X))
    disp("Reduced Feature Size:"), disp(size(Xpca))
    disp("Number of PCs Retained:"), disp(numPCs)
end



function [fr, bins] = preprocessSpikes(spikes, binSize)
    T = size(spikes, 2);
    bins = floor(T / binSize);
    spikesBinned = sum(reshape(spikes(:,1:bins*binSize), size(spikes,1), binSize, []), 2);
    fr = (1000 / binSize) * permute(spikesBinned, [1,3,2]);
end

function [allFeat, allPos] = preprocessForRegression(trials, binSize, windowSize)
    allFeat = []; allPos = [];
    windowBins = windowSize / binSize;
    
    for tr = 1:numel(trials)
        [fr, binCount] = preprocessSpikes(trials(tr).spikes, binSize);
        handPos = trials(tr).handPos(1:2, :);
        
        for t = windowBins:binCount-1
            featVec = fr(:, t-windowBins+1:t);
            allFeat = [allFeat; featVec(:)'];
            allPos = [allPos; handPos(:, t*binSize)'];
        end
    end
end