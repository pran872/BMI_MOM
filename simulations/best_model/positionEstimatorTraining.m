function modelParameters = positionEstimatorTraining(trainingData)

    classifierBinSize = 20;        
    classifierWindowSize = 300; 
    classifierStandardisation = true;
    classifierBias = false;   
    regressorBinSize = 20;         
    regressorWindowSize = 300;
    regressorStandardisation = true;
    regressorBias = true;
    
    [numTrials, numAngles] = size(trainingData);

    %% 1. Classifier Training
    [classifierFeatures, featureMask] = preprocessClassifierFeatures(trainingData, classifierBinSize, classifierWindowSize);
    classifierLabels = repelem(1:numAngles, numTrials)';
    
    %PCA
    [classifierFeaturesPCA, V_reduced_cls, X_mean_cls, X_std_cls] = pcaReduction(classifierFeatures, 0.95, classifierStandardisation, classifierBias);

    %Train LDA
    classifier = trainLDA(classifierFeaturesPCA, classifierLabels);

    %% 2. Regressor Training
    regressors = cell(1, numAngles);
    
    %Regressor for each angle with PCA applied
    for dir = 1:numAngles
        [allFeat, allPos] = preprocessForRegression(trainingData(:, dir), regressorBinSize, regressorWindowSize);
        [Xpca, V_reduced_reg, X_mean_reg, X_std_reg] = pcaReduction(allFeat, 0.60, regressorStandardisation, regressorBias); 
        Beta = Xpca \ allPos;

        regressors{dir} = struct(...
            'projMatrix', V_reduced_reg, ...
            'Beta', Beta, ...
            'mu', X_mean_reg, ...
            'std', X_std_reg, ...
            'binSize', regressorBinSize, ...
            'windowSize', regressorWindowSize);
    end

    %% 3. Return Model Parameters
    modelParameters = struct(...
        'classifier', classifier, ...
        'classifierPCA', struct(...
            'projMatrix', V_reduced_cls, ...
            'X_mean', X_mean_cls, ...
            'X_std', X_std_cls), ...
        'regressors', {regressors}, ...
        'featureMask', featureMask, ...
        'expectedFeatures', size(classifierFeaturesPCA, 2), ...
        'classifierParams', struct('binSize', classifierBinSize, 'windowSize', classifierWindowSize));%, ...
        % 'classificationAccPlus', 0, ...
        % 'classificationAccMinus', 0);

end

function [features, featureMask] = preprocessClassifierFeatures(data, binSize, windowSize)
    numNeurons = size(data(1,1).spikes, 1);
    features = [];
    
    %Get all features
    for angle = 1:size(data,2)
        for trial = 1:size(data,1)
            spikes = data(trial,angle).spikes(:,1:windowSize);
            [fr, ~] = preprocessSpikes(spikes, binSize);
            features = [features; fr(:)']; 
        end
    end
    
    %Threshold variance
    featVars = var(features);
    % threshold = prctile(featVars, 10);
    validFeatures = featVars > 1e-6;
    % disp(['Before filtering: ', num2str(size(features, 1)), ' x ', num2str(size(features, 2))]);
    features = features(:, validFeatures);
    % disp(['After filtering: ', num2str(size(features, 1)), ' x ', num2str(size(features, 2))]);
    
    %Remove low variance features
    featureMask = false(1, numNeurons*(windowSize/binSize));
    featureMask(1:length(validFeatures)) = validFeatures;
end

        
function ldaModel = trainLDA(X, Y)
    %LDA - Linear Discriminant Analysis

    classes = unique(Y);
    numClasses = length(classes);
    numFeatures = size(X, 2);
    meanTotal = mean(X, 1);
    Sw = zeros(numFeatures, numFeatures); %Within-class scatter
    Sb = zeros(numFeatures, numFeatures); %Between-class scatter
    
    classMeans = zeros(numClasses, numFeatures);
    classPriors = zeros(numClasses, 1);
    for i = 1:numClasses
        classData = X(Y == classes(i), :);
        classMean = mean(classData, 1);
        classMeans(i, :) = classMean;
        classPriors(i) = size(classData, 1) / size(X, 1);

        %Within-class scatter
        classScatter = (classData - classMean)' * (classData - classMean);
        Sw = Sw + classScatter;
        
        %Between-class scatter
        meanDiff = (classMean - meanTotal)';
        Sb = Sb + size(classData, 1) * (meanDiff * meanDiff');
    end

    [eigVecs, eigVals] = eig(Sb, Sw);
    [~, sortedIdx] = sort(diag(eigVals), 'descend');
    W = eigVecs(:, sortedIdx);

    %Trained LDA model
    ldaModel.W = W;
    ldaModel.classMeans = classMeans;
    ldaModel.classPriors = classPriors;
    ldaModel.classes = classes;
end


function [Xpca, V_reduced, X_mean, X_std] = pcaReduction(X, varianceThreshold, standardisation, bias)
    %Normalize data
    X_mean = mean(X, 1);
    if standardisation
        X_std = std(X, [], 1);
        X_std(X_std == 0) = 1;
        X_processed = (X - X_mean) ./ X_std;
    else
        X_processed = X - X_mean;
        X_std = [];
    end

    [U, S, V] = svd(X_processed, 'econ');
    singular_values = diag(S);
    explained_var = (singular_values .^ 2) / sum(singular_values .^ 2);
    cum_var = cumsum(explained_var);

    numPCs = find(cum_var >= varianceThreshold, 1, 'first');
    V_reduced = V(:, 1:numPCs);

    %Add bias term if needed
    if bias
        Xpca = [X_processed * V_reduced, ones(size(X,1),1)];
    else
        Xpca = X_processed * V_reduced;
    end
    
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

