function CombinedPipeline()
    clc; close all;
    global trainingData testData bestParams pcaCoeff pcaMean numPCs

    %% 1) Load Data
    if ~exist('trial', 'var')
        disp('Loading monkeydata_training.mat');
        load('monkeydata_training.mat', 'trial');
    end

    % Set random seed for reproducibility
    rng(2013);
    
    % Split into training/testing sets
    ix = randperm(length(trial));
    trainingData = trial(ix(1:50), :);
    testData = trial(ix(51:end), :);

    fprintf('Training set: %d trials, test set: %d trials.\n', size(trainingData, 1), size(testData, 1));
    
    %% 2) Set optimal parameters
    bestParams.binSize = 30;
    bestParams.historyBins = 12;
    bestParams.normalize = false;
    
    %% 3) Perform PCA for Dimensionality Reduction
    performPCA();
    
    %% 4) Train and evaluate SVR models per angle
    trainAngleSpecificSVRModels();
end

%% ===================== Manual PCA Implementation =====================
function performPCA()
    global trainingData numPCs pcaCoeff pcaMean
    
    % Define maximum PCs to retain
    numPCs = 80;
    
    % Collect all spike data into a matrix for PCA
    allSpikes = collectAllSpikes(trainingData);
    
    % Compute mean across all trials for centering
    pcaMean = mean(allSpikes, 1);
    
    % Centered data
    centeredData = allSpikes - pcaMean;
    
    % Compute covariance matrix
    covMatrix = (centeredData' * centeredData) / size(centeredData, 1);
    
    % Compute eigenvectors and eigenvalues
    [eigVectors, eigValues] = eig(covMatrix);
    
    % Sort eigenvalues in descending order
    [~, sortedIndices] = sort(diag(eigValues), 'descend');
    eigVectors = eigVectors(:, sortedIndices);
    
    % Select top numPCs principal components
    pcaCoeff = eigVectors(:, 1:numPCs);
    
    fprintf('PCA completed: Retained %d PCs\n', numPCs);
end

function allSpikes = collectAllSpikes(data)
    % Collects all spike data across trials for PCA
    [numTrials, numAngles] = size(data);
    numNeurons = size(data(1,1).spikes, 1);
    
    % First, count total number of time bins
    totalBins = 0;
    for angle = 1:numAngles
        for trial = 1:numTrials
            spikes = data(trial, angle).spikes;
            totalBins = totalBins + size(spikes, 2);
        end
    end
    
    % Preallocate matrix
    allSpikes = zeros(totalBins, numNeurons);
    currentIdx = 1;
    
    % Fill matrix
    for angle = 1:numAngles
        for trial = 1:numTrials
            spikes = data(trial, angle).spikes;
            numBins = size(spikes, 2);
            allSpikes(currentIdx:currentIdx+numBins-1, :) = spikes';
            currentIdx = currentIdx + numBins;
        end
    end
end

%% ===================== Train Angle-Specific SVR Models =====================
function trainAngleSpecificSVRModels()
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

%% ===================== Build Features (with PCA) =====================
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

%% ===================== Evaluate SVR Models =====================
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
