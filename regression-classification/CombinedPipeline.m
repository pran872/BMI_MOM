function CombinedPipeline()
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
    [Xtrain, Ytrain] = buildFeaturesPartialLDA(trainingData, bestParams.binSize, ...
                                            bestParams.historyBins, bestParams.normalize, 'train');
    
    % Train final LDA model
    finalModel = fitcdiscr(Xtrain(:, 2:end), Ytrain); % BUILT IN FUNCTION CHANGE IT
    
    % Evaluate on training set
    trainPreds = predict(finalModel, Xtrain(:, 2:end));
    finalTrainAcc = mean(trainPreds == Ytrain);
    disp(['Final train accuracy: ', num2str(finalTrainAcc * 100, '%.2f'), '%']);
    
    % Evaluate on test set
    % [finalTestAcc, binAccVec] = iterativeTestLDA(testData, finalModel, bestParams.binSize, ...
    %                                             bestParams.historyBins, bestParams.normalize);
    % disp(['Final test accuracy: ', num2str(finalTestAcc * 100, '%.2f'), '%']);

   %% 3) PCA
function performPCA()
    % This function performs PCA on the spike data to reduce dimensionality
    % before feeding into the regression model
    

    
    % Define parameters
    numPCs = min(70, size(trainingData(1,1).spikes, 1)); % Number of principal components to keep
    
    % Collect all training data spikes into a single matrix
    [numTrials, numAngles] = size(trainingData);
    spikeSamples = [];
    
    % Concatenate spike data from all trials and angles
    for angle = 1:numAngles
        for trial = 1:numTrials
            spikes = trainingData(trial, angle).spikes;
            % Reshape to have neurons in rows and time points in columns
            spikeSamples = [spikeSamples, spikes];
        end
    end
    
    % Compute PCA
    [pcaCoeff, pcaScore, ~, ~, explained] = pca(spikeSamples', 'NumComponents', numPCs);
    pcaMean = mean(spikeSamples, 2);
    
    % Display variance explained
    cumulativeVar = cumsum(explained);
    fprintf('Using %d PCs explains %.2f%% of variance\n', numPCs, cumulativeVar(numPCs));
    
    % Modify the buildFeatures function to incorporate PCA
    buildFeaturesOriginal = @buildFeatures;
    buildFeatures = @(data, binSize, historyBins, normalize, angleFilter) buildFeaturesWithPCA(data, binSize, historyBins, normalize, angleFilter, buildFeaturesOriginal);
end

function [X, Y] = buildFeaturesWithPCA(data, binSize, historyBins, normalize, angleFilter, originalFunction)
    % Modified version of buildFeatures that applies PCA transformation to the features
    % and filters for a specific angle if angleFilter is provided
    
    % First get features using the original function for all angles
    [Xall, Yall, angleIndices] = originalFunction(data, binSize, historyBins, normalize, true);
    
    % Filter for specific angle if requested
    if nargin >= 5 && ~isempty(angleFilter)
        angleIdx = angleIndices == angleFilter;
        X = Xall(angleIdx, :);
        Y = Yall(angleIdx, :);
    else
        X = Xall;
        Y = Yall;
    end
    
    % Extract only the spike features (exclude angle feature if present)
    if size(X, 2) > 1  % If there's more than just the spikes
        spikeFeatures = X(:, 2:end);
    else
        spikeFeatures = X;  % All features are spikes
    end
    
    % Reshape to format compatible with PCA
    [samples, totalFeatures] = size(spikeFeatures);
    numNeurons = totalFeatures / historyBins;
    
    % Apply PCA transformation to each time bin separately
    transformedFeatures = zeros(samples, numPCs * historyBins);
    
    for h = 1:historyBins
        % Extract features for this time bin across all neurons
        binStart = (h-1) * numNeurons + 1;
        binEnd = h * numNeurons;
        binFeatures = spikeFeatures(:, binStart:binEnd);
        
        % Center using the mean computed during PCA
        centered = binFeatures - pcaMean';
        
        % Project onto principal components
        projected = centered * pcaCoeff(:, 1:numPCs);
        
        % Store transformed features
        transformedStart = (h-1) * numPCs + 1;
        transformedEnd = h * numPCs;
        transformedFeatures(:, transformedStart:transformedEnd) = projected;
    end
    
    % Recombine with angle feature if it was present
    if size(X, 2) > 1
        angleFeature = X(:, 1);
        X = [angleFeature, transformedFeatures];
    else
        X = transformedFeatures;
    end
end

%% 4) Regression Pipeline with Per-Angle Models
% Results storage
results = [];

try
    performPCA()
    
    % Get number of angles
    [~, numAngles] = size(trainingData);
    
    % Create a model for each angle
    models = cell(numAngles, 1);
    
    % Train angle-specific models
    for angle = 1:numAngles
        fprintf('Training model for angle %d...\n', angle);
        
        % Build features for this specific angle
        [Xtrain, Ytrain] = buildFeatures(trainingData, bestParams.binSize, bestParams.historyBins, bestParams.normalize, angle);
        
        % No need for angle feature since each model is angle-specific
        % Just use the spike features
        
        % Train model for this angle
        models{angle}.x = fitrensemble(Xtrain, Ytrain(:,1));
        models{angle}.y = fitrensemble(Xtrain, Ytrain(:,2));
        
        fprintf('Model for angle %d trained successfully.\n', angle);
    end
    
    % Evaluate using angle-specific models
    [rmse] = evaluateAngleSpecificRegressors(testData, models, bestParams.binSize, bestParams.historyBins, bestParams.normalize);
    
    % Store results
    results = [results; {'rf_per_angle', bestParams.binSize, bestParams.historyBins, bestParams.normalize, rmse}];
    fprintf('RMSE with angle-specific models: %.4f\n', rmse);
    
catch ME
    disp(['Error: ', ME.message]);
    disp(getReport(ME));
    results = [results; {'rf_per_angle', bestParams.binSize, bestParams.historyBins, bestParams.normalize, NaN}];
end

%% Modified Helper Regression Functions
function [X, Y, angleIndices] = buildFeatures(data, binSize, historyBins, normalize, returnAngleIndices)
    [numTrials, numAngles] = size(data);
    numNeurons = size(data(1,1).spikes,1);

    X = [];
    Y = [];
    angleIndices = [];  % To track which angle each sample comes from

    for angle = 1:numAngles
        for trial = 1:numTrials
            spikes = data(trial, angle).spikes;
            Tms = size(spikes,2);
            nBins = ceil(Tms / binSize);
            binned = zeros(numNeurons, nBins);

            % Bin the spikes
            for bin = 1:nBins
                idxStart = (bin-1)*binSize + 1;
                idxEnd = min(bin*binSize, Tms);
                segment = spikes(:, idxStart:idxEnd);
                binMean = mean(segment,2);

                % Normalize partial bin
                binLength = idxEnd - idxStart + 1;
                if normalize && binLength < binSize
                    binMean = binMean * (binSize / binLength);
                end

                binned(:, bin) = binMean;
            end

            % Build sliding window features
            for binIdx = historyBins:nBins
                window = binned(:, binIdx-historyBins+1:binIdx);
                featRow = reshape(window, 1, []);

                % For angle-specific models, we can exclude the angle feature
                if nargin >= 5 && returnAngleIndices
                    % Include angle as a feature for mixed models
                    angleFeature = (angle - 1) * 40 + 30;
                    X = [X; angleFeature, featRow];
                else
                    X = [X; featRow];
                end
                
                handPosIdx = min(binSize * binIdx, Tms);
                xPos = data(trial, angle).handPos(1, handPosIdx);
                yPos = data(trial, angle).handPos(2, handPosIdx);
                Y = [Y; xPos, yPos];
                
                % Record which angle this sample came from
                angleIndices = [angleIndices; angle];
            end
        end
    end
end

function rmse = evaluateAngleSpecificRegressors(data, models, binSize, historyBins, normalize)
    % Evaluate using angle-specific models
    [~, numAngles] = size(data);
    allPreds = [];
    allTrueY = [];
    
    for angle = 1:numAngles
        % Get data for this angle
        [Xtest, Ytest] = buildFeatures(data(:, angle), binSize, historyBins, normalize, false);
        
        % Skip if no test data for this angle
        if isempty(Xtest)
            continue;
        end
        
        % Use the angle-specific model
        predX = predict(models{angle}.x, Xtest);
        predY = predict(models{angle}.y, Xtest);
        
        % Collect predictions and true values
        allPreds = [allPreds; [predX predY]];
        allTrueY = [allTrueY; Ytest];
    end
    
    % Calculate overall RMSE
    rmse = sqrt(mean(sum((allTrueY - allPreds).^2, 2)));
end

    %% Helper Classification Functions

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
        
            [numTrials, numAngles] = size(data);
            numNeurons = size(data(1, 1).spikes, 1);
        
            bigX = {};
            bigY = {};
            for a = 1:numAngles
                for tr = 1:numTrials
                    rawSpikes = data(tr, a).spikes;
                    Tms = size(rawSpikes, 2);
        
                    % Calculate number of full and partial bins
                    nFull = floor(Tms / binSize);
                    leftover = Tms - nFull * binSize;
                    nb = nFull + (leftover > 0);
        
                    % Create binned data
                    binned = zeros(numNeurons, nb, 'single');
                    
                    % Process full bins
                    for b = 1:nFull
                        st = (b-1) * binSize + 1;
                        ed = b * binSize;
                        seg = rawSpikes(:, st:ed);
                        binned(:, b) = mean(seg, 2);
                    end
                    
                    % Process leftover bin if exists
                    if leftover > 0.5*binSize
                        st = nFull * binSize + 1;
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