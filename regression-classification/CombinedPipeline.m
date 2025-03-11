function CombinedPipeline()
    clc; close all;

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

    
    disp(size(Xtrain))
    disp(size(Ytrain))
    
    % Evaluate on training set
    trainPreds = predict(finalModel, Xtrain(:, 2:end));
    disp(size(trainPreds))
    finalTrainAcc = mean(trainPreds == Ytrain);
    disp(['Final train accuracy: ', num2str(finalTrainAcc * 100, '%.2f'), '%']);
    
    % Evaluate on test set
    % [finalTestAcc, binAccVec] = iterativeTestLDA(testData, finalModel, bestParams.binSize, ...
    %                                             bestParams.historyBins, bestParams.normalize);
    % disp(['Final test accuracy: ', num2str(finalTestAcc * 100, '%.2f'), '%']);

    %% 3) Regression Pipeline
    % try
    %     % Build features
    %     [Xtrain, Ytrain] = buildFeatures(trainingData, bestParams.binSize, bestParams.historyBins, bestParams.normalize);

    %     % Train model
    %     model.x = fitrensemble(Xtrain,Ytrain(:,1));
    %     model.y = fitrensemble(Xtrain,Ytrain(:,2));

    %     % Evaluate
    %     [rmse] = evaluateRegressor(testData, model, bs, hb, normFlag);

    %     % Store results
    %     results = [results; {regType{1}, bs, hb, normFlag, rmse}];
    %     fprintf('RMSE: %.4f\n', rmse);
    % catch ME
    %     disp(['Error: ', ME.message]);
    %     results = [results; {regType{1}, bs, hb, normFlag, NaN}];
    % end


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
    function [X, Y] = buildFeatures(data, binSize, historyBins, normalize)
        [numTrials, numAngles] = size(data);
        numNeurons = size(data(1,1).spikes,1);
    
        X = [];
        Y = [];
    
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
    
                    % True angle as numeric feature
                    angleFeature = (angle - 1) * 40 + 30;
    
                    % Append features and labels
                    X = [X; angleFeature, featRow];
                    
                    handPosIdx = min(binSize * binIdx, Tms);
                    xPos = data(trial, angle).handPos(1, handPosIdx);
                    yPos = data(trial, angle).handPos(2, handPosIdx);
                    Y = [Y; xPos, yPos];
                end
            end
        end
    end

    function rmse = evaluateRegressor(data, model, binSize, historyBins, normalize)
        [Xtest, Ytest] = buildFeatures(data, binSize, historyBins, normalize);
        predX = predict(model.x, Xtest);
        predY = predict(model.y, Xtest);
        pred = [predX predY];
        rmse = sqrt(mean(sum((Ytest-pred).^2,2)));
    end

end