function RegressionPipeline()
    % Regression Pipeline Without PCA
    % - Predicts x,y coordinates simultaneously
    % - Supports ridge, SVR, and Random Forest (RF) regression methods

    clc; close all;

    %% Load Data
    if ~exist('trial', 'var')
        disp('Loading monkeydata_training.mat');
        load('monkeydata_training.mat', 'trial');
    end

    % Set random seed for reproducibility
    rng(2013);

    % Train/Test Split
    ix = randperm(length(trial));
    trainingData = trial(ix(1:50), :);
    testData = trial(ix(51:end), :);

    % Hyperparameter Grid
    binSizes = [20, 30, 40];
    historyBinsValues = [8, 10, 12];
    normalizeOptions = [true, false];
    regressors = {'rf'};

    % Results storage
    results = [];

    % Grid search
    for bs = binSizes
        for hb = historyBinsValues
            for normFlag = normalizeOptions
                for regType = regressors
                    disp(['Running ', regType{1}, '| BinSize=', num2str(bs), ...
                          ' | History=', num2str(hb), ' | Normalize=', num2str(normFlag)]);

                    try
                        % Build features
                        [Xtrain, Ytrain] = buildFeatures(trainingData, bs, hb, normFlag);

                        % Train model
                        model = trainRegressor(Xtrain, Ytrain, regType{1});

                        % Evaluate
                        [rmse] = evaluateRegressor(testData, model, bs, hb, normFlag);

                        % Store results
                        results = [results; {regType{1}, bs, hb, normFlag, rmse}];
                        fprintf('RMSE: %.4f\n', rmse);
                    catch ME
                        disp(['Error: ', ME.message]);
                        results = [results; {regType{1}, bs, hb, normFlag, NaN}];
                    end
                end
            end
        end
    end

    % Results Table
    resultsTable = cell2table(results, 'VariableNames', ...
        {'Regressor', 'BinSize', 'HistoryBins', 'Normalize', 'RMSE'});

    % Display sorted results
    resultsTable = sortrows(resultsTable, 'RMSE', 'ascend');
    disp('Top regression configurations:');
    disp(resultsTable(1:10, :));

end

%% Helper Functions

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



function model = trainRegressor(X, Y, regType)
    switch regType
        case 'ridge'
            model.x = fitrlinear(X,Y(:,1));
            model.y = fitrlinear(X,Y(:,2));
        case 'svr'
            model.x = fitrsvm(X,Y(:,1),'KernelFunction','gaussian');
            model.y = fitrsvm(X,Y(:,2),'KernelFunction','gaussian');
        case 'rf'
            model.x = fitrensemble(X,Y(:,1));
            model.y = fitrensemble(X,Y(:,2));
    end
end

function rmse = evaluateRegressor(data, model, binSize, historyBins, normalize)
    [Xtest, Ytest] = buildFeatures(data, binSize, historyBins, normalize);
    predX = predict(model.x, Xtest);
    predY = predict(model.y, Xtest);
    pred = [predX predY];
    rmse = sqrt(mean(sum((Ytest-pred).^2,2)));
end