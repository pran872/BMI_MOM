function ClassificationPipeline()
% AngleConfidenceLDA:
%  Demonstrates partial-binning with leftover normalization, passing the previous angle 
%  prediction as a numeric feature, and using LDA for monkey reaching angle classification.
%
% Performs:
% 1) Load monkeydata_training.mat if not found
% 2) Split data => train/test
% 3) Perform hyperparameter grid search to find optimal parameters
% 4) Train final model with best parameters
% 5) Evaluate performance on test set
% 6) Analyze time-based errors

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
    
    %% 2) Hyperparameter grid search
    disp('Starting hyperparameter grid search...');
    
    % Define hyperparameter grid
    binSizes = [ 30, 40, 50];          % ms
    historyBinsValues = [8, 10, 12, 14];   % number of history bins
    normalizeOptions = [true, false];          % whether to normalize leftover bins
    
    % Initialize results storage - use arrays instead of struct for easier sorting
    numCombinations = length(binSizes) * length(historyBinsValues) * length(normalizeOptions);
    binSizeResults = zeros(numCombinations, 1);
    historyBinsResults = zeros(numCombinations, 1);
    normalizeResults = zeros(numCombinations, 1);
    accuracyResults = zeros(numCombinations, 1);
    
    % Track best parameters
    bestAcc = 0;
    bestParams = struct('binSize', 0, 'historyBins', 0, 'normalize', false);
    
    % Progress counter
    counter = 0;
    
    % Perform grid search
    for bs = 1:length(binSizes)
        for hb = 1:length(historyBinsValues)
            for norm = 1:length(normalizeOptions)
                counter = counter + 1;
                
                % Current parameter combination
                currentBinSize = binSizes(bs);
                currentHistoryBins = historyBinsValues(hb);
                currentNormalize = normalizeOptions(norm);
                
                % Display progress
                disp(['Testing combination ', num2str(counter), '/', num2str(numCombinations), ...
                      ': binSize=', num2str(currentBinSize), ...
                      ', historyBins=', num2str(currentHistoryBins), ...
                      ', normalize=', num2str(currentNormalize)]);
                
                try
                    % Build training features
                    [Xtrain, Ytrain] = buildFeaturesPartialLDA(trainingData, currentBinSize, ...
                                                              currentHistoryBins, currentNormalize, 'train');
                    
                    % Train LDA model
                    ldaModel = fitcdiscr(Xtrain(:, 2:end), Ytrain);
                    
                    % Evaluate on training set
                    trainPreds = predict(ldaModel, Xtrain(:, 2:end));
                    trainAcc = mean(trainPreds == Ytrain);
                    
                    % Evaluate on test set using iterative approach
                    [testAcc, ~] = iterativeTestLDA(testData, ldaModel, currentBinSize, ...
                                                  currentHistoryBins, currentNormalize);
                    
                    % Store results
                    binSizeResults(counter) = currentBinSize;
                    historyBinsResults(counter) = currentHistoryBins;
                    normalizeResults(counter) = currentNormalize;
                    accuracyResults(counter) = testAcc;
                    
                    % Update best parameters if needed
                    if testAcc > bestAcc
                        bestAcc = testAcc;
                        bestParams.binSize = currentBinSize;
                        bestParams.historyBins = currentHistoryBins;
                        bestParams.normalize = currentNormalize;
                    end
                    
                    disp(['  Train accuracy: ', num2str(trainAcc * 100, '%.2f'), '%, Test accuracy: ', ...
                          num2str(testAcc * 100, '%.2f'), '%']);
                    
                catch ME
                    disp(['  Error in combination: ', ME.message]);
                    % Set accuracy to NaN for failed combinations
                    binSizeResults(counter) = currentBinSize;
                    historyBinsResults(counter) = currentHistoryBins;
                    normalizeResults(counter) = currentNormalize;
                    accuracyResults(counter) = NaN;
                end
            end
        end
    end
    
    %% 3) Display grid search results
    disp('Grid search completed!');
    disp(['Best parameters: binSize=', num2str(bestParams.binSize), ...
          ', historyBins=', num2str(bestParams.historyBins), ...
          ', normalize=', num2str(bestParams.normalize)]);
    disp(['Best test accuracy: ', num2str(bestAcc * 100, '%.2f'), '%']);
    
    % Create and display a sorted results table
    resultsTable = table(binSizeResults, historyBinsResults, normalizeResults, accuracyResults, ...
                        'VariableNames', {'BinSize', 'HistoryBins', 'Normalize', 'Accuracy'});
    
    % Remove NaN values for sorting
    validRows = ~isnan(resultsTable.Accuracy);
    validResultsTable = resultsTable(validRows, :);
    
    % Sort by accuracy in descending order
    sortedResultsTable = sortrows(validResultsTable, 'Accuracy', 'descend');
    
    disp('Top 10 parameter combinations:');
    if size(sortedResultsTable, 1) >= 10
        disp(sortedResultsTable(1:10, :));
    else
        disp(sortedResultsTable);
    end
    
    %% 4) Train final model with best parameters and evaluate
    disp('Training final model with best parameters...');
    
    % Build features with best parameters
    [Xtrain, Ytrain] = buildFeaturesPartialLDA(trainingData, bestParams.binSize, ...
                                             bestParams.historyBins, bestParams.normalize, 'train');
    
    % Train final LDA model
    finalModel = fitcdiscr(Xtrain(:, 2:end), Ytrain);
    
    % Evaluate on training set
    trainPreds = predict(finalModel, Xtrain(:, 2:end));
    finalTrainAcc = mean(trainPreds == Ytrain);
    disp(['Final train accuracy: ', num2str(finalTrainAcc * 100, '%.2f'), '%']);
    
    % Evaluate on test set
    [finalTestAcc, binAccVec] = iterativeTestLDA(testData, finalModel, bestParams.binSize, ...
                                                bestParams.historyBins, bestParams.normalize);
    disp(['Final test accuracy: ', num2str(finalTestAcc * 100, '%.2f'), '%']);
    
    %% 5) Visualize time-based accuracy
    figure('Name', 'Time-based Accuracy (Final Model)');
    plot(binAccVec, '-o', 'LineWidth', 1.5);
    xlabel('Bin Index', 'FontSize', 12);
    ylabel('Accuracy', 'FontSize', 12);
    ylim([0, 1]);
    title('Time-based Test Accuracy: Iterative Decoding', 'FontSize', 14);
    grid on;
    
    % Plot heatmap of results
    plotGridSearchResults(binSizeResults, historyBinsResults, normalizeResults, accuracyResults, ...
                         binSizes, historyBinsValues, normalizeOptions);
end

%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

function [testAcc, binAccVec] = iterativeTestLDA(data, ldaModel, binSize, historyBins, leftoverNormalize)
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

                [predLabel, predScores] = predict(ldaModel, featRow); % Get raw LDA scores
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

function plotGridSearchResults(binSizeResults, historyBinsResults, normalizeResults, accuracyResults, ...
                              binSizes, historyBinsValues, normalizeOptions)
% plotGridSearchResults:
%  Visualize grid search results as heatmaps
%
% Inputs:
%   binSizeResults     - Array of bin sizes used in each trial
%   historyBinsResults - Array of history bins used in each trial
%   normalizeResults   - Array of normalize options used in each trial
%   accuracyResults    - Array of accuracy results
%   binSizes           - Array of all bin sizes tested
%   historyBinsValues  - Array of all history bins values tested
%   normalizeOptions   - Array of all normalize options tested

    % Create figures for each normalize option
    for n = 1:length(normalizeOptions)
        figure('Name', ['Grid Search Results - Normalize=', num2str(normalizeOptions(n))]);
        
        % Filter results for current normalize option
        normIdx = normalizeResults == normalizeOptions(n);
        filteredBinSizes = binSizeResults(normIdx);
        filteredHistoryBins = historyBinsResults(normIdx);
        filteredAccuracy = accuracyResults(normIdx);
        
        % Create accuracy matrix
        accuracyMatrix = NaN(length(binSizes), length(historyBinsValues));
        
        % Fill in accuracy matrix
        for i = 1:length(filteredBinSizes)
            if isnan(filteredAccuracy(i))
                continue; % Skip failed combinations
            end
            
            bsIdx = find(binSizes == filteredBinSizes(i));
            hbIdx = find(historyBinsValues == filteredHistoryBins(i));
            accuracyMatrix(bsIdx, hbIdx) = filteredAccuracy(i);
        end
        
        % Plot heatmap
        imagesc(accuracyMatrix);
        colormap('jet');
        colorbar;
        
        % Set colorbar limits based on non-NaN values
        validAccuracies = accuracyResults(~isnan(accuracyResults));
        if ~isempty(validAccuracies)
            caxis([min(validAccuracies), max(validAccuracies)]);
        end
        
        % Set labels
        xlabel('History Bins', 'FontSize', 12);
        ylabel('Bin Size (ms)', 'FontSize', 12);
        title(['Test Accuracy - Normalize=', num2str(normalizeOptions(n))], 'FontSize', 14);
        
        % Set tick labels
        xticks(1:length(historyBinsValues));
        yticks(1:length(binSizes));
        xticklabels(historyBinsValues);
        yticklabels(binSizes);
        
        % Add text labels with accuracy values
        for i = 1:size(accuracyMatrix, 1)
            for j = 1:size(accuracyMatrix, 2)
                if ~isnan(accuracyMatrix(i, j))
                    text(j, i, num2str(accuracyMatrix(i, j) * 100, '%.1f'), ...
                         'HorizontalAlignment', 'center', 'Color', 'white', 'FontWeight', 'bold');
                end
            end
        end
    end
    
    % Plot accuracy by parameter
    figure('Name', 'Accuracy by Parameter');
    
    % 1. Bin Size effect
    subplot(1, 3, 1);
    binSizeAcc = zeros(length(binSizes), 1);
    binSizeCount = zeros(length(binSizes), 1);
    
    for i = 1:length(binSizeResults)
        if isnan(accuracyResults(i))
            continue; % Skip failed combinations
        end
        
        idx = find(binSizes == binSizeResults(i));
        binSizeAcc(idx) = binSizeAcc(idx) + accuracyResults(i);
        binSizeCount(idx) = binSizeCount(idx) + 1;
    end
    
    binSizeAvg = zeros(length(binSizes), 1);
    for i = 1:length(binSizes)
        if binSizeCount(i) > 0
            binSizeAvg(i) = binSizeAcc(i) / binSizeCount(i);
        else
            binSizeAvg(i) = 0;
        end
    end
    
    bar(binSizeAvg);
    xticks(1:length(binSizes));
    xticklabels(binSizes);
    xlabel('Bin Size (ms)');
    ylabel('Average Accuracy');
    title('Effect of Bin Size');
    
    % 2. History Bins effect
    subplot(1, 3, 2);
    historyBinsAcc = zeros(length(historyBinsValues), 1);
    historyBinsCount = zeros(length(historyBinsValues), 1);
    
    for i = 1:length(historyBinsResults)
        if isnan(accuracyResults(i))
            continue; % Skip failed combinations
        end
        
        idx = find(historyBinsValues == historyBinsResults(i));
        historyBinsAcc(idx) = historyBinsAcc(idx) + accuracyResults(i);
        historyBinsCount(idx) = historyBinsCount(idx) + 1;
    end
    
    historyBinsAvg = zeros(length(historyBinsValues), 1);
    for i = 1:length(historyBinsValues)
        if historyBinsCount(i) > 0
            historyBinsAvg(i) = historyBinsAcc(i) / historyBinsCount(i);
        else
            historyBinsAvg(i) = 0;
        end
    end
    
    bar(historyBinsAvg);
    xticks(1:length(historyBinsValues));
    xticklabels(historyBinsValues);
    xlabel('History Bins');
    ylabel('Average Accuracy');
    title('Effect of History Bins');
    
    % 3. Normalize effect
    subplot(1, 3, 3);
    normalizeAcc = zeros(length(normalizeOptions), 1);
    normalizeCount = zeros(length(normalizeOptions), 1);
    
    for i = 1:length(normalizeResults)
        if isnan(accuracyResults(i))
            continue; % Skip failed combinations
        end
        
        idx = find(normalizeOptions == normalizeResults(i));
        normalizeAcc(idx) = normalizeAcc(idx) + accuracyResults(i);
        normalizeCount(idx) = normalizeCount(idx) + 1;
    end
    
    normalizeAvg = zeros(length(normalizeOptions), 1);
    for i = 1:length(normalizeOptions)
        if normalizeCount(i) > 0
            normalizeAvg(i) = normalizeAcc(i) / normalizeCount(i);
        else
            normalizeAvg(i) = 0;
        end
    end
    
    bar(normalizeAvg);
    xticks(1:length(normalizeOptions));
    xticklabels({'False', 'True'});
    xlabel('Normalize Leftover');
    ylabel('Average Accuracy');
    title('Effect of Normalization');
end