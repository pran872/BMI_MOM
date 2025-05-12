% Authors: Nicolas Dehandschoewercker, Sonya Kalsi, Matthieu Pallud, Pranathi Poojary
function modelParameters = positionEstimatorTraining(trainingData, clsMethod, regMethod)
    classifierBinSize = 20;
    classifierWindowSize = 300;
    classifierStandardisation = true;
    classifierBias = false;   
    regressorBinSize = 20;         
    regressorWindowSize = 300;
    regressorStandardisation = true;
    
    [numTrials, numAngles] = size(trainingData);

    %% 1. Classifier Training
    [classifierFeatures, featureMask] = preprocessClassifierFeatures(trainingData, classifierBinSize, classifierWindowSize);
    classifierLabels = repelem(1:numAngles, numTrials)';
    
    %PCA
    [classifierFeaturesPCA, V_reduced_cls, X_mean_cls, X_std_cls, explained_var] = pcaReduction(classifierFeatures, 0.95, classifierStandardisation, classifierBias);
    
    switch clsMethod
        case 'knn'
            classifier = fitcknn(classifierFeaturesPCA, classifierLabels, ...
                'NumNeighbors', 10, ...
                'Distance', 'cosine', ...
                'Standardize', false);
        case 'lda'
            classifier = trainLDA(classifierFeaturesPCA, classifierLabels);
        case 'logistic'
            template = templateLinear('Learner', 'logistic', ...
                          'Regularization', 'ridge', ...
                          'Lambda', 1e-4, ...
                          'Solver', 'lbfgs');

            classifier = fitcecoc(classifierFeaturesPCA, classifierLabels, ...
                      'Learners', template, ...
                      'Coding', 'onevsall');
    
        otherwise
            error('Unknown classification method: %s', clsMethod);
    end

    %% 2. Regressor Training
    regressors = cell(1, numAngles);
    regressorBias = false;
    for dir = 1:numAngles
        [allFeat, allPos] = preprocessForRegression(trainingData(:, dir), regressorBinSize, regressorWindowSize);
        
        switch regMethod
            case 'linear'
                regressorBias = true;
                [Xpca, V_reduced, X_mu, X_std] = pcaReduction(allFeat, 0.6, regressorStandardisation, regressorBias); 
                regressors{dir} = trainLinearRegressor(Xpca, allPos, V_reduced, X_mu, X_std, regressorBinSize, regressorWindowSize);
            case 'svr'
                [Xpca, V_reduced, X_mu, X_std] = pcaReduction(allFeat, 0.6, regressorStandardisation, regressorBias); 
                regressors{dir} = trainSVRRegressor(Xpca, allPos, V_reduced, X_mu, X_std, regressorBinSize, regressorWindowSize);
            case 'knn'
                k = 5;
                [Xpca, V_reduced, X_mu, X_std] = pcaReduction(allFeat, 0.6, regressorStandardisation, regressorBias); 
                regressors{dir} = trainKNNRegressor(Xpca, allPos, V_reduced, X_mu, X_std, regressorBinSize, regressorWindowSize, k);
            case 'rf'
                [Xpca, V_reduced, X_mu, X_std] = pcaReduction(allFeat, 0.6, regressorStandardisation, regressorBias); 
                numTrees = 50;
                regressors{dir} = trainRFRegressor(Xpca, allPos, V_reduced, X_mu, X_std, regressorBinSize, regressorWindowSize, numTrees);
            otherwise
                error('Unsupported regression method: %s', method);
        end
    end
    
    [centroids_x,centroids_y] = computecentroids(trainingData);   


    %% 3. Return Model Parameters
    modelParameters = struct(...
        'classifier', classifier, ...
        'classifierPCA', struct(...
            'projMatrix', V_reduced_cls, ...
            'X_mean', X_mean_cls, ...
            'X_std', X_std_cls), ...
        'regressors', {regressors}, ...
        'centroids_x', centroids_x, ...
        'centroids_y', centroids_y, ...
        'featureMask', featureMask, ...
        'expectedFeatures', size(classifierFeaturesPCA, 2), ...
        'classifierParams', struct('binSize', classifierBinSize, 'windowSize', classifierWindowSize));

    %% 4. Plot
    % This plot requires stat and ml toolbox
    % plot_pca_lda_combined(classifier, classifierFeaturesPCA, classifierLabels, explained_var)

    % plot_pca(classifierFeaturesPCA, classifierLabels, explained_var)
    % plot_lda(classifier, classifierFeaturesPCA, classifierLabels)
    % plot_regressor_params(regressors, 20) % Weight magnitude against PCA feature index

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

function [Xpca, V_reduced, X_mean, X_std, explained_var] = pcaReduction(X, varianceThreshold, standardisation, bias)
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

function [centroids_x,centroids_y] = computecentroids(trainingData)
    [numTrials, numAngles] = size(trainingData);
    centroids_x = zeros(numAngles,1);
    centroids_y = zeros(numAngles,1);

    for angle_num = 1:numAngles
        final_x_positions = zeros(1, numTrials);
        final_y_positions = zeros(1, numTrials);

        for trial_num = 1:numTrials
            final_x_positions(trial_num) = trainingData(trial_num, angle_num).handPos(1, end);
            final_y_positions(trial_num) = trainingData(trial_num, angle_num).handPos(2, end);
        end

        % Centroid as mean position
        centroids_x(angle_num) = mean(final_x_positions);
        centroids_y(angle_num) = mean(final_y_positions);

    end

end

%% Other regressor models

function regressor = trainLinearRegressor(Xpca, Y, V, mu, std, binSize, windowSize)
    Beta = Xpca \ Y;
    regressor = struct(...
        'projMatrix', V, ...
        'Beta', Beta, ...
        'mu', mu, ...
        'std', std, ...
        'binSize', binSize, ...
        'windowSize', windowSize, ...
        'method', 'linear');
end

function regressor = trainSVRRegressor(Xpca, Y, V, mu, std, binSize, windowSize)
    modelX = fitrsvm(Xpca, Y(:,1), 'KernelFunction', 'linear', 'Standardize', true);
    modelY = fitrsvm(Xpca, Y(:,2), 'KernelFunction', 'linear', 'Standardize', true);
    regressor = struct(...
        'modelX', modelX, ...
        'modelY', modelY, ...
        'projMatrix', V, ...
        'mu', mu, ...
        'std', std, ...
        'binSize', binSize, ...
        'windowSize', windowSize, ...
        'method', 'svr');
end

function regressor = trainKNNRegressor(Xpca, Y, V, mu, std, binSize, windowSize, k)
    regressor = struct(...
        'Xpca', Xpca, ...
        'Y', Y, ...
        'projMatrix', V, ...
        'mu', mu, ...
        'std', std, ...
        'binSize', binSize, ...
        'windowSize', windowSize, ...
        'k', k, ...
        'method', 'knn');
end

function regressor = trainRFRegressor(Xpca, Y, V, mu, std, binSize, windowSize, numTrees)
    modelX = TreeBagger(numTrees, Xpca, Y(:,1), 'Method', 'regression', 'OOBPrediction', 'off');
    modelY = TreeBagger(numTrees, Xpca, Y(:,2), 'Method', 'regression', 'OOBPrediction', 'off');

    regressor = struct(...
        'modelX', modelX, ...
        'modelY', modelY, ...
        'projMatrix', V, ...
        'mu', mu, ...
        'std', std, ...
        'binSize', binSize, ...
        'windowSize', windowSize, ...
        'method', 'rf');
end

%% Plotting function - PCA/LDA used in report

function plot_pca(classifierFeaturesPCA, classifierLabels, explained_var)
    pc1_var = explained_var(1) * 100;
    pc2_var = explained_var(2) * 100;

    figure;
    colors = turbo(8);
    gscatter(classifierFeaturesPCA(:,1), classifierFeaturesPCA(:,2), classifierLabels, colors);

    xlab = sprintf('PC1 (%.2f\\%%)', pc1_var);
    ylab = sprintf('PC2 (%.2f\\%%)', pc2_var);
    xlabel(xlab, 'Interpreter', 'latex');
    ylabel(ylab, 'Interpreter', 'latex');
    % title("Fully local")
    angleLabels = {'$30^\circ$', '$70^\circ$', '$110^\circ$', '$150^\circ$', ...
                   '$190^\circ$', '$230^\circ$', '$310^\circ$', '$350^\circ$'};
    legend(angleLabels, 'Location', 'bestoutside', 'Interpreter', 'latex');
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    grid off;
    set(gca, 'FontSize', 14);
    % filename = sprintf('../figures/pca_best_model.pdf');
    % saveas(gcf, filename);
    % close(gcf);
end

function plot_lda(classifier, classifierFeaturesPCA, classifierLabels)
    lda_proj = classifierFeaturesPCA * classifier.W(:, 1:2);  % Project onto LDA1 and LDA2

    figure;
    colors = turbo(8);
    gscatter(lda_proj(:,1), lda_proj(:,2), classifierLabels, colors);

    xlabel('LDA 1', 'Interpreter', 'latex');
    ylabel('LDA 2', 'Interpreter', 'latex');
    title('LDA Projection of Classifier Features', 'Interpreter', 'latex');

    angleLabels = {'$30^\circ$', '$70^\circ$', '$110^\circ$', '$150^\circ$', ...
                '$190^\circ$', '$230^\circ$', '$310^\circ$', '$350^\circ$'};
    legend(angleLabels, 'Interpreter', 'latex', 'Location', 'bestoutside');

    grid off;
    set(gca, 'FontSize', 14, 'TickLabelInterpreter', 'latex');

end

function plot_pca_lda_combined(classifier, classifierFeaturesPCA, classifierLabels, explained_var)
    angleLabels = {'$30^\circ$', '$70^\circ$', '$110^\circ$', '$150^\circ$', ...
                   '$190^\circ$', '$230^\circ$', '$310^\circ$', '$350^\circ$'};
    colors = turbo(8);
    pc1_var = explained_var(1) * 100;
    pc2_var = explained_var(2) * 100;

    f = figure;
    f.Position = [100, 100, 900, 400];
    t = tiledlayout(2, 2, 'TileSpacing', 'compact');
    
    % PCA subplot (left)
    nexttile([2 1])
    % disp(size(classifierFeaturesPCA))
    gscatter(classifierFeaturesPCA(:,1), classifierFeaturesPCA(:,2), ...
             classifierLabels, colors, '.', 12, 'off');
    xlabel(sprintf('PC1 (%.2f\\%%)', pc1_var), 'Interpreter', 'latex');
    ylabel(sprintf('PC2 (%.2f\\%%)', pc2_var), 'Interpreter', 'latex');
    xlim(gca, [-12 14])
    ylim(gca, [-10 14]);
    set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex');
    grid off;
    box off;

    % LDA subplot (right)
    lda_proj = classifierFeaturesPCA * classifier.W(:, 1:2);
    % disp(size(lda_proj))
    nexttile([2 1])
    gscatter(lda_proj(:,1), lda_proj(:,2), classifierLabels, ...
             colors, '.', 12, 'off');
    xlabel('LDA 1', 'Interpreter', 'latex');
    yl = ylabel('LDA 2', 'Interpreter', 'latex');
    yl.Position(1) = yl.Position(1) + 0.15;  % reduce ylabel distance
    ylim(gca, [-0.75 0.8])
    xlim(gca, [-0.9 0.9])
    yticks(gca, [-0.5 0 0.5])
    set(gca, 'FontSize', 20, 'TickLabelInterpreter', 'latex');
    grid off;
    box off;

    nexttile(t, 1);
    legendHandles = gobjects(length(angleLabels), 1);
    hold on;
    for i = 1:length(angleLabels)
        legendHandles(i) = plot(NaN, NaN, '.', ...
            'Color', colors(i,:), ...
            'MarkerSize', 24);
    end
    hold off;

    %Share legend
    lgd = legend(legendHandles, angleLabels, ...
        'Interpreter', 'latex', ...
        'Orientation', 'horizontal', ...
        'Box', 'off');
    lgd.Layout.Tile = 'south';
    lgd.FontSize = 20;

    % exportgraphics(f, '../figures/pca_lda_combined3.pdf', 'ContentType', 'vector');
end

function plot_regressor_params(regressors, N)
    % regressors: cell array of regressor structs
    % N: number of top PCA components to inspect

    angles = {'$30^\circ$', '$70^\circ$', '$110^\circ$', '$150^\circ$', ...
              '$190^\circ$', '$230^\circ$', '$310^\circ$', '$350^\circ$'};

    for dir = 1:length(regressors)
        Beta = regressors{dir}.Beta;
        Beta = Beta(1:end-1, :); 

        importance = vecnorm(Beta, 2, 2);
        figure;
        bar(importance(1:N));
        title(sprintf('Feature Importance for %s', angles{dir}), 'Interpreter', 'latex');
        xlabel('PCA feature index', 'Interpreter', 'latex');
        ylabel('Combined weight magnitude', 'Interpreter', 'latex');
        set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14);

        totalWeight = sum(importance);
        topWeight = sum(importance(1:N));
        fprintf('Regressor for %s: %.1f%% of weight in first %d PCA features\n', ...
                angles{dir}, 100 * topWeight / totalWeight, N);
    end
end

%% Built-in PCA functions

function [Xpca, V_reduced, X_mean, X_std, explained_var] = built_in_pca(X, varianceThreshold, standardisation, bias, classifierLabels)
    X_mean = mean(X, 1);
    if standardisation
        X_std = std(X, [], 1);
        X_std(X_std == 0) = 1;  % avoid division by zero
        X_processed = (X - X_mean) ./ X_std;
    else
        X_std = [];
        X_processed = X - X_mean;
    end

    [coeff, score, ~, ~, explained, ~] = pca(X_processed);

    cumVar = cumsum(explained);
    numPCs = find(cumVar >= varianceThreshold * 100, 1, 'first');

    V_reduced = coeff(:, 1:numPCs);
    Xpca = score(:, 1:numPCs);

    if bias
        Xpca = [Xpca, ones(size(Xpca, 1), 1)];
    end

    explained_var = explained / 100;

    figure;
    gscatter(score(:,1), score(:,2), classifierLabels, turbo(numel(unique(classifierLabels))));
    xlabel(sprintf('PC1 (%.2f\\%%)', explained(1)), 'Interpreter', 'latex');
    ylabel(sprintf('PC2 (%.2f\\%%)', explained(2)), 'Interpreter', 'latex');
    title("Built in everythin + plot")

    % Optional: format legend with angle labels
    angleLabels = {'$30^\circ$', '$70^\circ$', '$110^\circ$', '$150^\circ$', ...
                '$190^\circ$', '$230^\circ$', '$310^\circ$', '$350^\circ$'};
    legend(angleLabels, 'Interpreter', 'latex', 'Location', 'bestoutside');

    grid off;
    set(gca, 'FontSize', 14, 'TickLabelInterpreter', 'latex');
end

function built_in_pca_plot(classifierFeatures, classifierLabels)
    % X is your data matrix: rows = samples, columns = features
    [coeff, score, latent, tsquared, explained, mu] = pca(classifierFeatures);

    % Plot PC1 vs PC2
    figure;
    gscatter(score(:,1), score(:,2), classifierLabels, turbo(numel(unique(classifierLabels))));
    xlabel(sprintf('PC1 (%.2f\\%%)', explained(1)), 'Interpreter', 'latex');
    ylabel(sprintf('PC2 (%.2f\\%%)', explained(2)), 'Interpreter', 'latex');
    title("Built in only plot");

    % Optional: format legend with angle labels
    angleLabels = {'$30^\circ$', '$70^\circ$', '$110^\circ$', '$150^\circ$', ...
                '$190^\circ$', '$230^\circ$', '$310^\circ$', '$350^\circ$'};
    legend(angleLabels, 'Interpreter', 'latex', 'Location', 'bestoutside');

    grid off;
    set(gca, 'FontSize', 14, 'TickLabelInterpreter', 'latex');
end



