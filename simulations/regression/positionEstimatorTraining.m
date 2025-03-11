function modelParameters = trainPipeline(trainingData, configFileName)
% trainPipeline:
%   1) Loads or receives 'trainingData' (nTrials x nAngles),
%   2) Reads classification+regression hyperparams from 'configFileName' (optional),
%   3) Builds separate classification + regression datasets (no PCA here),
%   4) Trains a classifier for angle, plus one regressor per angle for (x,y),
%   5) Returns 'modelParameters' for use in testing/inference.
%
% Example usage:
%   modelParams = trainPipeline(trainingData, 'myHyperparams.json');

    if nargin<2
        % If no configFileName, use defaults
        hyperparams = defaultHyperparams();
    else
        % Load from file, e.g. JSON or MAT. For demonstration:
        hyperparams = loadHyperparamsFromFile(configFileName);
    end

    disp('==== [trainPipeline] Using Hyperparams: ====');
    disp(hyperparams);

    [numTrials, numAngles] = size(trainingData);
    disp(['Training data has ', num2str(numTrials), ' trials, and ', num2str(numAngles), ' angles.']);
    disp(['Number of neurons: ', num2str(size(trainingData(1,1).spikes,1))]);

    %% =========== A) Classification pipeline ===========
    [minNumBinsC, truncatedAllC, meansC] = precomputeTruncatedData(trainingData, hyperparams.binSizeC);
    [X_class, Y_class] = buildClassificationDatasetNoPCA(trainingData, truncatedAllC, ...
        meansC, hyperparams.binSizeC, hyperparams.historyBinsC, minNumBinsC);

    % Train the classifier
    classifierModel = trainClassifier(X_class, Y_class, hyperparams.classifier);

    %% =========== B) Regression pipeline ===========
    [minNumBinsR, truncatedAllR, meansR] = precomputeTruncatedData(trainingData, hyperparams.binSizeR);
    [X_reg, Y_reg] = buildRegressionDatasetNoPCA(trainingData, truncatedAllR, ...
        meansR, hyperparams.binSizeR, hyperparams.historyBinsR, minNumBinsR);

    regModels = trainRegressors(X_reg, Y_reg, hyperparams.regressor);

    %% =========== C) Evaluate (Optional) ===========
    % You can do cross-validation or hold-out to measure classification accuracy or regression error
    % e.g., classificationAccuracy = evaluateClassifier(X_class, Y_class, classifierModel);
    % e.g., regressionError = evaluateRegression(X_reg, Y_reg, regModels);

    % Visualizations can also be placed here, e.g.:
    % plotConfusionMatrix(...), plotAnglePredictions(...), etc.

    %% =========== D) Store final model parameters ===========
    modelParameters = struct();
    % Classification info
    modelParameters.classifierModel   = classifierModel;
    modelParameters.binSizeC         = hyperparams.binSizeC;
    modelParameters.historyBinsC     = hyperparams.historyBinsC;
    modelParameters.neuronMeansC     = meansC;

    % Regression info
    modelParameters.regModels        = regModels;
    modelParameters.binSizeR         = hyperparams.binSizeR;
    modelParameters.historyBinsR     = hyperparams.historyBinsR;
    modelParameters.neuronMeansR     = meansR;

    disp('Done training pipeline with separate classification and regression (no PCA).');
end

%% ------------------------------------------------------------------------
function hyperparams = defaultHyperparams()
% defaultHyperparams: returns a struct of fallback hyperparams if config not provided

    hyperparams = struct();

    % Classification hyperparams
    hyperparams.binSizeC     = 40;  % ms
    hyperparams.historyBinsC = 12;
    hyperparams.classifier   = struct('kernel','gaussian','coding','onevsone');

    % Regression hyperparams
    hyperparams.binSizeR     = 10;  % ms
    hyperparams.historyBinsR = 10;
    hyperparams.regressor    = struct('kernel','gaussian');

end

function hp = loadHyperparamsFromFile(configFileName)
% A placeholder if you have a JSON or MAT file with hyperparams
% For demonstration, just call defaultHyperparams:
    hp = defaultHyperparams();
    disp(['(Placeholder) Loaded hyperparams from file: ', configFileName]);
end

%% ------------------------------------------------------------------------
function [classifierModel] = trainClassifier(X_class, Y_class, clfParams)
% e.g. use fitcecoc with kernel = clfParams.kernel, coding = clfParams.coding

    tSVM = templateSVM('KernelFunction', clfParams.kernel);
    classifierModel = fitcecoc(X_class, Y_class, ...
        'Learners', tSVM, 'Coding', clfParams.coding);
end

function [regModels] = trainRegressors(X_reg, Y_reg, regParams)
% train one model per angle, columns 1..2 => x,y, column 3 => angle
    numAngles = max(Y_reg(:,3));
    regModels = cell(numAngles,1);
    for a=1:numAngles
        mask = (Y_reg(:,3)==a);
        Xa   = X_reg(mask,:);
        xy   = Y_reg(mask,1:2);
        if isempty(Xa)
            regModels{a}=[];
            continue;
        end
        svmX = fitrsvm(Xa, xy(:,1), 'KernelFunction', regParams.kernel);
        svmY = fitrsvm(Xa, xy(:,2), 'KernelFunction', regParams.kernel);
        regModels{a} = struct('svmX',svmX,'svmY',svmY);
    end
end

%% ------------------------------------------------------------------------
function [minNumBins, truncatedAll, neuronMeans] = precomputeTruncatedData(trainingData, binSize)
% same approach as your code: find minNumBins, build truncatedAll, compute means

    [numTrials, numAngles] = size(trainingData);
    numNeurons = size(trainingData(1,1).spikes,1);
    totalSamples= numTrials*numAngles;

    minNumBins = Inf;
    sIdx=1;
    for a=1:numAngles
        for tr=1:numTrials
            Tms= size(trainingData(tr,a).spikes,2);
            nBins= floor(Tms/binSize);
            if nBins< minNumBins
                minNumBins= nBins;
            end
        end
    end

    truncatedAll= zeros(numNeurons, minNumBins*binSize, totalSamples, 'single');
    sIdx=1;
    for a=1:numAngles
        for tr=1:numTrials
            rawSpikes= trainingData(tr,a).spikes;
            truncatedAll(:,:, sIdx)= single(rawSpikes(:,1:(minNumBins*binSize)));
            sIdx= sIdx+1;
        end
    end

    allSpikesMean= squeeze(mean(mean(truncatedAll,2),3)); 
    neuronMeans= double(allSpikesMean);
end

%% ------------------------------------------------------------------------
function [Xclass, Yclass] = buildClassificationDatasetNoPCA(trainingData, truncatedAll, ...
    meansC, binSizeC, historyBinsC, minNumBinsC)

    [numTrials,numAngles] = size(trainingData);
    numNeurons= size(truncatedAll,1);
    totalSamples= numTrials*numAngles;
    samplesPerTrial= (minNumBinsC - historyBinsC+1);
    rowsAlloc= samplesPerTrial * totalSamples;

    Xclass= zeros(rowsAlloc, numNeurons*historyBinsC,'single');
    Yclass= zeros(rowsAlloc,1,'int32');

    rowPos=1; sIdx=1;
    for a=1:numAngles
        for tr=1:numTrials
            trialSpikes= truncatedAll(:,:, sIdx);
            sIdx=sIdx+1;
            for b= historyBinsC: minNumBinsC
                feat= zeros(1, numNeurons*historyBinsC,'single');
                colPos=1;
                for h=0:(historyBinsC-1)
                    binIdx= b-h;
                    st= (binIdx-1)* binSizeC+1;
                    ed= binIdx* binSizeC;
                    seg= trialSpikes(:, st:ed);
                    meanBin= mean(seg,2)- meansC;
                    feat(colPos:colPos+numNeurons-1)=single(meanBin);
                    colPos= colPos+ numNeurons;
                end
                Xclass(rowPos,:)=feat;
                Yclass(rowPos)=a;
                rowPos= rowPos+1;
            end
        end
    end

    Xclass= Xclass(1:rowPos-1,:);
    Yclass= Yclass(1:rowPos-1);
end

function [Xreg, Yreg] = buildRegressionDatasetNoPCA(trainingData, truncatedAll, ...
    meansR, binSizeR, historyBinsR, minNumBinsR)

    [numTrials,numAngles]= size(trainingData);
    numNeurons= size(truncatedAll,1);
    totalSamples= numTrials*numAngles;
    samplesPerTrial= (minNumBinsR- historyBinsR+1);
    rowsAlloc= samplesPerTrial * totalSamples;

    Xreg= zeros(rowsAlloc, numNeurons*historyBinsR,'single');
    Yreg= zeros(rowsAlloc, 3,'single');
    rowPos=1; sIdx=1;
    for a=1:numAngles
        for tr=1:numTrials
            trialSpikes= truncatedAll(:,:, sIdx);
            pos= trainingData(tr,a).handPos;
            sIdx=sIdx+1;
            for b= historyBinsR: minNumBinsR
                feat= zeros(1, numNeurons*historyBinsR,'single');
                colPos=1;
                for h=0:(historyBinsR-1)
                    binIdx= b-h;
                    st= (binIdx-1)*binSizeR+1;
                    ed= binIdx*binSizeR;
                    seg= trialSpikes(:, st:ed);
                    meanBin= mean(seg,2)- meansR;
                    feat(colPos: colPos+numNeurons-1)= single(meanBin);
                    colPos= colPos+ numNeurons;
                end
                Xreg(rowPos,:)= feat;
                xtrue= pos(1, b*binSizeR);
                ytrue= pos(2, b*binSizeR);
                Yreg(rowPos,:)=[xtrue, ytrue, a];
                rowPos= rowPos+1;
            end
        end
    end

    Xreg= Xreg(1:rowPos-1,:);
    Yreg= Yreg(1:rowPos-1,:);
end
