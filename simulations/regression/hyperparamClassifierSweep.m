function bestPCA = hyperparamClassifierSweep(trainingData)
% testPCAEffectsOnClassifier: A script focusing only on PCA changes
%
% We do:
%   1) Bin and truncate data safely
%   2) Train a multi-class SVM with a fixed kernel, boxC, etc.
%   3) Loop over different PCA variance cutoffs (and possibly a 'no PCA' option)
%   4) Evaluate classification accuracy on validation
%   5) Pick best PCA setting, retrain on (train+val), measure test accuracy
%
% OUTPUT:
%   bestPCA: struct with .bestCutoff, .bestValAcc, .finalTestAcc, .svmAngle, .pcaCoeff, etc.
%
% Example usage:
%   bestPCA = testPCAEffectsOnClassifier(trainingData);

    %% 1) Basic config
    binSize     = 20;     % ms per bin
    historyBins = 5;      % how many bins of history in the feature
    fracTrain   = 0.6;    % 60% train
    fracVal     = 0.2;    % 20% val
    % Remainder => test

    % For the classifier, we fix these hyperparams:
    kernelFun   = 'gaussian';   % or 'linear'
    boxC        = 1;
    coding      = 'onevsone';   % or 'onevsall'

    % PCA options to test
    % e.g. [0 => no PCA, 90 => 90% var, 95 => 95% var, 99 => 99% var]
    pcaCutoffs  = [0, 90, 95, 99];

    fprintf('testPCAEffectsOnClassifier:\n');
    fprintf(' binSize=%d, historyBins=%d, \n', binSize, historyBins);
    fprintf(' SVM: kernel=%s, boxC=%.1f, coding=%s\n', kernelFun, boxC, coding);

    %% 2) Stratified split
    [trainIdx, valIdx, testIdx, angleLookup] = angleStratSplit(trainingData, fracTrain, fracVal);

    %% 3) Determine minNumBins across entire dataset for robust binning
    [numTrials, numAngles] = size(trainingData);
    minNumBins = Inf;
    for a=1:numAngles
        for tr=1:numTrials
            Tms = size(trainingData(tr,a).spikes,2);
            nBins = floor(Tms/binSize);
            if nBins<minNumBins
                minNumBins= nBins;
            end
        end
    end
    fprintf('Truncating all trials to %d bins\n', minNumBins);

    %% 4) Loop over PCA cutoffs => measure val accuracy => pick best
    bestValAcc  = -inf;
    bestCutoff  = 0;
    bestModel   = struct();

    for cvar = pcaCutoffs
        modelC = trainClassifierWithPCA(trainingData, trainIdx, angleLookup,...
            binSize, minNumBins, historyBins, cvar, kernelFun, boxC, coding);

        accVal = evaluateClassifierAcc(trainingData, valIdx, angleLookup, modelC);

        fprintf('  PCA cutoff=%.1f => valAcc=%.4f\n', cvar, accVal);

        if accVal>bestValAcc
            bestValAcc = accVal;
            bestCutoff = cvar;
            bestModel  = modelC;
        end
    end

    fprintf('\nBest PCA cutoff=%.1f => valAcc=%.4f\n', bestCutoff, bestValAcc);

    %% 5) Retrain on (train+val), final test
    combinedIdx = [trainIdx; valIdx];
    finalModel  = trainClassifierWithPCA(trainingData, combinedIdx, angleLookup,...
        binSize, minNumBins, historyBins, bestCutoff, kernelFun, boxC, coding);
    testAcc = evaluateClassifierAcc(trainingData, testIdx, angleLookup, finalModel);

    fprintf('Final test accuracy=%.4f with PCA cutoff=%.1f\n', testAcc, bestCutoff);

    %% 6) Return best struct
    bestPCA.binSize        = binSize;
    bestPCA.historyBins    = historyBins;
    bestPCA.bestCutoff     = bestCutoff;
    bestPCA.bestValAcc     = bestValAcc;
    bestPCA.finalTestAcc   = testAcc;
    bestPCA.svmAngle       = finalModel.svmAngle;
    bestPCA.pcaCoeff       = finalModel.pcaCoeff;

end


%% ========================================================================
function [trainIdx, valIdx, testIdx, angleLookup] = angleStratSplit(trainingData, fracTrain, fracVal)
% Stratify by angle
[numTrials,numAngles] = size(trainingData);
N = numTrials*numAngles;
angleLookup = zeros(N,1,'int32');
pos=1;
for a=1:numAngles
    for tr=1:numTrials
        angleLookup(pos)=a;
        pos=pos+1;
    end
end

trainIdx= [];
valIdx  = [];
testIdx = [];

for a=1:numAngles
    subIdx= find(angleLookup==a);
    nSub= length(subIdx);
    rp= randperm(nSub);
    nTr= round(fracTrain*nSub);
    nVa= round(fracVal*nSub);

    tr= subIdx(rp(1:nTr));
    va= subIdx(rp(nTr+1 : nTr+nVa));
    te= subIdx(rp(nTr+nVa+1: end));

    trainIdx= [trainIdx; tr(:)];
    valIdx  = [valIdx;   va(:)];
    testIdx = [testIdx;  te(:)];
end
trainIdx= trainIdx(randperm(length(trainIdx)));
valIdx  = valIdx(randperm(length(valIdx)));
testIdx = testIdx(randperm(length(testIdx)));
end

%% ========================================================================
function modelC = trainClassifierWithPCA(trainingData, sampleIdx, angleLookup,...
    binSize, minNumBins, historyBins, pcaVarCut, kernelFun, boxC, coding)
% Train single classifier with a specified PCA cutoff, kernel, etc.

[X_class, Y_class, pcaCoeff] = buildClassData(trainingData, sampleIdx, angleLookup,...
    binSize, minNumBins, historyBins, pcaVarCut);

tSVM = templateSVM('KernelFunction', kernelFun, 'BoxConstraint', boxC);
svmAngle = fitcecoc(X_class, Y_class, 'Learners', tSVM, 'Coding', coding);

modelC.svmAngle  = svmAngle;
modelC.pcaCoeff  = pcaCoeff;
modelC.binSize   = binSize;
modelC.minNumBins= minNumBins;
modelC.historyBins=historyBins;
modelC.pcaVarCut = pcaVarCut;
modelC.kernelFun = kernelFun;
modelC.boxC      = boxC;
modelC.coding    = coding;
end

%% ========================================================================
function accVal = evaluateClassifierAcc(trainingData, sampleIdx, angleLookup, modelC)
% Evaluate classification accuracy

[X_class, Y_class] = buildClassData(trainingData, sampleIdx, angleLookup,...
    modelC.binSize, modelC.minNumBins, modelC.historyBins, modelC.pcaVarCut, modelC.pcaCoeff);

yhat = predict(modelC.svmAngle, X_class);
accVal= sum(yhat==Y_class)/numel(Y_class);
end

%% ========================================================================
function [X_class, Y_class, pcaCoeffOut] = buildClassData(trainingData, sampleIdx, angleLookup,...
    binSize, minNumBins, historyBins, pcaCutVar, pcaCoeffIn)
% buildClassData => classification dataset
% pcaCutVar>0 => apply new or existing PCA
% if pcaCutVar=0 => skip PCA (like a 'no PCA' mode)
% if pcaCoeffIn is given => apply existing PCA transform

if nargin<8
    pcaCoeffIn=[];
end

accumX = [];
accumY = [];
[numTrials, numAngles] = size(trainingData);
nSamples= length(sampleIdx);

for i=1:nSamples
    idx= sampleIdx(i);
    angleID= angleLookup(idx);
    [tr, a] = sampleToTrialAngle(idx, [numTrials,numAngles]);
    if a~=angleID, continue; end

    rawSpikes = trainingData(tr,a).spikes;
    truncated = rawSpikes(:, 1:(minNumBins*binSize));

    for b= historyBins : minNumBins
        feat = getHistFeature(truncated, b, historyBins, binSize);
        accumX= [accumX; feat']; %#ok<AGROW>
        accumY= [accumY; angleID];
    end
end

X_class= accumX;
Y_class= accumY;
pcaCoeffOut= [];

if pcaCutVar>0 && isempty(pcaCoeffIn)
    % compute new PCA
    [coeff, ~, ~, ~, explained] = pca(double(X_class));
    cumExp = cumsum(explained);
    dimP   = find(cumExp>=pcaCutVar,1,'first');
    if isempty(dimP), dimP = length(explained); end
    coeff  = coeff(:,1:dimP);
    X_class= double(X_class)* coeff;
    pcaCoeffOut= coeff;
elseif pcaCutVar>0 && ~isempty(pcaCoeffIn)
    % apply existing PCA
    X_class= double(X_class)* pcaCoeffIn;
    pcaCoeffOut= pcaCoeffIn;
else
    % pcaCutVar=0 => no PCA
    % do nothing
end

end

%% ========================================================================
function featRow = getHistFeature(truncatedSpikes, binIdx, historyBins, binSize)
% getHistFeature => single row for the last 'historyBins' bins up to binIdx
numNeurons= size(truncatedSpikes,1);
featRow= zeros(1, numNeurons*historyBins, 'single');
pos=1;
for h=0:(historyBins-1)
    b= binIdx - h;
    st= (b-1)*binSize +1;
    ed= b*binSize;
    seg= truncatedSpikes(:, st:ed);
    meanBin= mean(seg,2);
    featRow(pos:pos+numNeurons-1)= single(meanBin);
    pos= pos+ numNeurons;
end
end

%% ========================================================================
function [trialID, angleID] = sampleToTrialAngle(idx, dataSize)
% Flatten: sample i => angle=ceil(i/nTrials), trial = i - (angle-1)*nTrials
nTrials= dataSize(1);
angleID= ceil(idx/nTrials);
trialID= idx - (angleID-1)*nTrials;
end
