function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
% positionEstimator:
%   1) Gathers classification feature => picks angle => accumulates confidence
%   2) Gathers regression feature => picks that angle's regModel => gets x,y
%   3) Returns x,y, updates modelParameters for next call
%
% ARGS:
%   test_data: struct with .spikes => (neurons x T)
%   modelParameters: from trainPipeline, plus .angleConf for confidence
%
% OUTPUT:
%   [x,y], updated modelParameters (the function returns it, or you can store it with assignin)

    %% if first call, define angleConf
    if ~isfield(modelParameters, 'angleConf')
        K= length(modelParameters.regModels);
        modelParameters.angleConf= zeros(1,K);
    end

    % Some user-chosen constants
    confidenceDecay = 0.98;
    voteIncrement   = 1.0;

    %% =========== A) Classification pipeline (No PCA) ===========
    [angleWinner] = runClassifier(test_data.spikes, modelParameters);

    % angleConf update
    modelParameters.angleConf = modelParameters.angleConf * confidenceDecay;
    modelParameters.angleConf(angleWinner) = modelParameters.angleConf(angleWinner)+ voteIncrement;
    [~, bestAngle] = max(modelParameters.angleConf);

    %% =========== B) Regression pipeline (No PCA) ===========
    [xhat, yhat] = runRegressor(test_data.spikes, bestAngle, modelParameters);

    x= xhat; 
    y= yhat;
end

%% ------------------------------------------------------------------------
function angleWinner = runClassifier(spikes, modelParams)
% runClassifier: uses binSizeC, historyBinsC, meansC, and modelParams.classifierModel
    binSizeC     = modelParams.binSizeC;
    historyBinsC = modelParams.historyBinsC;
    meansC       = modelParams.neuronMeansC;
    svmAngle     = modelParams.classifierModel;  % we stored as 'classifierModel' or 'svmAngle'

    T   = size(spikes,2);
    totalBinsC= floor(T/binSizeC);
    if totalBinsC < historyBinsC
        angleWinner=1;
        return;
    end

    featClass= zeros(1, historyBinsC*size(spikes,1),'single');
    colPos=1;
    for h=0:(historyBinsC-1)
        b= totalBinsC-h;
        st= (b-1)*binSizeC+1;
        ed= b* binSizeC;
        seg= spikes(:, st:ed);
        meanBin= mean(seg,2)- meansC;
        nN= size(spikes,1);
        featClass(colPos: colPos+nN-1)= single(meanBin);
        colPos= colPos+ nN;
    end

    % pick angle
    anglePred= predict(svmAngle, featClass);
    angleWinner= double(anglePred);
end

function [x,y] = runRegressor(spikes, angleID, modelParams)
% runRegressor: uses binSizeR, historyBinsR, meansR, modelParams.regModels{angleID}
    binSizeR     = modelParams.binSizeR;
    historyBinsR = modelParams.historyBinsR;
    meansR       = modelParams.neuronMeansR;
    regModels    = modelParams.regModels;

    T= size(spikes,2);
    totalBinsR= floor(T/ binSizeR);
    if totalBinsR< historyBinsR
        x=0; y=0; return;
    end

    featReg= zeros(1, historyBinsR*size(spikes,1),'single');
    colPos=1;
    for hh=0:(historyBinsR-1)
        b= totalBinsR - hh;
        st= (b-1)* binSizeR+1;
        ed= b* binSizeR;
        seg= spikes(:, st:ed);
        meanBin= mean(seg,2)- meansR;
        nN= size(spikes,1);
        featReg(colPos: colPos+nN-1)= single(meanBin);
        colPos= colPos+ nN;
    end

    chosenReg = regModels{angleID};
    if isempty(chosenReg)
        x=0; y=0; return;
    end

    x= predict(chosenReg.svmX, featReg);
    y= predict(chosenReg.svmY, featReg);
end
