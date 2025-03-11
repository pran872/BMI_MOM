function [X, Y, pcaCoeff, metaInfo] = DataPreprocessing(rawData, config)
% DataPreprocessing:
%   1) Takes raw spiking data (rawData) with size (nTrials x nAngles), each cell having .spikes
%   2) Applies binning (using config.binSize, config.historyBins),
%   3) Optionally does PCA if config.doPCA == true,
%   4) Returns feature matrix X, label vector Y, PCA coefficient pcaCoeff, plus metaInfo if needed.
%
% ARGS:
%   rawData: (nTrials x nAngles) struct array with .spikes => (numNeurons x Tms)
%   config: struct with fields:
%       .binSize      (int, ms)
%       .historyBins  (int)
%       .doPCA        (bool)
%       .pcaVariance  (float) [optional] how much variance to keep
%
% OUTPUT:
%   X: (N x D) matrix of features, N = number of samples, D = dimension
%   Y: (N x 1) label vector indicating the angle or class
%   pcaCoeff: if doPCA==true, the PCA projection matrix (DxDsub), otherwise []
%   metaInfo: struct with extra details (like the binning logic, neuron means, etc.)

    if ~isfield(config,'binSize')
        error('config.binSize not defined');
    end
    if ~isfield(config,'historyBins')
        error('config.historyBins not defined');
    end
    if ~isfield(config,'doPCA')
        config.doPCA = false;
    end
    if ~isfield(config,'pcaVariance')
        config.pcaVariance = 95; % default
    end

    % A) Precompute minimum # of bins so we can truncate
    [numTrials, numAngles] = size(rawData);
    [minNumBins, truncatedAll] = truncateData(rawData, config.binSize);
    % truncatedAll => shape (numNeurons, (minNumBins*binSize), totalSamples)

    % B) Compute neuron means & build classification dataset
    neuronMeans = computeNeuronMeans(truncatedAll); 
    [Xraw, Yraw] = buildClassificationDataset(truncatedAll, rawData, ...
                    neuronMeans, config.binSize, config.historyBins, minNumBins);

    % C) Possibly do PCA
    pcaCoeff = [];
    X = Xraw;
    if config.doPCA
        [X, pcaCoeff] = doPCAonData(Xraw, config.pcaVariance);
    end

    Y = Yraw;

    % D) metaInfo
    metaInfo = struct();
    metaInfo.minNumBins = minNumBins;
    metaInfo.neuronMeans = neuronMeans;
    metaInfo.binSize = config.binSize;
    metaInfo.historyBins = config.historyBins;
    metaInfo.doPCA = config.doPCA;
end

%% ------------------------------------------------------------------------
function [minNumBins, truncatedAll] = truncateData(rawData, binSize)
% truncateData:
%   1) find minNumBins across all trials/angles => floor(Tms/binSize)
%   2) build truncatedAll => (numNeurons, minNumBins*binSize, totalSamples)

    [numTrials, numAngles] = size(rawData);
    numNeurons = size(rawData(1,1).spikes,1);
    totalSamples = numTrials * numAngles;

    minNumBins = Inf;
    sIdx = 1;
    % find minNumBins
    for a=1:numAngles
        for t=1:numTrials
            Tms = size(rawData(t,a).spikes,2);
            nBins = floor(Tms/binSize);
            if nBins<minNumBins
                minNumBins = nBins;
            end
        end
    end

    truncatedAll = zeros(numNeurons, minNumBins*binSize, totalSamples, 'single');
    sIdx=1;
    for a=1:numAngles
        for t=1:numTrials
            spikesT = rawData(t,a).spikes; 
            truncatedAll(:,:, sIdx) = single(spikesT(:,1: (minNumBins* binSize)));
            sIdx=sIdx+1;
        end
    end
end

function neuronMeans = computeNeuronMeans(truncatedAll)
    % truncatedAll => shape (numNeurons, minNumBins*binSize, totalSamples)
    neuronMeans = double( squeeze( mean(mean(truncatedAll,2),3) ) ); 
    % => (numNeurons x 1)
end

function [X, Y] = buildClassificationDataset(truncatedAll, rawData, neuronMeans, binSize, historyBins, minNumBins)
% buildClassificationDataset:
%   For each trial/angle, for each bin in [historyBins..minNumBins], build a feature row => X
%   label => angle => Y

    [numTrials, numAngles] = size(rawData);
    numNeurons= size(truncatedAll,1);
    totalSamples= numTrials* numAngles;
    samplesPerTrial= (minNumBins - historyBins+1);
    maxRows= samplesPerTrial * totalSamples;

    X = zeros(maxRows, numNeurons* historyBins, 'single');
    Y = zeros(maxRows,1,'int32');

    rowPos=1; 
    sIdx=1;
    for a=1:numAngles
        for t=1:numTrials
            spikesTr= truncatedAll(:,:, sIdx);
            sIdx=sIdx+1;

            for b= historyBins: minNumBins
                featRow= zeros(1, numNeurons* historyBins,'single');
                colPos=1;
                for h=0:(historyBins-1)
                    binIdx= b-h;
                    st= (binIdx-1)* binSize +1;
                    ed= binIdx* binSize;
                    seg= spikesTr(:, st:ed);
                    meanBin= mean(seg,2) - neuronMeans;
                    featRow(colPos: colPos+ numNeurons -1)= single(meanBin);
                    colPos= colPos+ numNeurons;
                end
                X(rowPos,:)= featRow;
                Y(rowPos)= a;  % angle label
                rowPos= rowPos+1;
            end
        end
    end

    X = X(1:rowPos-1,:);
    Y = Y(1:rowPos-1);
end

function [X_pca, pcaCoeff] = doPCAonData(Xraw, pcaVariance)
% doPCAonData => 
%   runs pca on Xraw, then picks top comps that achieve pcaVariance e.g. 95%
    [coeff, ~, ~, ~, explained] = pca(double(Xraw));
    csum= cumsum(explained);
    dimP= find(csum>= pcaVariance, 1, 'first');
    if isempty(dimP), dimP= length(explained); end

    coeff= coeff(:,1:dimP);
    X_pca= double(Xraw)* coeff;
    pcaCoeff= coeff;
end
