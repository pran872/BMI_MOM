function [x, y] = positionEstimator(test_data, modelParameters)
    classification_model = modelParameters.classificationModel;
    regression_models = modelParameters.regressionModels;
    
    classificationPreds = predict(classification_model, x);

    regression


    
end

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
