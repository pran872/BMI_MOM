function rmse = evaluateRegressor(testData, regressorModel)
    numAngles = length(regressorModel);
    angleRMSEs = zeros(1, numAngles);
    finalAnglePredictions = [];
    finalAngleActual = [];
    
    for angle = 1:numAngles
        % Get model parameters
        binSize = regressorModel(angle).binSize;
        windowSize = regressorModel(angle).windowSize;
        projMatrix = regressorModel(angle).projMatrix;
        Beta = regressorModel(angle).Beta;
        mu = regressorModel(angle).mu;  % Load feature mean
        
        allPredictions = [];
        allActual = [];
        [numTrials, ~] = size(testData);
        
        for trialNum = 1:numTrials
            spikes = double(testData(trialNum, angle).spikes);
            handPos = testData(trialNum, angle).handPos(1:2, :);  
            T = size(spikes, 2); 
            
            % Bin spikes and compute firing rate
            binCount = floor(T / binSize);
            spikesBinned = sum(reshape(spikes(:, 1:binCount * binSize), size(spikes,1), binSize, []), 2);
            spikesBinned = permute(spikesBinned, [1, 3, 2]);  
            fr = (1000 / binSize) * spikesBinned;
            
            % Process each time window
            windowBins = windowSize / binSize;
            for t = windowBins:binCount-1
                % Get and center features
                windowData = fr(:, t-windowBins+1:t);
                featVec = windowData(:)';  
                featVecCentered = featVec - mu;  % Critical: Apply centering
                
                % Project and predict
                featRed = featVecCentered * projMatrix;
                predictedPos = [featRed, 1] * Beta;  % Maintain bias term
                
                % Store results
                allPredictions(end+1, :) = predictedPos;
                allActual(end+1, :) = handPos(:, t * binSize)'; 
            end
        end
        
        % Calculate RMSE
        errors = allPredictions - allActual;
        angleRMSEs(angle) = sqrt(mean(errors(:).^2));
        if angle == 1
            finalAnglePredictions = allPredictions;
            finalAngleActual = allActual;
        end
    
        % Store results from the first angle for plotting
        if angle == 1
            finalAnglePredictions = allPredictions;
            finalAngleActual = allActual;
        end
    end
    
    % Print RMSE per angle
    for angle = 1:numAngles
        fprintf('Angle %d RMSE: %.4f\n', angle, angleRMSEs(angle));
    end
    
    % Compute overall RMSE
    rmse = mean(angleRMSEs);
    fprintf('Overall Regressor RMSE: %.4f\n', rmse);
    
    % Plot comparison of actual vs. predicted positions for first angle
    figure;
    plot(finalAngleActual(:, 1), finalAngleActual(:, 2), 'b-', 'LineWidth', 1.5);  % True trajectory
    hold on;
    plot(finalAnglePredictions(:, 1), finalAnglePredictions(:, 2), 'r--', 'LineWidth', 1.5);  % Predicted trajectory
    legend('Actual', 'Predicted');
    xlabel('X Position');
    ylabel('Y Position');
    title('Regressor Evaluation: Actual vs. Predicted Hand Position');
    grid on;
    hold off;
end