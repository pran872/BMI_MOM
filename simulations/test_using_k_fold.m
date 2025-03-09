function avg_RMSE = testFunction_KFold(teamName, k)
    teamName = 'Wiener'; % Folder name for model files
    k=5;

    % Load dataset
    load monkeydata_training.mat

    % Set random seed for reproducibility
    rng(2013);
    ix = randperm(length(trial)); % Shuffle trials

    addpath(teamName);

    % Define k-folds
    num_trials = length(trial);
    fold_size = floor(num_trials / k);
    RMSEs = zeros(k, 1); % Store RMSEs for each fold

    fprintf('Performing %d-fold cross-validation...\n', k);

    for fold = 1:k
        fprintf('Processing Fold %d/%d...\n', fold, k);

        % Define test and train splits
        test_idx = ix((fold-1)*fold_size + 1 : min(fold*fold_size, num_trials));
        train_idx = setdiff(ix, test_idx);

        trainingData = trial(train_idx, :);
        testData = trial(test_idx, :);

        % Train Model
        modelParameters = positionEstimatorTraining(trainingData);

        % Initialize RMSE calculation
        meanSqError = 0;
        n_predictions = 0;

        for tr = 1:size(testData,1)
            display(['Decoding block ', num2str(tr), ' out of ', num2str(size(testData,1))]);
            pause(0.001)

            for direc = randperm(8) 
                decodedHandPos = [];
                times = 320:20:size(testData(tr,direc).spikes, 2);

                for t = times
                    past_current_trial.trialId = testData(tr,direc).trialId;
                    past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
                    past_current_trial.decodedHandPos = decodedHandPos;
                    past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);

                    if nargout('positionEstimator') == 3
                        [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                        modelParameters = newParameters;
                    elseif nargout('positionEstimator') == 2
                        [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
                    end

                    decodedPos = [decodedPosX; decodedPosY];
                    decodedHandPos = [decodedHandPos decodedPos];

                    meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                end

                n_predictions = n_predictions + length(times);
            end
        end

        % Compute RMSE for this fold
        RMSEs(fold) = sqrt(meanSqError / n_predictions);
        fprintf('Fold %d RMSE: %.4f\n', fold, RMSEs(fold));
    end

    % Compute average RMSE across all folds
    avg_RMSE = mean(RMSEs);
    fprintf('Average RMSE across %d folds: %.4f\n', k, avg_RMSE);

    rmpath(genpath(teamName));
end
