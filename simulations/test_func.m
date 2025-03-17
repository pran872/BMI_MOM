% Test Script to give to the students, March 2015]
clc; clear all; close all;


teamName = '/Users/pranathipoojary/Imperial/BMI/BMI_MOM/simulations/regression_new'; %enter the name of the folder


RMSE = testFunction_for_students_MTb(teamName)
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

function RMSE = testFunction_for_students_MTb(teamName)
    load monkeydata_training.mat
    rng(2013);
    ix = randperm(length(trial));
    
    addpath(teamName);
    trainingData = trial(ix(1:80),:);
    testData = trial(ix(81:end),:);
    
    fprintf('Testing the continuous position estimator...\n')
    
    meanSqError = 0;
    n_predictions = 0;  
    
    % Create figure for all trajectories
    figure(1)
    hold on
    axis square
    grid
    
    
    addpath(teamName);

    modelParameters = positionEstimatorTraining(trainingData);
    
    for tr=1:size(testData,1)
        fprintf('Decoding block %d/%d\n', tr, size(testData,1));
        
        for direc=randperm(8) 
            decodedHandPos = [];
            times=320:20:size(testData(tr,direc).spikes,2);
            
            for t=times
                past_current_trial.trialId = testData(tr,direc).trialId;
                past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 

                if nargout('positionEstimator') == 3
                    [decodedPosX, decodedPosY, newParameters] = positionEstimator(...
                        past_current_trial, ...
                        modelParameters);
                    modelParameters = newParameters;
                elseif nargout('positionEstimator') == 2
                    [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
                end
                
                decodedPos = [decodedPosX; decodedPosY];
                decodedHandPos = [decodedHandPos decodedPos];
                
                meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            end
            
            % Add to main figure
            figure(1)
            plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r')
            plot(testData(tr,direc).handPos(1,times), testData(tr,direc).handPos(2,times), 'b')
            scatter(decodedHandPos(1, 1), decodedHandPos(2, 1), 50, 'g', 'filled')
            scatter(testData(tr, direc).handPos(1, 1), testData(tr, direc).handPos(2, 1), 50, 'k', 'filled')
            
            n_predictions = n_predictions+length(times);
        end
    end
    
    figure(1)
    legend('Decoded', 'Actual')
    RMSE = sqrt(meanSqError/n_predictions);
    fprintf("RMSE: %.4f\n", RMSE);
    
    rmpath(genpath(teamName))
end