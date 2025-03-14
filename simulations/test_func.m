% Test Script to give to the students, March 2015]
clc; clear all; close all;


teamName = 'simulations/regression'; %enter the name of the folder


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
    trainingData = trial(ix(1:50),:);
    testData = trial(ix(51:end),:);
    
    fprintf('Testing the continuous position estimator...\n')
    
    meanSqError = 0;
    n_predictions = 0;  
    plottedExample = false;  % Flag to track if we've plotted our example
    
    % Create figure for all trajectories
    figure(1)
    hold on
    axis square
    grid
    
    % Train Model
    modelParameters = positionEstimatorTraining(trainingData);
    
    % Create separate figure for single trial example
    exampleFig = figure(2);
    set(exampleFig, 'Position', [100 100 800 600]);
    
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
                
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
                decodedPos = [decodedPosX; decodedPosY];
                decodedHandPos = [decodedHandPos decodedPos];
                
                meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            end
            
            % Plot single example (first complete trial/direction)
            if ~plottedExample && tr == 1 && direc == 1
                figure(2)
                actualPos = testData(tr,direc).handPos(1:2,times);
                plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r-', 'LineWidth', 2)
                hold on
                plot(actualPos(1,:), actualPos(2,:), 'b--', 'LineWidth', 2)
                legend('Decoded', 'Actual', 'Location', 'best')
                title(sprintf('Trajectory Comparison\nTrial %d, Direction %d', tr, direc))
                xlabel('X Position (mm)')
                ylabel('Y Position (mm)')
                grid on
                axis equal
                hold off
                plottedExample = true;
            end
            
            % Add to main figure
            figure(1)
            plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r')
            plot(testData(tr,direc).handPos(1,times), testData(tr,direc).handPos(2,times), 'b')
            
            n_predictions = n_predictions+length(times);
        end
    end
    
    figure(1)
    legend('Decoded', 'Actual')
    RMSE = sqrt(meanSqError/n_predictions);
    fprintf('Overall RMSE: %.4f\n', RMSE);
    
    rmpath(genpath(teamName))
end