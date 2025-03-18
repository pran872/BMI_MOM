% Test Script to give to the students, March 2015]
clc; clear all; close all;


teamName = 'simulations/best_model'; %enter the name of the folder


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
    
    % Create figure for animation
    figure(2)
    animationAx = subplot(1,1,1);
    hold on
    axis square
    grid
    title('Trajectory Animation')
    
    addpath(teamName);

    modelParameters = positionEstimatorTraining(trainingData);
    
    for tr=1:size(testData,1)
        fprintf('Decoding block %d/%d\n', tr, size(testData,1));

        for direc=randperm(8) 
            decodedHandPos = [];
            times=320:20:size(testData(tr,direc).spikes,2);
            
            % Store positions for animation
            animatedActualPos = zeros(2, length(times));
            animatedDecodedPos = zeros(2, length(times));
            
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
                
                % Store positions for animation
                idx = find(times == t);
                animatedActualPos(:, idx) = testData(tr, direc).handPos(1:2, t);
                animatedDecodedPos(:, idx) = decodedPos;
            end
            
            % Add to main figure
            figure(1)
            plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r')
            plot(testData(tr,direc).handPos(1,times), testData(tr,direc).handPos(2,times), 'b')
            scatter(decodedHandPos(1, 1), decodedHandPos(2, 1), 50, 'g', 'filled')
            scatter(testData(tr, direc).handPos(1, 1), testData(tr, direc).handPos(2, 1), 50, 'k', 'filled')
            
            % Generate animation for the current trajectory
            figure(2)
            cla(animationAx)
            xlim(animationAx, [min(min(animatedActualPos(1,:)), min(animatedDecodedPos(1,:)))-5, ...
                  max(max(animatedActualPos(1,:)), max(animatedDecodedPos(1,:)))+5])
            ylim(animationAx, [min(min(animatedActualPos(2,:)), min(animatedDecodedPos(2,:)))-5, ...
                  max(max(animatedActualPos(2,:)), max(animatedDecodedPos(2,:)))+5])
            
            % Plot full trajectories as faded lines
            plot(animationAx, animatedActualPos(1,:), animatedActualPos(2,:), 'b:', 'LineWidth', 1)
            plot(animationAx, animatedDecodedPos(1,:), animatedDecodedPos(2,:), 'r:', 'LineWidth', 1)
            
            % Create point objects for animation
            actualPosMarker = scatter(animationAx, animatedActualPos(1,1), animatedActualPos(2,1), 100, 'b', 'filled');
            decodedPosMarker = scatter(animationAx, animatedDecodedPos(1,1), animatedDecodedPos(2,1), 100, 'r', 'filled');
            legend(animationAx, 'Actual Path', 'Decoded Path', 'Actual Position', 'Decoded Position')
            
            % Animate the trajectory
            for i = 1:length(times)
                % Update marker positions
                set(actualPosMarker, 'XData', animatedActualPos(1,i), 'YData', animatedActualPos(2,i));
                set(decodedPosMarker, 'XData', animatedDecodedPos(1,i), 'YData', animatedDecodedPos(2,i));
                
                % Add trace of past positions with decreasing marker size
                if i > 1
                    actualTrace = scatter(animationAx, animatedActualPos(1,1:i-1), animatedActualPos(2,1:i-1), 20, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
                    decodedTrace = scatter(animationAx, animatedDecodedPos(1,1:i-1), animatedDecodedPos(2,1:i-1), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
                end
                
                title(animationAx, sprintf('Time: %d ms, Trial: %d, Direction: %d', times(i), tr, direc))
                drawnow
                pause(0.1)
                
                % Remove trace markers (to be redrawn in next iteration)
                if i > 1
                    delete(actualTrace)
                    delete(decodedTrace)
                end
            end
            
            % Ask if user wants to continue or skip to next trajectory
            if tr < size(testData,1) || direc < 8
                choice = questdlg('Continue to next trajectory?', 'Animation Control', ...
                                  'Continue', 'Skip remaining', 'Continue');
                if strcmp(choice, 'Skip remaining')
                    break;
                end
            end
            
            n_predictions = n_predictions+length(times);
        end
        
        % If user chose to skip, break outer loop too
        if exist('choice', 'var') && strcmp(choice, 'Skip remaining')
            break;
        end
    end
    
    figure(1)
    legend('Decoded', 'Actual')
    RMSE = sqrt(meanSqError/n_predictions);
    fprintf("RMSE: %.4f\n", RMSE);
    
    rmpath(genpath(teamName))
end