%% Load and clear all
close all
clc
clear all
load monkeydata_training.mat
fprintf("PCA requires stat and ML toolbox. Need to remove\n")

rng(42); % set seed



%% Population Activity Matrix
num_neurons = size(trial(1, 1).spikes, 1);
num_trials = size(trial, 1);
num_angles = size(trial, 2);
angle_labels = [30, 70, 110, 150, 190, 230, 310, 350];
angles = repelem(angle_labels, num_trials);

pop_activity = zeros(num_neurons, num_trials * num_angles);  %(Neurons, Trials*Angles)
trial_idx = 1;
for angle_num = 1:num_angles
    for trial_num = 1:num_trials
        spike_data = trial(trial_num, angle_num).spikes;
        pop_activity(:, trial_idx) = mean(spike_data, 2);
        trial_idx = trial_idx + 1;
    end
end

%% Train

hand_pos_x_all = extract_hand_positions(trial, num_trials, num_angles, 'x');
hand_pos_y_all = extract_hand_positions(trial, num_trials, num_angles, 'y');

[train_data, val_data, test_data, train_idx, val_idx, test_idx]...
 = split_data(pop_activity, angles, 0.8, 0.1, 0.1);
fprintf("train_data size: %d\n", size(train_data, 2))
fprintf("test_data size: %d\n", size(test_data, 2))
fprintf("val_data size: %d\n", size(val_data, 2))
% Extract corresponding hand positions
hand_pos_x_train = hand_pos_x_all(train_idx);
hand_pos_y_train = hand_pos_y_all(train_idx);

hand_pos_x_val = hand_pos_x_all(val_idx);
hand_pos_y_val = hand_pos_y_all(val_idx);

hand_pos_x_test = hand_pos_x_all(test_idx);
hand_pos_y_test = hand_pos_y_all(test_idx);


k = 2; % Choose number of neighbors

predicted_x = knn_regression(train_data, hand_pos_x_train, test_data, k);
predicted_y = knn_regression(train_data, hand_pos_y_train, test_data, k);

corr_x = corr(predicted_x, hand_pos_x_test);
corr_y = corr(predicted_y, hand_pos_y_test);

fprintf('kNN Decoding Accuracy: X Corr = %.2f, Y Corr = %.2f\n', corr_x, corr_y);


% Compute RMSE for X position
% rmse_x = sqrt(mean((hand_pos_x_test - predicted_x).^2));

% % Compute RMSE for Y position
% rmse_y = sqrt(mean((hand_pos_y_test - predicted_y).^2));

% % Display RMSE values
% fprintf('kNN RMSE: X = %.2f, Y = %.2f\n', rmse_x, rmse_y);

% Compute Euclidean distance RMSE for both X and Y coordinates
meanSqError = mean(sum(( [hand_pos_x_test, hand_pos_y_test] - [predicted_x, predicted_y] ).^2, 2));

% Compute final RMSE
RMSE = sqrt(meanSqError);

% Display RMSE
fprintf('Combined RMSE: %.2f\n', RMSE);


figure; hold on;
plot(hand_pos_x_test, hand_pos_y_test, 'bo', 'MarkerSize', 6, 'DisplayName', 'Actual');
plot(predicted_x, predicted_y, 'ro', 'MarkerSize', 6, 'DisplayName', 'Predicted');
legend;
xlabel('X Position'); ylabel('Y Position');
title('kNN Decoding: Actual vs. Predicted Hand Trajectory');
grid on;
hold off;
