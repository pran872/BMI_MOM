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
% time_window = [100, 500];

pop_activity = zeros(num_neurons, num_trials * num_angles);  %(Neurons, Trials*Angles)
trial_idx = 1;
for angle_num = 1:num_angles
    for trial_num = 1:num_trials
        spike_data = trial(trial_num, angle_num).spikes;
        pop_activity(:, trial_idx) = mean(spike_data(:, :), 2);
        trial_idx = trial_idx + 1;
    end
end

[train_data, val_data, test_data] = split_data(pop_activity, 0.7, 0.15, 0.15);
pca_reduced_train = pca_reduce(train_data, true);


