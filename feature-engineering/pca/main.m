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

[train_data, val_data, test_data, train_angles, val_angles, test_angles]...
 = split_data(pop_activity, angles, 0.8, 0.1, 0.1);
fprintf("train_data size: %d\n", size(train_data, 2))
fprintf("test_data size: %d\n", size(test_data, 2))
fprintf("val_data size: %d\n", size(val_data, 2))

%Visualise PCA
% pca_reduced_train = pca_reduction(train_data, true, train_angles, ' - 0.8 split');

pca_reduced_train = pca_reduction(train_data);
disp(size(pca_reduced_train));


