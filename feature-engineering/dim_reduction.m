%% Load and clear all
close all
clc
clear all
load monkeydata_training.mat
fprintf("This script requires stat and ML toolbox. Need to remove\n")

%% Population Activity Matrix
num_neurons = 98; % Total neurons
num_trials = 100; % Total trials
num_angles = 8;   % Movement directions
% time_window = [150, 550 ];

pop_activity = zeros(num_neurons, num_trials * num_angles); 

trial_idx = 1; % Index to track trials in the matrix

for angle_num = 1:num_angles
    for trial_num = 1:num_trials
        % Extract spike data for all neurons in the given trial & angle
        spike_data = trial(trial_num, angle_num).spikes;
        pop_activity(:, trial_idx) = mean(spike_data(:, :), 2);
        trial_idx = trial_idx + 1;
    end
end

%% PCA
[coeff, score, ~, ~, explained] = pca(pop_activity');
fprintf('First PC explains %.2f%% variance\n', explained(1));
fprintf('Second PC explains %.2f%% variance\n', explained(2));
fprintf('Third PC explains %.2f%% variance\n', explained(3));

% Choose the number of PCs that explain 95% of variance
cumulative_variance = cumsum(explained);
num_pcs = find(cumulative_variance >= 95, 1);
fprintf("Reduced dimensions from %d to %d\n", length(cumulative_variance), num_pcs);

% Project population activity onto selected PCs
reduced_activity = score(:, 1:num_pcs); 
disp(num_pcs)

figure;
plot(cumulative_variance, '-o');
xlabel('Number of Principal Components');
ylabel('Explained Variance (%)');
title('Variance Explained by PCA');
grid on;

colors = lines(num_angles);
figure;
hold on;
trial_idx = 1;

for angle_num = 1:num_angles
    scatter(score(trial_idx:trial_idx+num_trials-1, 1), ...
            score(trial_idx:trial_idx+num_trials-1, 2), ...
            50, colors(angle_num, :), 'filled');
    trial_idx = trial_idx + num_trials;
end

xlabel('PC 1');
ylabel('PC 2');
title('PCA Visualization of Neural Population Activity');
legend(arrayfun(@(x) ['Angle ' num2str(x)], linspace(0, 360, num_angles), 'UniformOutput', false));
grid on;
hold off;

figure;
imagesc(coeff(:, 1:10)); % Shows neuron weightings for top PCs
xlabel('Principal Components');
ylabel('Neurons');
title('Neuron Contributions to Principal Components');
colorbar;
