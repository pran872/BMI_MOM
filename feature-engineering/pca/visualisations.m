%% Load and clear all
close all
clc
clear all
load monkeydata_training.mat
whos
disp(trial(1))

%% Raster Plots
% Select a specific trial and reaching angle
trial_num = 1; % Change this for different trials
angle_num = 3; % Change this for different reaching angles

% Extract spike data for the chosen trial
spike_data = trial(trial_num, angle_num).spikes; % 98 x 672 matrix
[num_neurons, num_time_bins] = size(spike_data); % Get matrix dimensions

% Create the raster plot
figure;
hold on;
for neuron_idx = 1:num_neurons
    % Find spike times (where spikes occurred)
    spike_times = find(spike_data(neuron_idx, :) > 0);
    
    % Plot each spike as a dot or short line
    for i = 1:length(spike_times)
        plot([spike_times(i), spike_times(i)], [neuron_idx - 0.4, neuron_idx + 0.4], 'k', 'LineWidth', 1);
    end
end

% Formatting the plot
xlabel('Time (bins)');
ylabel('Neural Unit (Neuron Index)');
title(['Population Raster Plot - Trial ', num2str(trial_num), ', Angle ', num2str(angle_num)]);
xlim([0, num_time_bins]); % Set x-axis limits
ylim([0, num_neurons + 1]); % Set y-axis limits
grid on;
hold off;

%% Multiple Trials Stacked Raster Plots
angle_num = 3; % Select reaching angle
num_trials = 10; % Number of trials to include

figure;
hold on;

for trial_num = 1:num_trials
    spike_data = trial(trial_num, angle_num).spikes; % Get spike data (98 x 672)
    [num_neurons, num_time_bins] = size(spike_data);

    % Plot spikes for each neuron in this trial
    for neuron_idx = 1:num_neurons
        spike_times = find(spike_data(neuron_idx, :) > 0); % Find spike times

        % Offset each trial so they are stacked
        for i = 1:length(spike_times)
            plot([spike_times(i), spike_times(i)], [neuron_idx + (trial_num - 1) * num_neurons - 0.4, ...
                                                    neuron_idx + (trial_num - 1) * num_neurons + 0.4], 'k', 'LineWidth', 1);
        end
    end
end

xlabel('Time (bins)');
ylabel('Neuron Index (Stacked Trials)');
title(['Stacked Raster Plot for ', num2str(num_trials), ' Trials, Angle ', num2str(angle_num)]);
xlim([0, num_time_bins]);
ylim([0, num_trials * num_neurons + 1]);
grid on;
hold off;

%% One neuron across all trials raster plot
angle_num = 3; % Select reaching angle
neuron_idx = 10; % Pick a specific neuron
num_trials = 10; % Number of trials

figure;
hold on;

for trial_num = 1:num_trials
    spike_data = trial(trial_num, angle_num).spikes(neuron_idx, :); % Get spikes for one neuron
    spike_times = find(spike_data > 0); % Find spike times

    % Plot spikes (each row is a different trial)
    for i = 1:length(spike_times)
        plot([spike_times(i), spike_times(i)], [trial_num - 0.4, trial_num + 0.4], 'k', 'LineWidth', 1);
    end
end

xlabel('Time (bins)');
ylabel('Trial Number');
title(['Single-Neuron Raster Plot - Neuron ', num2str(neuron_idx), ', Angle ', num2str(angle_num)]);
xlim([0, num_time_bins]);
ylim([0, num_trials + 1]);
grid on;
hold off;


%% PSTH
% Parameters
neuron_ids = [10, 20, 30];  % Neurons to analyze
angle_num = 3;              % Reaching angle (1-8)
num_trials = 10;            % Number of trials to include
bin_size = 20;              % Bin size in ms (e.g., 20 ms bins)
smoothing_sigma = 10;       % Smoothing window in ms (Gaussian kernel)

% Get time bins
time_bins = 0:bin_size:size(trial(1, angle_num).spikes, 2); % Time bins in steps

% Initialize matrix to store spike counts
psth_matrix = zeros(length(neuron_ids), length(time_bins)-1);

for n = 1:length(neuron_ids)
    neuron_idx = neuron_ids(n);
    spike_counts = zeros(num_trials, length(time_bins)-1);
    
    for trial_num = 1:num_trials
        % Get spike data for this neuron and trial
        spike_data = trial(trial_num, angle_num).spikes(neuron_idx, :);
        
        % Bin spike times
        spike_counts(trial_num, :) = histcounts(find(spike_data > 0), time_bins);
    end
    
    % Average across trials to get PSTH
    psth_matrix(n, :) = mean(spike_counts, 1) / (bin_size / 1000); % Convert to spikes/sec
end

% Create a Gaussian kernel for smoothing
window_size = round(5 * smoothing_sigma / bin_size); % Convert sigma to bin units
gauss_filter = fspecial('gaussian', [1, 2*window_size+1], smoothing_sigma / bin_size);

% Apply smoothing
smoothed_psth = conv2(psth_matrix, gauss_filter, 'same');

figure;
hold on;
colors = lines(length(neuron_ids)); % Generate distinct colors

for n = 1:length(neuron_ids)
    plot(time_bins(1:end-1), psth_matrix(n, :), 'Color', colors(n, :), 'LineWidth', 2);
    plot(time_bins(1:end-1), smoothed_psth(n, :), 'Color', colors(n, :), 'LineWidth', 2, 'LineStyle', '--');
end

xlabel('Time (bins)');
ylabel('Firing Rate (spikes/sec)');
title('Peri-Stimulus Time Histogram (PSTH)');
legend(arrayfun(@(x) ['Neuron ', num2str(x)], neuron_ids, 'UniformOutput', false));
grid on;
hold off;

%% Hand Position across trials
% Parameters
angle_num = 3;    % Select reaching angle (1-8)
num_trials = 10;  % Number of trials to plot
time_step = 1;    % Downsampling factor for clarity

% Define colors for different trials
colors = lines(num_trials);

% Create figure
figure;
hold on;

for trial_num = 1:num_trials
    % Extract hand position data for the chosen trial
    hand_pos = trial(trial_num, angle_num).handPos; % 3 x TimeBins (X, Y, Z)
    
    % Downsample time points for clarity
    x_pos = hand_pos(1, 1:time_step:end);
    y_pos = hand_pos(2, 1:time_step:end);
    
    % Plot trajectory for each trial
    plot(x_pos, y_pos, 'Color', colors(trial_num, :), 'LineWidth', 2);
end

% Formatting
xlabel('X Position');
ylabel('Y Position');
title(['Hand Position Across ', num2str(num_trials), ' Trials (Angle ', num2str(angle_num), ')']);
legend(arrayfun(@(x) ['Trial ', num2str(x)], 1:num_trials, 'UniformOutput', false));
grid on;
hold off;

%%
trial_num = 1; % Select trial number
angle_num = 3; % Select angle (1-8)

handPos = trial(trial_num, angle_num).handPos; % Get hand position (3×672 matrix)

figure;
plot(1:length(handPos(2,:)), handPos(2,:), 'LineWidth', 2); % X vs Y movement
xlabel('Time');
ylabel('Y Position');
title(['Reaching Angle ', num2str(angle_num)]);
grid on;

figure;
plot(1:length(handPos(1,:)), handPos(1,:), 'LineWidth', 2); % X vs Y movement
xlabel('Time');
ylabel('Y Position');
title(['Reaching Angle ', num2str(angle_num)]);
grid on;

%% Tuning map
% Parameters
neuron_ids = [10, 20, 30];  % Select neurons to analyze
num_angles = 8;             % Number of movement directions
num_trials = 10;            % Number of trials per direction
time_window = [100, 500];   % Time window (in bins) for computing firing rate

% Initialize storage for firing rates and standard deviations
firing_rates = zeros(length(neuron_ids), num_angles);
firing_std = zeros(length(neuron_ids), num_angles);

% Compute firing rates and standard deviations
for n = 1:length(neuron_ids)
    neuron_idx = neuron_ids(n);
    
    for angle_num = 1:num_angles
        spike_counts = zeros(num_trials, 1);
        
        for trial_num = 1:num_trials
            % Extract spike data for this neuron in the given trial & angle
            spike_data = trial(trial_num, angle_num).spikes(neuron_idx, :);
            
            % Compute total spikes within time window
            spike_counts(trial_num) = sum(spike_data(time_window(1):time_window(2)));
        end
        
        % Compute mean and standard deviation of firing rate
        firing_rates(n, angle_num) = mean(spike_counts) / ((time_window(2) - time_window(1)) / 1000); % Convert to spikes/sec
        firing_std(n, angle_num) = std(spike_counts) / ((time_window(2) - time_window(1)) / 1000); % Convert to spikes/sec
    end
end

% Define movement directions (assume 8 evenly spaced directions)
directions = linspace(0, 360, num_angles + 1);
directions = directions(1:end-1); % Remove duplicate 360-degree point

% Plot tuning curves with error bars
figure;
hold on;
colors = lines(length(neuron_ids)); % Generate distinct colors

for n = 1:length(neuron_ids)
    % Plot error bars (mean ± std)
    errorbar(directions, firing_rates(n, :), firing_std(n, :), 'o', 'Color', colors(n, :), 'LineWidth', 2, 'MarkerSize', 8);
    
    % Fit a smooth curve using interpolation
    smooth_dir = linspace(0, 360, 100);
    smooth_firing = interp1(directions, firing_rates(n, :), smooth_dir, 'pchip'); % Smooth interpolation
    plot(smooth_dir, smooth_firing, '-', 'Color', colors(n, :), 'LineWidth', 2, 'LineStyle', '--');
    plot(directions, firing_rates(n, :), '-', 'Color', colors(n, :), 'LineWidth', 2);
end

% Formatting
xlabel('Movement Direction (degrees)');
ylabel('Firing Rate (spikes/sec)');
title('Tuning Curves with Error Bars');
legend(arrayfun(@(x) ['Neuron ', num2str(x)], neuron_ids, 'UniformOutput', false));
xlim([0, 360]);
grid on;
hold off;

%% Mean Population Activity Over Time
time_bins = 0:bin_size:size(trial(1,1).spikes, 2);
pop_activity = zeros(length(time_bins)-1, num_neurons, num_trials);

for n = 1:num_neurons
    for trial_num = 1:num_trials
        spike_data = trial(trial_num, 3).spikes(n, :); % Pick angle 3 as an example
        pop_activity(:, n, trial_num) = histcounts(find(spike_data > 0), time_bins);
    end
end

% Compute mean firing rate across neurons and trials
mean_pop_activity = mean(mean(pop_activity, 3), 2) / (bin_size / 1000);

% Plot
figure;
plot(time_bins(1:end-1), mean_pop_activity, 'k', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Mean Population Firing Rate (spikes/sec)');
title('Mean Population Activity Over Time');
grid on;

%%
num_angles = 8;  % Number of movement directions
num_time_bins = size(trial(1,1).spikes, 2); % Total time bins (e.g., 576)
window_size = 50; % Smoothing window (adjustable)

% Initialize array to store main angles over time
main_angles = zeros(1, num_time_bins);

% Loop over time bins to compute the dominant movement angle
for t = 1:num_time_bins
    angle_firing_rates = zeros(1, num_angles); % Store mean firing rate per angle
    
    for angle_num = 1:num_angles
        firing_rates = []; % Store firing rates for this angle
        
        for trial_num = 1:num_trials
            % Get spike data for all neurons at this angle & trial
            spike_data = trial(trial_num, angle_num).spikes;
            
            % Fix indexing to prevent out-of-bounds errors
            start_idx = max(1, min(t - window_size, num_time_bins)); % Ensure ≥ 1
            end_idx = min(576, max(t + window_size, 1));   % Ensure ≤ num_time_bins
            
            % Compute mean firing rate within the adjusted time window
            valid_data = spike_data(:, start_idx:end_idx);
            
            % Ensure the data is not empty before computing mean
            if ~isempty(valid_data)
                firing_rates = [firing_rates; mean(valid_data, 2)];
            end
        end
        
        % Compute mean activity across trials for this angle
        if ~isempty(firing_rates)
            angle_firing_rates(angle_num) = mean(firing_rates, 'all');
        end
    end

    % Find the movement direction with highest activity at time t
    [~, main_angle_idx] = max(angle_firing_rates);
    main_angles(t) = (main_angle_idx - 1) * (360 / num_angles); % Convert to degrees
end

% Plot main angle over time
figure;
plot(1:num_time_bins, main_angles, 'k', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Main Movement Direction (Degrees)');
title('Main Movement Direction Over Time');
ylim([0, 360]);
yticks(0:45:360);
grid on;

