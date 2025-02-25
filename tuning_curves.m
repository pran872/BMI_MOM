% Define the reaching angles (in degrees)
angles_deg = [30, 70, 110, 150, 190, 230, 310, 350];
num_angles = length(angles_deg);


num_trials = size(trial, 1);               % should be 100 trials per angle
num_neurons = size(trial(1,1).spikes, 1);    % should be 98 neurons


meanRates = zeros(num_neurons, num_angles);
stdRates  = zeros(num_neurons, num_angles);

% Loop over each reaching angle and compute the time-averaged firing rates
for k = 1:num_angles
    rates = zeros(num_trials, num_neurons);  
    for n = 1:num_trials
        rates(n, :) = mean(trial(n, k).spikes, 2)';  
    end
    % Mean and std
    meanRates(:, k) = mean(rates, 1)';
    stdRates(:, k)  = std(rates, 0, 1)';
end

% Same y-axis
global_y_min = min(meanRates(:));
global_y_max = max(meanRates(:));
margin = 0.1 * (global_y_max - global_y_min);
y_limits = [global_y_min - margin, global_y_max + margin];


figure;
num_rows = 7;
num_cols = 14;

for i = 1:num_neurons
    subplot(num_rows, num_cols, i);
    % Plot tuning curve using error bars
    errorbar(angles_deg, meanRates(i,:), stdRates(i,:), 'o-', 'LineWidth', 1.5);
    xlim([min(angles_deg)-10, max(angles_deg)+10]); 
    % ylim(y_limits);    % toggle this to change the y-axis limits                         
    title(sprintf('Neuron %d', i), 'FontSize', 8);
    if i > (num_rows-1)*num_cols
        xlabel('Angle (°)', 'FontSize', 8);
    end
    if mod(i-1, num_cols) == 0
        ylabel('Firing Rate', 'FontSize', 8);
    end
    grid on;
end

sgtitle('Tuning Curves for Each Neuron');
