function filtered_neurons = filter_neurons_by_stdev(std_across_angles, mean_percentile, max_percentile, plot_flag)
% Filters neurons based on standard deviation thresholds across reaching angles
% 
% Inputs:
% - std_across_angles: [98x19] matrix of neural variability over time
% - mean_threshold: Threshold for mean variability
% - max_threshold: Threshold for peak variability
% - plot_flag: Boolean (true/false) to enable/disable plottings
%
% Output:
% - filtered_neurons: [98x1] binary vector (1 = keep, 0 = discard)

    neural_variability = std_across_angles; % (98x19 matrix)

    % Compute mean and max variability for each neuron
    mean_variability = mean(neural_variability, 2);
    max_variability = max(neural_variability, [], 2);
 % Compute absolute thresholds using percentiles
    mean_threshold = prctile(mean_variability, mean_percentile);
    max_threshold = prctile(max_variability, max_percentile);

    % Identify neurons that fail *both* thresholds (AND condition)
    discard_mask = (mean_variability < mean_threshold) & (max_variability < max_threshold);
    
    % Create binary vector: 1 = keep, 0 = discard
    filtered_neurons = ~discard_mask; % Invert to keep passing neurons

    % Display how many neurons are discarded
    disp(['Discarding ' num2str(sum(discard_mask)) ' neurons.']);

    % Plot results if plot_flag is true
    if plot_flag
        % Plot histogram of mean variability with threshold line
        figure;
        histogram(mean_variability, 20);
        xlabel('Mean Variability');
        ylabel('Neuron Count');
        title('Distribution of Neural Variability');
        xline(mean_threshold, 'r', 'LineWidth', 2, 'Label', ['Mean ' num2str(mean_percentile) '% Threshold']);

        % Plot histogram of max variability with threshold line
        figure;
        histogram(max_variability, 20);
        xlabel('Max Variability');
        ylabel('Neuron Count');
        title('Distribution of Peak Neural Variability');
        xline(max_threshold, 'r', 'LineWidth', 2, 'Label', ['Max ' num2str(max_percentile) '% Threshold']);

        % Plot retained neurons
        figure;
        imagesc(neural_variability(filtered_neurons == 1, :)); 
        colorbar;
        xlabel('Time Bins (50ms)');
        ylabel('Neuron Number (Retained)');
        title('Neural Variability After Filtering');
        colormap hot;
    end
end