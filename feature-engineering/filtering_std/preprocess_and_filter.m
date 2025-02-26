function filtered_neurons = preprocess_and_filter(trial, mean_percentile, max_percentile, plot_flag)
% Preprocesses neural data, computes firing rates, and filters neurons based on variability
% 
% Inputs:
% - trial: The struct array of neural spike data
% - mean_threshold: Threshold for mean variability
% - max_threshold: Threshold for peak variability
% - plot_flag: Boolean (true/false) to enable/disable plotting
%
% Output:
% - filtered_neurons: [98x1] binary vector (1 = keep, 0 = discard)

    num_neurons = size(trial(1,1).spikes, 1);
    num_trials = size(trial, 1); 
    num_angles = size(trial, 2);
    bin_size = 50; % 50ms bins

    % Determine the longest trial duration
    max_time_length = 0;
    for k = 1:num_angles
        for n = 1:num_trials
            max_time_length = max(max_time_length, size(trial(n, k).spikes, 2));
        end
    end

    num_bins = floor(max_time_length / bin_size);
    mean_firing_rates = zeros(num_neurons, num_bins, num_angles);

    for k = 1:num_angles
        for i = 1:num_neurons
            spikes_all_trials = zeros(num_trials, max_time_length);
            
            for n = 1:num_trials
                spike_train = trial(n, k).spikes(i, :); % Extract spike train
                T = length(spike_train); % Actual trial duration
                
                % Zero-pad the spike train to match max_time_length
                spikes_all_trials(n, 1:T) = spike_train;
            end
            
            % Compute mean firing rate for each time bin
            for b = 1:num_bins
                start_idx = (b-1) * bin_size + 1;
                end_idx = min(start_idx + bin_size - 1, max_time_length); % Adjust for last bin
                mean_firing_rates(i, b, k) = mean(mean(spikes_all_trials(:, start_idx:end_idx), 2), 'omitnan');
            end
        end
    end

    % Compute standard deviation across angles for each neuron and time bin
    std_across_angles = std(mean_firing_rates, 0, 3); 

    % Call filtering function
    filtered_neurons = filter_neurons_by_stdev(std_across_angles, mean_percentile, max_percentile, plot_flag);

    % Plot the heatmap if plot_flag is enabled
    if plot_flag
        figure;
        imagesc(std_across_angles);
        colorbar;
        xlabel('Time Bins (50ms)');
        ylabel('Neuron Number');
        title('Neural Firing Rate Variability Across Angles');
        colormap hot;
    end
end
