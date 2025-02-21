function [train_data, val_data, test_data] = split_data(pop_activity, train_ratio, val_ratio, test_ratio)
    % Inputs:
    %   pop_activity - Neural population activity matrix (Neurons, Trials*Angles)
    %   train_ratio - e.g., 0.7
    %   val_ratio - e.g., 0.15
    %   test_ratio - e.g., 0.15
    % Outputs:
    %   train_data, val_data, test_data - Split datasets
    
    assert(train_ratio + val_ratio + test_ratio == 1, 'Ratios must sum to 1')

    total_trials = size(pop_activity, 2);
    rand_indices = randperm(total_trials);
    fprintf('Checking seed is constant: %d\n', rand_indices(1));
    train_size = round(train_ratio * total_trials);
    val_size = round(val_ratio * total_trials);

    % Assign trials
    train_idx = rand_indices(1:train_size);
    val_idx = rand_indices(train_size+1:train_size+val_size);
    test_idx = rand_indices(train_size+val_size+1:end);

    % Split data
    train_data = pop_activity(:, train_idx);
    val_data = pop_activity(:, val_idx);
    test_data = pop_activity(:, test_idx);
end
