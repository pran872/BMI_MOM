function predicted_positions = knn_regression(train_data, train_labels, test_data, k)
    % Inputs:
    %   train_data    - (num_neurons x num_train_trials) Neural activity (features)
    %   train_labels  - (num_train_trials x 1) Actual X or Y positions
    %   test_data     - (num_neurons x num_test_trials) Neural activity (to predict)
    %   k             - Number of neighbors
    % Output:
    %   predicted_positions - (num_test_trials x 1) Predicted X or Y positions

    num_test_trials = size(test_data, 2);
    predicted_positions = zeros(num_test_trials, 1);

    for i = 1:num_test_trials
        % Compute Euclidean distances from test trial to all train trials
        distances = sum((train_data - test_data(:, i)).^2, 1); % Squared Euclidean distance
        [~, sorted_indices] = sort(distances); % Sort distances

        % Get the k nearest neighbors
        nearest_neighbors = sorted_indices(1:k);

        % Predict position as the mean of k-nearest neighbors' positions
        predicted_positions(i) = mean(train_labels(nearest_neighbors));
    end
end
