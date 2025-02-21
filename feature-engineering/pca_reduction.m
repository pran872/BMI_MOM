function [reduced_activity] = pca_reduction(pop_activity, visualise, angles, custom_title_suffix)
    % Inputs:
    %   pop_activity - Neural population activity matrix (Neurons, Trials*Angles)
    %                - Can be train, val, or test data
    %   visualise - Boolean to visualise the PCA
    %   angles - Angles of the data; this is required if visualise is true
    %   custom_title_suffix - Custom suffix for the title of the PCA plots e.g., " - 0.8 split"

    % Outputs:
    %   reduced_activity - Reduced activity matrix (Neurons, PCs)

    if nargin == 1  
        visualise = false;
    elseif nargin == 2 
        assert(false, 'Angles must be provided if visualise is true')
        angles = [];
    elseif nargin == 3 
        custom_title_suffix = '';
    end

    [coeff, score, ~, ~, explained] = pca(pop_activity');
    cumulative_variance = cumsum(explained);
    num_pcs = find(cumulative_variance >= 95, 1); % Explains 95% of variance
    fprintf("Reduced dimensions from %d to %d\n", length(cumulative_variance), num_pcs);

    reduced_activity = score(:, 1:num_pcs); 

    if visualise
        fprintf('First PC explains %.2f%% variance\n', explained(1));
        fprintf('Second PC explains %.2f%% variance\n', explained(2));
        fprintf('Third PC explains %.2f%% variance\n', explained(3));

        figure;
        plot(cumulative_variance, '-o');
        xlabel('Number of Principal Components');
        ylabel('Explained Variance (%)');
        title(sprintf('Variance Explained by PCA %s', custom_title_suffix));
        grid on;
        % exportgraphics(gcf, 'var_vs_pcs.png', 'Resolution', 300);

        unique_angles = unique(angles);
        num_angles = length(unique_angles);

        colors = parula(num_angles);

        figure; 
        hold on;
        for i = 1:num_angles
            idx = angles == unique_angles(i);
            scatter(score(idx, 1), score(idx, 2), 50, colors(i, :), 'filled');
        end
        xlabel('PC 1');
        ylabel('PC 2');
        title(sprintf('PCA Visualization of Neural Population Activity %s', custom_title_suffix));
        legend_labels = arrayfun(@(x) sprintf('%s°', num2str(x)), unique_angles, 'UniformOutput', false);
        legend(legend_labels, 'Location', 'BestOutside');

        grid on;
        % exportgraphics(gcf, 'og_pca.png', 'Resolution', 300);

        hold off;

    end

end  
        
