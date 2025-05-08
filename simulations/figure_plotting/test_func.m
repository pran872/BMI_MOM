% Test Script
% NOTE: FINAL MODELS - LDA AND LINEAR REGRESSION - DO NOT USE ANY TOOLBOXES BUT OTHER MODELS EXPLORED MAY USE
%       THE STATISTICS AND MACHINE LEARNING MATLAB TOOLBOX 
clc; clear all; close all;
teamName = '/Users/pranathipoojary/Imperial/BMI/BMI_MOM/simulations/figure_plotting'; %enter the name of the folder

tic;
RMSE = testFunction_for_students_MTb(teamName)
elapsedTime = toc;
fprintf('Elapsed time: %.4f seconds\n', elapsedTime);

%% Continuous Position Estimator Test Script
function RMSE = testFunction_for_students_MTb(teamName)
    
    clsMethod = 'lda'; % Options are ['lda', 'knn', 'logistic']
    regMethod = 'linear'; % Options are ['linear', 'knn', 'svr', 'rf']

    load monkeydata_training.mat
    rng(2013);
    ix = randperm(length(trial));
    
    addpath(teamName);
    trainingData = trial(ix(1:80),:);
    testData = trial(ix(81:end),:);
    
    fprintf('Testing the continuous position estimator...\n')
    regressors_mse = num2cell(zeros(1, 8));
    regressors_nums = num2cell(zeros(1, 8));
    meanSqError = 0;
    n_predictions = 0;  
    wrong_count = 0;
    
    figure(1)
    hold on
    axis square
    grid
    
    addpath(teamName);

    modelParameters = positionEstimatorTraining(trainingData, clsMethod, regMethod);
    pred_classes = [];
    true_classes = [];
    all_test_lda_feats = [];
    
    for tr=1:size(testData,1)
        fprintf('Decoding block %d/%d\n', tr, size(testData,1));

        for direc=randperm(8)
            decodedHandPos = [];
            times=320:20:size(testData(tr,direc).spikes,2);
            
            for t=times
                past_current_trial.trialId = testData(tr,direc).trialId;
                past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
                
                if nargout('positionEstimator') == 3
                    [decodedPosX, decodedPosY, newParameters] = positionEstimator(...
                        past_current_trial, ...
                        modelParameters);
                    modelParameters = newParameters;
                elseif nargout('positionEstimator') == 5
                    [decodedPosX, decodedPosY, newParameters, pred_classes, one_test_lda_proj] = positionEstimator(...
                        past_current_trial, ...
                        modelParameters, direc, pred_classes, clsMethod, regMethod);
                    modelParameters = newParameters;
                    if ~isempty(one_test_lda_proj)
                        all_test_lda_feats = [all_test_lda_feats; one_test_lda_proj]; 
                    end
                    if ~isempty(past_current_trial.decodedHandPos)
                        true_classes(end+1) = direc;

                        if ~isempty(pred_classes)
                            if direc ~= pred_classes(end)
                                wrong_count = wrong_count + 1;
                            end
                        end
                        
                    end
                end
                
                decodedPos = [decodedPosX; decodedPosY];
                decodedHandPos = [decodedHandPos decodedPos];

                regressors_mse{direc} = regressors_mse{direc} + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            end
            
            figure(1)
            plot(decodedHandPos(1,:), decodedHandPos(2,:), 'r')
            plot(testData(tr,direc).handPos(1,times), testData(tr,direc).handPos(2,times), 'b')
            
            regressors_nums{direc} = regressors_nums{direc} + length(times);
            n_predictions = n_predictions+length(times);
        end
        
    end

    RMSE = sqrt(meanSqError/n_predictions);

    % Metrics
    fprintf("RMSE: %.4f\n", RMSE);
    fprintf("Correct: %.4f\n", sum(pred_classes==true_classes))
    fprintf("Accuracy: %.4f\n", sum(pred_classes==true_classes)/length(pred_classes))
    fprintf("Wrong count: %d\n", wrong_count)

    for dir=1:8
        regressors_mse{dir} = sqrt(regressors_mse{dir}/regressors_nums{dir});
        fprintf("RMSE for regressor %d: %.4f\n", dir, regressors_mse{dir});
    end
    
    plot_hand_dirs_pretty()
    plot_confusion_matrix(true_classes, pred_classes)
    plot_lda_test(all_test_lda_feats, true_classes, pred_classes)
    plot_regressor_rmse_perfect_imperfect()

    plot_regressor_rmse(regressors_mse)
    % plot_confusion_matrix_heatmap(true_classes, pred_classes)
    
    rmpath(genpath(teamName))
end

%% Plot functions used in the report
function plot_hand_dirs_pretty()
    figure(1)
    xlabel('$X$ Hand Position (cm)', 'Interpreter', 'latex')
    ylabel('$Y$ Hand Position (cm)', 'Interpreter', 'latex')
    % set(gca, 'YTick', []); 
    % set(gca, 'YColor', 'none'); 
    legend('Predicted', 'Actual', 'Interpreter', 'latex')
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    grid off;
    set(gca, 'FontSize', 24);
end

function plot_confusion_matrix(true_classes, pred_classes)
    C_counts = confusionmat(true_classes, pred_classes);
    C = C_counts ./ sum(C_counts, 2) * 100;

    N = size(C,1);
    RGB = zeros(N, N, 3);

    blue = [0, 0.45, 0.74]; % MATLAB blue
    orange = [0.85, 0.33, 0.1]; % MATLAB orange

    %Asssigning colours
    for i = 1:N
        for j = 1:N
            if i == j
                RGB(i,j,:) = blue;
            else
                RGB(i,j,:) = orange;
            end
        end
    end

    %Plot squares
    figure;
    hold on;
    axis square;
    for i = 1:N
        for j = 1:N
            if C(i,j) ~= 0
                rectangle('Position', [j-1, N-i, 1, 1], ...
                    'FaceColor', squeeze(RGB(i,j,:)), ...
                    'EdgeColor', 'k');
            else
                rectangle('Position', [j-1, N-i, 1, 1], ...
                    'FaceColor', 'white', ...
                    'EdgeColor', 'k');
            end
        end
    end

    angles = {'30', '70', '110', '150', ...
          '190', '230', '310', '350'};
        
    set(gca, ...
        'XTick', 0.5:1:N-0.5, 'XTickLabel', angles, ...
        'YTick', 0.5:1:N-0.5, 'YTickLabel', fliplr(angles), ...
        'TickLabelInterpreter', 'latex', ...
        'FontSize', 20, ...
        'XTickLabelRotation', 0);

    xlabel('Predicted Angle ($^\circ$)', 'Interpreter', 'latex');
    ylabel('True Angle ($^\circ$)', 'Interpreter', 'latex');
    xlim([0 N]);
    ylim([0 N]);

    % Text in squares
    for i = 1:N
        for j = 1:N
            if C(i,j) ~= 0
                txt = sprintf('$%d\\%%$', round(C(i,j)));
                % txt = ['$' num2str(round(C(i,j))) '\%$'];
                text(j - 0.5, N - i + 0.5, txt, ...
                    'HorizontalAlignment', 'center', ...
                    'Interpreter', 'latex', ...
                    'FontSize', 16, ...
                    'Color', 'white');
            end
        end
    end
end

function plot_lda_test(all_test_lda_feats, true_classes, pred_classes)
    figure;
    g = gscatter(all_test_lda_feats(:,1), all_test_lda_feats(:,2), true_classes, ...
                 turbo(8), '.', 16);
    
    xlabel('LDA 1', 'Interpreter', 'latex');
    ylabel('LDA 2', 'Interpreter', 'latex');
    yl = ylabel('LDA 2', 'Interpreter', 'latex');
    yl.Position(1) = yl.Position(1) + 0.05;  % Reduce ylabel distance
    ylim(gca, [-0.7 0.8])
    xlim(gca, [-1 1.5])
    yticks(gca, [-0.5 0 0.5])
    % title('LDA Projection of Testing Features', 'Interpreter', 'latex');
    
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    set(gca, 'FontSize', 20);
    grid off;
    
    angleLabels = {'$30^\circ$', '$70^\circ$', '$110^\circ$', '$150^\circ$', ...
                   '$190^\circ$', '$230^\circ$', '$310^\circ$', '$350^\circ$'};
    
    %misclassified points
    misclassified_idx = find(pred_classes ~= true_classes);
    % disp(size(misclassified_idx))

    %misclassified points in red 'x'
    hold on;
    hMis = scatter(all_test_lda_feats(misclassified_idx,1), ...
                   all_test_lda_feats(misclassified_idx,2), ...
                   80, 'o', 'MarkerEdgeColor', 'r', 'LineWidth', 1.5);
    hold off;

    legend([g; hMis], [angleLabels, {'Misclassified'}], ...
       'Location', 'southeast', ...
       'Interpreter', 'latex', ...
       'Box', 'on', ...
       'FontSize', 14);

end

function plot_regressor_rmse_perfect_imperfect()
    % Hard coded values for LDA + linear regressor + centroid corr 
    % With perfect classification and imperfect classification

    angles = {'30', '70', '110', '150', ...
              '190', '230', '310', '350'};

    mse_perfect = [7.39 6.29 6.78 6.41 6.07 7.02 6.63 8.86];
    mse_imperfect = [7.39 6.29 15.37 14.78 6.07 7.02 6.63 8.86];
    mse_matrix = [mse_perfect(:), mse_imperfect(:)];

    figure;
    bar(mse_matrix, 'grouped');
    
    xticks(1:length(angles));
    xticklabels(angles);
    xlabel("Angle ($^\circ$)", 'Interpreter', 'latex', 'FontSize', 24);
    ylabel('RMSE (cm)', 'Interpreter', 'latex', 'FontSize', 24);
    legend({'Ideal classifier', 'Our classifier'}, ...
        'Interpreter', 'latex', 'Location', 'northeast', 'FontSize', 24);
    
    % title('Regression RMSE by Angle', 'Interpreter', 'latex');
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 24);
    box off;
end


%% Supplementary plot functions
function plot_confusion_matrix_heatmap(true_classes, pred_classes)
    C = confusionmat(true_classes, pred_classes);
    C = C ./ sum(C, 2); 
    C = C * 100;

    figure;
    imagesc(C);             
    colormap(turbo);       
    cb = colorbar;
    caxis([0 100]);
    cb.TickLabelInterpreter = 'latex';
    axis square;

    xticks(1:size(C,1));
    yticks(1:size(C,2));
    xlabel('Predicted Class', 'Interpreter', 'latex');
    ylabel('True Class', 'Interpreter', 'latex');
    ax = gca;
    ax.TickLabelInterpreter = 'latex';
    set(gca, 'FontSize', 20);
    hold on;

    for i = 1:size(C,1)
        for j = 1:size(C,2)
            if C(i,j) == 0
                rectangle('Position', [j-0.5, i-0.5, 1, 1], ...
                        'FaceColor', 'white', ...
                        'EdgeColor', 'k');
            end
        end
    end

    textStrings = num2str(C(:), '%.0f%%');
    textStrings = strtrim(cellstr(textStrings)); 
    textStrings(C(:) == 0) = {''};

    [x, y] = meshgrid(1:size(C,2), 1:size(C,1));  % Note: columns first, then rows
    text(x(:), y(:), textStrings(:), ...
        'HorizontalAlignment', 'center', ...
        'Color', 'white', ...
        'FontSize', 16, ...
        'Interpreter', 'latex');

end

function plot_confusionchart_percent(true_classes, pred_classes)
    figure;
    cm = confusionchart(true_classes, pred_classes, ...
        'Normalization', 'row-normalized', ...
        'Title', 'Confusion Matrix', ...
        'RowSummary', 'off', ...
        'ColumnSummary', 'off');

    cm.FontSize = 16;
    cm.FontName = 'Times';
    cm.XLabel = 'Predicted Class';
    cm.YLabel = 'True Class';

    cm.CellLabelFormat = '%.0f%';
end

function plot_regressor_rmse(regressors_mse)
    angles = {'30', '70', '110', '150', ...
              '190', '230', '310', '350'};

    mse_values = cellfun(@(x) x, regressors_mse);

    figure;
    bar(mse_values);
    xticks(1:length(angles));
    xticklabels(angles);
    ylabel('RMSE', 'Interpreter', 'latex');
    xlabel("Angle ($^\circ$)", 'Interpreter', 'latex')
    set(gca, 'TickLabelInterpreter', 'latex', 'FontSize', 14);
end
