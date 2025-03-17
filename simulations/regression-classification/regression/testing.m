clear all; close all; clc;


% Load dataset
load('monkeydata_training.mat');

% Split data (80% train, 20% test)
rng(4048);  % Set random seed for reproducibility
ix = randperm(length(trial));
trainData = trial(ix(1:80), :);
testData = trial(ix(81:end), :);

% Train regressors for each angle
fprintf('Training Regressors...\n');
regressorModel = trainRegressor(trainData);

% Evaluate the regressors
fprintf('Evaluating Regressors...\n');
rmse = evaluateRegressor(testData, regressorModel);
