clc; clear; close all; clear all;

load monkeydata_training.mat


teamName = 'simulations/wiener';
path = fullfile(pwd, teamName);

% add path to path 
addpath(path);

trainingData = trial(:,:);

bestModel = hyperparamClassifierSweep(trainingData);

