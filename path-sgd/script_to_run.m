clear all, close all, clc

% SETTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param = init();
dataset = param.DS_CIFAR10;
param.norm = 3;
param.maxIter = 8000;   % the number of updates
% Tunning parameters
param.dropout = 0;      % The amount of dropout. zero means no dropout
param.eta = 0.1;     % stepsize
param.lambda = 10^6;     % 

% DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[X_train, Y_train, X_test, Y_test] = load_datset(dataset, param);

% TRAINING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For switching between GPU and CPU, change line 23 in this file and lines
% 23 and 24 in backward.m file.
tic
layer = main(gpuArray(X_train), gpuArray(Y_train), param);
% layer = main(X_train, Y_train);
toc

evaluate(X_test, Y_test, layer, param.batchsize)

