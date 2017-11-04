clear all, close all, clc

% SETTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.path_normalized = 3; % path_normalized=false corresponds to SGD and path_normalized=true corresponds to Path-SGD
param.balanced = false;        % If balanced = false then the initial weights are unbalanced. See the paper for more details.
param.dropout = 0;            % The amount of dropout. zero means no dropout
param.batchsize = 100;
param.eta = 0.0001;          % stepsize
param.lambda = 10000;
param.maxIter = 10000;   % the number of updates

param.opt_type = 3; % cgd-pathnorm=3
param.dataset = 3; % 2 for cifar-10
% DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if param.dataset == 1
    load mnist.mat
    X_train = training.images;
    Y_train = training.labels;
    X_test = test.images;
    Y_test = test.labels;
    % transform
    X_train = squeeze(reshape(X_train,size(X_train,1)*size(X_train,2),1,size(X_train,3)));
    X_train = X_train'; % N by D matrix
    X_test = squeeze(reshape(X_test,size(X_test,1)*size(X_test,2),1,size(X_test,3)));
    X_test = X_test'; % N by D matrix
    clear training test
elseif param.dataset == 2 % cifar 10
    folder = '../dataset/mat/cifar-10';
    X_train = [];
    Y_train = [];
    for i = 1:5 % 5 batches
        filename = strcat(strcat(folder, '/data_batch_'), num2str(i));
        batch = load(filename);
        X_train = [X_train; im2double(batch.data)];
        Y_train = [Y_train; im2double(batch.labels)];
    end
    batch = load(strcat(folder, '/test_batch.mat'));
    X_test = im2double(batch.data);
    Y_test = im2double(batch.labels);
else % cifar-100
    folder = '../dataset/mat/cifar-100';
    data_train = load(strcat(folder, '/train.mat'));
    X_train = im2double(data_train.data);
    Y_train = im2double(data_train.fine_labels);
    data_test = load(strcat(folder, '/test.mat'));
    X_test = im2double(data_test.data);
    Y_test = im2double(data_test.fine_labels);
end

% TRAINING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For switching between GPU and CPU, change line 23 in this file and lines
% 23 and 24 in backward.m file.
tic
layer = main(gpuArray(X_train), gpuArray(Y_train), param);
% layer = main(X_train, Y_train);
toc
training_error = gather(layer{4}.classerr/param.batchsize);
loss_function = gather(layer{4}.objective);
loss_function = mean(loss_function);

% EVALUATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prediction_on_test = forward(layer, X_test, Y_test, 0);
prediction_on_test = gather(prediction_on_test{4}.classerr)*100/size(X_test,1);

fprintf('Pred-error = %.5f\n', prediction_on_test);
fprintf('Train-error = %.5f\n', training_error);
fprintf('Loss = %.5f\n', loss_function);
