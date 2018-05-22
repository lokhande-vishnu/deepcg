% SETTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reset(gpuDevice(1))
poolobj = gcp('nocreate');
delete(poolobj);
param = init();
param.norm = 2;
param.maxIter = 20000;   % the number of updates. 10 iterations took 125 seconds approximately including save.
param.mission = 2; %0-nothing, 1-tunning, 2-collect all train values, 3-path norm,
% Tunning parameters
param.dropout = 0;      % The amount of dropout. zero means no dropout
     % stepsize
param.lambda = 10^6;     % 
param.eta = 0.01;
% DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Tunning parameters
% parpool(close)
parpool(2)
tic
parfor dataset=1:2
    mkdir('baseline_models/',num2str(dataset));
    [X_train, Y_train, X_test, Y_test] = load_datset(dataset, param);
    param_local = param;
    % TRAINING
    param_local.X_test = gpuArray(X_test);
    param_local.Y_test = gpuArray(Y_test);
    [layer, train_values] = main(gpuArray(X_train), gpuArray(Y_train), param_local);
    % EVALUATION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    prediction_on_test = forward(layer, X_test, Y_test, 0);
    prediction_on_test = gather(prediction_on_test{4}.classerr)*100/size(X_test,1);
    % STORE
    filename = sprintf('baseline_models/%d/%d_%d.mat', dataset, dataset, param.norm);
    parsave_norm(filename, dataset, layer, param_local, prediction_on_test, train_values);
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    
