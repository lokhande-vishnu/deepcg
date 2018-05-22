% SETTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reset(gpuDevice(1))
poolobj = gcp('nocreate');
delete(poolobj);
param = init();
param.maxIter = 8000;   % the number of updates. 10 iterations took 125 seconds approximately including save.
% Tunning parameters
param.dropout = 0;      % The amount of dropout. zero means no dropout
     % stepsize
param.lambda = 10^6;     % 
norm_set = 1:3;
param.eta_set = [0.1, 0.1, 10^-6];
param.mission = 3; %0-nothing, 1-tunning, 2-collect all train values, 3-path norm,

% DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tunning parameters
% parpool(close)
parpool(2)
tic

parfor dataset=3:4
    mkdir('norms/', num2str(dataset));
    [X_train, Y_train, X_test, Y_test] = load_datset(dataset, param);
    for n = norm_set
        param_local = param;
        param_local.norm = n;
        param_local.eta = param.eta_set(n);
        % TRAINING
        [layer, train_values] = main(gpuArray(X_train), gpuArray(Y_train), param_local);
        % EVALUATION
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        prediction_on_test = forward(layer, X_test, Y_test, 0);
        prediction_on_test = gather(prediction_on_test{4}.classerr)*100/size(X_test,1);
        % STORE
        filename = sprintf('norms/%d/%d_%d.mat', dataset, dataset, n);
        parsave_norm(filename, dataset, layer, param_local, prediction_on_test, train_values);
    end
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    
