% SETTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
reset(gpuDevice(4))
poolobj = gcp('nocreate');
delete(poolobj);
param = init();
param.norm = param.CONSTRAINT_PATH_NORM;
param.maxIter = 8000;   % the number of updates. 10 iterations took 125 seconds approximately including save.


% DATASET
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataset = 4;
mkdir('models/',num2str(dataset));
[X_train, Y_train, X_test, Y_test] = load_datset(dataset, param);

% Tunning parameters
% parpool(close)
parpool(2)
tic
parfor log_lambda = param.lambda_set
    lambda = 10^log_lambda; 
    param_local = param;
    param_local.lambda = lambda;
    for dropout = param.dropout_set
        param_local.dropout = dropout;
        for log_eta = param.eta_set
            eta = 10^(-log_eta);
            % TRAINING
            param_local.eta = eta;
            [layer, train_values] = main(gpuArray(X_train), gpuArray(Y_train), param_local);
            % EVALUATION
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            layer = gather(layer);
            X_test_l = gather(X_test);
            Y_test_l = gather(Y_test);
            prediction_on_test = forward(layer, X_test_l, Y_test_l, 0);
            prediction_on_test = gather(prediction_on_test{4}.classerr)*100/size(X_test,1);
            % STORE
            filename = sprintf('models/%d/%d_lam%d_drop%.1f_eta%d.mat', ...
                dataset, dataset, log_lambda, dropout, -log_eta);
            parsave_eval(filename, dataset, log_lambda, dropout, log_eta, param_local,...
                prediction_on_test, train_values);
        end
    end    
end
toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



    
