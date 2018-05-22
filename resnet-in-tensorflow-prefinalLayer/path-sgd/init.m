function param = init()
    param.batchsize = 100;
    param.balanced = true;  % If balanced = false then the initial weights are unbalanced. 
                        % See the paper for more details.
    param.lambda_set = 6:8; % classification
    param.eta_set = 4:6;
    param.dropout_set = [0];
    
    %%%% CONSTANT %%%% 
    % Dataset
    param.DS_MNIST = 1;
    param.DS_CIFAR10 = 2;
    param.DS_CIFAR100 = 3;
    param.DS_SHVN = 4;
    % NORMS
    param.NORMAL_PATH_NORM = 1;
    param.SGD = 2;
    param.CONSTRAINT_PATH_NORM = 3;
    param.NUCLEAR_NORM = 4;
    param.FRO_NORM = 5;
    param.MIXED_NUCLEAR_FRO = 6;
end
