function [layer, train_values] = train(X, Y, layer, param)
    % This function optimizes the objective by SGD or Path-SGD (both with minibatches)
    %
    % Inputs
    % X:              An N * D matrix of data points
    % Y:              An N dimensional vector of corresponding labels
    % layer:          a depth-dimensional array such that layer{i} is the i-th layer of netural network
    % param:          a cell that includes parameters of the method
    %
    % Output
    % layer:          the trained neural network
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    eta = param.eta;          % stepsize
    lambda = param.lambda;
    maxIter = param.maxIter;   % the number of updates
    batchsize = param.batchsize;    % the size of the minibatch    
    
    fraction = 0.01; % this is just for printing
    printcount = 1;
    train_values = gpuArray(zeros(maxIter, 3));
    print_iters = round([fraction:fraction:1]*maxIter);
    depth = length(layer);
    for i=1:maxIter
%       if i == print_iters(printcount)
%           fprintf('Iteration number: %d/%d\n', i,maxIter);          
%           printcount = printcount + 1;
%       end
      ind = randperm( length(Y), batchsize);

      % Forward
      layer = forward( layer, X(ind,:), Y(ind), param.dropout);      
      % TRAINING ERROR
      if param.mission <= 2
          train_values(i, 1) = layer{4}.classerr/batchsize;
          train_values(i, 2) = sum(layer{depth}.objective);          
          prediction_on_test = forward(layer, param.X_test, param.Y_test, 0);
          train_values(i, 3) = prediction_on_test{4}.classerr *100;
      end
      % Backward
      layer = backward( layer, X(ind,:), Y(ind) );

      % Path-SGD
      switch param.norm
          case param.NORMAL_PATH_NORM % path norm
            gamma = path_scale( layer, depth );
            for j=2:depth
              layer{j}.W = layer{j}.W - eta * layer{j}.gradient ./ ( gamma{j-1}.in' * gamma{j}.out' );
              layer{j}.theta = layer{j}.theta - eta * layer{j}.gradientTheta ./ gamma{j}.out';
            end
          case param.SGD % sgd
            gamma = path_scale( layer, depth );
            for j=2:depth
              layer{j}.W = layer{j}.W - eta * layer{j}.gradient;
              layer{j}.theta = layer{j}.theta - eta * layer{j}.gradientTheta;
            end
            
          case param.CONSTRAINT_PATH_NORM % cgd - path norm 
            for j=2:depth 
              gamma = path_scale( layer, depth ); 
    %           path_norm = get_path_norm(layer, gamma)
              gamma_j = gamma{j-1}.in' * gamma{j}.out';
              bias_j = gamma{j}.out';
              wb_j = [gamma_j; bias_j]; % (m + 1) x n               
              
              gamma_j_root = wb_j.^0.5;
              gamma_j_zero = (gamma_j_root == 0);
              gamma_j_root_dezero = gamma_j_root + gamma_j_zero;

              wb_grad = [layer{j}.gradient; layer{j}.gradientTheta];
              norm_grad_j = norm(wb_grad, 'fro');
              normed_grad = wb_grad / norm_grad_j;

              c = normed_grad ./ gamma_j_root_dezero; %Maybe gamma = 0
              
              % get the path norm components from the biases
              temp_bias_path_norm = 0;
              for k_j = j+1:depth
                  temp_bias_path_norm = temp_bias_path_norm + layer{k_j}.theta.^2*gamma{k_j}.out;
              end
              
              lambda_j = lambda - temp_bias_path_norm;
%               if lambda_j < 0
%                   keyboard;
%               end
              s_j = -(lambda_j^0.5 ) * c;
              w_j = s_j(1:end-1, :);
              b_j = s_j(end, :);
              layer{j}.W = (1-eta)*layer{j}.W + eta * w_j;
              layer{j}.theta = (1-eta)*layer{j}.theta + eta * b_j;
              % for zero weight: don't update bias
              weight_zeros = (layer{j}.W == 0);
              layer{j}.W = layer{j}.W + (weight_zeros * 10^-7);
%               theta_zeros = (layer{j}.theta == 0); 
%               layer{j}.theta = layer{j}.theta + (theta_zeros * 10^-7);
%                       % Forward
%                        layer = forward( layer, X(ind,:), Y(ind),
%                        param.dropout);
%                       % Backward
%                        layer = backward( layer, X(ind,:), Y(ind) );
            end
          case param.NUCLEAR_NORM % cgd - nuclear norm
              for j = 2:depth
                wb_j = [layer{j}.gradient; layer{j}.gradientTheta]; % (m + 1) x n
                wb_j = -lambda * top_singular_vector(wb_j);
                w_j = wb_j(1:end-1, :);
                b_j = wb_j(end, :);
                layer{j}.W = (1-eta)*layer{j}.W + eta * w_j;
                layer{j}.theta = (1-eta)*layer{j}.theta + eta * b_j;
              end              
          case param.FRO_NORM % cgd - frobenius norm
              for j = 2:depth
                wb_j = [layer{j}.gradient; layer{j}.gradientTheta]; % (m + 1) x n
                wb_j = -lambda * wb_j ./ norm(wb_j, 'fro');
                w_j = wb_j(1:end-1, :);
                b_j = wb_j(end, :);
                layer{j}.W = (1-eta)*layer{j}.W + eta * w_j;
                layer{j}.theta = (1-eta)*layer{j}.theta + eta * b_j;
              end
          case param.MIXED_NUCLEAR_FRO % cgd - mixed - nuclear norm x fro
              % calculate st from gradient              
              frac_rank = 0.7;clear 
              for j = [2 4] % nuclear norm
                  lambda_n = min(size(layer{j}.gradient))*frac_rank;
                  wb_j = [layer{j}.gradient; layer{j}.gradientTheta]; % (m + 1) x n
                  wb_j = -lambda_n * top_singular_vector(wb_j);
                  w_j = wb_j(1:end-1, :);
                  b_j = wb_j(end, :);
                  layer{j}.W = (1-eta)*layer{j}.W + eta * w_j;
                  layer{j}.theta = (1-eta)*layer{j}.theta + eta * b_j;                  
              end
              j = 3;              
              wb_j = [layer{j}.gradient; layer{j}.gradientTheta]; % (m + 1) x n
              wb_j = lambda * wb_j ./ norm(wb_j, 'fro');
              w_j = wb_j(1:end-1, :);
              b_j = wb_j(end, :);
              layer{j}.W = (1-eta)*layer{j}.W + eta * w_j;
              layer{j}.theta = (1-eta)*layer{j}.theta + eta * b_j;
      end
      if param.mission == 3
        train_values(i, 3) = get_path_norm(layer, gamma);
      end
    end
end
% This function calculates the scaling factors for Path-SGD updates
function gamma = path_scale( layer, depth )
    gamma{1}.in = ones(1,size(layer{2}.W,1));
    gamma{depth}.out = ones(size(layer{depth}.W,2),1);
    for i=2:depth-1
        try
            gamma{i}.in = gamma{i-1}.in * abs(layer{i}.W).^2 + abs(layer{i}.theta) .^ 2; 
            gamma{depth-i+1}.out = abs(layer{depth-i+2}.W).^2 * gamma{depth-i+2}.out;
        catch
            keyboard;
        end
    end
end

function path_norm = get_path_norm(layer, gamma)
    min_layer = 4;
    path_norm_weights = gamma{min_layer-1}.in' * gamma{min_layer}.out';
    path_norm_weights = path_norm_weights .* (abs(layer{1, min_layer}.W).^2);
    path_norm_weights = sum(path_norm_weights(:)) ;

    path_norm_bias = layer{min_layer}.theta;
    path_norm_bias = sum(path_norm_bias(:).^2);
    
    path_norm = path_norm_weights + path_norm_bias;
end

function st = top_singular_vector(A)
    [U,S,V] = svds(A,1);
    ut = U(:, 1);
    vt = V(:, 1);
    st = ut * vt';
end

