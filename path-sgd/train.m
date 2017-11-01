function layer = train(X, Y, layer, param)
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

eta = 0.1;          % stepsize
batchSize = 100;    % the size of the minibatch
maxIter = 8000;   % the number of updates
fraction = 0.01; % this is just for printing
printcount = 1;

print_iters = round([fraction:fraction:1]*maxIter);

depth = length(layer);
for i=1:maxIter
%   i
  if i == print_iters(printcount)
      fprintf('Iteration number: %d/%d\n', i,maxIter);
      printcount = printcount + 1;
  end
  ind = randperm( length(Y), batchSize);

  % Forward
  layer = forward( layer, X(ind,:), Y(ind), param.dropout);
  
  % Backward
  layer = backward( layer, X(ind,:), Y(ind) );
  
  % Path-SGD
  if( param.path_normalized )
    gamma = path_scale( layer, depth );
    for j=2:depth
      layer{j}.W = layer{j}.W - eta * layer{j}.gradient ./ ( gamma{j-1}.in' * gamma{j}.out' );
      layer{j}.theta = layer{j}.theta - eta * layer{j}.gradientTheta ./ gamma{j}.out';
    end
  % SGD
  else
    for j=2:depth
      layer{j}.W = layer{j}.W - eta * layer{j}.gradient;
      layer{j}.theta = layer{j}.theta - eta * layer{j}.gradientTheta;
    end
  end

end

end

% This function calculates the scaling factors for Path-SGD updates
function gamma = path_scale( layer, depth )

gamma{1}.in = ones(1,size(layer{2}.W,1));
gamma{depth}.out = ones(size(layer{depth}.W,2),1);

for i=2:depth-1
  gamma{i}.in = gamma{i-1}.in * abs(layer{i}.W).^2 + abs(layer{i}.theta) .^ 2; 
  gamma{depth-i+1}.out = abs(layer{depth-i+2}.W).^2 * gamma{depth-i+2}.out;
end

end

