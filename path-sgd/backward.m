function layer = backward(layer, X, Y)
% This function returns the layer including the gradient with respect to current weights
%
% Inputs
% layer:    a depth-dimensional array such that layer{i} is the i-th layer of netural network
% X:        an N*D dimensional matrix that is the input to the network
% Y:        an N dimensional array that is the labels for each data point
%
% Output
% layer     :   the updated neural network including the gradient with respect to current weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

depth= length(layer);

% gradient with respect to the input of activation function on the top layer
layer{depth}.delta = exp( layer{depth}.logscore ) - layer{depth}.target;

% backpropagation
for i=depth:-1:2
%     layer{i}.gradient = gpuArray(zeros(size(layer{i}.W)));
%     layer{i}.gradientTheta = gpuArray(zeros(1,length(layer{i}.theta)));
    
    layer{i}.gradient = zeros(size(layer{i}.W));
    layer{i}.gradientTheta = zeros(1,length(layer{i}.theta));

    layer{i}.gradient(layer{i-1}.nonzero,layer{i}.nonzero) = layer{i-1}.act' * layer{i}.delta;
    layer{i}.gradientTheta(layer{i}.nonzero) = -sum( layer{i}.delta, 1);
    layer{i-1}.delta = (layer{i}.delta * layer{i}.W(layer{i-1}.nonzero,layer{i}.nonzero)') .* ( layer{i-1}.act > 0);
end

end

