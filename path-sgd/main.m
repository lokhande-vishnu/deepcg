function layer = main(X, Y)
% The main function. It gets the data points and labels as the input, trains a feedforward network on them and return the weights
%
% Input
% X:            An N * D matrix of data points
% Y:            An N dimensional vector of corresponding labels
%
% Output
% layer:        The learned neural network. layer{i}.W and layer{i}.threshold are the weights and thresholds of layer i (layer 1 is the input layer).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SETTING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.path_normalized = true; % path_normalized=false corresponds to SGD and path_normalized=true corresponds to Path-SGD
param.balanced = true;        % If balanced = false then the initial weights are unbalanced. See the paper for more details.
param.dropout = 0;            % The amount of dropout. zero means no dropout

% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H = [4000 4000];              % The structure of the network. H(i) is the number of hidden units in the i-th hidden layer
labels = unique(Y);           % The set of possible labels. The length of the vector is equal to the number of output units.
D = size(X, 2);               % The dimension of the data

layer0 = initialize(D, H, labels, param);   % Initializing the network with the given structure

% TRAINING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
layer = train( X, Y, layer0, param);        % training the given network on the given data points

end

