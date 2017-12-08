function layer = forward(layer, X, Y, dropout)
% This function calculates the output of all layers in the neural network
%
% Inputs
% layer:        a depth-dimensional array such that layer{i} is the i-th layer of netural network
% X:            an N*D dimensional matrix that is the input to the network
% Y:            An N dimensional vector of corresponding labels
% dropout:      The amount of dropout. zero means no dropout 
% Output
% layer:        the updated neural network including the output values of all layers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

depth = length(layer);

layer{1}.act = X;
layer{1}.nonzero = 1:size(X,2);
for i=2:depth
  % Dropout
  layer{i}.nonzero = find( rand( size(layer{i}.W,2),1) > layer{i}.dropout );
  % Calculates the activations
  layer{i}.act = max( 0, bsxfun( @minus, (1/(1-dropout)) * layer{i-1}.act * layer{i}.W(layer{i-1}.nonzero,layer{i}.nonzero), layer{i}.theta(layer{i}.nonzero) ) );
end

[~,I] = max( layer{depth}.act, [], 2 );
layer{depth}.target = ( repmat( Y, 1,length(layer{depth}.labels) ) == repmat( layer{depth}.labels', length(Y), 1) ) ;
layer{depth}.logscore = bsxfun(@minus, layer{depth}.act, logsumexp( layer{depth}.act, 2 ) );                              % Logarithm of scores of each class in softmax layer
layer{depth}.objective = - sum( layer{depth}.target .* layer{depth}.logscore, 2);                                         % The objective value
layer{depth}.classerr = length(I) - sum( layer{depth}.target( sub2ind( size(layer{depth}.target), 1:length(I), I') ) );   % The classification error

end


% calculates Y=log(sum(exp(X),varargin{:})) avoiding numerical errors as much as possible
function Y = logsumexp(X, d)
maxX = max(X, [], d);
Xshifted = bsxfun(@minus, X, maxX);
Y = bsxfun(@plus, log( sum( exp( Xshifted), d) ), maxX);
end
