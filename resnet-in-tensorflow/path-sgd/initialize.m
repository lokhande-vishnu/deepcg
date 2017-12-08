function layer = initialize( D, H, labels, param)
% This function gets the size of network, labels and parameters and create a neural network based on them
%
% Input
% D:          the dimension of the input data
% H:          an array such that H(i) is the number of units in hidden layer i.
% labels:     the set of class labels
% param:      a cell that includes parameters of the method
%
% Output
% layer:      a cell array of length depth such that layer{i} is the i-th layer of the neural net and layer{i}.W is the incomming  weights to the i-th layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


K = [D H length(labels)];                                         % K(i) is the number of units in layer i where the first layer is the input layer.
depth = length(K);

for i=2:depth
  layer{i}.dropout = param.dropout;
  layer{i}.theta = zeros(1,K(i));                                 
  layer{i}.W = (1/sqrt(K(i-1))) * randn(K(i-1),K(i));          % Balanced initialization
end

layer{1}.dropout = 0;
layer{depth}.dropout = 0;
layer{depth}.labels = labels;

if ( ~param.balanced )
  layer = makeUnbalanced( layer, param );                         % Make the weights unbalanced
end

end



% For each hidden layer with h hidden units, we pick h/4 hidden units randomly with replacement. Then for each unit, we multiply its incomming edges and divide its outgoing edges by 10c where c is chosen randomly from log-normal distribution.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function layer = makeUnbalanced( layer, param )

  for i=2:length(layer)-1
    for j=1:(size(layer{i}.W)/4)
      ind = randi(size(layer{i}.W,2));                            % picking a hidden unit randomly
	    scale = 10 * exp( randn( 1, 1 ) );                          % generating the scaling factor randomly
	    
      layer{i}.W(:,ind) = scale * layer{i}.W(:,ind);
    	layer{i}.theta(ind) = scale * layer{i}.theta(ind);
    	layer{i+1}.W(ind,:) = (1/scale) * layer{i+1}.W(ind,:);
    end
  end
end
