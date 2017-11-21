function evaluate(X_test, Y_test, layer, batchsize)
    
    % TRAINING ERROR
    training_error = gather(layer{4}.classerr/batchsize);
    loss_function = gather(layer{4}.objective);
    loss_function = mean(loss_function);

    % EVALUATION
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    prediction_on_test = forward(layer, X_test, Y_test, 0);
    prediction_on_test = gather(prediction_on_test{4}.classerr)*100/size(X_test,1);

    fprintf('Pred-error = %.5f\n', prediction_on_test);
    fprintf('Train-error = %.5f\n', training_error);
    fprintf('Loss = %.5f\n', loss_function);
end