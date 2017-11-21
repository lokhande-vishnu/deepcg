function [X_train, Y_train, X_test, Y_test] = load_datset(dataset_id, param)
    switch dataset_id
        case param.DS_MNIST
            folder = '../dataset/mat/mnist/';
            load(strcat(folder, 'mnist.mat'))
            X_train = training.images;
            Y_train = training.labels;
            X_test = test.images;
            Y_test = test.labels;
            % transform
            X_train = squeeze(reshape(X_train,size(X_train,1)*size(X_train,2),1,size(X_train,3)));
            X_train = X_train'; % N by D matrix
            X_test = squeeze(reshape(X_test,size(X_test,1)*size(X_test,2),1,size(X_test,3)));
            X_test = X_test'; % N by D matrix
            clear training test
        case param.DS_CIFAR10 % cifar 10
            folder = '../dataset/mat/cifar-10/';
            X_train = [];
            Y_train = [];
            for i = 1:5 % 5 batches
                filename = strcat(strcat(folder, 'data_batch_'), num2str(i));
                batch = load(filename);
                X_train = [X_train; im2double(batch.data)];
                Y_train = [Y_train; im2double(batch.labels)];
            end
            batch = load(strcat(folder, 'test_batch.mat'));
            X_test = im2double(batch.data);
            Y_test = im2double(batch.labels);
        case param.DS_CIFAR100 % cifar-100
            folder = '../dataset/mat/cifar-100/';
            data_train = load(strcat(folder, 'train.mat'));
            X_train = im2double(data_train.data);
            Y_train = im2double(data_train.fine_labels);
            data_test = load(strcat(folder, 'test.mat'));
            X_test = im2double(data_test.data);
            Y_test = im2double(data_test.fine_labels);
        case param.DS_SHVN
            folder = '../dataset/mat/svhn/';
            data_train = load(strcat(folder, 'train_32x32.mat'));
            im_size = size(data_train.X, 1) * size(data_train.X, 2) * size(data_train.X, 3);
            X_train = squeeze(reshape(data_train.X, im_size, size(data_train.X, 4)));
            X_train = im2double(X_train'); % transpose
            Y_train = data_train.y;
            data_test = load(strcat(folder, 'test_32x32.mat'));
            X_test = squeeze(reshape(data_test.X, im_size, size(data_test.X, 4)));
            X_test = im2double(X_test');
            Y_test = data_test.y;
    end
end