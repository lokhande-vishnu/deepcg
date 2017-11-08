param = init();
mat_folder = 'models/';
plot_folder = 'plots/';
csv_folder = 'stats/';
if ~exist(plot_folder, 'dir')
    mkdir(plot_folder);
end
if ~exist(csv_folder, 'dir')
    mkdir(csv_folder);
end

param.lambda_set = [6 7 8];
param.eta_set = [4 5 6 ];
param.dropout_set = [0];
stat_size = length(param.lambda_set) * length(param.dropout_set) * length(param.eta_set);
for dataset = 3
    ds_plot_folder = sprintf('%s%d/', plot_folder, dataset);
    if ~exist(ds_plot_folder, 'dir')
        mkdir(ds_plot_folder);
    end
    stats = zeros(stat_size, 5);
    count = 0;    
    for dropout = param.dropout_set        
        for log_eta = param.eta_set
            for log_lambda = param.lambda_set
                filename = sprintf('%d/%d_lam%d_drop%.1f_eta%d', ...
                    dataset, dataset, log_lambda, dropout, -log_eta);
                mat_name = strcat(strcat(mat_folder, filename), '.mat');
                plot_name = strcat(strcat(plot_folder, filename), '.jpg');
                [test_error, train_error] = process_mat(mat_name, plot_name);
                count = count + 1;
                stats(count, :) = [test_error, train_error, dropout, log_eta,log_lambda];
            end
        end
    end
    % savs as csv
%     sorted_stats = sortrows(stats, 1, 'ascend');
    csv_file = strcat(csv_folder, sprintf('ds_%d.csv', dataset));
    csvwrite(csv_file, stats);
    % print out
    fprintf('Finish dataset %d Sorted by test error:\n', dataset);
    display(stats);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [test_error, final_train_error] = process_mat(mat_name, plot_name)
    data = load(mat_name);
    test_error = gather(data.prediction_on_test);
    train_errors = gather(data.training_values(:, 1));
    final_train_error = train_errors(end);
    
    lag = 100;
    X = 1:size(data.training_values, 1);
    mavg_train = tsmovavg(train_errors,'s',lag,1);
    % save fig
    f = figure('Visible','off'); 
    set(f,'Visible','off','CreateFcn','set(f,''Visible'',''on'')')
    plot(X, mavg_train, 'b');
    xlabel('Iteration#');
    ylabel('Training error');
    saveas(f, plot_name)
    % imwrite
    close
end
