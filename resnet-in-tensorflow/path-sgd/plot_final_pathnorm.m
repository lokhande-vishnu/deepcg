param = init();
mat_folder = 'models/';
final_plot_folder = 'final_plots/';
if ~exist(final_plot_folder, 'dir')
    mkdir(final_plot_folder);
end

for dataset = 3
    ds_plot_folder = sprintf('%s%d/', final_plot_folder, dataset);
    if ~exist(ds_plot_folder, 'dir')
        mkdir(ds_plot_folder);
    end
    
    ds_plot_folder_lambda = sprintf('%s/lambda/', ds_final_plot_folder);
    if ~exist(ds_plot_folder, 'dir')
        mkdir(ds_plot_folder);
    end
    % For lambda
    for log_lambda = param.lambda_set
        for dropout = param.dropout_set
            filename = sprintf('%d/%d_lam%d_drop%.1f', ...
                dataset, dataset, log_lambda, dropout);
            stats = [];
            for log_eta = param.eta_set
                mat_name = sprintf('%s%s_eta-%d.mat', mat_folder, filename, log_eta);
                data = load(mat_name);
                errors = gather(data.training_values(:, 1));
                stats(:, end + 1) = errors;
            end
            save_name = sprintf('%s%s.jpg', ds_plot_folder_lambda, filename);
            plot_name = sprintf('DS %d Lambda %d Dropout %.1f', dataset, log_lambda, dropout);
            plot_curve(plot_name, save_name, stats, 1, param.eta_set);
        end
    end
    
    ds_plot_folder_lambda = sprintf('%s/lambda/', ds_final_plot_folder);
    if ~exist(ds_plot_folder, 'dir')
        mkdir(ds_plot_folder);
    end
    % For Eta
    for log_lambda = param.lambda_set
        for dropout = param.dropout_set
            filename = sprintf('%d/%d_lam%d_drop%.1f', ...
                dataset, dataset, log_lambda, dropout);
            stats = [];
            for log_eta = param.eta_set
                mat_name = sprintf('%s%s_eta-%d.mat', mat_folder, filename, log_eta);
                data = load(mat_name);
                errors = gather(data.training_values(:, 1));
                stats(:, end + 1) = errors;
            end
            save_name = sprintf('%s%s.jpg', ds_plot_folder_lambda, filename);
            plot_name = sprintf('DS %d Lambda %d Dropout %.1f', dataset, log_lambda, dropout);
            plot_curve(plot_name, save_name, stats, 1, param.eta_set);
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_curve(plot_name, save_name, stats, type, values)
    lag = 100;
    X = 1:size(stats, 1);
    sparse_x = X(1:50:end);
    switch type
        case 1
            x_name = 'eta';
        case 2
            x_name = 'lambda';
        case 3
            x_name = 'dropout';
    end
    plotStyle = {'b','k','r', 'g', 'y'};
    % save fig
    f = figure('Visible','off'); hold on;
    set(f,'Visible','off','CreateFcn','set(f,''Visible'',''on'')')
    for k=1:size(stats, 2)
        mavg_train = tsmovavg(stats(:, k),'s',lag,1);
        mavg_train = mavg_train(1:50:end); 
        plot(sparse_x, mavg_train, plotStyle{k});
        legendInfo{k} = sprintf('%s = %.1f', x_name, values(k)); % or whatever is appropriate
    end    
    title(plot_name);
    xlabel('Iteration#');
    ylabel('Training error');
    legend(legendInfo);
    saveas(f, save_name)
    hold off;
    close
end
