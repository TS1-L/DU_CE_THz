function [X_train, Y_train] = generate_training_input(SNR_collection, Mr_collection, dictionary, ~, num_sc, L, is_test)
% Generates the input/output pairs for training the DNN.
% It loads the pre-generated dataset, then for each sample, it runs OMP
% to get an initial, low-quality estimate 'x_hat', which is the network input.
% The true channel 'H' is the network target.
%
% Inputs:
%   SNR_collection - Array of SNR values (e.g., [0, 5, 10, 15, 20])
%   Mr_collection  - Array of Mr values (e.g., [16, 32, 64, 128])
%   dictionary     - The polar domain dictionary matrix (A_n_FID)
%   N              - Number of antenna elements (Nr)
%   num_sc         - Number of subcarriers
%   L              - Sparsity level for the OMP algorithm (number of paths)
%   is_test        - OPTIONAL boolean: if true, loads TEST_..._quicktest.mat files
%
% Outputs:
%   X_train        - Network inputs (the x_hat estimates), real-valued format
%   Y_train        - Network targets (the true H), real-valued format

    if nargin < 7
        is_test = false;
    end

    fprintf('Starting generation of training input data...\n');
    
    G_polar = size(dictionary, 2);
    channel_model = 'cluster';
    
    % Pre-allocate cell arrays to hold the data from different conditions
    total_conditions = length(SNR_collection) * length(Mr_collection);
    condition_count = 0;
    X_train_list = cell(1,total_conditions);
    Y_train_list = cell(1,total_conditions);
    for Mr = Mr_collection
        for SNR = SNR_collection
            condition_count = condition_count + 1;
            fprintf('(%d/%d) Processing condition: Mr = %d, SNR = %d dB\n', ...
                condition_count, total_conditions, Mr, SNR);

            % --- Load the appropriate dataset ---
            if is_test
                % OLD: dataset_name = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs.mat', Mr, SNR, channel_model, num_sc);
                % Replaced to load the TEST quicktest naming convention used in your pipeline
                dataset_name = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs_quicktest.mat', Mr, SNR, channel_model, num_sc);
            else
                dataset_name = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs.mat', Mr, SNR, channel_model, num_sc);
            end

            try
                data = load(dataset_name);
                Y_received = data.Y_list; % The noisy measurements from hardware
                H_true = data.H_list;     % The ground truth channel
                W = data.W;               % The measurement matrix
            catch ME
                fprintf('Failed to load: %s\n', dataset_name);
                fprintf('Please ensure all data generation scripts have been run for all SNR/Mr conditions.\n');
                rethrow(ME);
            end
            
            [num_samples, ~, ~] = size(Y_received);
            
            % --- Calculate the effective dictionary Phi for this condition ---
            Phi = W' * dictionary;
            
            % --- Initialize arrays for this batch ---
            x_hat_batch = zeros(num_samples, num_sc, G_polar, 'like', 1i);

            % --- Run OMP for each sample and each subcarrier ---
            % This is the most computationally intensive part
            % OLD: parfor i = 1:num_samples
            %       ...
            % Replaced: keep parfor but ensure compatibility with your environment; if parpool isn't available, this will run serially.
            parfor i = 1:num_samples
                temp_x_hat_sample = zeros(num_sc, G_polar, 'like', 1i);
                for n = 1:num_sc
                    y_vec = squeeze(Y_received(i, n, :));
                    
                    % Run OMP algorithm
                    [x_hat_omp, ~] = OMP(Phi, y_vec, L);
                    temp_x_hat_sample(n, :) = x_hat_omp;
                end
                x_hat_batch(i, :, :) = temp_x_hat_sample;
            end
            
            % Store the results
            X_train_list{condition_count} = x_hat_batch;
            Y_train_list{condition_count} = H_true;
        end
    end
    
    % --- Concatenate data from all conditions ---
    fprintf('Concatenating data from all conditions...\n');
    X_train_complex = cat(1, X_train_list{:});
    Y_train_complex = cat(1, Y_train_list{:});

    % --- Convert to real-valued format for the DNN ---
    X_train = C2R(X_train_complex);
    Y_train = C2R(Y_train_complex);

    fprintf('Training input generation complete.\n');
end