%% Calculate NMSE from Saved Model - Fixed Memory & Logic
clear; clc;
addpath("data/", "channels/", "functions/", "classes/","training outputs/");

% 1. Load Model & Params
model_file = 'trained_LAMP_model.mat';
if ~isfile(model_file)
    error('Model file not found.');
end
load(model_file, 'lamp_layers', 'norm_factor');
fprintf('Loaded model: %s\n', model_file);

fc = 300e9; c = 3e8; num_sc = 32; Nr = 256; d = (c/fc)/2;

% 2. Load Test Data
fprintf('Loading test data...\n');
SNR_collection = [0, 5, 10, 15, 20];
Mr_collection = [16, 32, 64, 128];
channel_model = 'cluster';

H_target_list = cell(1,length(Mr_collection)*length(SNR_collection));
H_input_list  = cell(1,length(Mr_collection)*length(SNR_collection)); % corrupt channels
i = 1;

for Mr = Mr_collection
    for SNR = SNR_collection
        % Use quicktest or normal files depending on what you have
        fname = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs.mat', Mr, SNR, channel_model, num_sc);
        if ~isfile(fname)
            % Try quicktest if normal not found
            fname = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs_quicktest.mat', Mr, SNR, channel_model, num_sc);
        end
        
        if isfile(fname)
            loaded = load(fname, 'H_list', 'Y_list', 'W');
            
            % Target
            H_perm = permute(loaded.H_list, [2, 3, 1]); 
            H_target_batch = cat(3, reshape(real(H_perm), num_sc, Nr, 1, []), ...
                                    reshape(imag(H_perm), num_sc, Nr, 1, []));
            
            % Input (corrupt Channel)
            Y_perm = permute(loaded.Y_list, [3, 2, 1]);
            [Mr_val, nsc, nsam] = size(Y_perm);
            Y_flat = reshape(Y_perm, Mr_val, []);
            H_corrupt_flat = loaded.W * Y_flat;
            H_corrupt_perm = permute(reshape(H_corrupt_flat, Nr, nsc, nsam), [2, 1, 3]);
            
            H_input_batch = cat(3, reshape(real(H_corrupt_perm), num_sc, Nr, 1, []), ...
                                   reshape(imag(H_corrupt_perm), num_sc, Nr, 1, []));
            
            H_target_list{i} = H_target_batch;
            H_input_list{i}  = H_input_batch;
            i = i + 1;
        end
    end
end

if isempty(H_target_list)
    error('No data found.');
end

% Combine all
H_target_all = cat(4, H_target_list{:});
H_input_all  = cat(4, H_input_list{:});

% Normalize using the SAME factor saved from training
H_target_all = H_target_all / norm_factor;
H_input_all  = H_input_all  / norm_factor;

[n_sc, nr, chan, total_samples] = size(H_target_all);
fprintf('Total Test Samples: %d\n', total_samples);

% 3. Batched Inference (Fixes Memory Crash)
batch_size = 50; 
total_mse = 0;
total_power = 0;

fprintf('Running inference in batches...\n');

% Pre-compute Dictionary A once
final_layer = lamp_layers{end};
% Ensure params are on CPU/Double for final calculation
thetas = double(gather(final_layer.Thetas));
ranges = double(gather(final_layer.Ranges));

A_complex = differentiable_manifold(thetas, ranges, Nr, d, fc, num_sc);
A_reshaped = reshape(A_complex, [Nr, num_sc, length(thetas)]);
A_perm = permute(A_reshaped, [1, 3, 2]);
Ar = real(A_perm); Ai = imag(A_perm);

for i = 1:batch_size:total_samples
    idx_end = min(i + batch_size - 1, total_samples);
    
    % Get Batch
    Input_batch_cpu  = H_input_all(:,:,:, i:idx_end);
    Target_batch_cpu = H_target_all(:,:,:, i:idx_end);
    
    % Move to dlarray
    X_batch = dlarray(single(Input_batch_cpu), 'SSCB');
    [~, ~, ~, b_size] = size(X_batch);
    G = length(thetas);
    
    % Init States
    h = dlarray(zeros(n_sc, G, 2, b_size, 'like', X_batch), 'SSCB');
    v = X_batch; % Input is v
    
    % Forward Pass
    for k = 1:length(lamp_layers)
        [h, v] = lamp_layers{k}.predict(h, v, X_batch);
    end
    
    % Reconstruct Output (Manual CPU calculation to be safe)
    h_raw = double(extractdata(h));
    h_raw = permute(h_raw, [1, 2, 4, 3]); % (n_sc, G, Batch, 2)
    
    hr = permute(h_raw(:,:,:,1), [2, 3, 1]); % (G, Batch, n_sc)
    hi = permute(h_raw(:,:,:,2), [2, 3, 1]);
    
    % Y = A * h
    yr = pagemtimes(Ar, hr) - pagemtimes(Ai, hi);
    yi = pagemtimes(Ar, hi) + pagemtimes(Ai, hr);
    
    % Shape back to (n_sc, Nr, 2, Batch)
    Y_pred_batch = cat(3, reshape(permute(yr, [3, 1, 2]), n_sc, Nr, 1, b_size), ...
                          reshape(permute(yi, [3, 1, 2]), n_sc, Nr, 1, b_size));
    
    % Error Calculation
    diff = Y_pred_batch - Target_batch_cpu;
    total_mse = total_mse + sum(diff.^2, 'all');
    total_power = total_power + sum(Target_batch_cpu.^2, 'all');
end

% 4. Final NMSE
nmse_linear = total_mse / total_power;
nmse_db = 10 * log10(nmse_linear);

save(fillfile('training outputs','NMSE_Results.mat'),"total_power","nmse_linear","nmse_db","total_mse");

fprintf('\n==============================\n');
fprintf('Final NMSE: %.4f dB\n', nmse_db);
fprintf('==============================\n');