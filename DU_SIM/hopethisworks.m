%% Train Deep Unfolding (Parametric LAMP) - Validated & Dual-Graph
% Features:
% - Automatic 80/20 Train/Val Split
% - Live MSE & NMSE Visualization
% - Fixes "Identity Mapping" bug (Uses corrupt Channel Input)
% - Batch Processing for Memory Safety

clear; clc; close all;

if ~exist(fullfile(pwd,'channels'),'dir'), mkdir(fullfile(pwd,'channels')); end
if ~exist(fullfile(pwd,'data'),'dir'), mkdir(fullfile(pwd,'data')); end
if ~exist(fullfile(pwd,'training outputs'),'dir'), mkdir(fullfile(pwd,'training outputs')); end

addpath("data/", "channels/", "functions/", "classes/","training outputs/");

% --- 0. GPU Setup ---
if canUseGPU
    g = gpuDevice(1);
    reset(g);
    fprintf('GPU Detected: %s (VRAM: %.2f GB)\n', g.Name, g.AvailableMemory/1e9);
    use_gpu = true;
else
    warning('No supported GPU found. Falling back to CPU.');
    use_gpu = false;
end

% --- 1. Configuration ---
network_lr = 1e-3;
dict_lr    = 1e-5;
grad_clip  = 1.0;
batch_size = 64;    

test = false; % Set to false for the full run

if test
    epochs = 2; num_layers = 2;
else
    epochs = 20; num_layers = 8;
end

% --- 2. Load Data & Prepare Input/Target Pairs ---
fprintf('Loading Data...\n');
SNR_collection = [0, 5, 10, 15, 20];
Mr_collection = [16, 32, 64, 128];
num_sc = 32; Nr = 256;
channel_model = 'cluster';

X_list = cell(1,length(Mr_collection)*length(SNR_collection)); % Inputs (Corrupt Channel)
Y_list = cell(1,length(Mr_collection)*length(SNR_collection)); % Targets (Clean Channel)
i = 1;
for Mr = Mr_collection
    for SNR = SNR_collection
        if test
            fname = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs_quicktest.mat', Mr, SNR, channel_model, num_sc);
        else
            fname = sprintf('data/data_%dBeams_%ddB-SNR_%s_%dscs.mat', Mr, SNR, channel_model, num_sc);
        end
        
        if isfile(fname)
            loaded = load(fname, 'H_list', 'Y_list', 'W');
            
            % --- A. Prepare Target (H_true) ---
            % Permute to (num_sc, Nr, Samples)
            H_perm = permute(loaded.H_list, [2, 3, 1]); 
            
            % --- B. Prepare Input (H_corrupt = W * Y) ---
            % Y_list: (Samples, num_sc, Mr) -> Permute to (Mr, num_sc, Samples)
            Y_perm = permute(loaded.Y_list, [3, 2, 1]);
            [Mr_val, nsc, nsam] = size(Y_perm);
            
            % Project back to antenna space
            Y_flat = reshape(Y_perm, Mr_val, []);
            H_corrupt_flat = loaded.W * Y_flat; % (Nr, TotalPoints)
            
            % Reshape back: (Nr, num_sc, Samples) -> Permute to (num_sc, Nr, Samples)
            H_corrupt_perm = permute(reshape(H_corrupt_flat, Nr, nsc, nsam), [2, 1, 3]);
            
            % --- C. Format (Complex -> 2 Channels) ---
            H_target = cat(3, reshape(real(H_perm), num_sc, Nr, 1, []), ...
                              reshape(imag(H_perm), num_sc, Nr, 1, []));
            H_input  = cat(3, reshape(real(H_corrupt_perm), num_sc, Nr, 1, []), ...
                              reshape(imag(H_corrupt_perm), num_sc, Nr, 1, []));
            
            Y_list{i} = single(H_target);
            X_list{i} = single(H_input);
            i = i + 1;
        end
    end
end

if isempty(Y_list)
    error('No data found. Run multi_data_gen.m first.');
end

% Combine all data
H_all = cat(4, Y_list{:});
X_all = cat(4, X_list{:});

% --- NORMALIZE ---
norm_factor = max(abs(H_all), [], 'all');
fprintf('Data Max: %.2e. Normalizing...\n', norm_factor);
H_all = H_all / norm_factor;
X_all = X_all / norm_factor;

% --- VALIDATION SPLIT (80/20) ---
num_total = size(H_all, 4);
val_split = 0.2;
num_val = floor(num_total * val_split);
num_train = num_total - num_val;

% Randomized Split
idx_rand = randperm(num_total);
idx_val = idx_rand(1:num_val);
idx_train = idx_rand(num_val+1:end);

% Create Data Objects (Move to GPU later in batches to save RAM)
Y_train = dlarray(H_all(:,:,:,idx_train), 'SSCB');
X_train = dlarray(X_all(:,:,:,idx_train), 'SSCB');
Y_val   = dlarray(H_all(:,:,:,idx_val), 'SSCB');
X_val   = dlarray(X_all(:,:,:,idx_val), 'SSCB');

fprintf('Split: Train=%d, Val=%d samples.\n', num_train, num_val);

% --- 3. Initialize Physics ---
fc = 300e9; c = 3e8; lambda_c = c/fc; d = lambda_c/2; 
s = 2; G_angle = s * Nr; beta = 1.2; rho_min = 3;
[grid_params, ~] = get_dictionary_parameters(Nr, d, lambda_c, G_angle, fc, beta, rho_min);

% --- 4. Build Network ---
lamp_layers = cell(num_layers, 1);
learnables = cell(num_layers, 1);

fprintf('Initializing %d Layers...\n', num_layers);
for k = 1:num_layers
    layer = ParametricLAMPLayer(grid_params, Nr, d, fc, num_sc, 64, "LAMP_"+k);
    if use_gpu
        layer.Thetas = gpuArray(single(layer.Thetas));
        layer.Ranges = gpuArray(single(layer.Ranges));
        layer.StepSize = gpuArray(single(layer.StepSize));
        layer.Threshold = gpuArray(single(layer.Threshold));
    else
        layer.Thetas = single(layer.Thetas);
        layer.Ranges = single(layer.Ranges);
        layer.StepSize = single(layer.StepSize);
        layer.Threshold = single(layer.Threshold);
    end
    lamp_layers{k} = layer;
    learnables{k}.Thetas = layer.Thetas;
    learnables{k}.Ranges = layer.Ranges;
    learnables{k}.StepSize = layer.StepSize;
    learnables{k}.Threshold = layer.Threshold;
end

% --- 5. VISUALIZATION SETUP ---
f = figure('Name', '6G Channel Estimation Training', 'Color', 'w');
tgroup = uitabgroup(f);
tab1 = uitab(tgroup, 'Title', 'Training Metrics');

% Subplot 1: MSE Loss
subplot(2,1,1, 'Parent', tab1);
lineLossTrain = animatedline('Color', '#0072BD', 'LineWidth', 1.5, 'DisplayName', 'Train MSE');
lineLossVal   = animatedline('Color', '#D95319', 'LineWidth', 2.0, 'Marker', 'o', 'MarkerFaceColor', 'w', 'DisplayName', 'Val MSE');
xlabel('Iteration'); ylabel('MSE Loss (dB)'); title('Model Convergence (MSE)');
legend('Location','northeast'); grid on;

% Subplot 2: NMSE (The real metric)
subplot(2,1,2, 'Parent', tab1);
lineNMSETrain = animatedline('Color', '#77AC30', 'LineWidth', 1.5, 'DisplayName', 'Train NMSE');
lineNMSEVal   = animatedline('Color', '#7E2F8E', 'LineWidth', 2.0, 'Marker', 's', 'MarkerFaceColor', 'w', 'DisplayName', 'Val NMSE');
xlabel('Epoch'); ylabel('NMSE (dB)'); title('Channel Estimation Accuracy (NMSE)');
legend('Location','northeast'); grid on;

% --- 6. Training Loop ---
vel_layers = cell(num_layers, 1);
num_batches_train = floor(num_train / batch_size);
iteration = 0;

fprintf('Starting Training...\n');
total_train_start = tic; 

for epoch = 1:epochs
    % Shuffle Training Data
    idx_shuff = randperm(num_train);
    X_train = X_train(:,:,:,idx_shuff);
    Y_train = Y_train(:,:,:,idx_shuff);
    
    % --- TRAIN BATCHES ---
    for b = 1:num_batches_train
        iteration = iteration + 1;
        idx_batch = (b-1)*batch_size + 1 : b*batch_size;
        
        X_batch = X_train(:,:,:,idx_batch);
        Y_batch = Y_train(:,:,:,idx_batch);
        
        if use_gpu; X_batch = gpuArray(X_batch); Y_batch = gpuArray(Y_batch); end
        
        % Calculate Gradients & Metrics
        [loss, grads, nmse_batch] = dlfeval(@model_loss_and_metrics, learnables, lamp_layers, X_batch, Y_batch, Nr, d, fc, num_sc);
        
        % Check divergence
        if isnan(extractdata(loss))
            error('Training diverged (NaN). Decrease Learning Rate.');
        end
        
        % Update Weights (Momentum)
        for k = 1:num_layers
            lg = grads{k};
            if isempty(vel_layers{k})
                z = cast(0, 'like', learnables{k}.Thetas);
                vel_layers{k} = struct('Thetas',z, 'Ranges',z, 'StepSize',z, 'Threshold',z);
            end
            
            % Update with Gradients
            [learnables{k}, vel_layers{k}] = apply_update(learnables{k}, vel_layers{k}, lg, dict_lr, network_lr, grad_clip);
        end
        
        % Update Live Training Graphs (every 10 iters)
        if mod(iteration, 10) == 0
            loss_db = 10*log10(double(extractdata(loss)));
            nmse_db = 10*log10(double(extractdata(nmse_batch)));
            
            addpoints(lineLossTrain, iteration, loss_db);
            addpoints(lineNMSETrain, epoch + (b/num_batches_train), nmse_db); % Smooth x-axis for NMSE
            drawnow limitrate;
        end
    end
    
    % --- VALIDATION PASS (End of Epoch) ---
    fprintf('Validating Epoch %d... ', epoch);
    [val_mse_db, val_nmse_db] = evaluate_validation(learnables, lamp_layers, X_val, Y_val, batch_size, Nr, d, fc, num_sc, use_gpu);
    
    % Plot Validation Points (aligned to Iteration count for MSE, Epoch for NMSE)
    addpoints(lineLossVal, iteration, val_mse_db);
    addpoints(lineNMSEVal, epoch + 1, val_nmse_db);
    drawnow;
    
    fprintf('Val NMSE: %.2f dB | Train Loss: %.2f dB\n', val_nmse_db, loss_db);
end

train_time = toc(total_train_start);

model_name = 'trained_LAMP_model.mat';
model_dir = 'training outputs';
model_figure = 'training_figure.fig';
save(fullfile(model_dir,model_name), 'lamp_layers', 'grid_params', 'train_time', 'norm_factor');
savefig(f,fullfile(model_dir,model_figure));
fprintf('Training Complete. Model & Training Figure Saved.\n');


%% --- Helper Functions ---

function [new_params, new_vel] = apply_update(params, vel, grads, lr_dict, lr_net, clip)
    % Helper to apply Momentum update cleanly
    new_params = params;
    new_vel = vel;
    
    % Clip Gradients
    g_th = max(min(real(grads.Thetas), clip), -clip);
    g_ra = max(min(real(grads.Ranges), clip), -clip);
    g_ss = max(min(grads.StepSize, clip), -clip);
    g_tr = max(min(grads.Threshold, clip), -clip);
    
    % Update Dictionary (Thetas/Ranges)
    new_vel.Thetas = 0.9*vel.Thetas - lr_dict*g_th;
    new_params.Thetas = params.Thetas + new_vel.Thetas;
    
    new_vel.Ranges = 0.9*vel.Ranges - lr_dict*g_ra;
    new_params.Ranges = max(params.Ranges + new_vel.Ranges, 1.0);
    
    % Update Gains (Step/Threshold)
    new_vel.StepSize = 0.9*vel.StepSize - lr_net*g_ss;
    new_params.StepSize = params.StepSize + new_vel.StepSize;
    
    new_vel.Threshold = 0.9*vel.Threshold - lr_net*g_tr;
    new_params.Threshold = max(params.Threshold + new_vel.Threshold, 1e-6);
end

function [val_mse_db, val_nmse_db] = evaluate_validation(learnables, layers, X_val, Y_val, b_size, Nr, d, fc, num_sc, use_gpu)
    % Runs inference on Validation Set in Batches (No Gradients)
    num_v = size(X_val, 4);
    total_sse = 0; % Sum Squared Error
    total_pow = 0; % Sum True Power
    
    % Temporarily update layer params for inference
    for k=1:length(layers)
        layers{k}.Thetas = learnables{k}.Thetas;
        layers{k}.Ranges = learnables{k}.Ranges;
        layers{k}.StepSize = learnables{k}.StepSize;
        layers{k}.Threshold = learnables{k}.Threshold;
    end
    
    % Pre-compute Dictionary (CPU is safer for reconstruction if memory tight)
    final = learnables{end};
    A_complex = differentiable_manifold(final.Thetas, final.Ranges, Nr, d, fc, num_sc);
    A_perm = permute(reshape(A_complex, [Nr, num_sc, length(final.Thetas)]), [1,3,2]);
    Ar = real(A_perm); Ai = imag(A_perm);
    
    for i = 1:b_size:num_v
        idx = i : min(i+b_size-1, num_v);
        X_b = X_val(:,:,:,idx);
        Y_b = Y_val(:,:,:,idx);
        if use_gpu; X_b = gpuArray(X_b); end
        
        % Forward
        h = dlarray(zeros(num_sc, length(final.Thetas), 2, length(idx), 'like', X_b), 'SSCB');
        v = X_b;
        for k = 1:length(layers)
            [h, v] = layers{k}.predict(h, v, X_b);
        end
        
        % Reconstruct
        h_raw = extractdata(h); % Move to CPU
        hr = permute(h_raw(:,:,1,:), [2,4,3,1]); % G x Batch x 1 x SC -> G x Batch x SC
        hr = permute(hr, [1,2,4,3]); % G x Batch x SC
        hi = permute(h_raw(:,:,2,:), [2,4,3,1]);
        hi = permute(hi, [1,2,4,3]);
        
        % A: Nr x G x SC
        % Y = Ar*hr ...
        % Using loops for validation safety (fast enough for val)
        Y_pred = zeros(num_sc, Nr, 2, length(idx), 'like', Y_b);
        Y_b_cpu = extractdata(Y_b);
        
        % Vectorized Reconstruction (simplified)
        % (Just calculating error on latent variables if needed, but here we do full)
        % Note: For speed, we approximate val loss using the last batch's forward pass logic
        % But correct way is full reconstruction:
        
        % Re-use the forward logic from model_loss but detached
        % ... (Skipped full expansion for brevity, using approximate metrics from last batch if needed, 
        %      but here is the scalar accumulation:)
        
        % Since we can't easily replicate the complex reconstruction in a helper without code dupe,
        % let's use a trick: run model_loss with "no gradients" on batches.
        [loss_val, ~, nmse_val] = dlfeval(@model_loss_and_metrics, learnables, layers, X_b, Y_b, Nr, d, fc, num_sc);
        
        batch_mse = double(extractdata(loss_val));
        batch_nmse = double(extractdata(nmse_val));
        
        % Approximate weighted average
        total_sse = total_sse + batch_mse * length(idx);
        total_pow = total_pow + (batch_mse / batch_nmse) * length(idx); % recover power
    end
    
    avg_mse = total_sse / num_v;
    avg_nmse = avg_mse / (total_pow / num_v); % This is an approximation for display
    
    val_mse_db = 10*log10(avg_mse);
    val_nmse_db = 10*log10(avg_nmse);
end

function [loss, gradients, nmse_linear] = model_loss_and_metrics(learnables, layers, X_in, Y_true, Nr, d, fc, num_sc)
    % Same as before but returns NMSE too
    batch_size = size(X_in, 4);
    G = length(learnables{1}.Thetas);
    
    h = dlarray(zeros(num_sc, G, 2, batch_size, 'like', X_in), 'SSCB');
    v = X_in; 
    
    for k = 1:length(layers)
        layers{k}.Thetas = learnables{k}.Thetas;
        layers{k}.Ranges = learnables{k}.Ranges;
        layers{k}.StepSize = learnables{k}.StepSize;
        layers{k}.Threshold = learnables{k}.Threshold;
        [h, v] = layers{k}.predict(h, v, X_in);
    end
    
    % Reconstruction
    final = learnables{end};
    A_complex = differentiable_manifold(final.Thetas, final.Ranges, Nr, d, fc, num_sc);
    A_reshaped = reshape(A_complex, [Nr, num_sc, G]);
    A_perm = permute(A_reshaped, [1, 3, 2]); 
    Ar = real(A_perm); Ai = imag(A_perm);
    
    h_raw = stripdims(h); 
    hr = permute(reshape(h_raw(:,:,1,:), num_sc, G, batch_size), [2, 3, 1]);
    hi = permute(reshape(h_raw(:,:,2,:), num_sc, G, batch_size), [2, 3, 1]);
    
    yr = pagemtimes(Ar, hr) - pagemtimes(Ai, hi);
    yi = pagemtimes(Ar, hi) + pagemtimes(Ai, hr);
    
    Y_pred = cat(3, reshape(permute(yr,[3,1,2]), num_sc, Nr, 1, batch_size), ...
                    reshape(permute(yi,[3,1,2]), num_sc, Nr, 1, batch_size));
    
    % Metrics
    diff = Y_pred - stripdims(Y_true);
    mse = sum(diff.^2, 'all') / numel(diff); % Mean Squared Error
    power = sum(stripdims(Y_true).^2, 'all') / numel(Y_true);
    
    loss = mse;
    nmse_linear = mse / power; % Normalized
    
    gradients = dlgradient(loss, learnables);
end