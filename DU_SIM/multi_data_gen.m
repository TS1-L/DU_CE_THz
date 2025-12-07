%% Multi-Condition Data Generation (Unique Channels per Condition)
%% NOTE: TEST DECISION IS SET BY Run_Scripts.m !!! MAKE SURE TO SET IT THERE OR ADD test = false/true; BASED ON YOUR GOAL

%%% clear; clc;

% Ensure folders exist
if ~exist(fullfile(pwd,'channels'),'dir'), mkdir(fullfile(pwd,'channels')); end
if ~exist(fullfile(pwd,'data'),'dir'), mkdir(fullfile(pwd,'data')); end
if ~exist(fullfile(pwd,'training outputs'),'dir'), mkdir(fullfile(pwd,'training outputs')); end

addpath ("data/","channels/","functions/","classes/","training outputs/");

% --- Parameters ---
SNR_collection = [0, 5, 10, 15, 20];
Mr_collection = [16, 32, 64, 128];
Nr = 256; 
num_sc = 32;
channel_model = 'cluster';

test = false;

if test
    samples_per_condition = 100;
    fprintf('--- RUNNING IN QUICK TEST MODE (100 samples/condition) ---\n');
else
    samples_per_condition = 1000;
    fprintf('--- RUNNING IN FULL MODE (1000 samples/condition) ---\n');
end

condition_count = 0;
total_conditions = length(SNR_collection) * length(Mr_collection);

for Mr = Mr_collection
    for SNR_dB = SNR_collection
        condition_count = condition_count + 1;
        fprintf('\n--- Processing Condition %d/%d: Mr=%d, SNR=%d dB ---\n', ...
            condition_count, total_conditions, Mr, SNR_dB);
        
        % 1. Determine filenames
        suffix = sprintf('%dBeams_%ddB-SNR_%s_%dscs', Mr, SNR_dB, channel_model, num_sc);
        if test; suffix = [suffix, '_quicktest']; end %#ok<AGROW>
        
        channel_filename = fullfile('channels', ['channel_' suffix '.mat']);
        data_filename    = fullfile('data', ['data_' suffix '.mat']);
        
        % 2. Generate UNIQUE Channels for this condition
        % We use a unique seed based on the condition count to ensure 
        % no two conditions share the same physical channels.
        unique_seed = 2026 + (condition_count * 2003); 
        
        fprintf('   Generating unique channels (Seed: %d)...\n', unique_seed);
        H_list = generate_channel_batch(samples_per_condition, Nr, num_sc, unique_seed);
        
        % Save the channel file
        save(channel_filename, 'H_list', '-v7.3');
        fprintf('   Saved: %s\n', channel_filename);
        
        % 3. Generate Measurements (Data)
        fprintf('   Generating measurements...\n');
        [data_num, ~, ~] = size(H_list);
        
        % Random Combiner W (Mr x Nr) - RF Chain Phase Shifts
        antennas_per_chain = Nr / Mr;
        W = zeros(Nr, Mr);
        rng(unique_seed + 911); % Ensure W is also deterministically random
        for m = 1:Mr
            start_idx = (m - 1) * antennas_per_chain + 1;
            end_idx = m * antennas_per_chain;
            phases = (2 * randi([0 1], antennas_per_chain, 1) - 1);
            W(start_idx:end_idx, m) = phases;
        end
        W = W / sqrt(antennas_per_chain);
        
        % Compute Received Signal Y
        % H_list is (N, num_sc, Nr) -> Permute to (Nr, num_sc, N)
        H_perm = permute(H_list, [3, 2, 1]); 
        H_flat = reshape(H_perm, Nr, []); % (Nr, num_sc*N)
        
        % Apply Combiner: Y = W' * H
        Y_flat = W' * H_flat; % (Mr, num_sc*N)
        
        % Add Noise
        sig_power = mean(abs(Y_flat(:)).^2);
        noise_var = sig_power / (10^(SNR_dB / 10));
        noise = sqrt(noise_var/2) * (randn(size(Y_flat)) + 1i*randn(size(Y_flat)));
        
        Y_noisy_flat = Y_flat + noise;
        
        % Reshape back to (N, num_sc, Mr)
        Y_3d = reshape(Y_noisy_flat, Mr, num_sc, data_num);
        Y_list = permute(Y_3d, [3, 2, 1]); % (N, num_sc, Mr)
        
        % 4. Save Final Dataset (Contains H_list so training script is self-contained)
        save(data_filename, 'H_list', 'Y_list', 'W', '-v7.3');
        fprintf('   Saved Dataset: %s\n', data_filename);
    end
end
fprintf('\nAll data generation complete.\n');