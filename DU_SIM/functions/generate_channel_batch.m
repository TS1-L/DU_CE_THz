function H_list = generate_channel_batch(num_samples, Nr, num_sc, random_seed)
% GENERATE_CHANNEL_BATCH Generates a batch of unique THz channels.
% This function replaces the standalone genChannel.m script.
%
% Inputs:
%   num_samples - Number of channels to generate (e.g., 1000)
%   Nr          - Number of receive antennas (e.g., 256)
%   num_sc      - Number of subcarriers (e.g., 32)
%   random_seed - Unique integer seed to ensure this batch is distinct
%
% Output:
%   H_list      - Complex channel matrix (num_samples, num_sc, Nr)

    % Set the specific seed for this batch
    rng(random_seed);

    % --- Simulation Parameters ---
    fc = 300 * 1e9;      
    fs = 15 * 1e9;       
    tau_max = 20 * 1e-9; 
    num_subpaths = 10;   
    num_clusters = 3;    
    d_min = 5;           
    d_max = 30;          
    AS = 4;              
    RS = 1;              

    % --- Derived Parameters ---
    c = 3e8;
    lambda_c = c / fc;
    d = lambda_c / 2;
    eta = fs / num_sc;
    Lp = num_clusters * num_subpaths;

    % --- Main Generation Loop ---
    H_list = zeros(num_samples, num_sc, Nr, 'like', 1i);
    
    % (Optional: suppress per-line print to reduce clutter in multi-gen)
    % fprintf('Generating %d channels with seed %d...\n', num_samples, random_seed);

    for i = 1:num_samples
        path_gains = sqrt(1 / 2) * (randn(Lp, 1) + 1i * randn(Lp, 1));
        taus = rand(Lp, 1) * tau_max;
        AoAs = zeros(Lp, 1);
        distances = zeros(Lp, 1);

        for nc = 1:num_clusters
            cluster_indices = (nc-1)*num_subpaths + 1 : nc*num_subpaths;
            mean_AoA = rand() * 360;
            mean_distance = d_min + (d_max - d_min) * rand();
            
            AoAs_cluster = laplace_rnd(mean_AoA, sqrt(AS^2/2), [num_subpaths, 1]);
            AoAs_cluster = max(min(AoAs_cluster, mean_AoA + 2*AS), mean_AoA - 2*AS);
            AoAs(cluster_indices) = AoAs_cluster / 180 * pi;
            
            distances_cluster = laplace_rnd(mean_distance, sqrt(RS^2/2), [num_subpaths, 1]);
            distances_cluster = max(min(distances_cluster, d_max), d_min);
            distances(cluster_indices) = distances_cluster;
        end
        
        for n = 1:num_sc
            fn = fc + (n - 1 - (num_sc - 1) / 2) * eta;
            H_sc = zeros(Nr, 1, 'like', 1i);
            for p = 1:Lp
                % Simple Near-Field Manifold Calculation Inline
                nn = (-(Nr-1)/2 : (Nr-1)/2)';
                r = sqrt(distances(p)^2 + (nn*d).^2 - 2*distances(p)*nn*d.*sin(AoAs(p)));
                at = exp(-1i*2*pi*fn*(r-distances(p))/c);
                
                multipath_delay_phase = exp(-1i * 2 * pi * fn * taus(p));
                H_sc = H_sc + path_gains(p) * multipath_delay_phase * at;
            end
            
            if norm(H_sc) > 0
                H_sc = H_sc / norm(H_sc);
            end
            H_list(i, n, :) = reshape(H_sc, [1, 1, Nr]);
        end
    end
end

% Helper for Laplace distribution
function y = laplace_rnd(mu, b, sz)
    u = rand(sz) - 0.5;
    y = mu - b * sign(u) .* log(1 - 2 * abs(u));
end