classdef ParametricLAMPLayer < nnet.layer.Layer
    % ParametricLAMPLayer: One iteration of Learned AMP (LAMP).
    % Superior to ISTA due to the Onsager Correction term.
    
    properties (Learnable)
        % Physical Parameters (Dictionary)
        Thetas
        Ranges
        
        % Trainable Gains
        StepSize      % Alpha in literature (Scalar)
        Threshold     % Lambda (Scalar)
    end
    
    properties
        Nr
        d
        fc
        num_sc
        M_beams % Number of measurements (Mr)
        G_polar % Number of atoms
    end
    
    methods
        function layer = ParametricLAMPLayer(grid_params, Nr, d, fc, num_sc, Mr, name)
            layer.Name = name;
            layer.Description = "Parametric Learned AMP";
            
            % Initialize Dictionary Params
            init_th = [grid_params.theta];
            init_r  = [grid_params.r];
            layer.Thetas = dlarray(init_th);
            layer.Ranges = dlarray(init_r);
            
            % Initialize Hyperparameters
            layer.StepSize = dlarray(1.0); % AMP usually starts with step 1.0
            layer.Threshold = dlarray(0.01); 
            
            layer.Nr = Nr;
            layer.d = d;
            layer.fc = fc;
            layer.num_sc = num_sc;
            layer.M_beams = Mr;
            layer.G_polar = length(init_th);
        end
        
        function [h_new, v_new] = predict(layer, h_old, v_old, y)
            % Inputs:
            %   h_old: Current sparse estimate (num_sc, G, 2, Batch)
            %   v_old: Current residual (num_sc, Mr, 2, Batch) 
            %   y:     Original Measurement (num_sc, Mr, 2, Batch)
            
            % 0. Strip labels for safe reshaping/permuting
            h_old_raw = stripdims(h_old);
            v_old_raw = stripdims(v_old);
            y_raw     = stripdims(y);
            
            % Get Batch Size dynamically
            batch_size = size(h_old_raw, 4);
            
            % --- 1. Build A ---
            A_complex = differentiable_manifold(layer.Thetas, layer.Ranges, ...
                                                layer.Nr, layer.d, layer.fc, layer.num_sc);
            % A_complex is (Nr*num_sc x G)
            
            % Reshape A: (Nr, num_sc, G) -> (Nr, G, num_sc)
            A_reshaped = reshape(A_complex, [layer.Nr, layer.num_sc, layer.G_polar]);
            A_perm = permute(A_reshaped, [1, 3, 2]); 
            Ar = real(A_perm); Ai = imag(A_perm);
            
            % --- 2. Calculate Onsager Term (b) ---
            h_mag = sqrt(h_old_raw(:,:,1,:).^2 + h_old_raw(:,:,2,:).^2 + 1e-9);
            sparsity_ratio = mean(h_mag > layer.Threshold, 'all');
            
            ratio = (layer.G_polar / layer.M_beams); 
            b = ratio * sparsity_ratio;
            
            % --- 3. Update Residual (v) ---
            % y_est = A * h_old
            % Prepare h for multiplication: (num_sc, G, 2, Batch) -> (G, Batch, num_sc)
            
            % Extract Real/Imag and Reshape to (num_sc, G, Batch)
            hr_3d = reshape(h_old_raw(:,:,1,:), layer.num_sc, layer.G_polar, batch_size);
            hi_3d = reshape(h_old_raw(:,:,2,:), layer.num_sc, layer.G_polar, batch_size);
            
            % Permute to (G, Batch, num_sc)
            hr = permute(hr_3d, [2, 3, 1]); 
            hi = permute(hi_3d, [2, 3, 1]);
            
            % Ar: (Nr, G, num_sc)
            % y_est_r = Ar*hr - Ai*hi
            yr_est = pagemtimes(Ar, hr) - pagemtimes(Ai, hi);
            yi_est = pagemtimes(Ar, hi) + pagemtimes(Ai, hr);
            
            % Output is (Nr, Batch, num_sc). Need (num_sc, Nr, 2, Batch).
            % Permute to (num_sc, Nr, Batch)
            yr_est = permute(yr_est, [3, 1, 2]); 
            yi_est = permute(yi_est, [3, 1, 2]);
            
            % Stack to (num_sc, Nr, 2, Batch)
            y_est = cat(3, reshape(yr_est, layer.num_sc, layer.Nr, 1, batch_size), ...
                           reshape(yi_est, layer.num_sc, layer.Nr, 1, batch_size));
            
            % v_new calculation
            v_new_raw = (y_raw - y_est) + (b .* v_old_raw);
            
            % --- 4. Update Estimate (h) ---
            % r = A' * v_new
            
            % Prepare v: (num_sc, Nr, 2, Batch) -> (Nr, Batch, num_sc)
            vr_3d = reshape(v_new_raw(:,:,1,:), layer.num_sc, layer.Nr, batch_size);
            vi_3d = reshape(v_new_raw(:,:,2,:), layer.num_sc, layer.Nr, batch_size);
            
            vr = permute(vr_3d, [2, 3, 1]);
            vi = permute(vi_3d, [2, 3, 1]);
            
            % A' = (Ar' - jAi')
            Ar_t = permute(Ar, [2, 1, 3]); % (G, Nr, num_sc)
            Ai_t = permute(Ai, [2, 1, 3]);
            
            % r_real = Ar'*vr + Ai'*vi
            rr = pagemtimes(Ar_t, vr) + pagemtimes(Ai_t, vi);
            ri = pagemtimes(Ar_t, vi) - pagemtimes(Ai_t, vr);
            
            % Output is (G, Batch, num_sc). Need (num_sc, G, 2, Batch)
            rr = permute(rr, [3, 1, 2]); % (num_sc, G, Batch)
            ri = permute(ri, [3, 1, 2]);
            
            r_stack = cat(3, reshape(rr, layer.num_sc, layer.G_polar, 1, batch_size), ...
                             reshape(ri, layer.num_sc, layer.G_polar, 1, batch_size));
            
            % Pre-threshold signal
            z = h_old_raw + layer.StepSize .* r_stack;
            
            % Soft Thresholding
            z_mag = sqrt(z(:,:,1,:).^2 + z(:,:,2,:).^2 + 1e-9);
            scale = relu(1 - layer.Threshold ./ z_mag);
            
            h_new_raw = z .* scale;
            
            % 5. Re-wrap in dlarray with labels for output
            h_new = dlarray(h_new_raw, 'SSCB');
            v_new = dlarray(v_new_raw, 'SSCB');
        end
    end
end