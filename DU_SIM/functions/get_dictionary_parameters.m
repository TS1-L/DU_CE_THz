function [params, dictionary] = get_dictionary_parameters(N, d, lamda, G_angle, fn, beta, rho_min)
% Reconstructs the dictionary AND returns the (theta, range) for each column.
% params: Struct array where params(i).theta and params(i).r are the values for column i.

    theta = linspace(-1 + 1/G_angle, 1 - 1/G_angle, G_angle);
    
    % --- 1. Far-field atoms ---
    rr_far = 1e5; 
    far_field_atoms = {};
    far_field_params = [];
    
    for i = 1:length(theta)
        th = theta(i);
        atom = polar_domain_manifold(N, d, fn, rr_far, asin(th));
        far_field_atoms{end+1} = atom; %#ok<AGROW>
        
        % Store params
        entry.theta = asin(th); % Store in radians
        entry.r = rr_far;
        entry.type = 'far';
        far_field_params = [far_field_params; entry]; %#ok<AGROW>
    end
    
    % --- 2. Near-field atoms ---
    Z = (N*d)^2 / (2 * lamda * beta^2);
    
    if rho_min > 0
        max_s = floor(Z / rho_min);
    else
        max_s = 100; 
    end
    
    near_field_atoms = {};
    near_field_params = [];
    
    s = 1;
    while Z/s >= rho_min
        for idx = 1:G_angle
            if abs(theta(idx)) <= 1
                rr_near = Z/s * (1 - theta(idx)^2);
                if rr_near > 0
                    atom = polar_domain_manifold(N, d, fn, rr_near, asin(theta(idx)));
                    near_field_atoms{end+1} = atom; %#ok<AGROW>
                    
                    % Store params
                    entry.theta = asin(theta(idx));
                    entry.r = rr_near;
                    entry.type = 'near';
                    near_field_params = [near_field_params; entry]; %#ok<AGROW>
                end
            end
        end
        s = s + 1;
    end
    
    % --- 3. Concatenate & Normalize ---
    all_atoms = [far_field_atoms, near_field_atoms];
    dictionary = cat(2, all_atoms{:});
    
    params = [far_field_params; near_field_params];
    
    % Normalize columns (Must match training!)
    for i = 1:size(dictionary, 2)
        nrm = norm(dictionary(:,i));
        if nrm > 0
            dictionary(:,i) = dictionary(:,i) / nrm;
        end
    end
end