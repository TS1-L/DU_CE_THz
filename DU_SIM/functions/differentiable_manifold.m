function A = differentiable_manifold(thetas, ranges, Nr, d, fc_center, num_sc)
    % DIFFERENTIABLE_MANIFOLD
    % Generates the Near-Field Polar Domain Dictionary matrix A.
    % Unlike the standard version, this supports Auto-Differentiation (AD).
    %
    % Inputs:
    %   thetas    - (1 x G) dlarray: Learnable angles [rad]
    %   ranges    - (1 x G) dlarray: Learnable distances [m]
    %   Nr        - integer: Number of antennas
    %   d         - double: Antenna spacing [m]
    %   fc_center - double: Center carrier frequency [Hz]
    %   num_sc    - integer: Number of subcarriers
    %
    % Output:
    %   A         - (Nr*num_sc x G) dlarray: The wideband dictionary matrix.
    %               Columns are normalized to unit l2-norm.

    % --- Constants ---
    c = 3e8; 
    fs = 15e9;          % System bandwidth (Must match simulation)
    eta = fs / num_sc;  % Subcarrier spacing
    
    % --- Antenna Indices ---
    % Create vector [-(N-1)/2, ..., (N-1)/2]
    % Fix: Cast to 'like' real(thetas) to ensure nn is strictly real,
    % avoiding complexity mismatch if thetas has become complex.
    nn = cast((-(Nr-1)/2 : (Nr-1)/2)', 'like', real(thetas)); 
    
    % --- 1. Tensor Expansion (Broadcasting) ---
    % We perform operations on tensors to handle G atoms and Nr antennas simultaneously.
    % Dimensions: (Nr, G, 1)
    
    % nn_exp: Antenna indices expanded to (Nr, 1, 1)
    nn_exp = reshape(nn, [Nr, 1, 1]);
    
    % th_exp: Angles expanded to (1, G, 1)
    th_exp = reshape(thetas, [1, length(thetas), 1]);
    
    % r_exp: Ranges expanded to (1, G, 1)
    r_exp  = reshape(ranges, [1, length(ranges), 1]);
    
    % --- 2. Subcarrier Loop ---
    % We generate the response for each subcarrier. 
    % Note: Using a cell array and cat(1) is efficient for dlarray graph construction.
    A_parts = cell(num_sc, 1);
    
    % Fix: Create j_unit safely. 
    % 1. Create a real-valued '1' with the same precision/type/device as thetas
    % 2. Multiply by standard 1i to create the complex unit
    one_val = cast(1, 'like', real(thetas));
    j_unit = 1i * one_val;
    
    for n = 1:num_sc
        % Calculate frequency for subcarrier n
        fn = fc_center + (n - 1 - (num_sc - 1) / 2) * eta;
        
        % --- SEE MATH SECTION [A] ---
        % Fresnel Phase Approximation: 
        % Phase = - (2*pi*f/c) * ( r_0 - n*d*sin(th) + n^2*d^2*cos^2(th)/(2*r0) - r_0 )
        % The linear r_0 term cancels out in the path difference (r - r_0).
        
        % Term 1: Far-Field Linear Phase (Steering Vector)
        % proportional to -n * d * sin(theta)
        term1 = -nn_exp .* d .* sin(th_exp);
        
        % Term 2: Near-Field Quadratic Phase (Focusing)
        % proportional to n^2 * d^2 * cos^2(theta) / (2 * r)
        % This term captures the spherical wavefront curvature.
        term2 = (nn_exp.^2 .* d^2 .* cos(th_exp).^2) ./ (2 * r_exp);
        
        % Total Path Difference (Distance)
        path_diff = term1 + term2;
        
        % Steering Vector (Nr x G)
        val = exp(-j_unit * 2 * pi * fn * path_diff / c);
        
        % --- Normalization ---
        % Physics dictates that the steering vector power is split among antennas.
        % l2-norm of a vector of Nr complex exponentials (mag 1) is sqrt(Nr).
        % We divide by sqrt(Nr) so that A'*A has diagonal entries of 1.
        val = val / sqrt(Nr); 
        
        A_parts{n} = val; 
    end
    
    % --- 3. Stack Wideband Matrix ---
    % Concatenate all subcarriers vertically.
    % Final Shape: (Nr * num_sc, G)
    A = cat(1, A_parts{:});
    
    % Wideband Normalization
    % Since we normalized each subblock to 1/sqrt(Nr), the total norm of the
    % wideband vector is sqrt(num_sc). We normalize again to unit norm.
    A = A / sqrt(num_sc);
end