function at = polar_domain_manifold(Nt, d, f, r0, theta0)
% Calculates the steering vector for a single path using the near-field 
% Fresnel approximation model.
%
% Inputs:
%   Nt     - Number of antenna elements
%   d      - Antenna spacing
%   f      - Carrier frequency
%   r0     - Distance to the user
%   theta0 - Angle of arrival
%
% Output:
%   at     - The [Nt x 1] complex steering vector

    c = 3e8;
    nn = (-(Nt-1)/2 : (Nt-1)/2)'; % Column vector for antenna indices
    
    % Fresnel approximation for near-field distance to each antenna element
    r = r0 - nn * d * sin(theta0) + nn.^2 .* d.^2 .* cos(theta0).^2 / (2 * r0);
    
    % Calculate the complex phase shift for each antenna
    at = exp(-1j * 2 * pi * f * (r - r0) / c) / sqrt(Nt);
end