function real_array = C2R(complex_array)
    % FORCE stack along 4th dimension.
    % This ensures that even if input is squeezed to H x Nr (N=1),
    % we get H x Nr x 1 x 2
    real_array = cat(4, real(complex_array), imag(complex_array));
end