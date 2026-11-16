function [K_field, diagnostics] = generate_random_field_optimized_v2( ...
    element_centers, layer_indices, layer_params, n_layers, n_sample_points)
% GENERATE_RANDOM_FIELD_OPTIMIZED_V2
% Scientifically corrected version of the production Scheme-C random-field generator.
%
% Fixes two issues in the old implementation:
%
% 1) Numerical stabilization is applied to the DIMENSIONLESS correlation
%    matrix, not as an absolute 1e-10 addition to the covariance matrix.
%
%       OLD: C = sigma^2 * R + 1e-10 * I
%       NEW: R_reg = R + jitter_rel * I
%
%    This prevents the numerical diagonal term from overwhelming the
%    physical variance when K is ~1e-9 to 1e-5 m/s.
%
% 2) If layer_params{i}.cov is the arithmetic coefficient of variation
%    of lognormal K, the exact lognormal mapping is used:
%
%       sigma_ln = sqrt(log(1 + COV^2))
%       mu_ln    = log(mean_K) - 0.5*sigma_ln^2
%       K        = exp(mu_ln + sigma_ln * G)
%
%    Therefore E[K] = mean_K and std(K)/E[K] = COV exactly in the
%    population model.
%
% Spatial correlation is intentionally kept identical to the original
% production implementation:
%
%       rho = exp(-h_dist/dh - |dz|/dv)
%
% Inputs:
%   element_centers : nElem x 3
%   layer_indices   : nElem x 1 values 1..n_layers
%   layer_params    : cell array; each layer has .mu, .cov, .dh, .dv
%   n_layers        : number of layers
%   n_sample_points : [nx ny nz]
%
% Outputs:
%   K_field         : nElem x 1 permeability field
%   diagnostics     : per-layer diagnostics
%
% IMPORTANT:
%   Set rng(seed,'twister') in the calling script before this function
%   if a reproducible realization is required.

n_elem = size(element_centers, 1);
K_field = zeros(n_elem, 1);

diagnostics = repmat(struct( ...
    'layer', [], ...
    'n_elements', [], ...
    'n_samples', [], ...
    'target_mean', [], ...
    'target_cov', [], ...
    'dh', [], ...
    'dv', [], ...
    'jitter_used', [], ...
    'used_cholesky', [], ...
    'K_min', [], ...
    'K_max', [], ...
    'K_mean_elements', [], ...
    'K_cov_elements', [], ...
    'n_clipped_low', [], ...
    'n_clipped_high', []), n_layers, 1);

x_min = min(element_centers(:,1));
x_max = max(element_centers(:,1));
y_min = min(element_centers(:,2));
y_max = max(element_centers(:,2));

fprintf('        V2 random-field generator\n');
fprintf('        Element-center X: [%.2f, %.2f]\n', x_min, x_max);
fprintf('        Element-center Y: [%.2f, %.2f]\n', y_min, y_max);

for i_layer = 1:n_layers

    mask = (layer_indices == i_layer);
    n_elem_layer = sum(mask);

    if n_elem_layer == 0
        warning('Layer %d has no elements.', i_layer);
        continue;
    end

    layer_centers = element_centers(mask, :);
    p = layer_params{i_layer};

    if p.mu <= 0
        error('Layer %d: mean K must be positive.', i_layer);
    end
    if p.cov < 0
        error('Layer %d: COV must be non-negative.', i_layer);
    end
    if p.dh <= 0 || p.dv <= 0
        error('Layer %d: dh and dv must be positive.', i_layer);
    end

    layer_z_min = min(layer_centers(:,3)) - 2;
    layer_z_max = max(layer_centers(:,3)) + 2;

    x_grid = linspace(x_min - 1, x_max + 1, n_sample_points(1));
    y_grid = linspace(y_min - 1, y_max + 1, n_sample_points(2));
    z_grid = linspace(layer_z_min, layer_z_max, n_sample_points(3));

    [X, Y, Z] = meshgrid(x_grid, y_grid, z_grid);

    sample_coords = [X(:), Y(:), Z(:)];
    n_samples = size(sample_coords, 1);

    fprintf('        Layer %d: %d elements, %d RF sample points\n', ...
        i_layer, n_elem_layer, n_samples);

    % ---------------------------------------------------------------------
    % Dimensionless correlation matrix.
    % ---------------------------------------------------------------------
    R = zeros(n_samples, n_samples);

    for i = 1:n_samples
        for j = i:n_samples
            dx = sample_coords(i,1) - sample_coords(j,1);
            dy = sample_coords(i,2) - sample_coords(j,2);
            dz = sample_coords(i,3) - sample_coords(j,3);

            h_dist = sqrt(dx^2 + dy^2);
            v_dist = abs(dz);

            rho = exp(-h_dist/p.dh - v_dist/p.dv);

            R(i,j) = rho;
            R(j,i) = rho;
        end
    end

    % Enforce exact symmetry before factorization.
    R = 0.5 * (R + R');

    % ---------------------------------------------------------------------
    % Relative numerical jitter on correlation scale.
    % Start tiny and increase only if Cholesky requires it.
    % ---------------------------------------------------------------------
    jitter_candidates = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7];
    use_chol = false;
    jitter_used = NaN;
    L = [];

    for jtry = 1:length(jitter_candidates)
        jitter_rel = jitter_candidates(jtry);

        [L_try, chol_flag] = chol( ...
            R + eye(n_samples) * jitter_rel, ...
            'lower');

        if chol_flag == 0
            L = L_try;
            use_chol = true;
            jitter_used = jitter_rel;
            break;
        end
    end

    if ~use_chol
        warning('Layer %d: Cholesky failed; using eigenvalue decomposition.', i_layer);

        [V, D] = eig(R);
        eigvals = real(diag(D));

        tol = max(abs(eigvals)) * 1e-12;
        eigvals(eigvals < tol) = 0;

        L = real(V) * diag(sqrt(eigvals));
        jitter_used = 0;
    end

    % Standard normal spatial field G has approximately unit marginal variance.
    xi = randn(n_samples, 1);
    G = L * xi;

    % ---------------------------------------------------------------------
    % Exact lognormal mapping from arithmetic mean + arithmetic COV.
    % ---------------------------------------------------------------------
    sigma_ln = sqrt(log(1 + p.cov^2));
    mu_ln = log(p.mu) - 0.5 * sigma_ln^2;

    log_K_samples = mu_ln + sigma_ln * G;
    K_samples = exp(log_K_samples);

    % Very broad physical safety bounds. Under a correct generator these
    % should almost never activate; counts are reported rather than hidden.
    low_bound = 1e-15;
    high_bound = 1e-2;

    n_clipped_low = sum(K_samples < low_bound);
    n_clipped_high = sum(K_samples > high_bound);

    K_samples(K_samples < low_bound) = low_bound;
    K_samples(K_samples > high_bound) = high_bound;

    if any(~isfinite(K_samples))
        error('Layer %d generated NaN/Inf K samples.', i_layer);
    end

    % ---------------------------------------------------------------------
    % Interpolate to element centers exactly as in the old production method.
    % ---------------------------------------------------------------------
    K_grid = reshape(K_samples, ...
        [n_sample_points(2), n_sample_points(1), n_sample_points(3)]);

    K_interp = interp3( ...
        X, Y, Z, K_grid, ...
        layer_centers(:,1), ...
        layer_centers(:,2), ...
        layer_centers(:,3), ...
        'linear', p.mu);

    if any(~isfinite(K_interp)) || any(K_interp <= 0)
        error('Layer %d produced invalid interpolated K.', i_layer);
    end

    K_field(mask) = K_interp;

    diagnostics(i_layer).layer = i_layer;
    diagnostics(i_layer).n_elements = n_elem_layer;
    diagnostics(i_layer).n_samples = n_samples;
    diagnostics(i_layer).target_mean = p.mu;
    diagnostics(i_layer).target_cov = p.cov;
    diagnostics(i_layer).dh = p.dh;
    diagnostics(i_layer).dv = p.dv;
    diagnostics(i_layer).jitter_used = jitter_used;
    diagnostics(i_layer).used_cholesky = use_chol;
    diagnostics(i_layer).K_min = min(K_interp);
    diagnostics(i_layer).K_max = max(K_interp);
    diagnostics(i_layer).K_mean_elements = mean(K_interp);
    diagnostics(i_layer).K_cov_elements = std(K_interp) / mean(K_interp);
    diagnostics(i_layer).n_clipped_low = n_clipped_low;
    diagnostics(i_layer).n_clipped_high = n_clipped_high;

    fprintf(['          target mean=%.3e, target COV=%.3f, ' ...
             'element mean=%.3e, element COV=%.3f\n'], ...
        p.mu, p.cov, mean(K_interp), std(K_interp)/mean(K_interp));

    fprintf('          relative jitter=%.1e, clipped low/high=%d/%d\n', ...
        jitter_used, n_clipped_low, n_clipped_high);
end

if any(~isfinite(K_field)) || any(K_field <= 0)
    error('Final K_field contains invalid values.');
end

end
