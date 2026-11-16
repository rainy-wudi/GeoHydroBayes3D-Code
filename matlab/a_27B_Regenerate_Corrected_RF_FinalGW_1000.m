%% =========================================================================
%% a_27B_Regenerate_Corrected_RF_FinalGW_1000.m
%% =========================================================================
% PURPOSE
% -------
% Rebuild the FINAL 50 x 20 = 1000 Abaqus INPs after correcting the
% random-field generator.
%
% SCIENTIFIC LOCKS
% ----------------
% 1) DO NOT resample the 50 macroparameter vectors.
%    This script reads the exact existing 50 groups from:
%       Final_ML_Dataset_FinalGW.csv
%
% 2) Use corrected generator:
%       generate_random_field_optimized_v2.m
%
% 3) Use deterministic explicit RF seeds for every P/R case.
%
% 4) Keep the finalized groundwater boundary:
%       downstream Hd = 451 m
%       upstream south Hs = 520 m
%       upstream north Hn = 560 m
%
% 5) Keep exact production discretization:
%       rf_sample_points = [20 20 25]
%       n_bins = 100
%
% 6) Never overwrite the old FinalGW database.
%
% HOW TO USE
% ----------
% First:
%       MODE = 'SMOKE';
% Run this script and then run the generated smoke BAT.
%
% Only after the smoke case is normal:
%       MODE = 'SMOKE';
% Run this script again to generate all 1000 corrected INPs.
%% =========================================================================

clc;
clear;


fprintf('\n');
fprintf('================================================================================\n');
fprintf(' STEP 27 - REGENERATE CORRECTED-RF FINALGW DATABASE\n');
fprintf('================================================================================\n');

%% =========================================================================
%% 1. USER MODE
%% =========================================================================

% FIRST RUN: leave as 'SMOKE'
% SECOND RUN, only after smoke passes: change to 'ALL'
MODE = 'SMOKE';

if ~ismember(MODE, {'SMOKE','ALL'})
    error('MODE must be either SMOKE or ALL.');
end

%% =========================================================================
%% 2. Paths
%% =========================================================================

CODE_DIR = fileparts(mfilename('fullpath'));
MODEL_DIR = fullfile(CODE_DIR, 'inputs');
addpath(CODE_DIR);

OLD_FINAL_DATASET = ...
    fullfile(CODE_DIR, '..', 'analysis', 'Final_ML_Dataset_CorrectedRF_FinalGW.csv');

ROOT_OUT = fullfile(...
    MODEL_DIR, ...
    'Corrected_RF_FinalGW_1000');

SMOKE_DIR = fullfile(...
    ROOT_OUT, ...
    'SMOKE');

ALL_INP_DIR = fullfile(...
    ROOT_OUT, ...
    'INP');

TEMPLATE_INP = fullfile(...
    MODEL_DIR, ...
    'Job-0.inp');

PTS_TOP_FILE = fullfile(...
    MODEL_DIR, ...
    'pts_tu1_top.txt');

PTS_MID_FILE = fullfile(...
    MODEL_DIR, ...
    'pts_tu2_mid.txt');

PTS_BOT_FILE = fullfile(...
    MODEL_DIR, ...
    'pts_tu3_bot.txt');

if ~exist(ROOT_OUT, 'dir')
    mkdir(ROOT_OUT);
end

if ~exist(SMOKE_DIR, 'dir')
    mkdir(SMOKE_DIR);
end

if ~exist(ALL_INP_DIR, 'dir')
    mkdir(ALL_INP_DIR);
end

addpath(MODEL_DIR);

%% =========================================================================
%% 3. Fixed settings
%% =========================================================================

N_PARAM = 50;
N_REAL = 20;
N_LAYERS = 3;

RF_SAMPLE_POINTS = [20, 20, 25];
N_BINS = 100;

E_VALUES = [21e6, 21e6, 21e6];
NU_VALUES = [0.25, 0.25, 0.25];

% Explicit reproducible database seed scheme.
% Truth seeds used later are separate from these.
RF_SEED_BASE = 2700000;

% Exact production design support, used only for validation.
PRIOR_LOW = [
    -9.0, -8.0, -7.0, 0.4, 50.0, 1.0
];

PRIOR_HIGH = [
    -7.0, -6.0, -5.0, 1.0, 300.0, 5.0
];

%% =========================================================================
%% 4. Fixed FinalGW boundary
%% =========================================================================

DOWNSTREAM_HEAD = 451.0;
UPSTREAM_SOUTH_HEAD = 520.0;
UPSTREAM_NORTH_HEAD = 560.0;
GAMMA_W = 9810.0;

TOP_Y_MIN = 2320.0;
TOP_X_MAX = 3325.0;

BOTTOM_Y_MAX = 870.0;
BOTTOM_X_MIN = 2380.0;

LEFT_X_MAX = 2450.0;
RIGHT_X_MIN = 3300.0;

%% =========================================================================
%% 5. Required-file/function checks
%% =========================================================================

required_files = {
    TEMPLATE_INP
    PTS_TOP_FILE
    PTS_MID_FILE
    PTS_BOT_FILE
    OLD_FINAL_DATASET
};

for i = 1:numel(required_files)
    if ~exist(required_files{i}, 'file')
        error('Required file not found:\n%s', required_files{i});
    end
end

if exist('read_inp_mesh_fast', 'file') ~= 2
    error('Cannot find read_inp_mesh_fast.m on MATLAB path.');
end

if exist('generate_random_field_optimized_v2', 'file') ~= 2
    error([ ...
        'Cannot find generate_random_field_optimized_v2.m.\n' ...
        'Put the corrected V2 generator on the MATLAB path first.' ...
    ]);
end

fprintf('\nFunction provenance:\n');
fprintf('  read_inp_mesh_fast:\n    %s\n', which('read_inp_mesh_fast'));
fprintf('  corrected RF generator:\n    %s\n', which('generate_random_field_optimized_v2'));
fprintf('  INP writer:\n    SELF-CONTAINED production 100-bin writer\n');

%% =========================================================================
%% 6. Freeze the EXACT original 50 macroparameter groups
%% =========================================================================

fprintf('\nReading exact 50 macroparameter groups from old FinalGW dataset...\n');

T = readtable(...
    OLD_FINAL_DATASET, ...
    'VariableNamingRule', ...
    'preserve');

X_NAMES = {
    'log10_K_Lower'
    'log10_K_Middle'
    'log10_K_Upper'
    'COV'
    'dh'
    'dv'
};

needed = [{'P_index'}; X_NAMES];

for i = 1:numel(needed)
    if ~ismember(needed{i}, T.Properties.VariableNames)
        error('Dataset missing required column: %s', needed{i});
    end
end

% Verify each P group has exactly 20 rows and one unique parameter vector.
P_VALUES = unique(T.P_index);

if numel(P_VALUES) ~= N_PARAM
    error('Expected 50 P groups, found %d.', numel(P_VALUES));
end

param_matrix = zeros(N_PARAM, 6);

for p = 1:N_PARAM

    Tp = T(T.P_index == p, :);

    if height(Tp) ~= N_REAL
        error('P%03d has %d rows, expected 20.', p, height(Tp));
    end

    Xp = [
        Tp.log10_K_Lower, ...
        Tp.log10_K_Middle, ...
        Tp.log10_K_Upper, ...
        Tp.COV, ...
        Tp.dh, ...
        Tp.dv
    ];

    x0 = Xp(1,:);

    if any(max(abs(Xp - x0), [], 1) > 1e-12)
        error('P%03d does not have a single consistent parameter vector.', p);
    end

    param_matrix(p,:) = x0;
end

inside = all(...
    param_matrix >= PRIOR_LOW & ...
    param_matrix <= PRIOR_HIGH, ...
    2);

if ~all(inside)
    error('At least one of the frozen 50 parameter groups lies outside production support.');
end

fprintf('Frozen parameter check: PASS (50 groups x 20 rows)\n');

param_table = array2table(...
    param_matrix, ...
    'VariableNames', X_NAMES);

param_table.P_index = (1:N_PARAM)';
param_table = movevars(param_table, 'P_index', 'Before', 1);

writetable(...
    param_table, ...
    fullfile(ROOT_OUT, 'Frozen_50_Macroparameter_Groups.csv'));

%% =========================================================================
%% 7. Rebuild exact geometry / layer assignment
%% =========================================================================

fprintf('\nRebuilding production geometry and KNN layer assignment...\n');

[node, elelist, element, etype] = read_inp_mesh_fast(TEMPLATE_INP);

element_centers = compute_element_centers_fast_local(...
    node, ...
    element);

pts_top = load(PTS_TOP_FILE);
pts_mid = load(PTS_MID_FILE);
pts_bot = load(PTS_BOT_FILE);

all_reference_pts = [
    pts_bot
    pts_mid
    pts_top
];

all_labels = [
    ones(size(pts_bot,1),1)
    2*ones(size(pts_mid,1),1)
    3*ones(size(pts_top,1),1)
];

idx = knnsearch(...
    all_reference_pts, ...
    element_centers);

layer_indices = all_labels(idx);

fprintf('Mesh summary:\n');
fprintf('  element type : %s\n', etype);
fprintf('  elements     : %d\n', numel(elelist));
fprintf('  Layer 1 Clay : %d\n', sum(layer_indices == 1));
fprintf('  Layer 2 Silt : %d\n', sum(layer_indices == 2));
fprintf('  Layer 3 Sand : %d\n', sum(layer_indices == 3));

if numel(elelist) ~= 58573
    warning('Element count differs from expected 58573.');
end

%% =========================================================================
%% 8. Select jobs
%% =========================================================================

if strcmp(MODE, 'SMOKE')

    jobs = [1, 1];

    output_dir = SMOKE_DIR;

    fprintf('\nMODE = SMOKE\n');
    fprintf('Only P001_R01 will be generated.\n');

else

    jobs = zeros(N_PARAM * N_REAL, 2);

    k = 0;

    for p = 1:N_PARAM
        for r = 1:N_REAL
            k = k + 1;
            jobs(k,:) = [p, r];
        end
    end

    output_dir = ALL_INP_DIR;

    fprintf('\nMODE = ALL\n');
    fprintf('All 1000 corrected INPs will be generated.\n');
end

%% =========================================================================
%% 9. Generate corrected fields + FinalGW INPs
%% =========================================================================

summary_rows = cell(size(jobs,1), 1);

total_tic = tic;

for ij = 1:size(jobs,1)

    p = jobs(ij,1);
    r = jobs(ij,2);

    job_name = sprintf('Job_P%03d_R%02d', p, r);

    final_inp = fullfile(...
        output_dir, ...
        [job_name, '.inp']);

    raw_inp = fullfile(...
        output_dir, ...
        [job_name, '_RAW.inp']);

    x = param_matrix(p,:);

    logK = x(1:3);
    mu_vals = 10.^logK;

    cov_val = x(4);
    dh_val = x(5);
    dv_val = x(6);

    % Unique reproducible seed for every P/R pair.
    rf_seed = RF_SEED_BASE + (p - 1) * 100 + r;

    fprintf('\n');
    fprintf('--------------------------------------------------------------------------------\n');
    fprintf('[%d/%d] %s\n', ij, size(jobs,1), job_name);
    fprintf('--------------------------------------------------------------------------------\n');
    fprintf('seed      = %d\n', rf_seed);
    fprintf('log10 K   = [%.6f, %.6f, %.6f]\n', logK(1), logK(2), logK(3));
    fprintf('linear K  = [%.6e, %.6e, %.6e]\n', mu_vals(1), mu_vals(2), mu_vals(3));
    fprintf('COV       = %.6f\n', cov_val);
    fprintf('dh / dv   = %.6f / %.6f m\n', dh_val, dv_val);

    if exist(final_inp, 'file')
        fprintf('Existing corrected INP found; skipping generation: %s\n', final_inp);

        % IMPORTANT:
        % Use the SAME field schema as generated records so struct arrays
        % can always be vertically concatenated during checkpoint/final save.
        rec = struct();

        rec.Job_Name = job_name;
        rec.P_index = p;
        rec.Realization = r;
        rec.RF_Seed = rf_seed;
        rec.Status = 'SKIPPED_EXISTING';

        rec.log10_K_Lower = logK(1);
        rec.log10_K_Middle = logK(2);
        rec.log10_K_Upper = logK(3);

        rec.K_Lower_Target = mu_vals(1);
        rec.K_Middle_Target = mu_vals(2);
        rec.K_Upper_Target = mu_vals(3);

        rec.COV = cov_val;
        rec.dh = dh_val;
        rec.dv = dv_val;

        % Statistics are unavailable when only an existing INP is detected.
        % Keep NaN placeholders so the structure schema remains identical.
        rec.Layer1_K_Mean = NaN;
        rec.Layer2_K_Mean = NaN;
        rec.Layer3_K_Mean = NaN;

        rec.Layer1_K_COV = NaN;
        rec.Layer2_K_COV = NaN;
        rec.Layer3_K_COV = NaN;

        rec.Layer1_Mean_Ratio = NaN;
        rec.Layer2_Mean_Ratio = NaN;
        rec.Layer3_Mean_Ratio = NaN;

        rec.Layer1_K_Min = NaN;
        rec.Layer2_K_Min = NaN;
        rec.Layer3_K_Min = NaN;

        rec.Layer1_K_Max = NaN;
        rec.Layer2_K_Max = NaN;
        rec.Layer3_K_Max = NaN;

        rec.Layer1_Jitter = NaN;
        rec.Layer2_Jitter = NaN;
        rec.Layer3_Jitter = NaN;

        rec.Layer1_ClipLow = NaN;
        rec.Layer2_ClipLow = NaN;
        rec.Layer3_ClipLow = NaN;

        rec.Layer1_ClipHigh = NaN;
        rec.Layer2_ClipHigh = NaN;
        rec.Layer3_ClipHigh = NaN;

        rec.Random_Field_Seconds = NaN;

        rec.Original_BC3_Nodes = NaN;
        rec.Downstream_BC_Nodes = NaN;
        rec.Upstream_BC_Nodes = NaN;
        rec.Skipped_Downstream_Nodes = NaN;
        rec.Skipped_Upstream_Nodes = NaN;
        rec.Non_BC3_Identity_PASS = NaN;

        summary_rows{ij} = rec;
        continue;
    end

    layer_params = cell(N_LAYERS, 1);

    for L = 1:N_LAYERS
        layer_params{L}.mu = mu_vals(L);
        layer_params{L}.cov = cov_val;
        layer_params{L}.sigma = cov_val * mu_vals(L); % metadata compatibility
        layer_params{L}.dh = dh_val;
        layer_params{L}.dv = dv_val;
    end

    rng(rf_seed, 'twister');

    t_rf = tic;

    [K_field, rf_diag] = generate_random_field_optimized_v2(...
        element_centers, ...
        layer_indices, ...
        layer_params, ...
        N_LAYERS, ...
        RF_SAMPLE_POINTS);

    rf_seconds = toc(t_rf);

    if any(~isfinite(K_field)) || any(K_field <= 0)
        error('%s produced invalid K values.', job_name);
    end

    layer_mean = zeros(1,3);
    layer_cov = zeros(1,3);
    layer_min = zeros(1,3);
    layer_max = zeros(1,3);

    fprintf('Actual element-field statistics:\n');

    for L = 1:3

        KL = K_field(layer_indices == L);

        layer_mean(L) = mean(KL);
        layer_cov(L) = std(KL) / mean(KL);
        layer_min(L) = min(KL);
        layer_max(L) = max(KL);

        fprintf(['  Layer %d: target mean=%.6e, actual mean=%.6e, ratio=%.4f, ' ...
                 'target COV=%.4f, actual COV=%.4f, K=[%.3e, %.3e]\n'], ...
            L, mu_vals(L), layer_mean(L), layer_mean(L)/mu_vals(L), ...
            cov_val, layer_cov(L), layer_min(L), layer_max(L));
    end

    fprintf('RF generation time = %.2f s\n', rf_seconds);

    % Save exact smoke field. For ALL mode, reproducibility is provided by
    % frozen parameters + explicit seed + V2 generator; avoid 1000 huge MAT files.
    if strcmp(MODE, 'SMOKE')
        save(...
            fullfile(SMOKE_DIR, [job_name, '_KField_V2.mat']), ...
            'K_field', ...
            'rf_diag', ...
            'layer_indices', ...
            'element_centers', ...
            'elelist', ...
            'layer_params', ...
            'rf_seed', ...
            'RF_SAMPLE_POINTS', ...
            '-v7.3');
    end

    % Exact 100-bin writer embedded locally to avoid old 6-input path conflict.
    generate_inp_prod100_local(...
        TEMPLATE_INP, ...
        raw_inp, ...
        elelist, ...
        K_field, ...
        E_VALUES, ...
        NU_VALUES, ...
        N_BINS);

    [...
        n_down, ...
        n_up, ...
        skip_down, ...
        skip_up, ...
        n_bc3_original, ...
        non_bc_identical ...
    ] = apply_finalgw_bc3(...
        raw_inp, ...
        final_inp, ...
        job_name, ...
        DOWNSTREAM_HEAD, ...
        UPSTREAM_SOUTH_HEAD, ...
        UPSTREAM_NORTH_HEAD, ...
        GAMMA_W, ...
        TOP_Y_MIN, ...
        TOP_X_MAX, ...
        BOTTOM_Y_MAX, ...
        BOTTOM_X_MIN, ...
        LEFT_X_MAX, ...
        RIGHT_X_MIN);

    if ~non_bc_identical
        error('%s failed non-BC3 identity check.', job_name);
    end

    if exist(raw_inp, 'file')
        delete(raw_inp);
    end

    fprintf('FinalGW boundary:\n');
    fprintf('  original BC-3 nodes = %d\n', n_bc3_original);
    fprintf('  downstream applied  = %d\n', n_down);
    fprintf('  upstream applied    = %d\n', n_up);
    fprintf('  non-BC3 identity    = PASS\n');

    rec = struct();

    rec.Job_Name = job_name;
    rec.P_index = p;
    rec.Realization = r;
    rec.RF_Seed = rf_seed;
    rec.Status = 'GENERATED';

    rec.log10_K_Lower = logK(1);
    rec.log10_K_Middle = logK(2);
    rec.log10_K_Upper = logK(3);

    rec.K_Lower_Target = mu_vals(1);
    rec.K_Middle_Target = mu_vals(2);
    rec.K_Upper_Target = mu_vals(3);

    rec.COV = cov_val;
    rec.dh = dh_val;
    rec.dv = dv_val;

    rec.Layer1_K_Mean = layer_mean(1);
    rec.Layer2_K_Mean = layer_mean(2);
    rec.Layer3_K_Mean = layer_mean(3);

    rec.Layer1_K_COV = layer_cov(1);
    rec.Layer2_K_COV = layer_cov(2);
    rec.Layer3_K_COV = layer_cov(3);

    rec.Layer1_Mean_Ratio = layer_mean(1) / mu_vals(1);
    rec.Layer2_Mean_Ratio = layer_mean(2) / mu_vals(2);
    rec.Layer3_Mean_Ratio = layer_mean(3) / mu_vals(3);

    rec.Layer1_K_Min = layer_min(1);
    rec.Layer2_K_Min = layer_min(2);
    rec.Layer3_K_Min = layer_min(3);

    rec.Layer1_K_Max = layer_max(1);
    rec.Layer2_K_Max = layer_max(2);
    rec.Layer3_K_Max = layer_max(3);

    rec.Layer1_Jitter = rf_diag(1).jitter_used;
    rec.Layer2_Jitter = rf_diag(2).jitter_used;
    rec.Layer3_Jitter = rf_diag(3).jitter_used;

    rec.Layer1_ClipLow = rf_diag(1).n_clipped_low;
    rec.Layer2_ClipLow = rf_diag(2).n_clipped_low;
    rec.Layer3_ClipLow = rf_diag(3).n_clipped_low;

    rec.Layer1_ClipHigh = rf_diag(1).n_clipped_high;
    rec.Layer2_ClipHigh = rf_diag(2).n_clipped_high;
    rec.Layer3_ClipHigh = rf_diag(3).n_clipped_high;

    rec.Random_Field_Seconds = rf_seconds;

    rec.Original_BC3_Nodes = n_bc3_original;
    rec.Downstream_BC_Nodes = n_down;
    rec.Upstream_BC_Nodes = n_up;
    rec.Skipped_Downstream_Nodes = skip_down;
    rec.Skipped_Upstream_Nodes = skip_up;
    rec.Non_BC3_Identity_PASS = non_bc_identical;

    summary_rows{ij} = rec;

    % Periodically checkpoint ALL-mode summary to reduce risk of loss.
    if strcmp(MODE, 'ALL') && (mod(ij,20) == 0 || ij == size(jobs,1))
        tmp_rows = summary_rows(~cellfun(@isempty, summary_rows));

        % All records use the same schema. orderfields adds one more
        % safeguard against field-order differences.
        for kk = 2:numel(tmp_rows)
            tmp_rows{kk} = orderfields(tmp_rows{kk}, tmp_rows{1});
        end

        tmp_struct = vertcat(tmp_rows{:});
        tmp_table = struct2table(tmp_struct);

        writetable(...
            tmp_table, ...
            fullfile(ROOT_OUT, 'Corrected_RF_Generation_Summary_CHECKPOINT.csv'));
    end
end

%% =========================================================================
%% 10. Save summaries / metadata
%% =========================================================================

valid_rows = summary_rows(~cellfun(@isempty, summary_rows));

for kk = 2:numel(valid_rows)
    valid_rows{kk} = orderfields(valid_rows{kk}, valid_rows{1});
end

summary_struct = vertcat(valid_rows{:});
summary_table = struct2table(summary_struct);

if strcmp(MODE, 'SMOKE')
    summary_csv = fullfile(...
        SMOKE_DIR, ...
        'Smoke_Generation_Summary.csv');
else
    summary_csv = fullfile(...
        ROOT_OUT, ...
        'Corrected_RF_Generation_Summary_1000.csv');
end

writetable(...
    summary_table, ...
    summary_csv);

metadata_path = fullfile(...
    ROOT_OUT, ...
    'Corrected_RF_Database_Metadata.mat');

save(...
    metadata_path, ...
    'param_matrix', ...
    'param_table', ...
    'RF_SEED_BASE', ...
    'RF_SAMPLE_POINTS', ...
    'N_BINS', ...
    'layer_indices', ...
    'element_centers', ...
    'DOWNSTREAM_HEAD', ...
    'UPSTREAM_SOUTH_HEAD', ...
    'UPSTREAM_NORTH_HEAD', ...
    'GAMMA_W', ...
    '-v7.3');

%% =========================================================================
%% 11. Generate Abaqus batch files
%% =========================================================================

if strcmp(MODE, 'SMOKE')

    bat_path = fullfile(...
        SMOKE_DIR, ...
        'Run_Smoke_P001_R01.bat');

    fid = fopen(bat_path, 'w');

    fprintf(fid, '@echo off\n');
    fprintf(fid, 'cd /d "%s"\n', SMOKE_DIR);
    fprintf(fid, 'echo ============================================================\n');
    fprintf(fid, 'echo CORRECTED RF SMOKE: Job_P001_R01\n');
    fprintf(fid, 'echo ============================================================\n');
    fprintf(fid, 'call abaqus job=Job_P001_R01 input=Job_P001_R01.inp cpus=4 interactive\n');
    fprintf(fid, 'echo.\n');
    fprintf(fid, 'echo Check .odb / .sta / .msg / .dat before MODE=ALL.\n');
    fprintf(fid, 'pause\n');

    fclose(fid);

else

    bat_path = fullfile(...
        ROOT_OUT, ...
        'Run_All_1000_Corrected_RF.bat');

    fid = fopen(bat_path, 'w');

    fprintf(fid, '@echo off\n');
    fprintf(fid, 'cd /d "%s"\n', ALL_INP_DIR);
    fprintf(fid, 'echo ============================================================\n');
    fprintf(fid, 'echo CORRECTED RF FINALGW - ALL 1000 ABAQUS JOBS\n');
    fprintf(fid, 'echo ============================================================\n');

    for p = 1:N_PARAM
        for r = 1:N_REAL

            job_name = sprintf('Job_P%03d_R%02d', p, r);

            fprintf(fid, 'echo.\n');
            fprintf(fid, 'if exist "%s.odb" (\n', job_name);
            fprintf(fid, '  echo SKIP %s - ODB exists\n', job_name);
            fprintf(fid, ') else (\n');
            fprintf(fid, '  echo RUN %s\n', job_name);
            fprintf(fid, '  call abaqus job=%s input=%s.inp cpus=4 interactive\n', ...
                job_name, job_name);
            fprintf(fid, ')\n');
        end
    end

    fprintf(fid, 'echo.\n');
    fprintf(fid, 'echo ALL 1000 JOBS FINISHED\n');
    fprintf(fid, 'pause\n');

    fclose(fid);
end

%% =========================================================================
%% 12. Final report
%% =========================================================================

fprintf('\n');
fprintf('================================================================================\n');
fprintf(' STEP 27 COMPLETE (%s)\n', MODE);
fprintf('================================================================================\n');

fprintf('Generation summary:\n  %s\n', summary_csv);
fprintf('Metadata:\n  %s\n', metadata_path);
fprintf('Elapsed generation time: %.2f min\n', toc(total_tic)/60);

if strcmp(MODE, 'SMOKE')

    fprintf('\nNEXT ACTION ONLY:\n');
    fprintf('  Run: %s\n', fullfile(SMOKE_DIR, 'Run_Smoke_P001_R01.bat'));
    fprintf('  Confirm the Abaqus ODB solves normally.\n');
    fprintf('  Then change MODE from SMOKE to ALL.\n');

else

    fprintf('\n1000 corrected INPs are ready.\n');
    fprintf('Do NOT mix these ODBs with the old FinalGW ODBs.\n');
end


%% =========================================================================
%% Local helper 1: compute element centers
%% =========================================================================

function centers = compute_element_centers_fast_local(...
    node, ...
    element)

n_elem = length(element);

centers = zeros(...
    n_elem, ...
    3);

node_map = containers.Map(...
    node(:, 1), ...
    1:size(node, 1));

for i = 1:n_elem

    conn = element{i};

    coords = zeros(...
        length(conn), ...
        3);

    for j = 1:length(conn)

        if isKey(...
                node_map, ...
                conn(j))

            idx = node_map(...
                conn(j));

            coords(j, :) = node(...
                idx, ...
                2:4);
        else

            error(...
                'Element %d references missing node %d.', ...
                i, ...
                conn(j));
        end
    end

    centers(i, :) = mean(...
        coords, ...
        1);
end

end

%% =========================================================================
%% Local helper 2: apply finalized FinalGW BC-3
%% =========================================================================

function [...
    n_down, ...
    n_up, ...
    skip_down, ...
    skip_up, ...
    n_bc3_original, ...
    non_bc_identical ...
] = apply_finalgw_bc3(...
    input_inp, ...
    output_inp, ...
    case_label, ...
    downstream_head, ...
    upstream_south_head, ...
    upstream_north_head, ...
    gamma_w, ...
    top_y_min, ...
    top_x_max, ...
    bottom_y_max, ...
    bottom_x_min, ...
    left_x_max, ...
    right_x_min)

lines = read_text_lines_local(...
    input_inp);

nodes = read_diceng_nodes_local(...
    lines);

[...
    bc_start, ...
    bc_end ...
] = find_bc3_block_local(...
    lines);

bc_ids = extract_bc3_node_ids_local(...
    lines, ...
    bc_start, ...
    bc_end);

n_bc3_original = numel(...
    bc_ids);

n = n_bc3_original;

xs = zeros(n, 1);
ys = zeros(n, 1);
zs = zeros(n, 1);

types = cell(n, 1);

for i = 1:n

    node_id = bc_ids(i);

    if ~isKey(...
            nodes, ...
            node_id)

        error(...
            'BC-3 node %d was not found in Part diceng.', ...
            node_id);
    end

    xyz = nodes(...
        node_id);

    x = xyz(1);
    y = xyz(2);
    z = xyz(3);

    xs(i) = x;
    ys(i) = y;
    zs(i) = z;

    types{i} = classify_boundary_local(...
        x, ...
        y, ...
        top_y_min, ...
        top_x_max, ...
        bottom_y_max, ...
        bottom_x_min, ...
        left_x_max, ...
        right_x_min);
end

other_mask = strcmp(...
    types, ...
    'OTHER');

if any(other_mask)

    error(...
        '%s has %d BC-3 nodes classified as OTHER. Check thresholds.', ...
        case_label, ...
        sum(other_mask));
end

up_mask = strcmp(...
    types, ...
    'UPSTREAM');

if ~any(up_mask)

    error(...
        '%s contains no UPSTREAM BC-3 nodes.', ...
        case_label);
end

y_up = ys(...
    up_mask);

y_min = min(...
    y_up);

y_max = max(...
    y_up);

new_bc = {};

new_bc{end+1, 1} = ...
    '** Name: BC-3 Type: corrected-random-field FinalGW groundwater boundary';

new_bc{end+1, 1} = sprintf(...
    '** Corrected RF case = %s', ...
    case_label);

new_bc{end+1, 1} = sprintf(...
    '** Downstream head = %.2f m', ...
    downstream_head);

new_bc{end+1, 1} = sprintf(...
    '** Upstream South head = %.2f m', ...
    upstream_south_head);

new_bc{end+1, 1} = sprintf(...
    '** Upstream North head = %.2f m', ...
    upstream_north_head);

new_bc{end+1, 1} = sprintf(...
    '** gamma_w = %.2f N/m3', ...
    gamma_w);

new_bc{end+1, 1} = ...
    '*Boundary';

n_down = 0;
n_up = 0;

skip_down = 0;
skip_up = 0;

for i = 1:n

    node_id = bc_ids(i);

    x = xs(i); %#ok<NASGU>
    y = ys(i);
    z = zs(i);

    btype = types{i};

    if strcmp(...
            btype, ...
            'DOWNSTREAM')

        h = downstream_head;

        if z > h

            skip_down = skip_down + 1;

            continue;
        end

        n_down = n_down + 1;

    elseif strcmp(...
            btype, ...
            'UPSTREAM')

        if y_max == y_min

            ratio = 0.5;

        else

            ratio = ...
                (y - y_min) / ...
                (y_max - y_min);
        end

        ratio = max(...
            0.0, ...
            min(...
                1.0, ...
                ratio));

        h = ...
            upstream_south_head + ...
            ratio * ...
            (...
                upstream_north_head - ...
                upstream_south_head);

        if z > h

            skip_up = skip_up + 1;

            continue;
        end

        n_up = n_up + 1;

    else

        % NOFLOW_TOP or NOFLOW_BOTTOM:
        % no pore-pressure entry is written.
        continue;
    end

    p = max(...
        0.0, ...
        gamma_w * ...
        (h - z));

    new_bc{end+1, 1} = sprintf(...
        'diceng-1.%d, 8, 8, %.6e', ...
        node_id, ...
        p);
end

new_lines = [
    lines(1:bc_start-1)
    new_bc
    lines(bc_end:end)
];

write_text_lines_local(...
    output_inp, ...
    new_lines);

% -------------------------------------------------------------------------
% Critical integrity check:
% after removing the BC-3 block, raw and FinalGW INPs must be identical.
% -------------------------------------------------------------------------
raw_stripped = strip_bc3_local(...
    lines);

final_lines = read_text_lines_local(...
    output_inp);

final_stripped = strip_bc3_local(...
    final_lines);

non_bc_identical = isequal(...
    raw_stripped, ...
    final_stripped);

end

%% =========================================================================
%% Local helper 3: classify boundary
%% =========================================================================

function btype = classify_boundary_local(...
    x, ...
    y, ...
    top_y_min, ...
    top_x_max, ...
    bottom_y_max, ...
    bottom_x_min, ...
    left_x_max, ...
    right_x_min)

if ...
        y >= top_y_min && ...
        x <= top_x_max

    btype = 'NOFLOW_TOP';

elseif ...
        y <= bottom_y_max && ...
        x >= bottom_x_min

    btype = 'NOFLOW_BOTTOM';

elseif ...
        x <= left_x_max

    btype = 'DOWNSTREAM';

elseif ...
        x >= right_x_min

    btype = 'UPSTREAM';

else

    btype = 'OTHER';
end

end

%% =========================================================================
%% Local helper 4: read diceng nodes
%% =========================================================================

function nodes = read_diceng_nodes_local(...
    lines)

nodes = containers.Map(...
    'KeyType', ...
    'double', ...
    'ValueType', ...
    'any');

inside_diceng = false;
inside_nodes = false;

for i = 1:numel(lines)

    s = strtrim(...
        lines{i});

    s_lower = lower(...
        s);

    if startsWith(...
            s_lower, ...
            '*part, name=diceng')

        inside_diceng = true;

        continue;
    end

    if inside_diceng

        if strcmp(...
                s_lower, ...
                '*node')

            inside_nodes = true;

            continue;
        end

        if inside_nodes

            if startsWith(...
                    s, ...
                    '*')

                break;
            end

            parts = strsplit(...
                s, ...
                ',');

            if numel(parts) >= 4

                node_id = str2double(...
                    strtrim(parts{1}));

                x = str2double(...
                    strtrim(parts{2}));

                y = str2double(...
                    strtrim(parts{3}));

                z = str2double(...
                    strtrim(parts{4}));

                if all(...
                        isfinite(...
                            [node_id, x, y, z]))

                    nodes(node_id) = [
                        x, ...
                        y, ...
                        z
                    ];
                end
            end
        end
    end
end

if nodes.Count == 0

    error(...
        'No nodes were found inside Part diceng.');
end

end

%% =========================================================================
%% Local helper 5: locate BC-3
%% =========================================================================

function [...
    bc_start, ...
    bc_end ...
] = find_bc3_block_local(...
    lines)

bc_start = [];

for i = 1:numel(lines)

    if contains(...
            lines{i}, ...
            '** Name: BC-3')

        bc_start = i;

        break;
    end
end

if isempty(...
        bc_start)

    error(...
        'BC-3 was not found in the generated INP.');
end

bc_end = [];

for i = bc_start+1:numel(lines)

    s = strtrim(...
        lines{i});

    if ...
            (startsWith(s, '** Name:') && i > bc_start + 1) || ...
            startsWith(lower(s), '*end step')

        bc_end = i;

        break;
    end
end

if isempty(...
        bc_end)

    error(...
        'Could not determine the end of BC-3.');
end

end

%% =========================================================================
%% Local helper 6: extract node IDs from BC-3
%% =========================================================================

function ids = extract_bc3_node_ids_local(...
    lines, ...
    bc_start, ...
    bc_end)

ids = [];

pattern = ...
    'diceng-1\.(\d+)\s*,\s*8\s*,\s*8\s*,';

for i = bc_start:bc_end-1

    token = regexp(...
        lines{i}, ...
        pattern, ...
        'tokens', ...
        'once');

    if ~isempty(token)

        ids(end+1, 1) = str2double(...
            token{1}); %#ok<AGROW>
    end
end

ids = unique(...
    ids, ...
    'sorted');

if isempty(...
        ids)

    error(...
        'No BC-3 node IDs were extracted.');
end

end

%% =========================================================================
%% Local helper 7: strip BC-3 for integrity comparison
%% =========================================================================

function stripped = strip_bc3_local(...
    lines)

[...
    bc_start, ...
    bc_end ...
] = find_bc3_block_local(...
    lines);

stripped = [
    lines(1:bc_start-1)
    lines(bc_end:end)
];

end

%% =========================================================================
%% Local helper 8: text I/O
%% =========================================================================

function lines = read_text_lines_local(...
    path)

fid = fopen(...
    path, ...
    'r');

if fid == -1

    error(...
        'Cannot open file:\n%s', ...
        path);
end

cleanup_obj = onCleanup(...
    @() fclose(fid));

lines = {};

while ~feof(fid)

    line = fgetl(fid);

    if ischar(line)

        lines{end+1, 1} = line; %#ok<AGROW>
    end
end

clear cleanup_obj;

end

function write_text_lines_local(...
    path, ...
    lines)

fid = fopen(...
    path, ...
    'w');

if fid == -1

    error(...
        'Cannot write file:\n%s', ...
        path);
end

cleanup_obj = onCleanup(...
    @() fclose(fid));

for i = 1:numel(lines)

    fprintf(...
        fid, ...
        '%s\n', ...
        lines{i});
end

clear cleanup_obj;

end


%% =========================================================================
%% Local helper 9: exact production 100-bin INP writer
%% =========================================================================
function generate_inp_prod100_local(...
    template_file, output_file, elelist, K_values, E_vals, nu_vals, n_bins)

if nargin < 7
    n_bins = 100;
end

[elelist, ia] = unique(elelist, 'stable');
K_values = K_values(ia);

min_logK = log10(min(K_values));
max_logK = log10(max(K_values));

log_edges = linspace(min_logK - 0.01, max_logK + 0.01, n_bins + 1);
bin_centers = 10.^(0.5 * (log_edges(1:end-1) + log_edges(2:end)));

bin_indices = discretize(K_values, 10.^log_edges);
bin_indices(K_values <= 10^log_edges(1)) = 1;
bin_indices(K_values >= 10^log_edges(end)) = n_bins;
bin_indices(isnan(bin_indices)) = 1;

unique_bins = unique(bin_indices);

fprintf('        Production K bins actually used: %d / %d\n', ...
    length(unique_bins), n_bins);

t_lines = readlines(template_file);

part_idx = find(contains(upper(t_lines), '*END PART'), 1, 'first');
assy_idx = find(contains(upper(t_lines), '*END ASSEMBLY'), 1, 'first');

if isempty(part_idx)
    error('Production writer: *END PART not found.');
end

if isempty(assy_idx)
    error('Production writer: *END ASSEMBLY not found.');
end

fid = fopen(output_file, 'w');

if fid == -1
    error('Cannot create INP:\n%s', output_file);
end

cleanup_obj = onCleanup(@() fclose(fid));

for i = 1:part_idx-1
    fprintf(fid, '%s\n', t_lines(i));
end

fprintf(fid, '** RANDOM FIELD SECTIONS (Production 100-bin Direct Write)\n');

for i = 1:length(unique_bins)

    b_id = unique_bins(i);
    curr_ids = elelist(bin_indices == b_id);

    fprintf(fid, '*Elset, elset=RF_ELSET_%03d\n', b_id);

    for j = 1:16:length(curr_ids)

        chunk = curr_ids(j:min(j+15, end));
        spec = repmat('%d, ', 1, length(chunk));

        fprintf(fid, [spec(1:end-2), '\n'], chunk);
    end

    fprintf(fid, ...
        '*Solid Section, elset=RF_ELSET_%03d, material=RF_MAT_%03d\n 1.,\n', ...
        b_id, b_id);
end

for i = part_idx:assy_idx
    fprintf(fid, '%s\n', t_lines(i));
end

fprintf(fid, '** RANDOM FIELD MATERIALS (Production 100-bin Direct Write)\n');

for i = 1:length(unique_bins)

    b_id = unique_bins(i);

    fprintf(fid, ...
        '*Material, name=RF_MAT_%03d\n*Density\n 2000.,\n*Elastic\n %.6e, %.4f\n', ...
        b_id, E_vals(1), nu_vals(1));

    fprintf(fid, ...
        '*Permeability, specific=9810.0\n %.6e,\n', ...
        bin_centers(b_id));
end

for i = assy_idx+1:length(t_lines)
    fprintf(fid, '%s\n', t_lines(i));
end

clear cleanup_obj;

end
