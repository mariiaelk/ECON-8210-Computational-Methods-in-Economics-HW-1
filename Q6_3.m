% This code performs value function iteration.

%----------------------------------------------------------------
% 0. Housekeeping
%----------------------------------------------------------------

clc
clear
close all

%----------------------------------------------------------------
% 1. Parameters of the model
%----------------------------------------------------------------

% Core parameters
bbeta = 0.97;
oomega_g = 0.2;
cchi = 1;
aalpha = 0.33;
ddelta = 0.1;
ppsi = 0.1;
tau_ss = 0.25;

pars = [bbeta; oomega_g; cchi; aalpha; ddelta; ppsi; tau_ss];
pars_names = {'bbeta', 'oomega_g', 'cchi', 'aalpha', 'ddelta', ...
    'ppsi', 'tau_ss'};

% Parameters of stochastic processes
% Taxes
ttau_val = [0.2; 0.25; 0.3];
ttau_prob = [0.9 0.1 0; ...
             0.05 0.9 0.05; ...
             0 0.1 0.9];
% Productivity
zz_val = [-0.0673; -0.0336; 0; 0.0336; 0.0673];
zz_prob = [0.9727 0.0273 0 0 0; ...
           0.0041 0.9806 0.0153 0 0; ...
           0 0.0082 0.9836 0.0082 0; ...
           0 0 0.0153 0.9806 0.0041; ...
           0 0 0 0.0273 0.9727];

%----------------------------------------------------------------
% 2. Parameters for Value Function Iteration
%----------------------------------------------------------------

% Number of grid points
kk_num = 100; % Capital
ii_num = 50; % Investment

% Grid coverage: steady state value +-x% of steady state value of capital
kk_cover = 0.3; 
ii_cover = 0.5;

% Tolerance level
tol = 1e-6;

% Maximum number of iterations
max_iter = 5000;  

%----------------------------------------------------------------
% 3. Compute steady state
%----------------------------------------------------------------

ss = Q6_2(pars,pars_names);

%----------------------------------------------------------------
% 3. Value function iteration
%----------------------------------------------------------------

% Create a grid
% Define the range
kk_lower_bound = ss.kk * 0.7; 
kk_upper_bound = ss.kk * 1.3;  
ii_lower_bound = ss.ii - ss.kk * 0.5; 
ii_upper_bound = ss.ii + ss.kk * 0.5; 

% Create the grid
kk_grid = linspace(kk_lower_bound, kk_upper_bound, kk_num)';
ii_grid = linspace(ii_lower_bound, ii_upper_bound, ii_num)';

% Create the matrices for states: kk, ii
% kk (inherited value of individual capital)
kk_matrix = repmat(kk_grid, 1, ii_num);
ii_matrix = repmat(ii_grid, 1, kk_num)';
% Repeat this base_matrix ii_num times horizontally
kk_matrix = repmat(kk_matrix, 1, ii_num);
ii_matrix = repmat(ii_matrix, 1, ii_num);

% Create matrix for ii'
ii_prime_matrix = []; % Initialize the final matrix
% Loop through each value in ii_grid
for i = 1:ii_num
    % Create a k_num by k_num^2 matrix filled with the current value of kk_grid
    temp_matrix = ii_grid(i) * ones(kk_num, ii_num);
    
    % Stack this matrix to the final matrix
    ii_prime_matrix = [ii_prime_matrix temp_matrix];
end

% Initial guess for value function
vv0 = log(ss.cc) - (cchi/2) * (ss.ll)^2;
vv0 = vv0 * ones(kk_num, ii_num);

% Create the 3D matrix by replicating vv0 along the 1st dimension
num_zz = size(zz_val, 1);
vv = repmat(vv0, [1, 1, num_zz]);
vv = permute(vv, [3, 1, 2]); % Rearrange dimensions so v(i,:,:) = vv0

% Initial guess for g_tilda
g_tilda0 = tau_ss*aalpha*(ss.ll^(1-aalpha))*kk_grid.^aalpha;
g_tilda0 = repmat(g_tilda0, 1, ii_num);
% Create the 3D matrix by replicating g_tilda along the 1st dimension
g_tilda = repmat(g_tilda0, [1, 1, num_zz]);
g_tilda = permute(g_tilda, [3, 1, 2]); % Rearrange dimensions
% First dimention is zz
% Second dimention is kk
% Third dimention are pairs (i,i') such that first we go over the grid for
% i for fixed i' = i1, then go over the grid for i for fixed i' = i2, etc.

% Initialize some matrices
% Define matrix names in a cell array
matrixNames = {'g_tilda_matrix','ll_sq_matrix', 'll_matrix', ...
    'cc_matrix', 'uu_cur_matrix','cont_matrix','obj_matrix','vv_new', ...
    'cont_expectation'};

% Loop over each matrix name
for i = 1:length(matrixNames)
    % Initialize each matrix
    temp_matrix0 = repmat(g_tilda0, 1, ii_num);
    temp_matrix = repmat(temp_matrix0, [1, 1, num_zz]);
    eval([matrixNames{i} ' = permute(temp_matrix, [3, 1, 2]);']);
end

% Start iterations
tic;
for iter = 1:max_iter

    % Compute optimal labor for each (z,k,i,i')
    for i = 1:num_zz
        ll_sq_matrix(i,:,:) = (((1-aalpha)*(1-tau_ss)/(cchi*tau_ss*aalpha))* ...
            squeeze(g_tilda_matrix(i,:,:)))./(squeeze(g_tilda_matrix(i,:,:))/(tau_ss*aalpha) - ... 
            ii_prime_matrix - squeeze(g_tilda_matrix(i,:,:)));
        ll_sq_matrix(i,:,:) = max(ll_sq_matrix(i,:,:), 0.1); % Replace negative values
        ll_matrix(i,:,:) = ll_sq_matrix(i,:,:).^(1/2);
    
        % Compute associated consumption values
        cc_matrix(i,:,:) = exp(zz_val(i))*(kk_matrix.^aalpha).* ...
            (squeeze(ll_matrix(i,:,:)).^(1-aalpha)) - ...
            ii_prime_matrix - squeeze(g_tilda_matrix(i,:,:));
        cc_matrix(i,:,:) = max(cc_matrix(i,:,:), 0.1); % Replace negative values
        
        % Current utility
        uu_cur_matrix(i,:,:) = log(cc_matrix(i,:,:)) - ...
            (cchi/2)*(ll_matrix(i,:,:).^2)/(1-tau_ss);
    end
        
    % Compute kk_prime for each (k,i,i')
    kk_prime_matrix = (1-ddelta)*kk_matrix + (1-(ppsi/2)* ...
        (ii_prime_matrix./ii_matrix - 1).^2).*ii_prime_matrix;
    kk_prime_matrix(kk_prime_matrix < 0) = 0.01; % Replace negative values

    % Perform linear interpolation
    % To compute continuation value for each (z',k'(k,i,i'),i')
    for i = 1:num_zz
            % Fix ii'
            for m = 1:ii_num
            % Extract part of kk_prime to be used for evaluation
            start_col = (m-1)*ii_num+1;
            end_col = start_col + ii_num - 1;
            kk_prime_part = kk_prime_matrix(:,start_col:end_col);
    
            % Interpolate to get values of vv
            cont_matrix(i,:,start_col:end_col) = interp1(kk_grid, vv(i,:, m), ...
                kk_prime_part, 'linear', 'extrap');
            end

    end

    
    % Compute expectation over z' for current z, evaluate objective, 
    % choose optimal ii_prime for each zz, kk, ii
    vv_new = zeros(size(vv)); % Initialize
    ii_prime_indices = zeros(size(vv)); % Initialize
    ll = zeros(size(vv)); % Initialize
    g_tilda_new = zeros(size(vv)); % Initialize

    for i = 1:num_zz
        % Fix z
        cont_expectation_temp = zeros(size(kk_prime_matrix)); % Initialize
        for q = 1:num_zz
            cont_expectation_temp = zz_prob(i,q)* ...
                squeeze(cont_matrix(q,:,:)) + cont_expectation_temp;
        end
        cont_expectation(i,:,:) = cont_expectation_temp;

        % Objective
        obj_matrix(i,:,:) = (1-bbeta)*uu_cur_matrix(i,:,:) + ...
                bbeta*cont_expectation(i,:,:);
        
        % Find optimal ii_prime for each zz, kk, ii
        % Fix ii
        for w = 1:ii_num
                % Obtain matrix for this ii
                % Define indices of columns we want to get
                ind = w + (0:(ii_num - 1)) * ii_num;
                obj_matrix_cur = squeeze(obj_matrix(i,:, ind));
                ll_matrix_cur = squeeze(ll_matrix(i,:, ind));
        
                % Find optimal ii_prime
                [vv_new(i,:,w), ii_prime_indices(i,:,w)] = max(obj_matrix_cur, [], 2);
                % Find associated labor
                ll(i,:,w) = ll_matrix_cur(sub2ind([kk_num ii_num], (1:kk_num)', ...
                    squeeze(ii_prime_indices(i,:,w))'));
        
        end

        % Find new g_tilda
        g_tilda_new(i,:,:) = tau_ss*aalpha*exp(zz_val(i))* ...
            (squeeze(ll(i,:,:)).^(1-aalpha)).*kk_matrix(:,1:ii_num).^aalpha;

    end

    % Check if value function and g_tilda converged
    if max(abs(vv_new - vv), [], 'all') < tol && ...
            max(abs(g_tilda_new - g_tilda), [], 'all') < tol
        fprintf('Iteration %d, value converged \n', iter);
        vv = vv_new;
        g_tilda = g_tilda_new;
        break;
        % Convergence of value function achieved
    else
        % Update guess for value function
        if mod(iter, 50) == 0
            fprintf('Iteration %d, value not converged \n', iter);
            err = max(max(abs(vv_new - vv), [], 'all'),...
                max(abs(g_tilda_new - g_tilda), [], 'all'));
            fprintf('Maximum error is %d \n', err);
        end
        
        vv = vv_new;
        g_tilda = g_tilda_new;

    end

end
toc

% Obtain policy functions
% Investment
ii_prime = ii_grid(ii_prime_indices); 
% Consumption
consumption = zeros(size(ii_prime));
for i = 1:num_zz
    consumption(i,:,:) = exp(zz_val(i))*(kk_matrix(:,1:ii_num).^aalpha).* ...
            (squeeze(ll(i,:,:)).^(1-aalpha)) - ...
            squeeze(ii_prime(i,:,:)) - squeeze(g_tilda(i,:,:));
end

% Make plots

if(false)
    
    current_path = pwd;
    adjusted_path = fullfile(current_path, 'figures');

    % Choose fixed values for the second and third arguments
    fixed_kk = round(kk_num/2);  % Fixed index for the second argument
    fixed_ii = round(ii_num/2);  % Fixed index for the third argument
    fixed_zz = 3;
    
    % Loop over each matrix and generate plots
    matrices = {vv, ii_prime, ll, consumption};
    titles = {'Value function', 'Investment', 'Labor', 'Consumption'};
    
    for m = 1:length(matrices)
        matrix = matrices{m};
        
        % Figure for capital
        fig = figure;
        hold on;
        % Extract values along kk_grid for fixed technology shock (fixed_zz) and investment (fixed_ii)
        values = squeeze(matrix(fixed_zz, :, fixed_ii)); % Extract values
        plot(kk_grid, values, 'LineWidth', 2); % Plot with thicker line
        title(['Plot of ' titles{m} ' with fixed technology shock and investment']);
        xlabel('Capital');
        ylabel(titles{m});
        % Adjust y-axis based on range of values
        y_min = min(values);
        y_max = max(values);
        if y_min == y_max
            ylim([y_min - 0.1, y_max + 0.1]); % Expand range slightly if they are equal
        else
            ylim([y_min, y_max]);
        end
        hold off;

        % Save the plot
        figname = fullfile(adjusted_path,[titles{m} '_Capital.png' ]);
        saveas(fig, figname);
    
        % Figure for investment
        fig = figure;
        hold on;
        % Extract values along ii_grid for fixed technology shock (fixed_zz) and capital (fixed_kk)
        values = squeeze(matrix(fixed_zz, fixed_kk, :)); % Extract values
        % Filter out points where ii_grid is between -0.04 and 0.04
        mask = (ii_grid < -0.04) | (ii_grid > 0.04); % Logical mask
        filtered_ii_grid = ii_grid(mask);
        filtered_values = values(mask);
        plot(filtered_ii_grid, filtered_values, 'LineWidth', 2); % Plot with thicker line
        title(['Plot of ' titles{m} ' with fixed technology shock and capital']);
        xlabel('Investment');
        ylabel(titles{m});
        % Adjust y-axis based on range of filtered values
        y_min = min(filtered_values);
        y_max = max(filtered_values);
        if y_min == y_max
            ylim([y_min - 0.1, y_max + 0.1]); % Expand range slightly if they are equal
        else
            ylim([y_min, y_max]);
        end
        hold off;

        % Save the plot
        figname = fullfile(adjusted_path,[titles{m} '_Investment.png' ]);
        saveas(fig, figname);

    end
end


