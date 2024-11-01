% This code computes Pareto efficient allocations

%----------------------------------------------------------------
% 0. Housekeeping
%----------------------------------------------------------------

clc
clear
close all

%----------------------------------------------------------------
% 1(a). Parameters for 3 by 3 case, no heterogeneity
%----------------------------------------------------------------

m = 3; % Number of goods
n = 3; % Number of agents

% Matrix of elasticity of substitution parameters for different agents
% Row corresponds to agent
% Column corresponds to good type
omegas = [-0.5 -0.5 -0.5; ...
          -0.5 -0.5 -0.5; ...
          -0.5 -0.5 -0.5];

% Vector of Pareto weights
lambdas = [1; 1; 1];

% Vector of aggregate endowment of each good
endowments = [1; 2; 5];

%----------------------------------------------------------------
% 1(b). Parameters for 3 by 3 case, heterogeneity
%----------------------------------------------------------------

m2 = 3; % Number of goods
n2 = 3; % Number of agents

% Matrix of elasticity of substitution parameters for different agents
% Row corresponds to agent
% Column corresponds to good type
omegas2 = [-0.5 -0.5 -0.5; ...
          -2 -3 -2.5; ...
          -0.2 -0.5 -0.7];

% Vector of Pareto weights
lambdas2 = [0.5; 1; 1.2];

% Vector of aggregate endowment of each good
endowments2 = [1; 2; 5];

%----------------------------------------------------------------
% 1(c). Parameters for 10 by 10 case, heterogeneity
%----------------------------------------------------------------

m3 = 10; % Number of goods
n3 = 10; % Number of agents

% Matrix of elasticity of substitution parameters for different agents
% Row corresponds to agent
% Column corresponds to good type
rng(123);
min_val = -3;
max_val = -0.2;
omegas3 = min_val + (max_val - min_val) * rand(n3, m3);

% Vector of Pareto weights
rng(123);
min_val = 0.5;
max_val = 1.5;
lambdas3 = min_val + (max_val - min_val) * rand(n3, 1);

% Vector of aggregate endowment of each good
rng(123);
min_val = 1;
max_val = 5;
endowments3 = min_val + (max_val - min_val) * rand(m3, 1);

%----------------------------------------------------------------
% 2. Compute optimal allocation
%----------------------------------------------------------------

% Options for optimizer
options = optimoptions('fsolve', ...
                       'Display', 'off', ...       % Display options
                       'TolFun', 1e-6, ...         % Set function tolerance
                       'TolX', 1e-6, ...           % Set solution tolerance
                       'MaxIterations', 100, ...   % Set maximum number of iterations
                       'Algorithm', 'trust-region'); % Choose 'trust-region' or 'levenberg-marquardt'

% Define the parameters for each case in a structure array
cases(1).n = n;
cases(1).m = m;
cases(1).omegas = omegas;
cases(1).lambdas = lambdas;
cases(1).endowments = endowments;
cases(1).label = 'No heterogeneity';

cases(2).n = n2;
cases(2).m = m2;
cases(2).omegas = omegas2;
cases(2).lambdas = lambdas2;
cases(2).endowments = endowments2;
cases(2).label = 'Heterogeneity';

cases(3).n = n3;
cases(3).m = m3;
cases(3).omegas = omegas3;
cases(3).lambdas = lambdas3;
cases(3).endowments = endowments3;
cases(3).label = 'Large system';

% Initialize storage for results
cons_results = cell(1, length(cases));
speeds = zeros(1, length(cases));

% Loop over each case
for i = 1:length(cases)
    % Set parameters for the current case
    pars.n = cases(i).n;
    pars.omegas = cases(i).omegas;
    pars.lambdas = cases(i).lambdas;
    pars.endowments = cases(i).endowments;
    
    % Clear functions to remove caching effects
    clear functions
    
    % Run the optimization and measure the time
    tic;
    cons_results{i} = opt_alloc(pars, options);
    speeds(i) = toc;
    
    % Create the table with optimal allocation
    T_alloc = array2table(round(cons_results{i}, 3));
    T_alloc.Properties.VariableNames = "Good " + string(1:cases(i).m);
    T_alloc.Properties.RowNames = "Agent " + string(1:cases(i).n);

    % Display results for the current case
    disp(['Case: ', cases(i).label]);
    disp(['Time taken: ', num2str(round(speeds(i),3)), ' seconds']);
    disp('Optimal allocation:');
    disp(T_alloc);
    disp(' ');

end

% Store results in a table for speed comparison
T_speed = array2table(round(speeds,3)', 'VariableNames', {'Speed'}, ...
                      'RowNames', {cases.label});

% Display the speed comparison table
disp('Speed comparison:');
disp(T_speed);

%----------------------------------------------------------------
% 3. Functions
%----------------------------------------------------------------

function eqq = syst_eq(x,pars)

    % Generate equations
    % Create a matrix where each column is equal to x (consumption of agent 1)
    xx = repmat(x', pars.n, 1);

    % Vectorized computation
    % 1. Compute the matrix of (lambda(1) ./ lambda)^(1 ./ omega) for each i and j
    lambdas_ratio = (pars.lambdas(1) ./ pars.lambdas).^(1 ./ pars.omegas);
    
    % 2. Compute the matrix of (x(i)^(omega(i,1) ./ omega(i,j))) for each i and j
    % Expand x to match dimensions for element-wise operations
    x_expanded = xx .^ (pars.omegas(1,:) ./ pars.omegas);
    
    % 3. Element-wise multiplication of lambda_ratio and x_expanded and summation across columns
    eqq = sum(lambdas_ratio .* x_expanded, 1)' - pars.endowments;

end

function cons = opt_alloc(pars, options)

    x0 = pars.endowments; % Initial guess

    % Solve for consumption of agent 1
    [x_star, err] = fsolve(@(x) syst_eq(x,pars), x0, options);
    
    % Solve for consumption of other agents
    % In resulting matrix cons row corresponds to agent, column to good
    xx = repmat(x_star', pars.n, 1);
    lambdas_ratio = (pars.lambdas(1) ./ pars.lambdas).^(1 ./ pars.omegas);
    x_expanded = xx .^ (pars.omegas(1,:) ./ pars.omegas);
    cons = lambdas_ratio .* x_expanded;

end