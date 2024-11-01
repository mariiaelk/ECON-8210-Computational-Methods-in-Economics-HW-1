% This code computes prices

%----------------------------------------------------------------
% 0. Housekeeping
%----------------------------------------------------------------

clc
clear
close all

%----------------------------------------------------------------
% 1. Parameters
%----------------------------------------------------------------

m = 3; % Number of goods
n = 3; % Number of agents

% Matrix of elasticity of substitution parameters for different agents
% Row corresponds to agent
% Column corresponds to good type
omegas = [-0.5 -0.5 -0.5; ...
          -0.2 -0.5 -0.9; ...
          -0.5 -0.5 -0.5];

% Matrix of individual endowments
% Row corresponds to agent
% Column corresponds to good type
end_ind = [2 1 1.5; ...
          0.2 0.1 1; ...
          0.1 0.1 0.1];

% Vector of alphas
alphas = [0.3; 0.1; 0.2];

% Options for optimizer
options = optimoptions('fsolve', ...
                       'Display', 'off', ...      % Display options
                       'TolFun', 1e-6, ...         % Set function tolerance
                       'TolX', 1e-6, ...           % Set solution tolerance
                       'MaxIterations', 100, ...   % Set maximum number of iterations
                       'Algorithm', 'trust-region'); % Choose 'trust-region' or 'levenberg-marquardt'

%----------------------------------------------------------------
% 2. Compute prices
%----------------------------------------------------------------

% Define the parameters for each case in a structure array
pars.n = n;
pars.m = m;
pars.end_ind = end_ind;
pars.alphas = alphas;
pars.omegas = omegas;

% Solve for prices and Pareto weights associated with competitive equilibrium
x0 = ones(m+n-1,1);
tic;
[x_star, err, exitflag] = fsolve(@(x) syst_eq(x,pars), x0, options);
speeds = toc;

if exitflag > 0
    
    % Get prices
    prices = x_star(n:end).^2;
    
    disp('Solution was obtained successfully');
    
    % Table with prices
    % Store results in a table for speed comparison
    T_prices = array2table(round(prices,3), ...
        'VariableNames', {'Prices'});
    T_prices.Properties.RowNames = "Good " + string(1:m); 
    
    % Display the speed comparison table
    disp('Solution for prices:');
    disp(T_prices);

else

    disp('No solution was obtained');

end

% Display time
disp(['Time taken: ', num2str(round(speeds,3)), ' seconds']);

%----------------------------------------------------------------
% 3. Functions
%----------------------------------------------------------------

function eqns_mat = syst_eq(x,pars)
    
    % First n - 1 elements of x correspond to lambdas
    lambdas = ones(pars.n, 1);
    lambdas(2:pars.n) = x(1:(pars.n-1)).^2;

    % The rest of elements of x correspond to etas
    etas = x(pars.n:end).^2;

    % Generate equations
    eqns = zeros(pars.n+pars.m-1, 1);
    eqns_mat = zeros(pars.n+pars.m-1, 1);

    % First set of equations: budget constraints
    sum1_mat = sum(etas .* pars.end_ind')';
    sum2_mat = sum(repmat(etas,1,pars.n) .* ...
        ((etas./ pars.alphas) * (1./lambdas')).^(1 ./ pars.omegas'),1)';
    eqns_mat(1:(pars.n-1)) = sum1_mat(2:end) - sum2_mat(2:end);

    % Second set of equations: market clearing
    sum3_mat = sum(((etas./ pars.alphas) * (1./lambdas')).^(1 ./ pars.omegas'),2);
    sum4_mat = sum(pars.end_ind,1)';
    eqns_mat(pars.n:end) = sum3_mat - sum4_mat;

end