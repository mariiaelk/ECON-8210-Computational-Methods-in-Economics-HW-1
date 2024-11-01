% This code compares 4 optimization methods (with multiple initial guesses).
%   1) Newton-Raphson
%   2) BFGS
%   3) Steepest descent
%   4) Conjugate descent

% The function we are going to minimize is
% 100*(y-x^2)^2 + (1-x)^2

%----------------------------------------------------------------
% 0. Housekeeping
%----------------------------------------------------------------

clc
clear
close all

%----------------------------------------------------------------
% 1. Function and some optimization parameters
%----------------------------------------------------------------

% Function
ffn = @(x) (100*(x(2)-x(1)^2)^2 + (1-x(1))^2);

% Initial guesses to be considered
init_vec = [0 0; 0.9 0.9; 1 0; 0 1; 3 3; 5 5; -5 -3];

% Tolerance for convergence
tol = 1e-6;

% Maximum number of iterations
max_iter = 5000;   

% Initial step size for steepest descent and conjugate descent
step0 = 0.1;               

%----------------------------------------------------------------
% 2. Auxiliary computations
%----------------------------------------------------------------

% Initialize object to store the results
x_length = size(init_vec,2);
numCases = size(init_vec, 1);     
res = zeros(4, x_length);  
speed = zeros(4, 1);  

% Get expressions for gradient and Hessian as functions
syms x y                            
f = 100*(y-x^2)^2 + (1-x)^2;                     

% Compute the gradient and Hessian symbolically
grad_f = gradient(f, [x, y]);        % Gradient of f
hess_f = hessian(f, [x, y]);         % Hessian of f

% Convert symbolic gradient and Hessian to MATLAB functions
grad_f_func = matlabFunction(grad_f, 'Vars', {[x; y]});
hess_f_func = matlabFunction(hess_f, 'Vars', {[x; y]});

% New function handle to compute gradient and hessian for step size
grad_step = @(x, dir, st) dir.' * grad_f_func(x + st * dir);
hess_step = @(x, dir, st) dir.' * hess_f_func(x + st * dir) * dir;

%----------------------------------------------------------------
% 2. Minimize objective: Newton-Raphson
%----------------------------------------------------------------
fprintf('Newton-Raphson: \n');

tic;
xx_star = NaN; % Object to store minimizer
for ccase = 1:numCases

    % Initial point to be used in this iteration
    xx0 = init_vec(ccase,:)';

    xx = newton_raphson_symb(xx0,grad_f_func,hess_f_func,max_iter, ...
        tol, true);
    
    if isnan(xx_star)

        xx_star = xx;

    else
        
        % Check if new solution attains lower value of the objective
        if (ffn(xx) - ffn(xx_star)) <  - tol
            xx_star = xx;
        end

    end

end

speed(1) = toc;
res(1,1) = xx_star(1);
res(1,2) = xx_star(2);

%----------------------------------------------------------------
% 3. Minimize objective: BFGS
%----------------------------------------------------------------
fprintf('BFGS: \n');

tic;
xx_star = NaN; % Object to store minimizer
for ccase = 1:numCases

    % Initial point to be used in this iteration
    xx0 = init_vec(ccase,:)';

    xx = BFGS_symb(xx0,grad_f_func,hess_f_func,max_iter,tol);

    if isnan(xx_star)

        xx_star = xx;

    else
        
        % Check if new solution attains lower value of the objective
        if (ffn(xx) - ffn(xx_star)) <  - tol
            xx_star = xx;
        end

    end

end

speed(2) = toc;
res(2,1) = xx_star(1);
res(2,2) = xx_star(2);

%----------------------------------------------------------------
% 4. Minimize objective: Steepest descent
%----------------------------------------------------------------
fprintf('Steepest descent: \n');

tic;
xx_star = NaN; % Object to store minimizer

% Start the main part of algorithm
for ccase = 1:numCases

    % Initial point to be used in this iteration
    xx0 = init_vec(ccase,:)';

    xx = steepest_descent(xx0,step0,grad_f_func,grad_step,hess_step,...
    max_iter,tol);
    
    if isnan(xx_star)

        xx_star = xx;

    else
        
        % Check if new solution attains lower value of the objective
        if (ffn(xx) - ffn(xx_star)) <  - tol
            xx_star = xx;
        end

    end

end

speed(3) = toc;
res(3,1) = xx_star(1);
res(3,2) = xx_star(2);

%----------------------------------------------------------------
% 5. Minimize objective: Conjugate descent
%----------------------------------------------------------------

fprintf('Conjugate descent: \n');

tic;
xx_star = NaN; % Object to store minimizer
for ccase = 1:numCases

    % Initial point to be used in this iteration
    xx0 = init_vec(ccase,:)';
    step = step0;

    xx = conjugate_descent(xx0,step0,grad_f_func,grad_step,hess_step,...
    max_iter,tol);

    if isnan(xx_star)

        xx_star = xx;

    else
        
        % Check if new solution attains lower value of the objective
        if (ffn(xx) - ffn(xx_star)) <  - tol
            xx_star = xx;
        end

    end

end

speed(4) = toc;
res(4,1) = xx_star(1);
res(4,2) = xx_star(2);

%----------------------------------------------------------------
% 6. Print tables with results
%----------------------------------------------------------------

% Create a table with minimizers
T_comp = array2table(round(res,2));          
T_comp.Properties.VariableNames = {'x','y'}; 
T_comp.Properties.RowNames = {'Newton-Raphson', 'BFGS', ...
    'Steepest descent', 'Conjugate descent'}; 

    
% Display the table
disp(' ');
disp('Minimizers');
disp(' ');
disp(T_comp);

% Create a table with speed comparison
T_speed = array2table(round(speed,3));          
T_speed.Properties.VariableNames = {'Speed'}; 
T_speed.Properties.RowNames = {'Newton-Raphson', 'BFGS', ...
    'Steepest descent', 'Conjugate descent'}; 

    
% Display the table
disp(' ');
disp('Speed comparison');
disp(' ');
disp(T_speed);

%----------------------------------------------------------------
% 7. Algorithms
%----------------------------------------------------------------

function xx = newton_raphson_symb(xx0,grad_f_func,hess_f_func, ...
    max_iter,tol,disp_conv)
    
    xx = xx0; % Initial guess

    for iter = 1:max_iter
        
        grad = grad_f_func(xx); % Evaluate the gradient
        hess = hess_f_func(xx); % Evaluate the Hessian

        % Check for convergence
        if norm(grad) < tol
            if disp_conv
                fprintf('Converged in %d iterations\n', iter);
            end
            break;
        end
    
        % Update xx using the Newton-Raphson step
        xx = xx - hess \ grad;

    end

    if iter == max_iter && disp_conv
        fprintf('Algorithm did not converge \n');
    end

end

function xx = BFGS_symb(xx0,grad_f_func,hess_f_func,max_iter,tol)
    
    xx = xx0; % Initial guess
    x_length = size(xx,1);

    grad = grad_f_func(xx); % Evaluate the gradient at initial guess
    hess = hess_f_func(xx); % Evaluate the Hessian at initial guess
    invhess = inv(hess); % Inverse of Hessian

    for iter = 1:max_iter
        
        % Check for convergence
        if norm(grad) < tol
            fprintf('Converged in %d iterations\n', iter);
            break;
        end

        % Compute inverse of Hessian for next iteration
        diff_xx = -invhess*grad;
        xx = xx + diff_xx; % Update xx using the Newton-Raphson step
        diff_grad = grad_f_func(xx) - grad;
        grad = grad + diff_grad;
        denom = diff_grad.'*diff_xx;
        invhess = (eye(x_length) - diff_xx*diff_grad.'/denom)*invhess* ...
            (eye(x_length) - diff_grad*diff_xx.'/denom) + ...
            diff_xx*diff_xx.'/denom;

     end
    
    if iter == max_iter
        fprintf('Algorithm did not converge \n');
    end

end

function xx = steepest_descent(xx0,step0,grad_f_func,grad_step,hess_step,...
    max_iter,tol)
    
    xx = xx0; % Initial guess
    step = step0; % Initial step size to be used in this iteration

    for iter = 1:max_iter
        
    grad = grad_f_func(xx); % Evaluate the gradient

        % Check for convergence
        if norm(grad) < tol
            fprintf('Converged in %d iterations\n', iter);
            break;
        end

        % Compute direction of next step: negative gradient
        direction = - grad/norm(grad); 

        % Determine optimal step size 
        grad_step_cur = @(st) grad_step(xx, direction, st);
        hess_step_cur = @(st) hess_step(xx, direction, st);
        step = newton_raphson_symb(step,grad_step_cur,hess_step_cur, ...
            max_iter,tol,false);

        % Update candidate xx
        xx = xx + step * direction;

    end

    if iter == max_iter
        fprintf('Algorithm did not converge \n');
    end

end

function xx = conjugate_descent(xx0,step0,grad_f_func,grad_step,hess_step,...
    max_iter,tol)
    
    xx = xx0; % Initial guess
    step = step0; % Initial step size to be used in this iteration

    grad = grad_f_func(xx); % Evaluate the gradient at initial guess
    grad_sq = grad.'*grad;
    direction = - grad/norm(grad); % direction for initial step 

    for iter = 1:max_iter

        % Check for convergence
        if norm(grad) < tol
            fprintf('Converged in %d iterations\n', iter);
            break;
        end

        % Update the variable xx in the direction of the negative gradient
        xx = xx + step * direction;

        % Compute bbeta and dd for next iteration
        % New gradient
        grad_new = grad_f_func(xx);
        grad_new_sq = grad_new.'*grad_new;
        % Use Fletcher-Reeves for bbeta
        bbeta = grad_new_sq/grad_sq;
        % Compute direction
        direction = - grad_new + bbeta*direction;
        % Compute step size
        grad_step_cur = @(st) grad_step(xx, direction, st);
        hess_step_cur = @(st) hess_step(xx, direction, st);
        step = newton_raphson_symb(step,grad_step_cur,hess_step_cur, ...
            max_iter,tol,false);
        % Record gradient for next iteration
        grad = grad_new;
        grad_sq = grad_new_sq;

     end
    
    if iter == max_iter
        fprintf('Algorithm did not converge \n');
    end

end