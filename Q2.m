% This code computes itergral using 4 methods.
%   1) Midpoint
%   2) Trapezoid
%   3) Simpson's
%   4) Monte Carlo

% The function we are going to use for comparing them is 
% f(x) = -exp(-rrho*x+exp(-llambda*x)-1) on [A,T]

%----------------------------------------------------------------
% 0. Housekeeping
%----------------------------------------------------------------

clc
clear
close all

%----------------------------------------------------------------
% 1. Parameters and function
%----------------------------------------------------------------

% Number of intervals to be used
intNvec = [10;100;1000;5000];

% Parameters for defining the integral
A = 0;
T = 100;
rrho = 0.04;
llambda = 0.02;

% Function
ffn = @(x) (-exp(-rrho*x+exp(-llambda*x)-1));

% Initialize object to store the results
numCases = length(intNvec);     
res = zeros(4, numCases);  
speed = zeros(4, numCases);  

%----------------------------------------------------------------
% 2. Compute integral
%----------------------------------------------------------------

for ccase = 1:numCases

    % Number of intervals to be used in this iteration
    intN = intNvec(ccase);

    vval = 0;

    % Midpoint
    clear functions % Remove cashing effects for fair comparison
    tic;
    for currentInt=1:intN
        
        % Current interval bounds
        a = A + (currentInt - 1) * (T - A) / intN;
        b = A + currentInt * (T - A) / intN;

        % Value of the integral
        vval = vval + (b-a) * ffn((a+b)/2);

    end

    speed(1,ccase) = toc;
    res(1,ccase) = vval;
    vval = 0;

    % Trapezoid
    clear functions % Remove cashing effects for fair comparison
    tic;
    for currentInt=1:intN
        
        % Current interval bounds
        a = A + (currentInt - 1) * (T - A) / intN;
        b = A + currentInt * (T - A) / intN;

        % Value of the integral
        vval = vval + ((b-a)/2) * (ffn(a)+ffn(b));

    end

    speed(2,ccase) = toc;
    res(2,ccase) = vval;
    vval = 0;

    % Simpson rule
    clear functions % Remove cashing effects for fair comparison
    tic;
    for currentInt=1:intN
        
        % Current interval bounds
        a = A + (currentInt - 1) * (T - A) / intN;
        b = A + currentInt * (T - A) / intN;

        % Value of the integral
        vval = vval + ((b-a)/6) * (ffn(a)+4*ffn((a+b)/2)+ffn(b));

    end

    speed(3,ccase) = toc;
    res(3,ccase) = vval;

    % Monte Carlo
    clear functions % Remove cashing effects for fair comparison
    tic;

    % Draw points at which function will be evaluated
    rng(123);
    ppoints = A + (T - A) * rand(intN, 1);

    % Value of the integral
    vval = T*mean(ffn(ppoints));

    speed(4,ccase) = toc;
    res(4,ccase) = vval;

end

%----------------------------------------------------------------
% 3. Print tables with results
%----------------------------------------------------------------

% Create a table with results for integral value
T_comp = array2table(round(res,3));          
T_comp.Properties.VariableNames = "N = " + string(intNvec); 
T_comp.Properties.RowNames = {'Midpoint', 'Trapezoid', 'Simpson', 'Monte Carlo'}; 

    
% Display the table
disp(' ');
disp('Computed values of intergral');
disp(' ');
disp(T_comp);

% Create a table with speed comparison
T_speed = array2table(round(speed,3));          
T_speed.Properties.VariableNames = "N = " + string(intNvec); 
T_speed.Properties.RowNames = {'Midpoint', 'Trapezoid', 'Simpson', 'Monte Carlo'}; 

    
% Display the table
disp(' ');
disp('Speed comparison');
disp(' ');
disp(T_speed);