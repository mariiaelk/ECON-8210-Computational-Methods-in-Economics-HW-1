% This function computes deterministic steady state of the model.

%% Inputs:
%        (2) pars: parameters of the model
%        (3) pars_names: names of parameters of the model

%% Function
function ss = Q6_2(pars,pars_names)

    % Load parameters
    for i = 1:length(pars)
        eval([pars_names{i} ' = pars(i);']);
    end

    % Compute steady state
    % Rental rate of capital
    ss.rr = 1/bbeta - (1-ddelta);
    
    % Capital to labor
    kk_ll = (ss.rr/aalpha)^(1/(aalpha-1));
    
    % Wage
    ss.ww = (1-aalpha)*kk_ll^aalpha;

    % Labor to output
    ss.ll_yy = kk_ll^(-aalpha);

    % Capital to output
    ss.kk_yy = kk_ll*ss.ll_yy;

    % Investment to output
    ss.ii_yy = ddelta*ss.kk_yy;
   
    % Government expenditures to output
    ss.gov_yy = tau_ss*ss.ww*ss.ll_yy;

    % Consumption to output
    ss.cc_yy = 1 - ss.ii_yy - ss.gov_yy;
    
    % Output
    ss.yy = ((1-tau_ss)*ss.ww/(cchi*ss.cc_yy*ss.ll_yy))^0.5;

    % Recover variables before they were divided by output
    ss.ll = ss.ll_yy*ss.yy;
    ss.kk = ss.kk_yy*ss.yy;
    ss.ii = ss.ii_yy*ss.yy;
    ss.gov = ss.gov_yy*ss.yy;
    ss.cc = ss.cc_yy*ss.yy;

end