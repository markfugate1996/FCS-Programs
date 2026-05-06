function globalFit = fitCrossGlobal(tblCell, linkParams, initialGuessCell)

% tblCell: cell array of tables

% linkParams: logical [1x3] → which parameters are global
% like this: linkParams = [true false true];
% order of params: [N0 tauD0 k0]

%sample usage: Ffits = fitCrossGlobalTriplet(FData, [false true false])

% initialGuessCell: cell array of initial guesses per dataset (optional)
%Like this:
%initialGuessCell = {
%    [1.1  0.01  5.0 ];   % <-- global guesses taken from here
%    [999  0.02  999 ];   % global entries ignored
%    [999  0.03  999  ]
%};



nSets = numel(tblCell);
nParam = 3;

if nargin < 2 || isempty(linkParams)
    linkParams = false(1,nParam);
end

if nargin < 3
    initialGuessCell = [];
end

%% ------------------------
% Build parameter indexing
%% ------------------------

% Count how many global parameters
nGlobal = sum(linkParams);
nLocalPerSet = sum(~linkParams);

% Total number of fit parameters
nTotal = nGlobal + nSets*nLocalPerSet;

% Build mapping structure
paramMap = cell(nSets,1);

currentIdx = 1;

% Assign global indices
globalIdx = zeros(1,nParam);
for p = 1:nParam
    if linkParams(p)
        globalIdx(p) = currentIdx;
        currentIdx = currentIdx + 1;
    end
end

% Assign local indices per dataset
for s = 1:nSets
    localIdx = zeros(1,nParam);
    for p = 1:nParam
        if ~linkParams(p)
            localIdx(p) = currentIdx;
            currentIdx = currentIdx + 1;
        else
            localIdx(p) = globalIdx(p);
        end
    end
    paramMap{s} = localIdx;
end

%% ------------------------
% Build initial guess
%% ------------------------

p0 = zeros(nTotal,1);

for s = 1:nSets
    
    tbl = tblCell{s};
    xData = tbl.tau;
    yData = tbl.Cross;
    
    if isempty(initialGuessCell)
        N0 = 1/max(yData);
        tauD0 = median(xData);
        k0 = 5;

        guess = [N0 tauD0 k0];
    else
        guess = initialGuessCell{s};
    end
    
    idx = paramMap{s};
    
    for p = 1:nParam
        if p0(idx(p)) == 0
            p0(idx(p)) = guess(p);
        end
    end
end

%% ------------------------
% Model function
%% ------------------------

modelFun = @(p,x) ...
    (1./p(1)) .* ...
    (1 + x./p(2)).^-1 .* ...
    (1 + x./(p(3).^2 .* p(2))).^(-0.5);

%% ------------------------
% Global residual function
%% ------------------------

residFun = @(p) globalResidual(p, tblCell, paramMap, modelFun);

%% ------------------------
% Bounds (expanded)
%% ------------------------

lbSingle = [1e-5 1e-9 1];
ubSingle = [1e4 1 1e05];

lb = -inf(nTotal,1);
ub = inf(nTotal,1);

for s = 1:nSets
    idx = paramMap{s};
    lb(idx) = lbSingle;
    ub(idx) = ubSingle;
end

%% ------------------------
% Fit
%% ------------------------

options = optimoptions('lsqnonlin','Display','off');

[pFit,resnorm,residual,exitflag,output,lambda,J] = ...
    lsqnonlin(residFun, p0, lb, ub, options);

%% ------------------------
% Covariance
%% ------------------------

dof = length(residual) - length(pFit);
reducedChi2 = resnorm / dof;

covMatrix = reducedChi2 * (J' * J) \ eye(length(pFit));
paramErrors = sqrt(diag(covMatrix));

%% ------------------------
% Store Results
%% ------------------------

globalFit.pFit = pFit;
globalFit.paramErrors = paramErrors;
globalFit.covariance = covMatrix;
globalFit.reducedChi2 = reducedChi2;

% Per-dataset unpacked results

for s = 1:nSets
    
    idx = paramMap{s};
    ps = pFit(idx);
    errs = paramErrors(idx);
    
    globalFit.dataset(s).N        = ps(1);
    globalFit.dataset(s).N_err    = errs(1);
    
    globalFit.dataset(s).tauD     = ps(2);
    globalFit.dataset(s).tauD_err = errs(2);
    
    globalFit.dataset(s).k        = ps(3);
    globalFit.dataset(s).k_err    = errs(3);
   
end



%% ------------------------
% Combined Plot
%% ------------------------

figure;
hold on;

colors = lines(nSets);  % distinct colors

for s = 1:nSets
    
    tbl = tblCell{s};
    x = tbl.tau;
    y = tbl.Cross;
    sigma = tbl.CrossErr;
    
    valid = sigma > 0;
    x = x(valid);
    y = y(valid);
    sigma = sigma(valid);
    
    idx = paramMap{s};
    ps = pFit(idx);
    
    yModel = modelFun(ps, x);
    
    % Plot data
    h = errorbar(x, y, sigma, 'o', ...
        'Color', colors(s,:), ...
        'MarkerFaceColor', colors(s,:), ...
        'DisplayName', sprintf('Data %d', s));
    
    % Plot fit
    semilogx(x, yModel, '-', ...
        'Color', colors(s,:), ...
        'LineWidth', 2, ...
        'DisplayName', sprintf('Fit %d', s));
end

set(gca,'XScale','log')
xlabel('\tau');
ylabel('Cross');
title('Global Fit: 1-Component + Triplet');
legend show
grid on



end




%% ------------------------
% Residual helper
%% ------------------------



function r = globalResidual(p, tblCell, paramMap, modelFun)

r = [];

for s = 1:numel(tblCell)
    
    tbl = tblCell{s};
    
    x = tbl.tau;
    y = tbl.Cross;
    sigma = tbl.CrossErr;
    
    valid = sigma > 0;
    x = x(valid);
    y = y(valid);
    sigma = sigma(valid);
    
    idx = paramMap{s};
    ps = p(idx);
    
    yModel = modelFun(ps, x);
    
    r = [r; (y - yModel)./sigma];
end

end
