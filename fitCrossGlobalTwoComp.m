function globalFit = fitCrossGlobalTwoComp(tblCell, linkParams, initialGuessCell)

% parameter order
% [N tauD1 tauD2 f q k]

nSets = numel(tblCell);
nParam = 6;

if nargin < 2 || isempty(linkParams)
    linkParams = false(1,nParam);
end

if nargin < 3
    initialGuessCell = [];
end

%% ------------------------
% Build parameter indexing
%% ------------------------

nGlobal = sum(linkParams);
nLocalPerSet = sum(~linkParams);

nTotal = nGlobal + nSets*nLocalPerSet;

paramMap = cell(nSets,1);

currentIdx = 1;

globalIdx = zeros(1,nParam);
for p = 1:nParam
    if linkParams(p)
        globalIdx(p) = currentIdx;
        currentIdx = currentIdx + 1;
    end
end

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
% Initial guess
%% ------------------------

p0 = zeros(nTotal,1);

for s = 1:nSets
    
    tbl = tblCell{s};
    
    xData = tbl.tau;
    yData = tbl.Cross;
    
    if isempty(initialGuessCell)
        
        N0 = 1/max(yData);
        tauD1 = median(xData)/5;
        tauD2 = median(xData)*2;
        f0 = 0.5;
        q0 = 1;
        k0 = 5;
        
        guess = [N0 tauD1 tauD2 f0 q0 k0];
        
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

modelFun = @(p,x) twoCompModel(p,x);

%% ------------------------
% Residual function
%% ------------------------

residFun = @(p) globalResidual(p, tblCell, paramMap, modelFun);

%% ------------------------
% Bounds
%% ------------------------

lbSingle = [1e-5 1e-9 1e-9 0 0 1];
ubSingle = [1e4 1 10 1 100 1e3];

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

for s = 1:nSets
    
    idx = paramMap{s};
    
    ps = pFit(idx);
    errs = paramErrors(idx);
    
    globalFit.dataset(s).N = ps(1);
    globalFit.dataset(s).N_err = errs(1);
    
    globalFit.dataset(s).tauD1 = ps(2);
    globalFit.dataset(s).tauD1_err = errs(2);
    
    globalFit.dataset(s).tauD2 = ps(3);
    globalFit.dataset(s).tauD2_err = errs(3);
    
    globalFit.dataset(s).fraction = ps(4);
    globalFit.dataset(s).fraction_err = errs(4);
    
    globalFit.dataset(s).brightnessRatio = ps(5);
    globalFit.dataset(s).brightnessRatio_err = errs(5);
    
    globalFit.dataset(s).k = ps(6);
    globalFit.dataset(s).k_err = errs(6);
    
end

%% ------------------------
% Plot
%% ------------------------

figure;
hold on;

colors = lines(nSets);

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
    
    yModel = modelFun(ps,x);
    
    errorbar(x,y,sigma,'o',...
        'Color',colors(s,:),...
        'MarkerFaceColor',colors(s,:));
    
    semilogx(x,yModel,'-','Color',colors(s,:),'LineWidth',2);
    
end

set(gca,'XScale','log')

xlabel('\tau')
ylabel('Cross')

title('Global Fit: Two Diffusing Species')

grid on

end



%% ------------------------
% Two component model
%% ------------------------

function y = twoCompModel(p,x)

N = p(1);
tau1 = p(2);
tau2 = p(3);
f = p(4);
q = p(5);
k = p(6);

g1 = (1 + x./tau1).^(-1) .* ...
     (1 + x./(k.^2 .* tau1)).^(-0.5);

g2 = (1 + x./tau2).^(-1) .* ...
     (1 + x./(k.^2 .* tau2)).^(-0.5);

numerator = (1-f).*g1 + (q.^2).*f.*g2;

denom = ((1-f) + q*f).^2;

y = (1./N) .* numerator ./ denom;

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
    
    yModel = modelFun(ps,x);
    
    r = [r; (y - yModel)./sigma];
    
end

end