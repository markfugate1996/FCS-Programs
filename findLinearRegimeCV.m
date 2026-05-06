function [bestXmin, bestIdx, cvError, slope, intercept] = findLinearRegimeCV(x, y, sigmaY, kfold, minPts)
% findLinearRegimeCV
%
% Determines where a high-X linear regime begins using
% weighted k-fold cross-validation.
%
% INPUTS
%   x        : Nx1 vector (independent variable)
%   y        : Nx1 vector (dependent variable)
%   sigmaY   : Nx1 vector (std dev of y)
%   kfold    : number of folds (default = 5)
%   minPts   : minimum points required for fitting (default = 4)
%
% OUTPUTS
%   bestXmin : X value where linear regime begins
%   bestIdx  : index of cutoff in sorted data
%   cvError  : cross-validated weighted error for each cutoff
%   slope    : slope of final weighted fit on selected regime
%   intercept: intercept of final weighted fit on selected regime

    if nargin < 4 || isempty(kfold)
        kfold = 5;
    end
    if nargin < 5 || isempty(minPts)
        minPts = 4;
    end

    % Ensure column vectors
    x = x(:); 
    y = y(:); 
    sigmaY = sigmaY(:);

    % Sort by increasing x
    [x, idx] = sort(x);
    y = y(idx);
    sigmaY = sigmaY(idx);

    N = length(x);
    cutIdx = (1:(N-minPts))';
    cvError = nan(size(cutIdx));

    for c = 1:length(cutIdx)

        startIdx = cutIdx(c);

        xSub = x(startIdx:end);
        ySub = y(startIdx:end);
        sSub = sigmaY(startIdx:end);

        nSub = length(xSub);
        if nSub < minPts
            continue
        end

        foldID = mod(0:nSub-1, kfold) + 1;
        totalErr = 0;

        for k = 1:kfold

            test  = (foldID == k);
            train = ~test;

            w = 1 ./ sSub(train).^2;
            Xmat = [xSub(train) ones(sum(train),1)];

            % Weighted least squares
            W = diag(w);
            beta = (Xmat' * W * Xmat) \ (Xmat' * W * ySub(train));

            % Predict test set
            yPred = beta(1)*xSub(test) + beta(2);

            % Weighted squared error
            totalErr = totalErr + ...
                sum(((ySub(test) - yPred)./sSub(test)).^2);
        end

        cvError(c) = totalErr;
    end

    % Select optimal cutoff
    [~, bestLocalIdx] = min(cvError);
    bestIdx = cutIdx(bestLocalIdx);
    bestXmin = x(bestIdx);

    % Final weighted fit on optimal regime
    xFinal = x(bestIdx:end);
    yFinal = y(bestIdx:end);
    sFinal = sigmaY(bestIdx:end);

    w = 1 ./ sFinal.^2;
    Xmat = [xFinal ones(length(xFinal),1)];
    W = diag(w);
    beta = (Xmat' * W * Xmat) \ (Xmat' * W * yFinal);

    slope = beta(1);
    intercept = beta(2);

end
