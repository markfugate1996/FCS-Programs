function fitResults = fitCrossVsTauTriplet(tbl, initialGuess)
%FITCROSSVSTAU Fits Cross vs tau data using weighted nonlinear least squares
% and returns covariance matrix and parameter uncertainties.
%
%   fitResults = fitCrossVsTau(tbl)
%   fitResults = fitCrossVsTau(tbl, initialGuess)

    % Check required variables
    requiredVars = {'tau','Cross','CrossErr'};
    for v = requiredVars
        if ~ismember(v{1}, tbl.Properties.VariableNames)
            error('Table is missing required variable: %s', v{1});
        end
    end

    xData = tbl.tau;
    yData = tbl.Cross;
    sigma = tbl.CrossErr;

    % Remove zero or negative uncertainties
    valid = sigma > 0;
    xData = xData(valid);
    yData = yData(valid);
    sigma = sigma(valid);

    % Define the model function
    modelFun = @(p, x) ...
        (1 + (p(4).*exp(-x./p(5))) ./ (1 - p(4))) .* ...
        (1./p(1)) .* ...
        (1 + x./p(2)).^-1 .* ...
        (1 + x./(p(3).^2 .* p(2))).^(-0.5);
    
    % Initial guess [N, tauD, k]
    if nargin < 2 || isempty(initialGuess)
        N0 = 1/max(yData);
        tauD0 = median(xData);
        k0 = 5;
        trpFrac = 0;
        trpTau = 10^-6;
        initialGuess = [N0, tauD0, k0, trpFrac, trpTau];
    end

    % Proper weighted residuals (divide by sigma)
    residFun = @(p) (yData - modelFun(p, xData)) ./ sigma;

    % bounds
    lb = [0.00001, 10^-12, 0, 0, 0];
    ub = [10000, 100, Inf, 0.99, 1];

    % Perform nonlinear least-squares fit and get Jacobian
    options = optimoptions('lsqnonlin','Display','off');
    [pFit,resnorm,residual,exitflag,output,lambda,J] = ...
        lsqnonlin(residFun, initialGuess, lb, ub, options);

    % Degrees of freedom
    n = length(yData);
    m = length(pFit);
    dof = n - m;

    % Reduced chi-squared
    reducedChi2 = resnorm / dof;

    % Covariance matrix (numerically stable)
    covMatrix = reducedChi2 * (J' * J) \ eye(m);

    % Parameter uncertainties (1-sigma)
    paramErrors = sqrt(diag(covMatrix));

    % Store results
    fitResults.N = pFit(1);
    fitResults.tauD = pFit(2);
    fitResults.k = pFit(3);
    fitResults.trpFrac = pFit(4);
    fitResults.trpTau = pFit(5);

    fitResults.paramErrors = paramErrors;
    fitResults.covariance = covMatrix;
    fitResults.resnorm = resnorm;
    fitResults.reducedChi2 = reducedChi2;

    fitResults.yFit = modelFun(pFit, xData);
    fitResults.residuals = residual;

    % -------- Plot (keeping your semilog fix) --------
    figure;
    h = errorbar(xData, yData, sigma, 'o', 'MarkerFaceColor','b');
    set(get(h,'Parent'), 'XScale', 'log')
    hold on;
    semilogx(xData, fitResults.yFit, 'r-', 'LineWidth', 2);
    xlabel('\tau');
    ylabel('Cross');
    title('1-component with Trplt');
    legend('Data','Fit');
    grid on;

    % Print results
    fprintf('\nFit Results:\n');
    fprintf('N       = %.6g ± %.6g\n', pFit(1), paramErrors(1));
    fprintf('tauD    = %.6g ± %.6g\n', pFit(2), paramErrors(2));
    fprintf('k       = %.6g ± %.6g\n', pFit(3), paramErrors(3));
    fprintf('trpFrac = %.6g ± %.6g\n', pFit(4), paramErrors(4));
    fprintf('trpTau  = %.6g ± %.6g\n', pFit(5), paramErrors(5));
    fprintf('Reduced chi^2 = %.4f\n\n', reducedChi2);

end
