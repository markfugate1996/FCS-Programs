function [bestXmin, bestIdx, cvError, slope, slopeErr, intercept, interceptErr] = findLinearRegimeCV3(x, y, sigmaY, kfold, minPts)

    % sample usage:
    % findLinearRegimeCV2(C, [Ffits.dataset.N], [Ffits.dataset.N_err])

    if nargin < 5 || isempty(minPts)
        minPts = 2;
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

    %test each possible cutoff value
    for c = 1:length(cutIdx)

        startIdx = cutIdx(c);

        xSub = x(startIdx:end);
        ySub = y(startIdx:end);
        sSub = sigmaY(startIdx:end);

        %number of points in the subset. Skip the loop if
        %there are too few points
        nSub = length(xSub);
        if nSub < minPts
            continue
        end

        kfold = nSub;
        foldID = 1:nSub;
        %foldID = mod(0:nSub-1, kfold) + 1;
        totalErr = 0;

        for k = 1:kfold

            test  = (foldID == k);
            train = ~test;

            Xtrain = [xSub(train) ones(sum(train),1)];

            % weights = 1/sigma^2
            w = 1 ./ sSub(train).^2;

            % weighted linear fit
            [beta, stdBeta] = lscov(Xtrain, ySub(train), w);

            % prediction on test fold
            yPred = beta(1)*xSub(test) + beta(2);

            % chi^2 prediction error
            totalErr = totalErr + ...
                sum(((ySub(test) - yPred)./sSub(test)).^2);

        end
        %divide by nsub to make sure we aren't unfairly preferring 
        % smaller datasets
        cvError(c) = totalErr/nSub;

    end

    % Select optimal cutoff
    [~, bestLocalIdx] = min(cvError);
    bestIdx = cutIdx(bestLocalIdx);
    bestXmin = x(bestIdx);

    %% Final weighted fit on optimal region

    xFinal = x(bestIdx:end);
    yFinal = y(bestIdx:end);
    sFinal = sigmaY(bestIdx:end);

    Xfinal = [xFinal ones(length(xFinal),1)];
    wFinal = 1 ./ sFinal.^2;

    [beta, stdBeta] = lscov(Xfinal, yFinal, wFinal);

    slope = beta(1);
    intercept = beta(2);

    slopeErr = stdBeta(1);
    interceptErr = stdBeta(2);

    %% ------------------------
    % Plotting section
    %% ------------------------

    figure;

    % ---- Top panel: Data + fit ----
    subplot(2,1,1); hold on;

    % All data
    errorbar(x, y, sigmaY, 'ko', 'MarkerFaceColor','k');

    % Highlight linear regime
    errorbar(xFinal, yFinal, sFinal, 'ro', 'MarkerFaceColor','r');

    % Plot fitted line
    xFit = linspace(min(xFinal), max(xFinal), 200);
    yFit = slope*xFit + intercept;
    plot(xFit, yFit, 'r-', 'LineWidth',2);

    % Vertical cutoff line
    xline(bestXmin, '--b', 'LineWidth',1.5);

    % ---- Derived quantity ----
    % ISS manual reports 1 nM = 0.6 molecules/fL
    Veff = slope * 1.6611;
    VeffErr = slopeErr * 1.6611;

    % ---- Text box ----
    xText = min(xFinal) + 0.05*(max(xFinal)-min(xFinal));
    yText = max(yFinal);

    txt = sprintf(['Slope = %.4g ± %.2g\n' ...
                   'V_{eff} = %.4g ± %.2g fL'], ...
                   slope, slopeErr, Veff, VeffErr);

    text(xText, yText, txt, ...
        'VerticalAlignment','top', ...
        'BackgroundColor','w', ...
        'EdgeColor','k', ...
        'Margin',6, ...
        'FontSize',10);

    xlabel('x');
    ylabel('y');
    title(sprintf('Best linear regime: x >= %.4g', bestXmin));
    legend('All data','Linear regime','Weighted fit','Cutoff','Location','best');
    grid on;

    % ---- Bottom panel: CV error ----
    subplot(2,1,2); hold on;

    plot(x(cutIdx), cvError, '-o','LineWidth',1.5);
    xline(bestXmin, '--r','LineWidth',1.5);

    xlabel('Candidate X_{min}');
    ylabel('Cross-validated weighted error');
    title('Cross-validation error vs cutoff');
    grid on;

end