function plotCrossVsTau(tbl)
%PLOTCROSSVSTAU Plots Cross vs tau from a table with error bars
%
%   plotCrossVsTau(tbl)
%
%   Inputs:
%       tbl - MATLAB table containing the variables:
%             'tau'      : x-axis values (log scale)
%             'Cross'    : y-axis values
%             'CrossErr' : y-axis error values
%
%   The function produces a semilogarithmic plot (log10 x-axis, linear y-axis).

    % Check that required variables exist
    requiredVars = {'tau','Cross','CrossErr'};
    for v = requiredVars
        if ~ismember(v{1}, tbl.Properties.VariableNames)
            error('Table is missing required variable: %s', v{1});
        end
    end

    % Extract data
    x = tbl.tau;
    y = tbl.Cross;
    err = tbl.CrossErr;

    % Create the plot with error bars
    figure;
    errorbar(x, y, err, 'o', 'MarkerSize', 6, 'MarkerFaceColor', 'b', 'LineWidth', 1.2);
    set(gca, 'XScale', 'log'); % semilog X-axis

    % Labels and title
    xlabel('\tau');
    ylabel('Cross');
    title('Cross vs Tau with Error Bars');
    grid on;
end
