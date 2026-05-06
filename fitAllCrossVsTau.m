function combinedResults = fitAllCrossVsTauCell(dataCell, initialGuess)
%FITALLCROSSVSTAUCell Fits all tables inside a cell array
%
%   combinedResults = fitAllCrossVsTauCell(dataCell)
%   combinedResults = fitAllCrossVsTauCell(dataCell, initialGuess)
%
%   Input:
%       dataCell    - cell array where each cell contains a table
%       initialGuess - optional [N, tauD, k] starting guess
%
%   Output:
%       combinedResults - struct containing:
%           .individual   -> cell array of individual fit results
%           .N            -> array of fitted N values
%           .tauD         -> array of fitted tauD values
%           .k            -> array of fitted k values
%           .N_err        -> uncertainties
%           .tauD_err
%           .k_err
%           .reducedChi2  -> array of reduced chi^2 values

    if ~iscell(dataCell)
        error('Input must be a cell array of tables.');
    end

    nFits = numel(dataCell);

    % Preallocate arrays
    N_vals = zeros(nFits,1);
    tauD_vals = zeros(nFits,1);
    k_vals = zeros(nFits,1);

    N_err = zeros(nFits,1);
    tauD_err = zeros(nFits,1);
    k_err = zeros(nFits,1);

    chi2_vals = zeros(nFits,1);

    % Store individual results
    individual = cell(nFits,1);

    for i = 1:nFits
        tbl = dataCell{i};

        fprintf('Fitting dataset %d of %d\n', i, nFits);

        % Call your existing fit function
        if nargin < 2
            fitRes = fitCrossVsTau(tbl);
        else
            fitRes = fitCrossVsTau(tbl, initialGuess);
        end

        % Store full result
        individual{i} = fitRes;

        % Extract parameters
        N_vals(i) = fitRes.N;
        tauD_vals(i) = fitRes.tauD;
        k_vals(i) = fitRes.k;

        N_err(i) = fitRes.paramErrors(1);
        tauD_err(i) = fitRes.paramErrors(2);
        k_err(i) = fitRes.paramErrors(3);

        chi2_vals(i) = fitRes.reducedChi2;
    end

    % Combine into output struct
    combinedResults.individual = individual;

    combinedResults.N = N_vals;
    combinedResults.tauD = tauD_vals;
    combinedResults.k = k_vals;

    combinedResults.N_err = N_err;
    combinedResults.tauD_err = tauD_err;
    combinedResults.k_err = k_err;

    combinedResults.reducedChi2 = chi2_vals;

end
