function [lags, G] = compute_ACF(photonTimes, dt0, m, nStages)
% COMPUTE_ACF
% Computes the multi-tau autocorrelation function for photon arrival times.
%
% Inputs:
%   photonTimes : vector of photon arrival times (seconds)
%   dt0         : base time bin width for stage 1
%   m           : number of bins per stage
%   nStages     : total number of multi-tau stages
%
% Outputs:
%   lags        : vector of lag times (seconds)
%   G           : autocorrelation function G(tau)
%
% Example:
%   [lags, G] = compute_ACF(photonTimes, 1e-6, 16, 6);
%   loglog(lags, G); xlabel('Lag (s)'); ylabel('G(\tau)');

    % Preallocate cell arrays
    lagsCell = cell(nStages,1);
    acfCell = cell(nStages,1);
    
    % Stage 1 binning: cover all photon times
    edges = 0:dt0:max(photonTimes);
    counts = histcounts(photonTimes, edges);
    
    dt = dt0;  % current bin width
    
    for stage = 1:nStages
        
        % Compute ACF for this stage
        [acf_stage, tau_stage] = correlateStage(counts, dt);
        
        % Store results
        lagsCell{stage} = tau_stage;
        acfCell{stage} = acf_stage;
        
        % Prepare counts for next stage (merge every 2 bins)
        if stage < nStages
            if mod(length(counts),2) ~= 0
                counts = [counts, 0];  % pad with zero if odd length
            end
            counts = 0.5*(counts(1:2:end) + counts(2:2:end));
            dt = dt*2;  % double bin width
        end
    end
    
    % Concatenate results sequentially
    lags = [];
    G = [];
    for stage = 1:nStages
        lags = [lags, lagsCell{stage}];
        G    = [G, acfCell{stage}];
    end

    %% Nested function: compute ACF for one stage
    function [acf_stage, tau_stage] = correlateStage(countsVec, dtStage)
        N = length(countsVec);
        % FFT length
        Nfft = 2^nextpow2(2*N);
        % Zero-mean counts
        x = countsVec - mean(countsVec);
        % FFT-based autocorrelation
        F = fft(x, Nfft);
        acf_full = ifft(abs(F).^2);
        acf_stage = real(acf_full(1:N));
        tau_stage = (0:N-1) * dtStage;
        % Normalize: avoid NaN if zero-lag is zero
        if acf_stage(1) ~= 0
            acf_stage = acf_stage / acf_stage(1);
        else
            acf_stage = zeros(size(acf_stage));
        end
    end
end
