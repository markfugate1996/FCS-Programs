"""
fcs_noise.py
============
Analytical noise covariance of the FCS correlation estimator, and the Fisher
information / Cramer-Rao bounds it implies, for the *direct* (pairwise) photon
correlator used in this suite.

Purpose
-------
To rank lag-binning schemes by how much information the resulting correlation
curve carries about the model parameters, we need the full covariance matrix
Sigma of the estimator G(tau) -- not just its diagonal -- because neighbouring
log-spaced lag channels are strongly correlated and a diagonal (per-point sigma)
treatment systematically over-counts their joint information.  With Sigma in
hand the (generalized-least-squares) Fisher information matrix is

        I(theta) = J^T Sigma^-1 J ,     J[k,p] = d G(tau_k) / d theta_p ,

and the parameter covariance / Cramer-Rao lower bounds follow from I^-1.

Method and the approximations it rests on
-----------------------------------------
The fluorescence signal is treated as a stationary process whose intensity
fluctuations dI have autocovariance C(u) = <I>^2 * G(|u|), with G(tau) the
(baseline-zero) model correlation returned by fcs_models.  Two noise sources are
combined:

1.  Number-fluctuation ("particle") noise -- the covariance of the sample
    autocovariance of a stationary Gaussian process.  This is the classical
    result of Bartlett (1946); its multivariate form (covariance between two
    different lags) follows from the fourth-order Gaussian moment via the
    Isserlis/Wick theorem (Isserlis 1918).  The connected fourth-order term
    factorizes into products of two-point functions, and after integrating over
    the measurement time T the covariance between lags tau_i and tau_j collapses
    to

        Cov_sig[G(tau_i), G(tau_j)]  ~=  (1/T) * [ R_G(tau_i - tau_j)
                                                 + R_G(tau_i + tau_j) ] ,

        R_G(s) = integral_{-inf}^{inf} G(|x|) G(|x + s|) dx .

    R_G is the autocorrelation of the (even-extended) model correlation curve;
    it is computed once by FFT and sampled at every lag sum and difference.
    This is exactly the noise structure derived for FCS by Qian (1990),
    Wohland, Rigler & Vogel (2001) and, most completely, Saffarian & Elson
    (2003); here it is evaluated numerically on an arbitrary lag grid rather
    than via their closed-form special cases, so it works for any model in
    fcs_models.MODELS.

2.  Shot ("particle-number-independent") noise -- the Poisson statistics of
    photon detection (Saleh 1978; Koppel 1974).  Added on the diagonal as
    Var_shot[G(tau_k)] = (1 + G(tau_k)) / n_pairs(tau_k), with the number of
    photon pairs at lag tau_k over time T approximated by
    n_pairs ~= R^2 * T * dtau_k  (R = detected count rate, dtau_k = bin width).
    Signal and shot terms carry the correct opposite dependence on brightness:
    particle noise is count-rate independent (the high-count-rate limit of
    Koppel 1974), shot noise falls as 1/R^2.

Assumptions made explicit (and where they fail)
-----------------------------------------------
*   Gaussian factorization of the 4th-order moment.  Valid when the number of
    molecules in the volume is not too small; it drops the non-Gaussian
    "particle noise at large lag / small N" correction that Saffarian & Elson
    (2003) add via an apparent molecule number M_App.  Distrust the analytical
    Sigma at very low N and long lag, and check against the simulator below.
*   Shot noise is taken diagonal.  A direct correlator with non-overlapping lag
    bins that are wide compared with the hardware timing resolution has weak
    off-diagonal shot-noise coupling; that coupling is neglected here.  (This is
    also why the direct estimator has a more tractable covariance than a
    multi-tau correlator, whose triangular averaging couples channels --
    cf. Schaetzel 1990.)
*   Sigma is evaluated at the fitted parameters theta_hat.  Because it depends
    on theta it should, for rigorous GLS, be iterated once with the fit
    (Enderlein et al. 2005); for binning *comparison* at fixed theta_hat this is
    not needed.
*   Finite bin-width averaging of G within each channel is approximated by the
    bin-centre value; refine by averaging G over the bin if bins are wide
    relative to the curvature of G.

Validation
----------
simulate_estimator_covariance() synthesises Gaussian intensity traces with the
prescribed autocovariance by spectral (circulant) synthesis, optionally
Poisson-samples them, forms the correlation estimator on the same lag grid over
many realizations, and returns the empirical mean and covariance.  Comparing
that empirical Sigma with noise_covariance() validates the *implementation* of
the Bartlett/Wick term (and, with Poisson sampling, the shot term).  It does
NOT test the Gaussian assumption itself -- that needs a particle-diffusion
simulation, deliberately left as a heavier, separate step.

References
----------
Bartlett MS (1946) J. R. Stat. Soc. Suppl. 8:27-41.  (variance of sample
    autocovariances; see also Priestley, Spectral Analysis and Time Series,
    1981, sec. 5.3.)
Isserlis L (1918) Biometrika 12:134-139.  (Gaussian moment / Wick factorization)
Koppel DE (1974) Phys. Rev. A 10:1938-1945.  (statistical accuracy of FCS; SNR)
Saleh BEA (1978) Photoelectron Statistics, Springer.  (photodetection shot noise)
Qian H (1990) Biophys. Chem. 38:49-57.  ("On the statistics of FCS"; variance)
Wohland T, Rigler R, Vogel H (2001) Biophys. J. 80:2987-2999.  (the standard
    deviation in FCS; block/Monte-Carlo estimate; covariance in fitting)
Saffarian S, Elson EL (2003) Biophys. J. 84:2030-2042.  (analytical SD and bias
    of the correlation function, including large-lag particle noise)
Schaetzel K (1990) Quantum Opt. 2:287-305.  (noise on multi-tau correlation data)
Enderlein J, Gregor I, Patra D, Dertinger T, Kaupp UB (2005) ChemPhysChem
    6:2324-2336.  (FCS accuracy; correlated-noise weighting in fitting)
Wahl M, Gregor I, Patting M, Enderlein J (2003) Opt. Express 11:3583-3591.
    (fast FCS from photon arrival times -- this suite's correlator)
Laurence TA, Fore S, Yeh AT (2006) Opt. Lett. 31:829-831.  (fast, flexible
    direct photon-correlation algorithm; long-lag hybrid coarsening)

This module is numpy/scipy only -- no GUI or fcs_main dependency -- and reuses
fcs_fisher for the FIM -> CRLB summaries so the reporting is consistent with the
per-fit Fisher section.  fcs_models is imported only to turn a model + parameter
dict into a G(tau) callable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import fcs_fisher
import fcs_models


# ─────────────────────────────────────────────────────────────────────────────
# Model correlation as a plain callable
# ─────────────────────────────────────────────────────────────────────────────

def model_g_callable(
    model: "fcs_models.FCSModel",
    params: Dict[str, float],
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap an fcs_models correlation model + parameter dict as a vectorized
    G(tau) callable (baseline-zero convention, i.e. G = g - 1 = G0*shape + ...).

    The returned function accepts an array of lag times (s) and returns G at
    those lags, evaluated at ``params``.  Lags are treated as |tau| so the
    caller may pass negative or zero arguments (needed when building R_G).
    """
    names = model.param_names()
    kw = {n: float(params[n]) for n in names}

    def g(tau: np.ndarray) -> np.ndarray:
        tau = np.abs(np.asarray(tau, dtype=np.float64))
        # tau = 0 is finite for these models (G(0) = amplitude); guard against a
        # model that divides by tau by nudging exact zeros to a tiny value.
        tau = np.where(tau == 0.0, 1e-300, tau)
        return np.asarray(model.func(tau, **kw), dtype=np.float64)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# R_G(s): autocorrelation of the model correlation curve
# ─────────────────────────────────────────────────────────────────────────────

def _autocorrelation_grid(
    g: Callable[[np.ndarray], np.ndarray],
    s_max: float,
    dt: float,
    pad: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tabulate R_G(s) = integral G(|x|) G(|x+s|) dx for s in [0, s_max].

    The integral is dominated by |x| out to a few correlation times, so we
    evaluate G on a uniform grid of step ``dt`` extending to ``pad*s_max`` on
    each side and take the FFT autocorrelation.  ``dt`` should resolve the
    fastest variation of G (i.e. the smallest characteristic time of the model),
    NOT the smallest lag -- R_G is smooth on the scale of the correlation time.

    Returns (s_grid, R_grid) with s_grid = [0, dt, 2dt, ...] up to >= s_max.
    Units of R_G are (G^2 * time), i.e. the dt factor is included.
    """
    if dt <= 0 or s_max <= 0:
        raise ValueError("s_max and dt must be positive.")

    half = pad * s_max
    n_half = int(np.ceil(half / dt))
    # Symmetric grid x in [-n_half*dt, +n_half*dt].
    x = np.arange(-n_half, n_half + 1) * dt
    gx = g(x)                                   # G(|x|), even in x

    # Linear (non-circular) autocorrelation via FFT with zero padding.
    n = gx.size
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    F = np.fft.rfft(gx, nfft)
    ac_full = np.fft.irfft(F * np.conj(F), nfft)          # sum_x G(x)G(x+lag)
    ac_full *= dt                                          # -> integral dx

    # ac_full[lag] corresponds to shift = lag*dt for lag = 0,1,2,...
    n_s = int(np.ceil(s_max / dt)) + 1
    s_grid = np.arange(n_s) * dt
    R_grid = ac_full[:n_s]
    return s_grid, R_grid


def _pick_timescales(model: "fcs_models.FCSModel",
                     params: Dict[str, float]) -> Tuple[float, float]:
    """
    Heuristic smallest/largest characteristic time of a correlation model, used
    to choose the R_G integration step and range.  Looks for any parameter whose
    name marks a time (unit 's': tau_D*, tau_R, ...); falls back to sane values.
    """
    times = []
    for p in model.params:
        if p.unit == "s":
            v = params.get(p.name, None)
            if v is not None and np.isfinite(v) and v > 0:
                times.append(float(v))
    if not times:
        return 1e-6, 1.0
    return min(times), max(times)


# ─────────────────────────────────────────────────────────────────────────────
# Covariance matrix of the estimator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoiseCovariance:
    """Result of noise_covariance()."""
    tau: np.ndarray                 # lag grid (s)
    sigma: np.ndarray               # full covariance (n x n)
    sigma_signal: np.ndarray        # Bartlett/Wick particle-noise part
    var_shot: np.ndarray            # diagonal shot-noise variances
    cond: float                     # condition number of sigma
    meta: dict                      # counts, T, etc. for the record


def noise_covariance(
    model: "fcs_models.FCSModel",
    params: Dict[str, float],
    tau: Sequence[float],
    T: float,
    count_rate: float,
    bin_widths: Optional[Sequence[float]] = None,
    *,
    dt: Optional[float] = None,
    include_shot: bool = True,
) -> NoiseCovariance:
    """
    Analytical covariance of the correlation estimator G(tau) at the given lags.

    Parameters
    ----------
    model, params : the correlation model and parameter values (theta_hat).
    tau           : lag grid (s) of the scheme being evaluated (bin centres).
    T             : total effective measurement time (s).
    count_rate    : detected photon rate R (counts/s) for the channel.
    bin_widths    : width dtau_k of each lag bin (s), for the shot-noise pair
                    count.  If None, inferred from the spacing of ``tau``.
    dt            : integration step for R_G; defaults to (smallest model
                    timescale)/12, which resolves G's fastest variation.
    include_shot  : add the Poisson shot-noise diagonal (set False to inspect
                    the pure particle-noise structure).

    Returns
    -------
    NoiseCovariance
    """
    tau = np.asarray(tau, dtype=np.float64)
    n = tau.size
    if n == 0:
        raise ValueError("tau is empty.")
    g = model_g_callable(model, params)

    t_min, t_max = _pick_timescales(model, params)
    if dt is None:
        dt = t_min / 12.0
    s_max = float(2.0 * tau.max() + 8.0 * t_max)      # need R_G up to tau_i+tau_j
    s_grid, R_grid = _autocorrelation_grid(g, s_max=s_max, dt=dt)

    def R_of(s: np.ndarray) -> np.ndarray:
        s = np.abs(np.asarray(s, dtype=np.float64))    # R_G is even
        return np.interp(s, s_grid, R_grid, left=R_grid[0], right=0.0)

    # Signal covariance: (1/T) [ R_G(tau_i - tau_j) + R_G(tau_i + tau_j) ].
    ti = tau[:, None]
    tj = tau[None, :]
    sigma_signal = (R_of(ti - tj) + R_of(ti + tj)) / float(T)

    # Shot-noise diagonal.
    var_shot = np.zeros(n)
    if include_shot:
        if bin_widths is None:
            dtau = _infer_bin_widths(tau)
        else:
            dtau = np.asarray(bin_widths, dtype=np.float64)
        Gvals = g(tau)
        n_pairs = np.maximum(count_rate ** 2 * np.maximum(T - tau, 0.0) * dtau, 1e-300)
        var_shot = (1.0 + Gvals) / n_pairs

    sigma = sigma_signal.copy()
    if include_shot:
        sigma[np.diag_indices(n)] += var_shot

    # Symmetrise (guard tiny asymmetry from interpolation) and condition number.
    sigma = 0.5 * (sigma + sigma.T)
    try:
        cond = float(np.linalg.cond(sigma))
    except np.linalg.LinAlgError:
        cond = float("inf")

    meta = {
        "T": float(T), "count_rate": float(count_rate),
        "dt": float(dt), "n_lags": int(n),
        "include_shot": bool(include_shot),
        "model": model.key,
    }
    return NoiseCovariance(tau=tau, sigma=sigma, sigma_signal=sigma_signal,
                           var_shot=var_shot, cond=cond, meta=meta)


def _infer_bin_widths(tau: np.ndarray) -> np.ndarray:
    """
    Approximate lag-bin widths from bin-centre spacing on a (log-spaced) grid.
    Uses the geometric midpoints between neighbours; end bins mirror their
    single neighbour.  Prefer passing explicit ``bin_widths`` from the actual
    tau_edges when available.
    """
    tau = np.asarray(tau, dtype=np.float64)
    n = tau.size
    if n == 1:
        return tau.copy()
    edges = np.empty(n + 1)
    edges[1:-1] = np.sqrt(tau[:-1] * tau[1:])          # geometric midpoints
    edges[0] = tau[0] ** 2 / edges[1]
    edges[-1] = tau[-1] ** 2 / edges[-2]
    return np.diff(edges)


# ─────────────────────────────────────────────────────────────────────────────
# Jacobian and Fisher information
# ─────────────────────────────────────────────────────────────────────────────

def model_jacobian(
    model: "fcs_models.FCSModel",
    params: Dict[str, float],
    tau: Sequence[float],
    free_names: Sequence[str],
    *,
    rel_step: float = 1e-4,
) -> np.ndarray:
    """
    Central-difference Jacobian J[k, p] = d G(tau_k) / d theta_p at ``params``,
    for the free parameters ``free_names``.  The models carry no analytic
    derivatives, so finite differences are used with a per-parameter relative
    step (absolute floor for parameters near zero).
    """
    tau = np.asarray(tau, dtype=np.float64)
    names = model.param_names()
    base = {n: float(params[n]) for n in names}
    J = np.empty((tau.size, len(free_names)))

    for p, pname in enumerate(free_names):
        v = base[pname]
        h = rel_step * abs(v) if abs(v) > 0 else rel_step
        up, dn = dict(base), dict(base)
        up[pname] = v + h
        dn[pname] = v - h
        g_up = np.asarray(model.func(tau, **up), dtype=np.float64)
        g_dn = np.asarray(model.func(tau, **dn), dtype=np.float64)
        J[:, p] = (g_up - g_dn) / (2.0 * h)
    return J


@dataclass
class InformationResult:
    """Result of fisher_information()."""
    free_names: List[str]
    fim: np.ndarray                 # J^T Sigma^-1 J  (information matrix)
    param_cov: np.ndarray           # inv(fim)  (parameter covariance / CRLB^2)
    crlb: Dict[str, float]          # per-parameter 1-sigma bound
    logdet_fim: float               # D-optimality
    cond_sigma: float               # conditioning of the data covariance
    used_pinv: bool                 # whether a pseudo-inverse was needed


def fisher_information(
    model: "fcs_models.FCSModel",
    params: Dict[str, float],
    tau: Sequence[float],
    sigma: np.ndarray,
    free_names: Sequence[str],
    *,
    jitter_rel: float = 0.0,
) -> InformationResult:
    """
    Generalized-least-squares Fisher information I = J^T Sigma^-1 J and the
    parameter covariance / Cramer-Rao bounds it implies.

    Sigma^-1 J is obtained by solving Sigma X = J (never by forming Sigma^-1
    explicitly).  When Sigma is ill-conditioned -- the expected signature of a
    binning scheme so fine that adjacent channels are nearly redundant -- an
    optional relative Tikhonov ``jitter`` may be added to the diagonal, and a
    pseudo-inverse is used as a last resort with ``used_pinv=True`` flagged.
    """
    tau = np.asarray(tau, dtype=np.float64)
    free_names = list(free_names)
    S = np.asarray(sigma, dtype=np.float64)
    if jitter_rel > 0.0:
        S = S + jitter_rel * np.mean(np.diag(S)) * np.eye(S.shape[0])

    J = model_jacobian(model, params, tau, free_names)

    used_pinv = False
    try:
        X = np.linalg.solve(S, J)                  # Sigma^-1 J
    except np.linalg.LinAlgError:
        X = np.linalg.pinv(S) @ J
        used_pinv = True

    fim = J.T @ X
    fim = 0.5 * (fim + fim.T)

    try:
        param_cov = np.linalg.inv(fim)
    except np.linalg.LinAlgError:
        param_cov = np.linalg.pinv(fim)
        used_pinv = True

    crlb = {free_names[i]: float(np.sqrt(max(param_cov[i, i], 0.0)))
            for i in range(len(free_names))}
    sign, logdet = np.linalg.slogdet(fim)
    try:
        cond_sigma = float(np.linalg.cond(S))
    except np.linalg.LinAlgError:
        cond_sigma = float("inf")

    return InformationResult(
        free_names=free_names, fim=fim, param_cov=param_cov, crlb=crlb,
        logdet_fim=float(logdet if sign > 0 else -np.inf),
        cond_sigma=cond_sigma, used_pinv=used_pinv,
    )


def information_report_lines(info: InformationResult,
                             weighted: bool = True) -> List[str]:
    """
    Render the information/CRLB summary using fcs_fisher's formatter, so a
    binning-comparison report reads the same as the per-fit Fisher section.
    fcs_fisher works from a *parameter covariance*, which here is inv(FIM).
    """
    fi = fcs_fisher.fisher_from_covariance(info.param_cov, info.free_names,
                                           weighted=weighted)
    lines = fcs_fisher.format_report_lines(fi)
    lines.append("")
    lines.append(f"  cond(Sigma_data)         : {info.cond_sigma:.6g}")
    if info.used_pinv:
        lines.append("  (data covariance was ill-conditioned; pseudo-inverse used)")
    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Monte-Carlo validation (spectral synthesis of a Gaussian intensity trace)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_estimator_covariance(
    model: "fcs_models.FCSModel",
    params: Dict[str, float],
    tau: Sequence[float],
    T: float,
    count_rate: float,
    *,
    dt: Optional[float] = None,
    n_real: int = 200,
    poisson: bool = True,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate noise_covariance() by direct simulation.

    Synthesises ``n_real`` stationary Gaussian intensity traces of duration T
    and mean R with autocovariance C(u) = R^2 * G(|u|) via circulant (FFT)
    spectral synthesis, optionally Poisson-samples them into photon counts per
    dt bin, forms the correlation estimator on the lag grid ``tau``, and returns
    the empirical mean curve and covariance across realizations.

    Compare the returned covariance with noise_covariance(...).sigma:  agreement
    validates the Bartlett/Wick implementation (and, with ``poisson=True``, the
    shot-noise term).  This does not test the Gaussian assumption itself.

    Notes
    -----
    *   dt must resolve both the fastest correlation decay and the smallest lag;
        it defaults to min(tau)/4.
    *   Lags are snapped to the nearest multiple of dt.
    *   This is a validation tool: keep T and n_real modest.
    """
    tau = np.asarray(tau, dtype=np.float64)
    g = model_g_callable(model, params)
    rng = np.random.default_rng(seed)

    if dt is None:
        dt = float(tau.min()) / 4.0
    # Keep the *total duration* exactly T so the empirical variance matches the
    # analytical (1/T) prefactor; choose an FFT-friendly length and absorb the
    # rounding into a sub-bin adjustment of dt.
    from scipy.fft import next_fast_len
    n_t = int(next_fast_len(int(np.ceil(T / dt))))
    dt = T / n_t

    # Target autocovariance on the circular grid, then its (non-negative) PSD.
    lags = np.arange(n_t) * dt
    # Even, circular ACF: C(k) for k and n_t-k identified.
    c = (count_rate ** 2) * g(np.minimum(lags, (n_t * dt) - lags))
    psd = np.fft.rfft(c).real
    psd = np.clip(psd, 0.0, None)                  # enforce a valid spectrum
    amp = np.sqrt(psd)

    # Lag indices for the estimator.
    lag_idx = np.round(tau / dt).astype(int)
    lag_idx = np.clip(lag_idx, 1, n_t - 1)

    G_real = np.empty((n_real, tau.size))
    for r in range(n_real):
        # Spectral synthesis of a real Gaussian process with the given PSD.
        # Complex white noise with unit power per bin (E|z|^2 = 1) so that
        # E|spec[f]|^2 = psd[f]; DC and Nyquist bins are real.
        z = (rng.standard_normal(amp.size)
             + 1j * rng.standard_normal(amp.size)) * np.sqrt(0.5)
        z[0] = z[0].real
        if n_t % 2 == 0:
            z[-1] = z[-1].real
        spec = amp * z
        dI = np.fft.irfft(spec, n_t)
        dI *= np.sqrt(n_t)                          # spectral-synthesis normalisation
        I = count_rate + dI
        I = np.clip(I, 0.0, None)

        if poisson:
            counts = rng.poisson(I * dt).astype(np.float64)
            trace = counts / dt                     # back to intensity units
        else:
            trace = I

        m = trace.mean()
        d = trace - m
        # Normalised correlation estimator G(tau) = <dI(t) dI(t+tau)> / <I>^2.
        for j, k in enumerate(lag_idx):
            prod = d[:-k] * d[k:] if k > 0 else d * d
            G_real[r, j] = prod.mean() / (m * m)

    mean_curve = G_real.mean(axis=0)
    emp_cov = np.cov(G_real, rowvar=False)
    return mean_curve, emp_cov


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test / usage example
# ─────────────────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """
    Minimal end-to-end check on the single-species 3D-Gaussian model: build the
    analytical covariance, its Fisher information, and compare the diagonal
    against a short Monte-Carlo run.  Run:  python fcs_noise.py
    """
    model = fcs_models.MODELS["diffusion_3d_1comp"]
    params = model.defaults()
    params.update({"G0": 0.05, "tau_D": 3e-4, "offset": 0.0})   # ~20 molecules

    tau = np.logspace(-6, -1, 40)      # 1 us .. 100 ms, 40 log-spaced lags
    T = 10.0                            # 10 s acquisition
    R = 2.0e5                           # 200 kHz

    nc = noise_covariance(model, params, tau, T=T, count_rate=R)
    free = [n for n in ("G0", "tau_D") if n in model.param_names()]
    info = fisher_information(model, params, tau, nc.sigma, free)

    print(f"cond(Sigma)         = {nc.cond:.3e}")
    print(f"CRLB (analytical)   : " +
          "  ".join(f"{k}={v:.3e}" for k, v in info.crlb.items()))
    print(f"log|FIM|            = {info.logdet_fim:.4g}")

    # Cheap MC diagonal check: short trace, lags where tau_D is well resolved.
    T_mc = 0.3
    tau_mc = np.logspace(-4.2, -3.0, 6)
    dt_mc = params["tau_D"] / 40.0
    nc_mc = noise_covariance(model, params, tau_mc, T=T_mc, count_rate=R,
                             include_shot=False, dt=dt_mc)
    _, emp = simulate_estimator_covariance(
        model, params, tau_mc, T=T_mc, count_rate=R, dt=dt_mc,
        n_real=400, poisson=False, seed=0)
    ratio = np.sqrt(np.diag(emp) / np.diag(nc_mc.sigma))
    print("sqrt(var_MC / var_analytical) on diagonal "
          "(~1 validates the implementation):")
    print("  " + "  ".join(f"{x:.2f}" for x in ratio))


if __name__ == "__main__":
    _selftest()
