"""
fcs_corr.py
===========
Fluorescence Correlation Spectroscopy — correlation functions.

Segmented estimation
--------------------
The primary computation path divides the photon stream into the maximum
number of non-overlapping segments whose duration is at least
_MIN_SEGMENT_FACTOR × tau_max.  Each segment yields an independent G(τ)
curve; the mean is reported as the correlation and the standard deviation
as the uncertainty.  A warning is issued when fewer than _MIN_SEGMENTS
segments are available, since the std estimate becomes unreliable.

Three computation backends
--------------------------
  "perbin"          Per-bin searchsorted.  Fully vectorised; O(n_bins × N
                    log N).  Fast for all dataset sizes.  Default.

  "twopointer"      Wahl two-pointer algorithm — direct equivalent of the
                    MATLAB tcspc_crosscorr_directed.m implementation.
                    O(N²) worst case in pure Python; JIT-compiled with
                    numba when installed (20–100× faster).

  "wiener_khinchin" Not yet implemented — placeholder in dialog.

Public API
----------
    segment_times(times_s, seg_duration_s)         -> list[np.ndarray]
    compute_segmented(timesA_s, timesB_s,
                      tau_edges, method)            -> (tau, G_mean, G_std, n_seg)
    compute_crosscorr(timesA_s, timesB_s,
                      tau_edges, method)            -> (tau, G_mean, G_std, n_seg)
    compute_autocorr(times_s, tau_edges, method)   -> (tau, G_mean, G_std, n_seg)
    build_tau_edges(tau_min_s, tau_max_s)           -> np.ndarray
    plot_correlation(tau, G_mean, G_std, ...)       -> (fig, ax)
    run_correlation_dialog(fcs_data)                -> shows dialog + plot
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fcs_plottools

from fcs_reader import FCSData, read_fcs
import fcs_lifetime
import fcs_export

# ── Optional numba ────────────────────────────────────────────────────────────

try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    def _njit(fn=None, *args, **kwargs):
        if fn is not None:
            return fn
        return lambda f: f
    _NUMBA = False

_NUMBA_THRESHOLD = 50_000
Method = Literal["perbin", "twopointer", "wiener_khinchin"]

# ── Segmentation constants ────────────────────────────────────────────────────

# Minimum segment duration as a multiple of tau_max.
# Below this, the long-lag bins within a segment are poorly sampled.
_MIN_SEGMENT_FACTOR = 10

# Warn the user when fewer than this many segments are available.
_MIN_SEGMENTS = 5


# ── Lag axis ──────────────────────────────────────────────────────────────────

_POINTS_PER_DECADE = 20


def build_tau_edges(tau_min_s: float, tau_max_s: float) -> np.ndarray:
    """
    Build a log-spaced array of lag bin edges (~30 per decade).

    Parameters
    ----------
    tau_min_s, tau_max_s : float
        Minimum and maximum lag times in seconds.

    Returns
    -------
    np.ndarray  — monotonically increasing edge array.
    """
    if tau_min_s <= 0 or tau_max_s <= tau_min_s:
        raise ValueError(
            f"Need 0 < tau_min < tau_max; got {tau_min_s:.3g}, {tau_max_s:.3g}"
        )
    log_min = np.log10(tau_min_s)
    log_max = np.log10(tau_max_s)
    n = max(10, int(round((log_max - log_min) * _POINTS_PER_DECADE)) + 1)
    return np.logspace(log_min, log_max, n)


# ── Segmentation ──────────────────────────────────────────────────────────────

def segment_times(
    times_s: np.ndarray,
    seg_duration_s: float,
) -> List[np.ndarray]:
    """
    Split a sorted photon arrival time array into non-overlapping segments
    of exactly seg_duration_s seconds.

    Photons in any trailing partial segment (shorter than seg_duration_s)
    are discarded so that every segment has the same duration and the
    (T − τ) normalisation factor is identical across segments.

    Parameters
    ----------
    times_s : np.ndarray
        Sorted photon arrival times in seconds.
    seg_duration_s : float
        Duration of each segment in seconds.

    Returns
    -------
    list of np.ndarray
        Each element is a time array with times re-zeroed to the start of
        that segment (so normalisation is computed correctly per segment).
    """
    t_start = times_s[0]
    t_end   = times_s[-1]
    total   = t_end - t_start
    n_segs  = int(total // seg_duration_s)

    segments = []
    for k in range(n_segs):
        lo = t_start + k * seg_duration_s
        hi = lo + seg_duration_s
        mask = (times_s >= lo) & (times_s < hi)
        seg  = times_s[mask] - lo   # re-zero to segment start
        if len(seg) > 1:
            segments.append(seg)

    return segments


# ── Correlator backends ───────────────────────────────────────────────────────

def _correlate_perbin(
    timesA: np.ndarray,
    timesB: np.ndarray,
    tau_edges: np.ndarray,
) -> np.ndarray:
    """
    Per-bin vectorised cross-correlator.

    For each lag bin k, two numpy.searchsorted calls find, for every A
    photon simultaneously, how many B photons fall in the window
    [tA + tau_edges[k], tA + tau_edges[k+1]).  No Python loop over photons.

    Complexity: O(n_bins × N log N).
    """
    nBins  = len(tau_edges) - 1
    counts = np.zeros(nBins, dtype=np.float64)
    for k in range(nBins):
        lo = np.searchsorted(timesB, timesA + tau_edges[k],     side='left')
        hi = np.searchsorted(timesB, timesA + tau_edges[k + 1], side='left')
        counts[k] = float(np.sum(hi - lo))
    return counts


def _correlate_twopointer_numpy(
    timesA: np.ndarray,
    timesB: np.ndarray,
    tau_edges: np.ndarray,
) -> np.ndarray:
    """
    Wahl two-pointer cross-correlator — pure Python inner loop.

    Direct translation of tcspc_crosscorr_directed.m.  j_start advances
    monotonically so total pointer work is O(N_A + N_B), but the number
    of pairs P counted can be O(N²), making overall complexity O(N² log
    n_bins) in the worst case.
    """
    nBins  = len(tau_edges) - 1
    counts = np.zeros(nBins, dtype=np.float64)
    maxTau = tau_edges[-1]
    minTau = tau_edges[0]
    NB     = len(timesB)
    j_start = 0

    for i in range(len(timesA)):
        tA = timesA[i]
        while j_start < NB and timesB[j_start] < tA + minTau:
            j_start += 1
        j = j_start
        while j < NB:
            dt = timesB[j] - tA
            if dt > maxTau:
                break
            idx = np.searchsorted(tau_edges, dt, side='right') - 1
            if 0 <= idx < nBins:
                counts[idx] += 1.0
            j += 1

    return counts


@_njit(cache=True)
def _correlate_twopointer_numba(
    timesA: np.ndarray,
    timesB: np.ndarray,
    tau_edges: np.ndarray,
) -> np.ndarray:
    """
    Wahl two-pointer cross-correlator — numba JIT-compiled.

    Identical algorithm to _correlate_twopointer_numpy; uses an inline
    binary search because numpy.searchsorted is unavailable in nopython
    mode.  Typical speedup: 20–100× over the pure Python version.
    """
    nBins   = len(tau_edges) - 1
    counts  = np.zeros(nBins, dtype=np.float64)
    NB      = len(timesB)
    maxTau  = tau_edges[-1]
    minTau  = tau_edges[0]
    j_start = 0

    for i in range(len(timesA)):
        tA = timesA[i]
        while j_start < NB and timesB[j_start] < tA + minTau:
            j_start += 1
        j = j_start
        while j < NB:
            dt = timesB[j] - tA
            if dt > maxTau:
                break
            left, right = 0, nBins - 1
            while left <= right:
                mid = (left + right) >> 1
                if dt < tau_edges[mid]:
                    right = mid - 1
                elif dt >= tau_edges[mid + 1]:
                    left = mid + 1
                else:
                    counts[mid] += 1.0
                    break
            j += 1

    return counts


def _correlate(
    timesA: np.ndarray,
    timesB: np.ndarray,
    tau_edges: np.ndarray,
    method: Method,
) -> np.ndarray:
    """Dispatch to the requested backend, returning raw pair counts."""
    if method == "perbin":
        return _correlate_perbin(timesA, timesB, tau_edges)
    if method == "twopointer":
        N = max(len(timesA), len(timesB))
        if _NUMBA and N >= _NUMBA_THRESHOLD:
            return _correlate_twopointer_numba(timesA, timesB, tau_edges)
        return _correlate_twopointer_numpy(timesA, timesB, tau_edges)
    if method == "wiener_khinchin":
        raise NotImplementedError(
            "Wiener–Khinchin FFT correlator is not yet implemented."
        )
    raise ValueError(f"Unknown method: {method!r}")


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize(
    counts: np.ndarray,
    timesA: np.ndarray,
    timesB: np.ndarray,
    tau_edges: np.ndarray,
) -> np.ndarray:
    """
    Normalise raw pair counts to G(τ) with baseline 0.

    G(τ) = counts / [rateA × rateB × (T − τ) × Δτ]  −  1

    Returns G as a float64 array; bins where norm → 0 are set to NaN.
    """
    tau        = 0.5 * (tau_edges[:-1] + tau_edges[1:])
    bin_widths = np.diff(tau_edges)
    T          = (max(timesA[-1], timesB[-1])
                  - min(timesA[0],  timesB[0]))
    rateA = len(timesA) / T
    rateB = len(timesB) / T
    norm  = rateA * rateB * (T - tau) * bin_widths
    with np.errstate(invalid='ignore', divide='ignore'):
        G = np.where(norm > 0, counts / norm - 1.0, np.nan)
    return G


# ── Segmented computation (core) ──────────────────────────────────────────────

def compute_segmented(
    timesA_s: np.ndarray,
    timesB_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute a cross-correlation, optionally with per-segment uncertainty.

    When segment=False (default), the full photon streams are correlated
    in one pass.  G_std is all-NaN and n_segs=1, so plot_correlation
    draws only the black dots with no grey uncertainty lines.

    When segment=True, the streams are divided into the maximum number of
    non-overlapping segments of duration >= _MIN_SEGMENT_FACTOR x tau_max.
    Each segment yields an independent G(tau); the mean and standard
    deviation across segments are returned.  Retained for future use but
    should be interpreted with caution -- see module docstring.

    For autocorrelation, pass the same array for both timesA_s and timesB_s.

    Parameters
    ----------
    timesA_s, timesB_s : np.ndarray
        Sorted photon arrival times in seconds.
    tau_edges : np.ndarray
        Lag bin edges in seconds (from build_tau_edges).
    method : Method
        Computation backend.
    segment : bool
        If False (default), correlate the full dataset as one block.
        If True, segment and return mean +/- std across segments.

    Returns
    -------
    tau    : np.ndarray -- bin centre lag times (seconds)
    G_mean : np.ndarray -- G(tau), baseline 0
    G_std  : np.ndarray -- std across segments, or all-NaN if segment=False
    n_segs : int        -- number of segments used (1 if segment=False)
    """
    tau = 0.5 * (tau_edges[:-1] + tau_edges[1:])

    if not segment:
        # Full-dataset path
        t_start = max(timesA_s[0], timesB_s[0])
        t_end   = min(timesA_s[-1], timesB_s[-1])
        maskA   = (timesA_s >= t_start) & (timesA_s <= t_end)
        maskB   = (timesB_s >= t_start) & (timesB_s <= t_end)
        segA    = timesA_s[maskA] - t_start
        segB    = timesB_s[maskB] - t_start
        counts  = _correlate(segA, segB, tau_edges, method)
        G_mean  = _normalize(counts, segA, segB, tau_edges)
        G_std   = np.full_like(G_mean, np.nan)
        return tau, G_mean, G_std, 1

    # Segmented path
    tau_max        = tau_edges[-1]
    seg_duration_s = _MIN_SEGMENT_FACTOR * tau_max
    t_start = max(timesA_s[0], timesB_s[0])
    t_end   = min(timesA_s[-1], timesB_s[-1])
    total   = t_end - t_start
    n_segs  = max(1, int(total // seg_duration_s))

    G_segments = []
    for k in range(n_segs):
        lo = t_start + k * seg_duration_s
        hi = lo + seg_duration_s
        maskA = (timesA_s >= lo) & (timesA_s < hi)
        maskB = (timesB_s >= lo) & (timesB_s < hi)
        segA  = timesA_s[maskA] - lo
        segB  = timesB_s[maskB] - lo
        if len(segA) < 2 or len(segB) < 2:
            continue
        counts = _correlate(segA, segB, tau_edges, method)
        G_segments.append(_normalize(counts, segA, segB, tau_edges))

    if not G_segments:
        raise ValueError(
            "No valid segments could be computed.  "
            "The dataset may be too short relative to tau_max."
        )

    G_stack = np.array(G_segments)
    n_valid = len(G_stack)
    G_mean  = np.nanmean(G_stack, axis=0)
    G_std   = np.nanstd(G_stack, axis=0, ddof=1) if n_valid > 1 \
              else np.full_like(G_mean, np.nan)

    return tau, G_mean, G_std, n_valid



# ── Public API ────────────────────────────────────────────────────────────────

def compute_crosscorr(
    timesA_s: np.ndarray,
    timesB_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Directed cross-correlation G_AB(tau) with optional segmented uncertainty.

    Returns
    -------
    tau, G_mean, G_std, n_segments
    """
    return compute_segmented(timesA_s, timesB_s, tau_edges, method, segment)


def compute_autocorr(
    times_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Autocorrelation G(tau) with optional segmented uncertainty.

    Returns
    -------
    tau, G_mean, G_std, n_segments
    """
    return compute_segmented(times_s, times_s, tau_edges, method, segment)


# ── Plotting ──────────────────────────────────────────────────────────────────

_CORR_LABEL = {
    "auto_ch1": "Autocorr Ch1",
    "auto_ch2": "Autocorr Ch2",
    "cross":    "Cross-corr Ch1→Ch2",
}
_METHOD_LABEL = {
    "perbin":          "per-bin searchsorted",
    "twopointer":      "two-pointer (Wahl)",
    "wiener_khinchin": "Wiener–Khinchin",
}


def _export_correlation(
    fcs_data: FCSData,
    tau: np.ndarray,
    G_mean: np.ndarray,
    G_std: np.ndarray,
    corr_type: str,
    method: Method,
    tau_min_s: float,
    tau_max_s: float,
    n_segs: int,
    gate_min_ns: Optional[float] = None,
    gate_max_ns: Optional[float] = None,
) -> None:
    """Write one file's plotted correlation curve to a CSV."""
    cols: dict = {
        "tau_s":  tau,
        "tau_ms": tau * 1e3,
        "G":      G_mean,
    }
    if np.isfinite(G_std).any():
        cols["G_std"] = G_std
    meta = {
        "type":      corr_type,
        "method":    method,
        "tau_min_s": f"{tau_min_s:.6g}",
        "tau_max_s": f"{tau_max_s:.6g}",
        "n_segments": n_segs,
        # Measurement summary — recorded so downstream CPS / brightness
        # analyses can be done from the correlation file alone, without the
        # (often large) source .fcs file.  These are the raw measurement
        # values for the whole acquisition, independent of any time gate.
        "cps_ch1":            f"{fcs_data.count_rate_ch1_hz:.6g}",
        "cps_ch2":            f"{fcs_data.count_rate_ch2_hz:.6g}",
        "acquisition_time_s": f"{fcs_data.duration_s:.6g}",
        "n_photons_ch1":      len(fcs_data.ch1_deltas),
        "n_photons_ch2":      len(fcs_data.ch2_deltas),
    }
    if gate_min_ns is not None:
        meta["gate_min_ns"] = f"{gate_min_ns:.3f}"
        meta["gate_max_ns"] = f"{gate_max_ns:.3f}"
    fcs_export.safe_export(
        fcs_data, "correlation", cols, meta=meta, suffix=corr_type,
    )


def plot_correlation(
    tau: np.ndarray,
    G_mean: np.ndarray,
    G_std: np.ndarray,
    corr_type: str,
    fcs_data: FCSData,
    tau_min_s: float,
    tau_max_s: float,
    n_segs: int,
    method: Method = "perbin",
    gate_min_ns: Optional[float] = None,
    gate_max_ns: Optional[float] = None,
    show: bool = True,
    export: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a segmented correlation function with uncertainty bounds.

    The mean G(τ) is drawn as black dots.  The ±1σ band (mean ± std) is
    drawn as light grey lines above and below.

    Parameters
    ----------
    tau      : lag times in seconds
    G_mean   : mean normalised correlation (baseline 0)
    G_std    : standard deviation across segments
    corr_type: 'auto_ch1', 'auto_ch2', or 'cross'
    fcs_data : source FCSData (for title / photon counts)
    tau_min_s, tau_max_s : lag range in seconds
    n_segs   : number of segments used (shown in subtitle)
    method   : backend used (shown in annotation)
    gate_min_ns, gate_max_ns : gate window in ns, or None if not gated
    show     : call plt.show() if True

    Returns
    -------
    fig, ax
    """
    # ── Optional CSV export of the plotted data ───────────────────────────────
    if export:
        _export_correlation(
            fcs_data, tau, G_mean, G_std, corr_type, method,
            tau_min_s, tau_max_s, n_segs, gate_min_ns, gate_max_ns,
        )

    fig, ax = plt.subplots(figsize=(9, 4.5))

    label = _CORR_LABEL.get(corr_type, corr_type)
    tau_ms = tau * 1e3   # display in milliseconds

    # ── ±1σ bounds ────────────────────────────────────────────────────────────
    # Draw upper and lower uncertainty lines in light grey before the mean
    # dots so the dots sit on top visually.
    have_std = not np.all(np.isnan(G_std))
    if have_std:
        upper = G_mean + G_std
        lower = G_mean - G_std

        # Mask NaN so semilogx doesn't leave gaps in the grey lines
        mask_u = np.isfinite(upper)
        mask_l = np.isfinite(lower)

        ax.semilogx(tau_ms[mask_u], upper[mask_u],
                    color="lightgrey", linewidth=1.0,
                    label=f"±1σ  (n={n_segs} segments)")
        ax.semilogx(tau_ms[mask_l], lower[mask_l],
                    color="lightgrey", linewidth=1.0)

    # ── Mean correlation ──────────────────────────────────────────────────────
    mask_m = np.isfinite(G_mean)
    ax.semilogx(
        tau_ms[mask_m], G_mean[mask_m],
        color="black",
        linestyle="none",
        marker=".",
        markersize=4,
        label=label,
    )

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")

    # ── Labels and title ──────────────────────────────────────────────────────
    ax.set_xlabel("Lag time τ (ms)", fontsize=12)
    ax.set_ylabel("G(τ)", fontsize=12)

    n_ch1 = len(fcs_data.ch1_deltas)
    n_ch2 = len(fcs_data.ch2_deltas)
    title = f"Correlation — {fcs_data.filepath.name}"
    gate_str = (f"  ·  gate: {gate_min_ns:.2f}–{gate_max_ns:.2f} ns"
                if gate_min_ns is not None else "")
    subtitle = (
        f"{label}  ·  "
        f"τ: {tau_min_s*1e3:.3g}–{tau_max_s*1e3:.3g} ms  ·  "
        f"{n_segs} segments"
        f"{gate_str}  ·  "
        f"Ch1: {n_ch1:,}  Ch2: {n_ch2:,} photons"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlim(tau_min_s * 1e3, tau_max_s * 1e3)

    # Y limits: headroom above peak, small margin below zero
    g_finite = G_mean[mask_m]
    if have_std:
        upper_finite = (G_mean + G_std)[np.isfinite(G_mean + G_std)]
        lower_finite = (G_mean - G_std)[np.isfinite(G_mean - G_std)]
        y_top = upper_finite.max() if len(upper_finite) else g_finite.max()
        y_bot = lower_finite.min() if len(lower_finite) else g_finite.min()
    else:
        y_top = g_finite.max() if len(g_finite) else 1.0
        y_bot = g_finite.min() if len(g_finite) else 0.0
    span = y_top - y_bot if y_top != y_bot else 1.0
    ax.set_ylim(y_bot - 0.05 * span, y_top + 0.15 * span)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.3g}")
    )
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
    ax.legend(fontsize=10, framealpha=0.85)

    # Bottom-right: method + numba status
    method_str = _METHOD_LABEL.get(method, method)
    if method == "twopointer":
        method_str += f" [{'numba' if _NUMBA else 'numpy'}]"
    fig.text(0.99, 0.01, method_str,
             ha="right", va="bottom", fontsize=7, color="grey")

    fig.tight_layout()
    if show:
        #plt.show was static; now dynamic w/ fcs_plottools
        #plt.show()
        fcs_plottools.show_figure(fig, ax)
    return fig, ax



# ── Time gating ───────────────────────────────────────────────────────────────

def apply_time_gate(
    times_s: np.ndarray,
    micro_ns: np.ndarray,
    gate_min_ns: float,
    gate_max_ns: float,
) -> np.ndarray:
    """
    Filter a photon arrival time array by microtime gate.

    Keeps only photons whose microtime (arrival time within the laser cycle)
    falls in [gate_min_ns, gate_max_ns).  Returns the corresponding subset
    of the macrotime array times_s.

    Parameters
    ----------
    times_s     : np.ndarray — absolute photon arrival times (seconds)
    micro_ns    : np.ndarray — microtime of each photon (nanoseconds),
                               same length as times_s
    gate_min_ns : float — lower gate edge (ns, inclusive)
    gate_max_ns : float — upper gate edge (ns, exclusive)

    Returns
    -------
    np.ndarray — filtered times_s, sorted (inherits sort order of input)
    """
    mask = (micro_ns >= gate_min_ns) & (micro_ns < gate_max_ns)
    return times_s[mask]


# ── Per-file compute + overlay (batch / combined) ─────────────────────────────

def compute_correlation_for(
    fcs_data: FCSData,
    params: dict,
    parent=None,
) -> Optional[dict]:
    """
    Compute G(tau) for a single file from a parameter dict.

    *params* uses the same keys the dialog persists: ``tau_min_ms``,
    ``tau_max_ms``, ``corr_type`` ('cross' | 'auto_ch1' | 'auto_ch2'),
    ``method``, ``segment`` (bool), and ``gate`` (bool).  When ``gate`` is
    True the interactive microtime gate is shown for *this* file (the gate is
    file-specific), so in a batch each file gets its own gate window.

    Returns
    -------
    dict with keys tau, G_mean, G_std, n_segs, tau_min_s, tau_max_s,
    gate_min_ns, gate_max_ns — or None if the user cancelled the gate, the
    gate was too narrow, or the computation failed.
    """
    from tkinter import messagebox

    corr_type = params["corr_type"]
    method    = params["method"]
    segment   = params["segment"]
    use_gate  = params["gate"]
    tau_min_s = float(params["tau_min_ms"]) * 1e-3
    tau_max_s = float(params["tau_max_ms"]) * 1e-3
    tau_edges = build_tau_edges(tau_min_s, tau_max_s)

    times_ch1 = fcs_data.ch1_times_s
    times_ch2 = fcs_data.ch2_times_s

    if use_gate:
        gate = fcs_lifetime.select_gate(
            fcs_data,
            title=f"Set time gate — {fcs_data.filepath.name}",
        )
        if gate is None:
            return None   # user cancelled the gate window for this file
        gate_min_ns, gate_max_ns = gate
        times_ch1 = apply_time_gate(
            times_ch1, fcs_data.ch1_micro_ns, gate_min_ns, gate_max_ns)
        times_ch2 = apply_time_gate(
            times_ch2, fcs_data.ch2_micro_ns, gate_min_ns, gate_max_ns)
        if len(times_ch1) < 10 or len(times_ch2) < 10:
            messagebox.showerror(
                "Gate too narrow",
                f"Gate {gate_min_ns:.2f}–{gate_max_ns:.2f} ns retains only "
                f"{len(times_ch1):,} Ch1 and {len(times_ch2):,} Ch2 photons "
                f"in {fcs_data.filepath.name}.\n\nWiden the gate and try again.",
                parent=parent,
            )
            return None
    else:
        gate_min_ns = gate_max_ns = None

    try:
        if corr_type == "cross":
            tau, G_mean, G_std, n_segs = compute_crosscorr(
                times_ch1, times_ch2, tau_edges, method, segment)
        elif corr_type == "auto_ch1":
            tau, G_mean, G_std, n_segs = compute_autocorr(
                times_ch1, tau_edges, method, segment)
        else:
            tau, G_mean, G_std, n_segs = compute_autocorr(
                times_ch2, tau_edges, method, segment)
    except NotImplementedError as e:
        messagebox.showerror("Not implemented", str(e), parent=parent)
        return None
    except Exception as e:
        messagebox.showerror("Computation error",
                             f"{fcs_data.filepath.name}:\n{e}", parent=parent)
        return None

    return {
        "tau": tau, "G_mean": G_mean, "G_std": G_std, "n_segs": n_segs,
        "tau_min_s": tau_min_s, "tau_max_s": tau_max_s,
        "gate_min_ns": gate_min_ns, "gate_max_ns": gate_max_ns,
    }


def plot_correlation_overlay(
    results,
    corr_type: str,
    method: Method,
    tau_min_s: float,
    tau_max_s: float,
    show: bool = True,
    export: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Overlay G(tau) curves from several files on one semilog-x axes.

    Each file is drawn in its own colour.  The per-file ±1σ uncertainty band
    is omitted (it would clutter a multi-file overlay); the mean curve is
    drawn for every file.  When *export* is True each file's curve is written
    to its own CSV exactly as in the single-file path.

    Parameters
    ----------
    results : sequence of (FCSData, result_dict)
        result_dict as returned by compute_correlation_for.
    corr_type, method : shared correlation type / backend (for the title).
    tau_min_s, tau_max_s : shared lag range in seconds (for the x-limits).

    Returns
    -------
    fig, ax
    """
    results = list(results)
    if not results:
        raise ValueError("plot_correlation_overlay requires at least one result.")

    colours = fcs_plottools.palette(len(results))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    label = _CORR_LABEL.get(corr_type, corr_type)

    y_tops: list[float] = []
    y_bots: list[float] = []
    for (d, res), colour in zip(results, colours):
        tau    = res["tau"]
        G_mean = res["G_mean"]
        tau_ms = tau * 1e3
        mask   = np.isfinite(G_mean)
        ax.semilogx(
            tau_ms[mask], G_mean[mask],
            color=colour, linewidth=1.0, marker=".", markersize=3.5,
            alpha=0.9, label=d.filepath.name,
        )
        if mask.any():
            y_tops.append(float(np.nanmax(G_mean[mask])))
            y_bots.append(float(np.nanmin(G_mean[mask])))
        if export:
            _export_correlation(
                d, tau, G_mean, res["G_std"], corr_type, method,
                res["tau_min_s"], res["tau_max_s"], res["n_segs"],
                res["gate_min_ns"], res["gate_max_ns"],
            )

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Lag time τ (ms)", fontsize=12)
    ax.set_ylabel("G(τ)", fontsize=12)
    ax.set_title(
        f"Correlation overlay — {len(results)} files\n"
        f"{label}  ·  τ: {tau_min_s*1e3:.3g}–{tau_max_s*1e3:.3g} ms  ·  "
        f"{_METHOD_LABEL.get(method, method)}",
        fontsize=10,
    )
    ax.set_xlim(tau_min_s * 1e3, tau_max_s * 1e3)
    if y_tops:
        y_top = max(y_tops)
        y_bot = min(y_bots)
        span  = y_top - y_bot if y_top != y_bot else 1.0
        ax.set_ylim(y_bot - 0.05 * span, y_top + 0.15 * span)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3g}"))
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.85, title="File")

    method_str = _METHOD_LABEL.get(method, method)
    if method == "twopointer":
        method_str += f" [{'numba' if _NUMBA else 'numpy'}]"
    fig.text(0.99, 0.01, method_str,
             ha="right", va="bottom", fontsize=7, color="grey")

    fig.tight_layout()
    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── Dialog ────────────────────────────────────────────────────────────────────

_defaults: dict = {
    "tau_min_ms": 0.01,
    "tau_max_ms": 1000.0,
    "corr_type":  "cross",
    "method":     "perbin",
    "segment":    False,
    "gate":       False,
}


def run_correlation_dialog(fcs_data: FCSData, export: bool = False,
                           *, collect_only: bool = False):
    """
    Show a parameter dialog, then compute and plot the segmented correlation.
    Settings persist between calls within the same session.

    If *collect_only* is True the dialog gathers and returns the chosen
    parameter dict (the same keys ``_defaults`` uses) without computing or
    plotting anything, and the file-specific warnings are skipped.  This is
    used by the batch/overlay path to ask for parameters once and then apply
    them to every selected file via :func:`compute_correlation_for`.  Returns
    None if the user cancels.
    """
    import tkinter as tk
    from tkinter import messagebox

    result_box: dict = {"params": None}

    dialog = tk.Toplevel()
    dialog.title("Correlation — options")
    dialog.geometry("340x520")
    dialog.resizable(False, True)
    dialog.grab_set()

    pad = dict(padx=12, pady=4)

    tk.Label(dialog, text="Correlation options",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    # ── Correlation type ──────────────────────────────────────────────────────
    type_frame = tk.LabelFrame(dialog, text="Type", padx=10, pady=4)
    type_frame.pack(fill="x", **pad)

    type_var = tk.StringVar(value=_defaults["corr_type"])
    for text, value in [
        ("Cross-correlation  Ch1 → Ch2", "cross"),
        ("Autocorrelation  Ch1",         "auto_ch1"),
        ("Autocorrelation  Ch2",         "auto_ch2"),
    ]:
        tk.Radiobutton(type_frame, text=text, variable=type_var,
                       value=value, anchor="w").pack(fill="x")

    # ── Lag range ─────────────────────────────────────────────────────────────
    range_frame = tk.LabelFrame(dialog, text="Lag range (ms)", padx=10, pady=6)
    range_frame.pack(fill="x", **pad)

    row = tk.Frame(range_frame)
    row.pack(fill="x")
    tk.Label(row, text="Min:", width=5, anchor="e").pack(side="left")
    min_var = tk.StringVar(value=str(_defaults["tau_min_ms"]))
    tk.Entry(row, textvariable=min_var, width=10).pack(side="left", padx=(4, 16))
    tk.Label(row, text="Max:", width=5, anchor="e").pack(side="left")
    max_var = tk.StringVar(value=str(_defaults["tau_max_ms"]))
    tk.Entry(row, textvariable=max_var, width=10).pack(side="left", padx=4)

    # ── Segmentation toggle ───────────────────────────────────────────────────
    seg_frame = tk.LabelFrame(dialog, text="Uncertainty", padx=10, pady=4)
    seg_frame.pack(fill="x", **pad)

    segment_var = tk.BooleanVar(value=_defaults["segment"])
    seg_cb = tk.Checkbutton(
        seg_frame,
        text="Segment and average  (experimental)",
        variable=segment_var,
        anchor="w",
    )
    seg_cb.pack(fill="x")

    # Info label: shows expected segment count; only visible when ticked
    seg_info_var = tk.StringVar(value="")
    seg_info_label = tk.Label(seg_frame, textvariable=seg_info_var,
                              font=("Helvetica", 9), fg="grey", anchor="w")

    def _update_seg_info(*_):
        """Recompute expected segment count and show/hide the info label."""
        if not segment_var.get():
            seg_info_label.pack_forget()
            return
        try:
            tau_max_s  = float(max_var.get()) * 1e-3
            seg_dur    = _MIN_SEGMENT_FACTOR * tau_max_s
            total_s    = fcs_data.duration_s
            n          = int(total_s // seg_dur)
            warn       = "  ⚠ few segments" if 0 < n < _MIN_SEGMENTS else ""
            seg_info_var.set(
                f"  {n} segment{'s' if n != 1 else ''}  "
                f"({seg_dur*1e3:.3g} ms each){warn}"
                if n > 0 else
                f"  dataset too short for tau_max  ({total_s:.1f} s available)"
            )
        except ValueError:
            seg_info_var.set("")
        seg_info_label.pack(fill="x")

    segment_var.trace_add("write", _update_seg_info)
    max_var.trace_add("write", _update_seg_info)
    _update_seg_info()


    # ── Time gating ───────────────────────────────────────────────────────────
    gate_frame = tk.LabelFrame(dialog, text="Time gating", padx=10, pady=4)
    gate_frame.pack(fill="x", **pad)

    gate_var = tk.BooleanVar(value=_defaults["gate"])
    tk.Checkbutton(
        gate_frame,
        text="Apply microtime gate  (set interactively on histogram)",
        variable=gate_var,
        anchor="w",
    ).pack(fill="x")

    # ── Method ────────────────────────────────────────────────────────────────
    method_frame = tk.LabelFrame(dialog, text="Method", padx=10, pady=4)
    method_frame.pack(fill="x", **pad)

    method_var = tk.StringVar(value=_defaults["method"])
    tk.Radiobutton(method_frame,
                   text="Per-bin searchsorted  (fast, all sizes)",
                   variable=method_var, value="perbin",
                   anchor="w").pack(fill="x")

    tp_label = ("Two-pointer / Wahl  (numba — fast)"
                if _NUMBA else
                "Two-pointer / Wahl  (pure Python — slow for N > 5k)")
    tk.Radiobutton(method_frame, text=tp_label,
                   variable=method_var, value="twopointer",
                   anchor="w").pack(fill="x")

    tk.Radiobutton(method_frame,
                   text="Wiener–Khinchin / FFT  (coming soon)",
                   variable=method_var, value="wiener_khinchin",
                   anchor="w", state="disabled", fg="grey").pack(fill="x")

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=10)

    def _on_compute():
        try:
            tau_min_ms = float(min_var.get())
            tau_max_ms = float(max_var.get())
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Lag min and max must be numbers.",
                                 parent=dialog)
            return
        if tau_min_ms <= 0 or tau_max_ms <= tau_min_ms:
            messagebox.showerror("Invalid range",
                                 "Need:  0 < lag min < lag max.",
                                 parent=dialog)
            return

        corr_type = type_var.get()
        method    = method_var.get()
        segment   = segment_var.get()
        use_gate  = gate_var.get()
        tau_max_s = tau_max_ms * 1e-3

        params = {
            "tau_min_ms": tau_min_ms,
            "tau_max_ms": tau_max_ms,
            "corr_type":  corr_type,
            "method":     method,
            "segment":    segment,
            "gate":       use_gate,
        }

        # File-specific warnings only matter when we compute here (single file).
        # In collect_only mode the same parameters are reused across many files,
        # so these per-file checks are skipped.
        if not collect_only:
            # Slow-path warning for two-pointer without numba
            if method == "twopointer" and not _NUMBA:
                N = max(len(fcs_data.ch1_deltas), len(fcs_data.ch2_deltas))
                if N > 5_000:
                    if not messagebox.askyesno(
                        "Slow computation",
                        f"{N:,} photons with two-pointer (pure Python) may take "
                        f"several minutes.\n\nProceed anyway?",
                        parent=dialog,
                    ):
                        return

            # Check segment count and warn if low (only when segmentation enabled)
            if segment:
                seg_dur    = _MIN_SEGMENT_FACTOR * tau_max_s
                n_expected = int(fcs_data.duration_s // seg_dur)
                if n_expected < _MIN_SEGMENTS:
                    if not messagebox.askyesno(
                        "Few segments",
                        f"Only {n_expected} segment{'s' if n_expected != 1 else ''} "
                        f"can be formed with tau_max = {tau_max_ms:.3g} ms.\n\n"
                        f"Uncertainty estimates will be unreliable with fewer than "
                        f"{_MIN_SEGMENTS} segments.\n\n"
                        f"Consider reducing tau_max or using a longer dataset.\n\n"
                        f"Proceed anyway?",
                        parent=dialog,
                    ):
                        return

        # Persist
        _defaults.update(params)

        dialog.destroy()

        if collect_only:
            result_box["params"] = params
            return

        res = compute_correlation_for(fcs_data, params)
        if res is None:
            return

        plot_correlation(
            res["tau"], res["G_mean"], res["G_std"],
            corr_type=corr_type,
            fcs_data=fcs_data,
            tau_min_s=res["tau_min_s"],
            tau_max_s=res["tau_max_s"],
            n_segs=res["n_segs"],
            method=method,
            gate_min_ns=res["gate_min_ns"],
            gate_max_ns=res["gate_max_ns"],
            export=export,
        )

    tk.Button(btn_frame, text="Compute", width=12,
              command=_on_compute, pady=4).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel", width=10,
              command=dialog.destroy, pady=4).pack(side="left", padx=6)

    dialog.wait_window()
    return result_box["params"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fcs_corr.py <file.fcs> [tau_min_ms] [tau_max_ms] [method]")
        print("  method: perbin (default) | twopointer")
        sys.exit(1)
    d    = read_fcs(sys.argv[1])
    tmin = float(sys.argv[2]) if len(sys.argv) > 2 else _defaults["tau_min_ms"]
    tmax = float(sys.argv[3]) if len(sys.argv) > 3 else _defaults["tau_max_ms"]
    meth = sys.argv[4]        if len(sys.argv) > 4 else "perbin"

    edges = build_tau_edges(tmin * 1e-3, tmax * 1e-3)
    tau, G_mean, G_std, n_segs = compute_crosscorr(
        d.ch1_times_s, d.ch2_times_s, edges, meth)
    print(f"Segments used: {n_segs}")
    plot_correlation(tau, G_mean, G_std, "cross", d,
                     tmin * 1e-3, tmax * 1e-3, n_segs, method=meth)
