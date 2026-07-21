"""
fcs_corr.py
===========
Fluorescence Correlation Spectroscopy — correlation functions.

Segment-and-average uncertainty
-------------------------------
Segment-and-average is the primary way error bars (G_std) are determined,
and is on by default.  The photon stream is divided into fixed-duration
segments of _SEGMENT_DURATION_S (default 1 s).  Any trailing remainder is
absorbed into the final segment, which is therefore between 1x and 2x the
nominal duration — no short segment is ever produced, so a stub tail cannot
contaminate the statistics.

Each segment yields an independent G(τ).  The reported uncertainty is the
STANDARD ERROR of that ensemble,

    G_std = std(G_segments, ddof=1) / sqrt(n_segs)

i.e. the uncertainty on the reported curve — which is what a fit weight
needs — not the population spread of a single segment.

By default the reported G(τ) itself is estimated from the FULL trace
(full_trace_G=True), with the segments used only for G_std; this avoids the
boundary effects of cutting the stream, at the cost of one extra pass.  Set
full_trace_G=False to report the mean of the segment curves instead.

Because a segment of duration D only supports lags well below D, a 1 s
segment gives trustworthy statistics out to roughly 100 ms.  G_std is
allowed to grow honestly beyond that rather than being capped.

Three computation backends
--------------------------
  "perbin"          Per-bin searchsorted.  Fully vectorised; O(n_bins × N
                    log N).  Fast for all dataset sizes.  Default.

  "twopointer"      Wahl two-pointer algorithm — direct equivalent of the
                    MATLAB tcspc_crosscorr_directed.m implementation.
                    O(N²) worst case in pure Python; JIT-compiled with
                    numba when installed (20–100× faster).

  "wiener_khinchin" Not yet implemented — placeholder in dialog.

Cross-correlation symmetry
--------------------------
ALL cross-correlations are symmetrised: G(τ) = ½[G_AB(τ) + G_BA(τ)], matching
the (Ch1×Ch2 + Ch2×Ch1)/2 convention of ISS VistaVision.  The estimator itself
is directed, so symmetrisation is a deliberate, explicit step -- performed
WITHIN each segment, so the across-segment scatter measures the symmetrised
quantity that is actually reported.

Public API
----------
    segment_bounds(t_start, t_end, seg_duration_s)  -> list[(lo, hi)]
    compute_segmented(timesA_s, timesB_s,
                      tau_edges, method)            -> (tau, G_mean, G_std, n_seg)
    compute_crosscorr(timesA_s, timesB_s,
                      tau_edges, method)            -> (tau, G_mean, G_std, n_seg)
    compute_crosscorr_symmetric(timesA_s, timesB_s,
                      tau_edges, method)            -> (tau, G_mean, G_std, n_seg)
    compute_autocorr(times_s, tau_edges, method)   -> (tau, G_mean, G_std, n_seg)
    multitau_lag_grid(dt0, m, coarsen, tau_max)     -> np.ndarray
    compute_multitau(times_a, times_b, ...)         -> (tau, G_mean, G_std, n_seg)
    build_tau_edges(tau_min_s, tau_max_s)           -> np.ndarray
    plot_correlation(tau, G_mean, G_std, ...)       -> (fig, ax)
    compute_correlation_for(fcs_data, params)       -> result dict | None
    plot_correlation_overlay(results, ...)          -> (fig, ax)
    run_correlation_dialog(fcs_data, collect_only=) -> params | None
"""

from __future__ import annotations

import warnings
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

# ── Optional multipletau ──────────────────────────────────────────────────────
# The multipletau package (Paul Mueller, BSD-3) provides the reference
# implementation of the Schaetzel multiple-tau algorithm.  It is an optional
# dependency: if it is missing the "multitau_pkg" method is disabled in the
# dialog and everything else still runs.
#
#   pip install multipletau
#
# Verified behaviour of multipletau 0.4.1 (measured, not assumed):
#   * correlate()/autocorrelate() consume BINNED INTENSITY, not arrival times.
#   * The lag grid is hard-wired base-2 (measured successive-lag ratio 2.000);
#     there is no coarsening-factor parameter.  This is why the in-house
#     compute_multitau (arbitrary `coarsen`, for ISS-grid matching) is kept
#     alongside rather than replaced.
#   * `m` = points per octave; level 0 emits lags 0..m, each later octave
#     emits m/2+1..m at twice the spacing.  m must be an even integer.
#   * normalize=True ALREADY subtracts the 1: G decays to 0 at large lag,
#     matching this suite's baseline-0 convention.  Do NOT subtract again.
#   * Only the positive direction is computed; swap the inputs for the
#     reverse.  Symmetrisation is therefore ours to do (see `symmetric`).
#   * The tau=0 row carries the shot-noise spike (G(0) = 1/<k> for a Poisson
#     trace, ~2000x the neighbouring lags) and is dropped.

try:
    import multipletau as _multipletau
    _MULTIPLETAU = True
except ImportError:
    _multipletau = None
    _MULTIPLETAU = False

_NUMBA_THRESHOLD = 50_000
Method = Literal["perbin", "twopointer", "wiener_khinchin",
                 "multitau", "multitau_pkg"]

# ── Segmentation constants ────────────────────────────────────────────────────

# Default segment duration (s) for the segment-and-average error estimate.
# The stream is cut into fixed-length segments of this duration; the trailing
# remainder is absorbed into the final segment rather than forming a short one.
_SEGMENT_DURATION_S = 1.0

# Minimum segment duration as a multiple of tau_max.
# Below this, the long-lag bins within a segment are poorly sampled.
_MIN_SEGMENT_FACTOR = 10

# Photon arrival times are quantised to the macrotime (laser) period, so lag
# times below a few grid periods cannot be resolved without the microtime.
# Warn when the requested tau_min comes within this many periods of the grid.
_GRID_WARN_FACTOR = 10

# Warn the user when fewer than this many segments are available.
_MIN_SEGMENTS = 5


# ── Lag axis ──────────────────────────────────────────────────────────────────

_POINTS_PER_DECADE = 20

# ── Time estimation ───────────────────────────────────────────────────────────

def estimate_corr_time(
    N: int,
    n_bins: int,
    method: Method,
    n_segs: int = 1,
    tau_max_s: float = 1.0,
    total_s: float = 1.0,
) -> str:
    """
    Return a rough human-readable estimate of the correlation computation time.

    Parameters
    ----------
    N        : number of photons (max of ch1, ch2)
    n_bins   : number of lag bins
    method   : backend to be used
    n_segs   : number of segments (1 for unsegmented)
    tau_max_s: maximum lag in seconds
    total_s  : total dataset duration in seconds

    Returns
    -------
    str — e.g. "~3 s" or "~2 min" or "< 1 s"
    """
    import math

    # N per segment (photons split roughly evenly across time)
    N_per_seg = N / n_segs if n_segs > 1 else N

    if method == "perbin":
        # O(n_bins × N_seg × log N_seg): searchsorted is very fast in numpy;
        # empirical constant ~5 ns per (bin × photon) on typical hardware.
        _PERBIN_NS = 5e-9   # seconds per bin-photon operation
        ops = n_bins * N_per_seg * math.log2(max(N_per_seg, 2))
        t_seg = _PERBIN_NS * ops
    elif method == "twopointer":
        if _NUMBA and N_per_seg >= _NUMBA_THRESHOLD:
            # Numba JIT: ~10 ns per photon-pair scan step
            _TP_NUMBA_NS = 10e-9
        else:
            # Pure Python: ~300 ns per photon-pair step
            _TP_NUMBA_NS = 300e-9
        # Average B photons in window per A photon ≈ N_B × (tau_max / T)
        lag_frac = min(tau_max_s / max(total_s / n_segs, tau_max_s), 1.0)
        pairs_per_photon = N_per_seg * lag_frac
        t_seg = _TP_NUMBA_NS * N_per_seg * pairs_per_photon
    else:
        return "unknown"

    total_t = t_seg * n_segs

    if total_t < 1.0:
        return "< 1 s"
    elif total_t < 60:
        return f"~{int(round(total_t))} s"
    elif total_t < 3600:
        mins = total_t / 60
        return f"~{mins:.1f} min"
    else:
        hrs = total_t / 3600
        return f"~{hrs:.1f} h"


# ── Progress window ───────────────────────────────────────────────────────────

class _ProgressWindow:
    """
    Lightweight tkinter progress window for long correlation computations.

    Usage
    -----
    pw = _ProgressWindow(parent, total_steps, title="Computing…")
    pw.step(completed, label="Segment 3 / 10")   # update bar + ETA
    pw.close()

    The window is non-blocking: call pw.update() or pw.step() regularly so
    the event loop gets processed and the Cancel button stays responsive.

    pw.cancelled() returns True if the user clicked Cancel.
    """

    def __init__(self, parent, total_steps: int, title: str = "Computing…"):
        import tkinter as tk
        from tkinter import ttk

        self._cancelled = False
        self._total     = max(1, total_steps)
        self._t_start   = None   # set on first step() call

        # Use Toplevel if a root already exists; otherwise create a root.
        # In the normal dialog flow, the main app root is always alive here.
        root = tk._get_default_root("create Toplevel")  # type: ignore[attr-defined]
        self._win = tk.Toplevel(root) if root is not None else tk.Tk()
        self._win.title(title)
        self._win.geometry("360x130")
        self._win.resizable(False, False)
        self._win.grab_set()
        self._win.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Status label (e.g. "Segment 3 / 10")
        self._status_var = tk.StringVar(value="Starting…")
        tk.Label(self._win, textvariable=self._status_var,
                 font=("Helvetica", 10), anchor="w",
                 padx=16).pack(fill="x", pady=(14, 2))

        # Progress bar
        self._bar = ttk.Progressbar(
            self._win, orient="horizontal", length=320,
            mode="determinate", maximum=self._total,
        )
        self._bar.pack(padx=16, pady=4)

        # ETA label
        self._eta_var = tk.StringVar(value="")
        tk.Label(self._win, textvariable=self._eta_var,
                 font=("Helvetica", 9), fg="grey", anchor="w",
                 padx=16).pack(fill="x")

        # Cancel button
        tk.Button(self._win, text="Cancel", width=10,
                  command=self._on_cancel).pack(pady=8)

        self._win.update()

    def _on_cancel(self):
        self._cancelled = True

    def cancelled(self) -> bool:
        return self._cancelled

    def step(self, completed: int, label: str = ""):
        import time
        if self._t_start is None:
            self._t_start = time.monotonic()

        self._bar["value"] = completed
        if label:
            self._status_var.set(label)

        # ETA: elapsed / fraction_done × remaining_fraction
        if completed > 0:
            elapsed  = time.monotonic() - self._t_start
            fraction = completed / self._total
            if fraction > 0:
                eta_s = elapsed / fraction * (1 - fraction)
                if eta_s < 60:
                    self._eta_var.set(f"ETA: {int(eta_s)} s")
                else:
                    self._eta_var.set(f"ETA: {eta_s/60:.1f} min")

        self._win.update()

    def close(self):
        try:
            self._win.destroy()
        except Exception:
            pass


def build_tau_edges(
    tau_min_s: float,
    tau_max_s: float,
    points_per_decade: int = _POINTS_PER_DECADE,
    grid_period_s: Optional[float] = None,
) -> np.ndarray:
    """
    Build a log-spaced array of lag bin edges.

    Parameters
    ----------
    tau_min_s, tau_max_s : float
        Minimum and maximum lag times in seconds.
    points_per_decade : int
        Number of lag bins per decade of lag time (default: _POINTS_PER_DECADE).
        Reducing this linearly reduces n_bins and therefore linearly reduces
        the compute time for the perbin backend.  Typical values:
          20  — full resolution (default)
          10  — half the bins; adequate for fitting standard models
           5  — coarse; useful for a quick preview
    grid_period_s : float, optional
        Photon-timing grid period in seconds — the macrotime (laser) period.
        When supplied, every bin edge is snapped to the nearest multiple of this
        period so each bin spans a whole number of timing-grid points.  This is
        required for correct normalisation: arrival times are quantised to the
        macrotime grid, so pair lags fall only on that grid.  Log edges that
        land between grid points make the continuous bin width disagree with the
        true number of lag points in the bin, injecting a systematic,
        reproducible per-bin bias into G(τ) (identical in auto- and
        cross-correlation, and large enough at short lag to drive G negative).
        Snapping removes it, because the bin width then equals n_grid·period
        exactly and the existing normalisation is correct.  Edges that collapse
        onto the same grid point are merged, so the shortest bins are never
        narrower than one period.

    Returns
    -------
    np.ndarray  — monotonically increasing edge array.
    """
    if tau_min_s <= 0 or tau_max_s <= tau_min_s:
        raise ValueError(
            f"Need 0 < tau_min < tau_max; got {tau_min_s:.3g}, {tau_max_s:.3g}"
        )
    ppd = max(2, int(points_per_decade))
    log_min = np.log10(tau_min_s)
    log_max = np.log10(tau_max_s)
    n = max(10, int(round((log_max - log_min) * ppd)) + 1)
    edges = np.logspace(log_min, log_max, n)

    if grid_period_s and grid_period_s > 0:
        if tau_min_s < _GRID_WARN_FACTOR * grid_period_s:
            warnings.warn(
                f"tau_min ({tau_min_s * 1e9:.1f} ns) is within "
                f"{_GRID_WARN_FACTOR}x the photon-timing grid period "
                f"({grid_period_s * 1e9:.1f} ns). Lags below "
                f"~{_GRID_WARN_FACTOR * grid_period_s * 1e9:.0f} ns cannot be "
                f"resolved without microtime; the shortest bins are unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Snap every edge onto the timing grid; merge any that collapse so no
        # bin is narrower than one period.
        edges = np.round(edges / grid_period_s) * grid_period_s
        edges = np.unique(edges)
        edges = edges[edges >= grid_period_s]
        if edges.size < 2:
            raise ValueError(
                "After snapping to the timing grid, fewer than two edges "
                "remain; increase tau_max (or tau_min) relative to the grid "
                f"period ({grid_period_s * 1e9:.1f} ns)."
            )
    return edges


def thin_photons(times_s: np.ndarray, keep_every: int) -> np.ndarray:
    """
    Uniformly thin a photon arrival time array by retaining every k-th photon.

    This reduces N — and therefore computation time — while preserving the
    full temporal range of the dataset (so normalisation is unaffected).
    The correlation amplitude G(τ) is unchanged; only the noise floor rises
    (as 1/√N_kept).

    Parameters
    ----------
    times_s    : sorted photon arrival time array (seconds)
    keep_every : decimation factor k; keep photons at indices 0, k, 2k, …
                 k=1 returns the original array unchanged.

    Returns
    -------
    np.ndarray — thinned, sorted arrival time array.
    """
    k = max(1, int(keep_every))
    if k == 1:
        return times_s
    return times_s[::k]

# ── Correlator backends ───────────────────────────────────────────────────────

def _correlate_perbin(
    timesA: np.ndarray,
    timesB: np.ndarray,
    tau_edges: np.ndarray,
    progress_cb=None,
    progress_offset: int = 0,
) -> np.ndarray:
    """
    Per-bin vectorised cross-correlator.

    For each lag bin k, two numpy.searchsorted calls find, for every A
    photon simultaneously, how many B photons fall in the window
    [tA + tau_edges[k], tA + tau_edges[k+1]).  No Python loop over photons.

    Complexity: O(n_bins × N log N).

    Parameters
    ----------
    progress_cb : callable(completed: int, label: str) | None
        Called after each bin. completed is the absolute step count
        (progress_offset + bins done so far).
    progress_offset : int
        Added to the completed count; used when this call is one segment
        in a multi-segment computation.
    """
    nBins  = len(tau_edges) - 1
    counts = np.zeros(nBins, dtype=np.float64)
    for k in range(nBins):
        lo = np.searchsorted(timesB, timesA + tau_edges[k],     side='left')
        hi = np.searchsorted(timesB, timesA + tau_edges[k + 1], side='left')
        counts[k] = float(np.sum(hi - lo))
        if progress_cb is not None:
            progress_cb(progress_offset + k + 1, f"Bin {k + 1} / {nBins}")
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


def _correlate_twopointer_chunked(
    timesA: np.ndarray,
    timesB: np.ndarray,
    tau_edges: np.ndarray,
    progress_cb=None,
    progress_offset: int = 0,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """
    Two-pointer correlator with chunked progress reporting.

    The numba/numpy twopointer backends process all photons in one call,
    making mid-run callbacks impossible.  This wrapper slices timesA into
    chunks of chunk_size photons and calls the backend on each chunk,
    accumulating counts and reporting progress between chunks.

    Note: because the two-pointer j_start state is NOT preserved across
    chunks (timesB is always searched from the beginning for each chunk),
    correctness is maintained — j_start is re-derived via searchsorted at
    the start of each chunk.  This adds a small O(log N_B) overhead per
    chunk, which is negligible compared to the inner-loop work.

    Parameters
    ----------
    progress_cb : callable(completed: int, label: str) | None
    progress_offset : int
    chunk_size  : number of A-photons per chunk (default 10 000)
    """
    nBins   = len(tau_edges) - 1
    counts  = np.zeros(nBins, dtype=np.float64)
    NA      = len(timesA)
    n_chunks = max(1, (NA + chunk_size - 1) // chunk_size)

    use_numba = _NUMBA and NA >= _NUMBA_THRESHOLD

    for ci in range(n_chunks):
        lo_i = ci * chunk_size
        hi_i = min(lo_i + chunk_size, NA)
        chunkA = timesA[lo_i:hi_i]

        # For each chunk: restrict timesB to a window that can contain
        # any pair with chunkA.  This avoids redundant work on large datasets.
        minTau = tau_edges[0]
        maxTau = tau_edges[-1]
        b_lo = np.searchsorted(timesB, chunkA[0]  + minTau, side='left')
        b_hi = np.searchsorted(timesB, chunkA[-1] + maxTau, side='right')
        chunkB = timesB[b_lo:b_hi]

        if use_numba:
            counts += _correlate_twopointer_numba(chunkA, chunkB, tau_edges)
        else:
            counts += _correlate_twopointer_numpy(chunkA, chunkB, tau_edges)

        if progress_cb is not None:
            done = ci + 1
            progress_cb(
                progress_offset + done,
                f"Photon chunk {done} / {n_chunks}",
            )

    return counts


# Number of twopointer chunks for progress reporting (approximate chunk size)
_TP_CHUNK_SIZE = 10_000


def _correlate(
    timesA: np.ndarray,
    timesB: np.ndarray,
    tau_edges: np.ndarray,
    method: Method,
    progress_cb=None,
    progress_offset: int = 0,
) -> np.ndarray:
    """
    Dispatch to the requested backend, returning raw pair counts.

    Parameters
    ----------
    progress_cb : callable(completed: int, label: str) | None
        Progress callback; passed through to the backend.
    progress_offset : int
        Step offset added to completed counts (for multi-segment calls).
    """
    if method == "perbin":
        return _correlate_perbin(
            timesA, timesB, tau_edges, progress_cb, progress_offset)
    if method == "twopointer":
        return _correlate_twopointer_chunked(
            timesA, timesB, tau_edges, progress_cb, progress_offset,
            chunk_size=_TP_CHUNK_SIZE)
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


# ── Segmentation (core) ───────────────────────────────────────────────────────

def segment_bounds(
    t_start: float,
    t_end: float,
    seg_duration_s: float = _SEGMENT_DURATION_S,
) -> List[Tuple[float, float]]:
    """
    Split the span [t_start, t_end) into fixed-duration segments.

    Every segment is exactly *seg_duration_s* long except the final one, which
    absorbs the remainder and is therefore between 1x and 2x seg_duration_s.
    No segment is ever shorter than seg_duration_s, so a short trailing tail
    can never produce a photon-starved G(tau) that would corrupt the across-
    segment statistics.

    If the span is shorter than one segment, a single segment covering the
    whole span is returned (the caller is responsible for deciding whether a
    lone segment is enough to quote an uncertainty).

    Returns
    -------
    list of (lo, hi) bounds in seconds; always at least one entry.
    """
    total = float(t_end) - float(t_start)
    if total <= 0:
        raise ValueError("Empty time span; cannot segment.")
    if seg_duration_s <= 0:
        raise ValueError("Segment duration must be positive.")

    n = int(total // seg_duration_s)
    if n < 1:
        return [(float(t_start), float(t_end))]

    bounds = [(t_start + k * seg_duration_s,
               t_start + (k + 1) * seg_duration_s)
              for k in range(n - 1)]
    # Final segment runs to t_end, absorbing the remainder.
    bounds.append((t_start + (n - 1) * seg_duration_s, float(t_end)))
    return bounds


# ── Segmented computation (core) ──────────────────────────────────────────────

def compute_segmented(
    timesA_s: np.ndarray,
    timesB_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = True,
    progress_cb=None,
    *,
    seg_duration_s: float = _SEGMENT_DURATION_S,
    full_trace_G: bool = True,
    symmetric: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Correlate, using segment-and-average for the uncertainty.

    Segment-and-average is the primary uncertainty estimator.  The stream is
    cut into fixed-duration segments (see :func:`segment_bounds`); each yields
    an independent G(tau).  The scatter across those independent estimates is
    the empirical uncertainty of the mean:

        G_std = std(G_segments, ddof=1) / sqrt(n_segs)

    i.e. the *standard error*, not the population standard deviation -- it is
    the uncertainty on the reported curve, which is what a fit weight needs.

    The reported curve itself depends on *full_trace_G*:

      full_trace_G=True (default)
          G is estimated once from the whole stream; the segments are used
          only to derive G_std.  Avoids the boundary effects of cutting the
          stream (pairs spanning a segment edge are lost from every segment),
          at the cost of one extra correlation pass.

      full_trace_G=False
          G is the mean of the per-segment curves.  Cheaper (no extra pass)
          and G_std is then exactly the standard error of the reported mean,
          but long-lag bins are biased low by the lost boundary-spanning
          pairs.

    Note that with full_trace_G=True the reported G and the quoted G_std come
    from slightly different estimators; the std is a faithful measure of the
    run-to-run scatter of a segment-length estimate, rescaled to the mean.

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
        If True (default), segment and return the standard error in G_std.
        If False, correlate the full stream in one pass; G_std is all-NaN.
    seg_duration_s : float
        Segment duration in seconds.
    full_trace_G : bool
        See above.  Ignored when segment=False (the full trace is the only
        estimate available).
    symmetric : bool
        If True, each segment reports 0.5 * (G_AB + G_BA) -- the symmetrised
        cross-correlation -- so the across-segment scatter is measured on the
        same quantity that is reported.  See
        :func:`compute_crosscorr_symmetric`.
    progress_cb : callable(completed: int, label: str) | None
        Optional progress callback.

    Returns
    -------
    tau    : np.ndarray -- bin centre lag times (seconds)
    G_mean : np.ndarray -- G(tau), baseline 0
    G_std  : np.ndarray -- standard error across segments; all-NaN when
                           segment=False or when only one segment exists
    n_segs : int        -- number of segments contributing to G_std
    """
    tau    = 0.5 * (tau_edges[:-1] + tau_edges[1:])
    n_bins = len(tau_edges) - 1

    t_start = max(timesA_s[0], timesB_s[0])
    t_end   = min(timesA_s[-1], timesB_s[-1])

    # Steps reported by ONE directed pass over one segment.  Symmetrisation
    # runs two such passes, so a symmetric segment reports twice this.  The
    # dialog's progress budget must use the same arithmetic or the bar will
    # either strand short of 100% or overshoot.
    if method == "perbin":
        steps_per_pass = n_bins
    else:
        n_segs_guess = max(1, len(segment_bounds(t_start, t_end,
                                                 seg_duration_s))) if segment else 1
        avg_N_seg = max(len(timesA_s), len(timesB_s)) // n_segs_guess
        steps_per_pass = max(1, (avg_N_seg + _TP_CHUNK_SIZE - 1)
                                // _TP_CHUNK_SIZE)
    steps_per_seg = steps_per_pass * (2 if symmetric else 1)

    def _curve(segA, segB, cb):
        """G(tau) for one (sub)trace, symmetrised if requested."""
        counts = _correlate(segA, segB, tau_edges, method, cb, 0)
        G_ab   = _normalize(counts, segA, segB, tau_edges)
        if not symmetric:
            return G_ab
        # Reverse direction B->A on the SAME photons.  Symmetrising inside the
        # segment means the across-segment scatter is measured on the reported
        # quantity directly -- no assumption that the two directions are
        # independent estimates (they are not; they are the same photons read
        # both ways).
        rev_cb = None
        if cb is not None:
            def rev_cb(completed, label, _off=steps_per_pass):
                cb(_off + completed, label)
        counts_ba = _correlate(segB, segA, tau_edges, method, rev_cb, 0)
        G_ba      = _normalize(counts_ba, segB, segA, tau_edges)
        return 0.5 * (G_ab + G_ba)

    def _full_trace(cb):
        maskA = (timesA_s >= t_start) & (timesA_s <= t_end)
        maskB = (timesB_s >= t_start) & (timesB_s <= t_end)
        return _curve(timesA_s[maskA] - t_start,
                      timesB_s[maskB] - t_start, cb)

    # ── Unsegmented path ─────────────────────────────────────────────────────
    if not segment:
        G_mean = _full_trace(progress_cb)
        return tau, G_mean, np.full_like(G_mean, np.nan), 1

    # ── Segmented path ───────────────────────────────────────────────────────
    bounds = segment_bounds(t_start, t_end, seg_duration_s)
    n_segs = len(bounds)

    G_segments = []
    for k, (lo, hi) in enumerate(bounds):
        maskA = (timesA_s >= lo) & (timesA_s < hi)
        maskB = (timesB_s >= lo) & (timesB_s < hi)
        segA  = timesA_s[maskA] - lo
        segB  = timesB_s[maskB] - lo
        if len(segA) < 2 or len(segB) < 2:
            warnings.warn(
                f"Segment {k + 1}/{n_segs} ({lo:.3g}-{hi:.3g} s) has too few "
                f"photons (Ch1 {len(segA)}, Ch2 {len(segB)}) and was skipped; "
                f"the quoted uncertainty is based on the remaining segments.",
                RuntimeWarning,
            )
            continue

        seg_cb = None
        if progress_cb is not None:
            offset = k * steps_per_seg
            def seg_cb(completed, label, _k=k, _n=n_segs, _off=offset):
                progress_cb(
                    _off + completed,
                    f"Segment {_k + 1} / {_n}  —  {label}",
                )

        G_segments.append(_curve(segA, segB, seg_cb))

    if not G_segments:
        raise ValueError(
            "No valid segments could be computed.  "
            "The dataset may be too short, or the segment duration too small."
        )

    G_stack = np.array(G_segments)
    n_valid = len(G_stack)

    # Standard ERROR of the segment ensemble: the uncertainty on the mean.
    if n_valid > 1:
        with np.errstate(invalid="ignore"):
            G_std = (np.nanstd(G_stack, axis=0, ddof=1)
                     / np.sqrt(float(n_valid)))
    else:
        # A single segment carries no scatter information.  NaN (not zero) so
        # downstream fits treat the point as unweighted rather than infinitely
        # well determined.
        G_std = np.full(n_bins, np.nan)

    if full_trace_G:
        full_cb = None
        if progress_cb is not None:
            base = n_segs * steps_per_seg
            def full_cb(completed, label, _off=base):
                progress_cb(_off + completed, f"Full trace  —  {label}")
        G_mean = _full_trace(full_cb)
    else:
        G_mean = np.nanmean(G_stack, axis=0)

    return tau, G_mean, G_std, n_valid



# ── Public API ────────────────────────────────────────────────────────────────

def compute_crosscorr(
    timesA_s: np.ndarray,
    timesB_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = True,
    progress_cb=None,
    *,
    seg_duration_s: float = _SEGMENT_DURATION_S,
    full_trace_G: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Directed cross-correlation G_AB(tau) with segment-and-average uncertainty.

    Returns
    -------
    tau, G_mean, G_std, n_segments
    """
    return compute_segmented(
        timesA_s, timesB_s, tau_edges, method, segment, progress_cb,
        seg_duration_s=seg_duration_s, full_trace_G=full_trace_G)


def compute_crosscorr_symmetric(
    timesA_s: np.ndarray,
    timesB_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = True,
    progress_cb=None,
    *,
    seg_duration_s: float = _SEGMENT_DURATION_S,
    full_trace_G: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Symmetric cross-correlation ½[G_AB(τ) + G_BA(τ)].

    Averages the forward (A→B) and reverse (B→A) directed cross-correlations.
    For a time-symmetric process the two directions estimate the same G(τ);
    averaging them removes any small asymmetry bias.  This matches the
    (Ch1×Ch2 + Ch2×Ch1)/2 convention used by ISS VistaVision and most
    commercial correlators.

    The two directions are averaged at the level of the *normalised* G(τ),
    not the raw counts, since each direction carries its own normalisation.

    Symmetrisation happens *within each segment*, so the across-segment scatter
    is measured on the symmetrised quantity that is actually reported.  The
    previous implementation correlated the whole stream in each direction and
    combined the two standard deviations as sqrt(s_ab² + s_ba²)/2, which treats
    A→B and B→A as independent estimates.  They are not independent -- they are
    the same photons read in two directions, and for a time-symmetric process
    they are strongly correlated -- so that formula understated the
    uncertainty.  Expect slightly larger, and honest, error bars.

    Returns
    -------
    tau, G_mean, G_std, n_segments
    """
    return compute_segmented(
        timesA_s, timesB_s, tau_edges, method, segment, progress_cb,
        seg_duration_s=seg_duration_s, full_trace_G=full_trace_G,
        symmetric=True)


def compute_autocorr(
    times_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = True,
    progress_cb=None,
    *,
    seg_duration_s: float = _SEGMENT_DURATION_S,
    full_trace_G: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Autocorrelation G(tau) with segment-and-average uncertainty.

    Returns
    -------
    tau, G_mean, G_std, n_segments
    """
    return compute_segmented(
        times_s, times_s, tau_edges, method, segment, progress_cb,
        seg_duration_s=seg_duration_s, full_trace_G=full_trace_G)


# ── Plotting ──────────────────────────────────────────────────────────────────

_CORR_LABEL = {
    "auto_ch1": "Autocorr Ch1",
    "auto_ch2": "Autocorr Ch2",
    "cross":    "Ch1xCh2",
}
def multitau_lag_grid(dt0: float, m: int, coarsen: int,
                      tau_max: float) -> np.ndarray:
    """
    The canonical multi-tau lag grid (s), ascending.

    The grid is a pure function of (dt0, m, coarsen, tau_max) and does NOT
    depend on the trace length: octaves are terminated by tau_max alone.  This
    is what lets segments of different durations -- in particular the final,
    over-long segment produced by :func:`segment_bounds` -- share one grid and
    be averaged element-wise.
    """
    taus: List[float] = []
    dt = float(dt0)
    first = True
    while dt * m <= tau_max:
        k0 = 1 if first else (m // coarsen + 1)
        taus.extend(k * dt for k in range(k0, m + 1))
        dt *= coarsen
        first = False
    return np.asarray(sorted(taus), float)


def _multitau_curve(a_times, b_times, dt0, m, coarsen, tau_max, T):
    """
    One multi-tau pass over a single (sub)trace; returns (tau_s, G).

    The returned tau axis is always :func:`multitau_lag_grid`, regardless of
    T.  Octaves the trace is too short to support yield NaN rather than
    silently shortening the grid, so every segment is element-wise comparable.
    """
    nb = int(T / dt0) + 2
    a = np.bincount((a_times / dt0).astype(np.int64), minlength=nb).astype(np.float64)
    b = np.bincount((b_times / dt0).astype(np.int64), minlength=nb).astype(np.float64)
    taus: List[float] = []
    Gs:   List[float] = []
    dt = dt0
    first = True
    while dt * m <= tau_max:
        k0 = 1 if first else (m // coarsen + 1)
        for k in range(k0, m + 1):
            n = a.size - k
            taus.append(k * dt)
            # Too few bins left at this octave to estimate the lag: report NaN
            # and keep the grid intact.
            if n <= 0 or a.size < 2 * m + 1:
                Gs.append(float("nan"))
                continue
            C  = float(np.dot(a[:n], b[k:k + n]))
            Sa = float(a[:n].sum())
            Sb = float(b[k:k + n].sum())
            Gs.append(C * n / (Sa * Sb) - 1.0 if Sa > 0.0 and Sb > 0.0
                      else float("nan"))
        n_keep = (a.size // coarsen) * coarsen
        if n_keep >= coarsen:
            a = a[:n_keep].reshape(-1, coarsen).sum(axis=1)
            b = b[:n_keep].reshape(-1, coarsen).sum(axis=1)
        else:
            a = np.zeros(0, dtype=np.float64)
            b = np.zeros(0, dtype=np.float64)
        dt *= coarsen
        first = False
    order = np.argsort(taus)
    return np.asarray(taus, float)[order], np.asarray(Gs, float)[order]


def compute_multitau(
    times_a: np.ndarray,
    times_b: Optional[np.ndarray],
    base_dt_s: float,
    tau_max_s: float,
    m: int = 8,
    coarsen: int = 8,
    segment: bool = True,
    duration_s: Optional[float] = None,
    progress_cb=None,
    *,
    seg_duration_s: float = _SEGMENT_DURATION_S,
    full_trace_G: bool = True,
    symmetric: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Multi-tau correlation on binned intensity with symmetric normalisation.

    The classic Schätzel multi-tau scheme: photons are binned into intensity
    bins of width ``base_dt_s``; G is evaluated at lags k = 1..m of the current
    bin width; the intensity is then coarsened by ``coarsen`` and the next
    octave evaluated, up to ``tau_max_s``.  Symmetric ("delayed-monitor")
    normalisation is used — the scheme hardware multi-tau correlators (including
    ISS VistaVision) use — so the output can be compared to theirs directly.

    Because it correlates *binned intensities* it never divides by a continuous
    log-bin width, so it is immune to the timing-grid normalisation artifact
    that affects arbitrary log edges (see :func:`build_tau_edges`).

    To reproduce the ISS lag grid, choose ``base_dt_s`` = the ISS base bin
    (e.g. 1 µs), ``m = 8``, ``coarsen = 8``: lags 1-8 at the base width, then
    2-8 at 8x, then 2-8 at 64x, and so on.

    Parameters
    ----------
    times_a, times_b : np.ndarray
        Photon arrival times (s).  Pass ``times_b=None`` (or the same array)
        for an autocorrelation.
    base_dt_s : float
        Base intensity-bin width (s) = the shortest lag.
    tau_max_s : float
        Largest lag to reach (s).
    m : int
        Channels per octave (default 8).
    coarsen : int
        Intensity coarsening factor between octaves (default 8).
    segment : bool
        If True (default), split into fixed-duration segments (see
        :func:`segment_bounds`) and return the standard error across them in
        G_std; else a single full-trace curve with an all-NaN G_std.
    seg_duration_s : float
        Segment duration in seconds.
    full_trace_G : bool
        If True (default), G is estimated once from the whole trace and the
        segments contribute only G_std.  See :func:`compute_segmented`.
    symmetric : bool
        If True, each curve is 0.5 * (G_AB + G_BA).
    duration_s : float, optional
        Measurement duration (s); defaults to the span of the photon times.

    Returns
    -------
    (tau_s, G_mean, G_std, n_segs)
    """
    if times_b is None:
        times_b = times_a
    m = max(2, int(m))
    coarsen = max(2, int(coarsen))
    t0 = float(min(times_a[0], times_b[0]))
    a_all = np.asarray(times_a, float) - t0
    b_all = np.asarray(times_b, float) - t0
    T = float(duration_s) if duration_s else float(max(a_all[-1], b_all[-1]))

    tau_grid = multitau_lag_grid(base_dt_s, m, coarsen, tau_max_s)
    if tau_grid.size == 0:
        raise ValueError(
            f"No multi-tau lags fit between base bin {base_dt_s:.3g} s and "
            f"tau_max {tau_max_s:.3g} s with m={m}."
        )

    def _curve(aseg, bseg, span):
        tau, G = _multitau_curve(aseg, bseg, base_dt_s, m, coarsen,
                                 tau_max_s, span)
        if not symmetric:
            return G
        _, G_ba = _multitau_curve(bseg, aseg, base_dt_s, m, coarsen,
                                  tau_max_s, span)
        return 0.5 * (G + G_ba)

    # ── Unsegmented path ─────────────────────────────────────────────────────
    if not segment:
        G_mean = _curve(a_all, b_all, T)
        if progress_cb is not None:
            progress_cb(1, "multi-tau full trace")
        return tau_grid, G_mean, np.full(tau_grid.size, np.nan), 1

    # ── Segmented path ───────────────────────────────────────────────────────
    bounds = segment_bounds(0.0, T, seg_duration_s)
    n_segs = len(bounds)

    rows: List[np.ndarray] = []
    for s, (lo, hi) in enumerate(bounds):
        aseg = a_all[(a_all >= lo) & (a_all < hi)] - lo
        bseg = b_all[(b_all >= lo) & (b_all < hi)] - lo
        if aseg.size >= 2 and bseg.size >= 2:
            rows.append(_curve(aseg, bseg, hi - lo))
        else:
            warnings.warn(
                f"Multi-tau segment {s + 1}/{n_segs} ({lo:.3g}-{hi:.3g} s) has "
                f"too few photons and was skipped.",
                RuntimeWarning,
            )
        if progress_cb is not None:
            progress_cb(s + 1, f"multi-tau segment {s + 1}/{n_segs}")

    if not rows:
        raise ValueError("Multi-tau produced no valid segments.")
    stack   = np.vstack(rows)
    n_valid = int(stack.shape[0])

    # Standard ERROR across segments; NaN (not zero) for a lone segment.
    if n_valid > 1:
        with np.errstate(invalid="ignore"):
            G_std = (np.nanstd(stack, axis=0, ddof=1)
                     / np.sqrt(float(n_valid)))
    else:
        G_std = np.full(tau_grid.size, np.nan)

    if full_trace_G:
        G_mean = _curve(a_all, b_all, T)
        if progress_cb is not None:
            progress_cb(n_segs + 1, "multi-tau full trace")
    else:
        G_mean = np.nanmean(stack, axis=0)

    return tau_grid, G_mean, G_std, n_valid


# ── multipletau package backend ───────────────────────────────────────────────

def _bin_intensity(times_s: np.ndarray, dt0: float, n_bins: int) -> np.ndarray:
    """Bin arrival times (s) into a regular intensity trace of n_bins bins."""
    idx = (times_s / dt0).astype(np.int64)
    idx = idx[(idx >= 0) & (idx < n_bins)]
    return np.bincount(idx, minlength=n_bins).astype(np.float64)


def _even_m(m: int) -> int:
    """multipletau requires an even m; round up and keep it >= 2."""
    m = max(2, int(m))
    return m if m % 2 == 0 else m + 1


def _multitau_pkg_curve(a_times, b_times, dt0, m, tau_max, T,
                        compress="average", is_auto=False):
    """
    One multipletau pass over a single (sub)trace; returns (tau_s, G).

    Photons are binned to a regular grid of width dt0, then handed to
    multipletau.  The tau=0 row is dropped (it is the shot-noise spike, not
    signal) and lags beyond tau_max are truncated.

    The returned grid depends on the trace length T, because multipletau adds
    octaves for as long as the data supports them.  Callers must therefore NOT
    assume positional alignment between curves from traces of different
    lengths -- align by tau value (see _align_to_grid).
    """
    if not _MULTIPLETAU:
        raise NotImplementedError(
            "The 'multipletau' package is not installed.\n\n"
            "Install it with:    pip install multipletau\n\n"
            "Or choose a different correlation method."
        )

    n_bins = int(T / dt0) + 2
    if n_bins < 4 * m:
        raise ValueError(
            f"Trace of {T:.4g} s at a {dt0:.3g} s base bin gives only "
            f"{n_bins} bins, too few for m={m}."
        )

    a = _bin_intensity(np.asarray(a_times, float), dt0, n_bins)
    # NB: identity (`b_times is a_times`) is NOT a usable test for an
    # autocorrelation -- compute_multitau_pkg subtracts a common t0 from both
    # streams, so even an autocorrelation arrives here as two distinct arrays
    # holding identical data.  The caller must say so explicitly.
    b = a if is_auto else _bin_intensity(np.asarray(b_times, float),
                                         dt0, n_bins)

    # multipletau guards against an all-zero trace with (core.py:412-414):
    #
    #     if np.abs(traceavg1) / np.median(np.abs(v)) < ZERO_CUTOFF: ...
    #
    # For photon-counting data binned at tau_min the MEDIAN BIN IS EMPTY, so
    # that divides by zero and numpy emits a RuntimeWarning on every call.
    # The median of a Poisson trace is 0 whenever the mean is below ln(2), i.e.
    # whenever the count rate is below ln(2)/dt0 -- about 69 kHz per channel at
    # a 10 us bin.  Ordinary FCS count rates sit well under that, so the
    # warning fires on essentially every run.
    #
    # It is benign: x/0 -> inf, and `inf < ZERO_CUTOFF` is False, so the check
    # PASSES and the correlation proceeds untouched.  The normalisation uses
    # the trace MEAN, not the median, and the mean is unaffected by sparsity.
    #
    # Rather than mute it blindly, we test the condition multipletau is
    # actually trying to catch -- a trace with no photons in it -- and raise a
    # clear error ourselves.  Only then is the (now meaningless) divide-by-zero
    # in its heuristic suppressed.
    if a.mean() <= 0.0 or b.mean() <= 0.0:
        raise ValueError(
            f"Cannot correlate an empty intensity trace over {T:.4g} s "
            f"(Ch1 mean {a.mean():.4g}, Ch2 mean {b.mean():.4g} counts/bin). "
            f"Check the gate, the thinning factor, and the channel selection."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        if is_auto:
            out = _multipletau.autocorrelate(
                a, m=m, deltat=dt0, normalize=True, copy=False,
                compress=compress)
        else:
            out = _multipletau.correlate(
                a, b, m=m, deltat=dt0, normalize=True, copy=False,
                compress=compress)

    tau = np.asarray(out[:, 0], float)
    G   = np.asarray(out[:, 1], float)

    # normalize=True already subtracts the 1 (verified against multipletau
    # 0.4.1: an uncorrelated Poisson trace gives a large-lag baseline of
    # 0.000000).  No further offset is applied here.
    keep = (tau > 0) & (tau <= tau_max)
    return tau[keep], G[keep]


def _align_to_grid(tau_ref: np.ndarray, tau_seg: np.ndarray,
                   G_seg: np.ndarray) -> np.ndarray:
    """
    Place G_seg onto tau_ref, matching by lag VALUE rather than by position.

    multipletau's grid grows with trace length, so the over-long final segment
    from segment_bounds() can carry more octaves than the nominal ones.  Lags
    are exact binary multiples of dt0 and so compare exactly, but a tolerance
    is used anyway to stay safe against float drift.  Lags present in tau_ref
    but absent from this segment become NaN, which nanmean/nanstd then skip --
    no segment is ever silently discarded.
    """
    out = np.full(tau_ref.size, np.nan)
    if tau_seg.size == 0:
        return out
    idx = np.searchsorted(tau_seg, tau_ref)
    idx = np.clip(idx, 0, tau_seg.size - 1)
    close = np.isclose(tau_seg[idx], tau_ref, rtol=1e-9, atol=0.0)
    out[close] = G_seg[idx[close]]
    return out


def compute_multitau_pkg(
    times_a: np.ndarray,
    times_b: Optional[np.ndarray],
    base_dt_s: float,
    tau_max_s: float,
    m: int = 16,
    segment: bool = True,
    duration_s: Optional[float] = None,
    progress_cb=None,
    *,
    seg_duration_s: float = _SEGMENT_DURATION_S,
    full_trace_G: bool = True,
    symmetric: bool = False,
    compress: str = "average",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Multi-tau correlation using the `multipletau` package as the backend.

    The reference implementation of the Schaetzel multiple-tau scheme.  The lag
    grid is hard-wired base-2; `m` (channels per octave) is the only grid knob.
    For an ISS-matched grid with an arbitrary coarsening factor use
    :func:`compute_multitau` instead.

    Uncertainty follows the suite-wide segment-and-average policy: see
    :func:`compute_segmented` for the meaning of `segment`, `seg_duration_s`
    and `full_trace_G`.

    Parameters
    ----------
    times_a, times_b : np.ndarray
        Photon arrival times (s).  Pass times_b=None for an autocorrelation.
    base_dt_s : float
        Base intensity-bin width (s) = the shortest lag = multipletau's deltat.
    tau_max_s : float
        Largest lag to keep (s).
    m : int
        Channels per octave.  Forced to an even integer (multipletau requires
        it); an odd request is rounded UP.
    symmetric : bool
        If True, report 0.5 * (G_AB + G_BA).  multipletau computes only the
        positive direction, so the reverse is obtained by swapping the inputs
        -- exactly as the package's own documentation prescribes.
    compress : str
        multipletau's register-propagation strategy: 'average' (default,
        carries the Schaetzel binning bias), or 'first'/'second' (bias-free,
        noisier).  See https://doi.org/10.1063/1.3491098.

    Returns
    -------
    (tau_s, G_mean, G_std, n_segs)
    """
    if not _MULTIPLETAU:
        raise NotImplementedError(
            "The 'multipletau' package is not installed.\n\n"
            "Install it with:    pip install multipletau\n\n"
            "Or choose a different correlation method."
        )
    # Decide this BEFORE subtracting t0 below: that subtraction produces new
    # arrays, after which the two streams are no longer the same object and an
    # autocorrelation is indistinguishable from a cross-correlation of two
    # identical traces.
    is_auto = (times_b is None) or (times_b is times_a)
    if times_b is None:
        times_b = times_a
    m_eff = _even_m(m)
    if m_eff != int(m):
        warnings.warn(
            f"multipletau requires an even number of channels; m={int(m)} "
            f"was rounded up to {m_eff}.",
            RuntimeWarning,
        )

    t0 = float(min(times_a[0], times_b[0]))
    a_all = np.asarray(times_a, float) - t0
    b_all = np.asarray(times_b, float) - t0
    T = float(duration_s) if duration_s else float(max(a_all[-1], b_all[-1]))

    def _curve(aseg, bseg, span):
        tau, G = _multitau_pkg_curve(aseg, bseg, base_dt_s, m_eff,
                                     tau_max_s, span, compress,
                                     is_auto=is_auto)
        if not symmetric:
            return tau, G
        # Reached only for a cross-correlation, so is_auto is False here.
        tau_r, G_ba = _multitau_pkg_curve(bseg, aseg, base_dt_s, m_eff,
                                          tau_max_s, span, compress,
                                          is_auto=False)
        # Same trace length both ways, so the grids are identical; assert it
        # rather than trust it.
        if tau_r.size != tau.size or not np.allclose(tau_r, tau):
            raise RuntimeError(
                "multipletau returned different lag grids for A->B and B->A "
                "on the same trace; cannot symmetrise."
            )
        return tau, 0.5 * (G + G_ba)

    # ── Unsegmented path ─────────────────────────────────────────────────────
    if not segment:
        tau, G = _curve(a_all, b_all, T)
        if progress_cb is not None:
            progress_cb(1, "multipletau full trace")
        return tau, G, np.full(tau.size, np.nan), 1

    # ── Segmented path ───────────────────────────────────────────────────────
    bounds = segment_bounds(0.0, T, seg_duration_s)
    n_segs = len(bounds)

    seg_curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for s, (lo, hi) in enumerate(bounds):
        aseg = a_all[(a_all >= lo) & (a_all < hi)] - lo
        bseg = b_all[(b_all >= lo) & (b_all < hi)] - lo
        if aseg.size >= 2 and bseg.size >= 2:
            seg_curves.append(_curve(aseg, bseg, hi - lo))
        else:
            warnings.warn(
                f"multipletau segment {s + 1}/{n_segs} ({lo:.3g}-{hi:.3g} s) "
                f"has too few photons and was skipped.",
                RuntimeWarning,
            )
        if progress_cb is not None:
            progress_cb(s + 1, f"multipletau segment {s + 1}/{n_segs}")

    if not seg_curves:
        raise ValueError("multipletau produced no valid segments.")

    # Reference grid = the LONGEST segment's grid.  segment_bounds() puts the
    # remainder in the final segment, so that one is always the longest and
    # carries the most octaves -- a deterministic choice, unlike "whichever
    # segment happened to be computed first".
    ref_i   = int(np.argmax([len(t) for t, _ in seg_curves]))
    tau_ref = seg_curves[ref_i][0]

    stack   = np.vstack([_align_to_grid(tau_ref, t, G) for t, G in seg_curves])
    n_valid = int(stack.shape[0])

    if n_valid > 1:
        with np.errstate(invalid="ignore"):
            G_std = (np.nanstd(stack, axis=0, ddof=1)
                     / np.sqrt(float(n_valid)))
    else:
        G_std = np.full(tau_ref.size, np.nan)

    if full_trace_G:
        tau_full, G_full = _curve(a_all, b_all, T)
        G_mean = _align_to_grid(tau_ref, tau_full, G_full)
        if progress_cb is not None:
            progress_cb(n_segs + 1, "multipletau full trace")
    else:
        G_mean = np.nanmean(stack, axis=0)

    return tau_ref, G_mean, G_std, n_valid


_METHOD_LABEL = {
    "perbin":          "per-bin searchsorted",
    "twopointer":      "two-pointer (Wahl)",
    "wiener_khinchin": "Wiener–Khinchin",
    "multitau":        "multi-tau (ISS-style)",
    "multitau_pkg":    "multi-tau (multipletau pkg)",
}

def _method_annotation(method: Method,
                       mt_channels: Optional[int] = None,
                       mt_coarsen: Optional[int] = None,
                       mt_compress: Optional[str] = None) -> str:
    """
    Human-readable method string, including the parameters that change the
    numbers.  Single source of truth for the plot annotation, the overlay
    annotation and the CSV `method_label` header, so the three cannot drift
    apart.
    """
    s = _METHOD_LABEL.get(method, method)
    if method == "twopointer":
        s += f" [{'numba' if _NUMBA else 'numpy'}]"
    elif method == "multitau" and mt_channels:
        s += f" [m={mt_channels}, coarsen x{mt_coarsen}]"
    elif method == "multitau_pkg" and mt_channels:
        s += f" [m={mt_channels}, base-2, compress={mt_compress}]"
    return s


def _cps_meta(n1: int, n2: int, T: float) -> dict:
    """
    CPS / acquisition-time header fields, consumed by fcs_fit._cps_from_meta
    (and hence fcs_noise's shot-noise term).

    Counts and T must describe the photons ACTUALLY correlated — i.e. after
    microtime gating and thinning — and T must be computed the same way the
    correlator computes it (span of the filtered arrays), so that the shot-noise
    pair count matches the estimator's own rateA·rateB·(T−τ)·Δτ normalisation.
    Returns {} when the fields cannot be trusted, so the fit report says
    "not available" rather than reporting a wrong rate.
    """
    if T <= 0 or (n1 == 0 and n2 == 0):
        return {}
    return {
        "acquisition_time_s": f"{T:.6g}",
        "n_photons_ch1": n1,
        "n_photons_ch2": n2,
        "cps_ch1": f"{n1 / T:.6g}",
        "cps_ch2": f"{n2 / T:.6g}",
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
    n_used_ch1: Optional[int] = None,
    n_used_ch2: Optional[int] = None,
    T_used: Optional[float] = None,
    *,
    seg_duration_s: Optional[float] = None,
    full_trace_G: Optional[bool] = None,
    mt_channels: Optional[int] = None,
    mt_coarsen: Optional[int] = None,
    mt_compress: Optional[str] = None,
) -> None:
    """
    Write one file's plotted correlation curve to a CSV.

    Mirrors the inline export in :func:`plot_correlation` so single-file and
    batch/overlay exports use an identical column and metadata layout.

    The header records how G and G_std were produced, not just that they were.
    Two of those facts changed meaning at the same time as this suite gained
    segment-and-average, WITHOUT the column names changing:

      * G_std used to be the population standard deviation across segments;
        it is now the STANDARD ERROR (smaller by sqrt(n_segments) -- a factor
        of ~3.2 at the default ten 1 s segments).  Anything weighting a fit by
        1/G_std^2 would be off by ~10x between an old file and a new one.
      * A "cross" export used to be a DIRECTED Ch1->Ch2 estimate on the
        single-file path; it is now symmetrised.

    A reader cannot tell old from new by looking at the numbers, so the header
    says which it is.  Files written before this patch simply lack these keys,
    which is itself the signal that they are the old convention.
    """
    cols: dict = {
        "tau_s":  tau,
        "tau_ms": tau * 1e3,
        "G":      G_mean,
    }
    have_std = bool(np.isfinite(G_std).any())
    if have_std:
        cols["G_std"] = G_std

    meta = {
        "type":         corr_type,
        "method":       method,
        "method_label": _method_annotation(method, mt_channels,
                                           mt_coarsen, mt_compress),
        "tau_min_s":    f"{tau_min_s:.6g}",
        "tau_max_s":    f"{tau_max_s:.6g}",
        "n_segments":   n_segs,
    }

    if corr_type == "cross":
        meta["symmetrized"] = "yes: G = 0.5*(G_Ch1Ch2 + G_Ch2Ch1), per segment"

    # Segment-and-average provenance.
    if seg_duration_s is not None and n_segs > 1:
        meta["segmented"]          = "yes"
        meta["segment_duration_s"] = f"{seg_duration_s:.6g}"
        meta["segment_note"] = (
            "fixed-duration segments; the final segment absorbs the "
            "remainder and may be up to 2x segment_duration_s"
        )
    elif n_segs <= 1:
        meta["segmented"] = "no"

    if full_trace_G is not None and n_segs > 1:
        meta["G_source"] = ("full trace (segments used only for G_std)"
                            if full_trace_G else "mean of the per-segment G")

    if have_std:
        meta["G_std_definition"] = (
            "standard error of the mean: "
            "std(G_segments, ddof=1) / sqrt(n_segments)"
        )

    # Multi-tau grid parameters: without these the lag grid cannot be
    # reconstructed from the file alone.
    if method in ("multitau", "multitau_pkg") and mt_channels is not None:
        meta["mt_channels_per_octave"] = mt_channels
    if method == "multitau" and mt_coarsen is not None:
        meta["mt_coarsen_factor"] = mt_coarsen
    if method == "multitau_pkg":
        meta["mt_coarsen_factor"] = 2      # multipletau is hard-wired base-2
        if mt_compress is not None:
            meta["mt_compress"] = mt_compress
    if None not in (n_used_ch1, n_used_ch2, T_used):
            meta.update(_cps_meta(n_used_ch1, n_used_ch2, T_used))
    
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
    n_used_ch1: Optional[int] = None,
    n_used_ch2: Optional[int] = None,
    T_used: Optional[float] = None,
    *,
    seg_duration_s: Optional[float] = None,
    full_trace_G: Optional[bool] = None,
    mt_channels: Optional[int] = None,
    mt_coarsen: Optional[int] = None,
    mt_compress: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a segmented correlation function with uncertainty bounds.

    The reported G(τ) is drawn as black dots.  The grey band is ±1 STANDARD
    ERROR -- the uncertainty ON THE PLOTTED CURVE -- not the segment-to-segment
    spread, which is larger by sqrt(n_segments).

    Parameters
    ----------
    tau      : lag times in seconds
    G_mean   : reported correlation (baseline 0)
    G_std    : standard error across segments (std/sqrt(n)), or all-NaN
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
            n_used_ch1=n_used_ch1, n_used_ch2=n_used_ch2, T_used=T_used,
            seg_duration_s=seg_duration_s, full_trace_G=full_trace_G,
            mt_channels=mt_channels, mt_coarsen=mt_coarsen,
            mt_compress=mt_compress,
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

        # "±1σ" would read as the segment-to-segment spread; this band is the
        # standard error of the mean, smaller by sqrt(n_segments).
        ax.semilogx(tau_ms[mask_u], upper[mask_u],
                    color="lightgrey", linewidth=1.0,
                    label=f"±1 SEM  (n={n_segs} segments)")
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
    if n_segs > 1 and seg_duration_s:
        seg_str = f"{n_segs} × {seg_duration_s:.3g} s segments"
        if full_trace_G is False:
            seg_str += " (G = segment mean)"
    elif n_segs > 1:
        seg_str = f"{n_segs} segments"
    else:
        seg_str = "unsegmented — no error bars"
    subtitle = (
        f"{label}  ·  "
        f"τ: {tau_min_s*1e3:.3g}–{tau_max_s*1e3:.3g} ms  ·  "
        f"{seg_str}"
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

    # Bottom-right: method + the parameters that change the numbers
    method_str = _method_annotation(method, mt_channels, mt_coarsen,
                                    mt_compress)
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
    show_progress: bool = True,
) -> Optional[dict]:
    """
    Compute G(tau) for a single file from a parameter dict.

    *params* uses the same keys the dialog persists: ``tau_min_ms``,
    ``tau_max_ms``, ``corr_type`` ('cross' | 'auto_ch1' | 'auto_ch2'),
    ``method``, ``segment`` (bool), ``gate`` (bool), and the speed controls
    ``points_per_decade`` and ``thin_factor``.  This is the batch counterpart
    of the single-file Compute path, so it honours the same bins-per-decade,
    photon-thinning, gating, and progress-window behaviour.

    When ``gate`` is True the interactive microtime gate is shown for *this*
    file (the gate is file-specific), so in a batch each file gets its own
    gate window.  When *show_progress* is True a per-file progress window with
    an ETA and a Cancel button is shown during the computation; cancelling
    skips this file (the function returns None) and the batch continues with
    the next one.

    Returns
    -------
    dict with keys tau, G_mean, G_std, n_segs, tau_min_s, tau_max_s,
    gate_min_ns, gate_max_ns — or None if the user cancelled the gate or the
    progress window, the gate was too narrow, or the computation failed.
    """
    from tkinter import messagebox

    corr_type   = params["corr_type"]
    method      = params["method"]
    segment     = params["segment"]
    use_gate    = params["gate"]
    ppd         = max(2, int(params.get("points_per_decade", _POINTS_PER_DECADE)))
    seg_duration_s = float(params.get("seg_duration_s", _SEGMENT_DURATION_S))
    full_trace_G   = bool(params.get("full_trace_G", True))
    thin_factor = max(1, int(params.get("thin_factor", 1)))

    tau_min_s = float(params["tau_min_ms"]) * 1e-3
    tau_max_s = float(params["tau_max_ms"]) * 1e-3
    is_multitau = method in ("multitau", "multitau_pkg")
    mt_m        = max(2, int(params.get("mt_channels", 8)))
    mt_coarsen  = max(2, int(params.get("mt_coarsen", 8)))
    mt_compress = params.get("mt_compress", "average")
    if is_multitau:
        tau_edges, n_bins = None, 0
    else:
        try:
            tau_edges = build_tau_edges(tau_min_s, tau_max_s, ppd,
                                        grid_period_s=fcs_data.macrotime_period_s)
        except ValueError as e:
            messagebox.showerror("Invalid range", str(e), parent=parent)
            return None
        n_bins = len(tau_edges) - 1

    # ── Photon stream selection (with optional, file-specific gating) ─────────
    times_ch1 = fcs_data.ch1_times_s
    times_ch2 = fcs_data.ch2_times_s

    if use_gate:
        gate = fcs_lifetime.select_gate(fcs_data)
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

    # ── Photon thinning (decimation) ─────────────────────────────────────────
    if thin_factor > 1:
        times_ch1 = thin_photons(times_ch1, thin_factor)
        times_ch2 = thin_photons(times_ch2, thin_factor)

    # ── Progress-step bookkeeping (mirrors the single-file path) ─────────────
    N = max(len(times_ch1), len(times_ch2))
    if segment:
        seg_dur  = _MIN_SEGMENT_FACTOR * tau_max_s
        n_segs0  = max(1, int(fcs_data.duration_s // seg_dur))
    else:
        n_segs0 = 1
    if is_multitau:
        total_steps = n_segs0
    elif method == "perbin":
        total_steps = n_segs0 * n_bins
    else:
        N_per_seg   = max(1, N // n_segs0)
        total_steps = n_segs0 * max(1, (N_per_seg + _TP_CHUNK_SIZE - 1)
                                       // _TP_CHUNK_SIZE)
    # ── Optional per-file progress window (fully guarded) ────────────────────
    pw = None
    progress_cb = None
    if show_progress:
        try:
            pw = _ProgressWindow(
                None, total_steps,
                title=f"Computing — {fcs_data.filepath.name}",
            )

            def progress_cb(completed: int, label: str):
                if pw.cancelled():
                    raise KeyboardInterrupt("User cancelled")
                pw.step(completed, label)
        except Exception:
            # If the window can't be created for any reason, compute silently
            # rather than failing the whole batch.
            pw = None
            progress_cb = None

    try:
        if is_multitau:
            if corr_type == "cross":
                _a, _b = times_ch1, times_ch2
            elif corr_type == "auto_ch1":
                _a = _b = times_ch1
            else:
                _a = _b = times_ch2
            if method == "multitau_pkg":
                tau, G_mean, G_std, n_segs = compute_multitau_pkg(
                    _a, _b, tau_min_s, tau_max_s, m=mt_m,
                    segment=segment, duration_s=fcs_data.duration_s,
                    progress_cb=progress_cb,
                    seg_duration_s=seg_duration_s, full_trace_G=full_trace_G,
                    symmetric=(corr_type == "cross"),
                    compress=mt_compress)
            else:
                tau, G_mean, G_std, n_segs = compute_multitau(
                    _a, _b, tau_min_s, tau_max_s, m=mt_m, coarsen=mt_coarsen,
                    segment=segment, duration_s=fcs_data.duration_s,
                    progress_cb=progress_cb,
                    seg_duration_s=seg_duration_s, full_trace_G=full_trace_G,
                    symmetric=(corr_type == "cross"))
        elif corr_type == "cross":
            tau, G_mean, G_std, n_segs = compute_crosscorr_symmetric(
                times_ch1, times_ch2, tau_edges, method, segment,
                progress_cb=progress_cb,
                seg_duration_s=seg_duration_s, full_trace_G=full_trace_G)
        elif corr_type == "auto_ch1":
            tau, G_mean, G_std, n_segs = compute_autocorr(
                times_ch1, tau_edges, method, segment, progress_cb=progress_cb,
                seg_duration_s=seg_duration_s, full_trace_G=full_trace_G)
        else:
            tau, G_mean, G_std, n_segs = compute_autocorr(
                times_ch2, tau_edges, method, segment, progress_cb=progress_cb,
                seg_duration_s=seg_duration_s, full_trace_G=full_trace_G)
    except KeyboardInterrupt:
        if pw is not None:
            pw.close()
        return None
    except NotImplementedError as e:
        if pw is not None:
            pw.close()
        messagebox.showerror("Not implemented", str(e), parent=parent)
        return None
    except Exception as e:
        if pw is not None:
            pw.close()
        messagebox.showerror("Computation error",
                             f"{fcs_data.filepath.name}:\n{e}", parent=parent)
        return None

    if pw is not None:
        pw.close()

    return {
        "tau": tau, "G_mean": G_mean, "G_std": G_std, "n_segs": n_segs,
        "tau_min_s": tau_min_s, "tau_max_s": tau_max_s,
        "gate_min_ns": gate_min_ns, "gate_max_ns": gate_max_ns,
        "n_used_ch1": len(times_ch1), "n_used_ch2": len(times_ch2),
        "T_used": float(max(times_ch1[-1], times_ch2[-1])
                        - min(times_ch1[0], times_ch2[0])),
        # Provenance, so the batch export writes the same header as the
        # single-file path rather than re-deriving it from _defaults (which
        # the user may have changed since).
        "seg_duration_s": seg_duration_s if segment else None,
        "full_trace_G":   full_trace_G if segment else None,
        "mt_channels":    mt_m if is_multitau else None,
        "mt_coarsen":     mt_coarsen if method == "multitau" else None,
        "mt_compress":    mt_compress if method == "multitau_pkg" else None,
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

    Each file is drawn in its own colour; the per-file ±1σ band is omitted to
    keep a multi-file overlay readable.  When *export* is True each file's
    curve is written to its own CSV exactly as in the single-file path.

    Parameters
    ----------
    results : sequence of (FCSData, result_dict)
        result_dict as returned by :func:`compute_correlation_for`.
    corr_type, method : shared correlation type / backend (for the title).
    tau_min_s, tau_max_s : shared lag range in seconds (for the x-limits).

    Returns
    -------
    fig, ax
    """
    results = list(results)
    if not results:
        raise ValueError("plot_correlation_overlay requires at least one result.")

    # Distinct colour per file.  Use fcs_plottools.palette when present, else
    # fall back to a matplotlib colormap so this never hard-depends on it.
    _palette = getattr(fcs_plottools, "palette", None)
    if callable(_palette):
        colours = _palette(len(results))
    else:
        cmap = plt.get_cmap("tab10")
        colours = [cmap(i % 10) for i in range(len(results))]

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
                n_used_ch1=res["n_used_ch1"], n_used_ch2=res["n_used_ch2"],
                T_used=res["T_used"],
                seg_duration_s=res.get("seg_duration_s"),
                full_trace_G=res.get("full_trace_G"),
                mt_channels=res.get("mt_channels"),
                mt_coarsen=res.get("mt_coarsen"),
                mt_compress=res.get("mt_compress"),
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

    _first = next((r for _d, r in results), {}) if results else {}
    method_str = _method_annotation(method,
                                    _first.get("mt_channels"),
                                    _first.get("mt_coarsen"),
                                    _first.get("mt_compress"))
    fig.text(0.99, 0.01, method_str,
             ha="right", va="bottom", fontsize=7, color="grey")

    fig.tight_layout()
    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── Dialog ────────────────────────────────────────────────────────────────────

_defaults: dict = {
    "tau_min_ms":       0.01,
    "tau_max_ms":       10.0,
    "corr_type":        "cross",
    "method":           "perbin",
    "segment":          True,
    "gate":             False,
    "points_per_decade": _POINTS_PER_DECADE,
    "thin_factor":      1,
    "mt_channels":      8,
    "mt_coarsen":       8,
    "seg_duration_s":   _SEGMENT_DURATION_S,
    "full_trace_G":     True,
    "mt_compress":      "average",
}


def run_correlation_dialog(fcs_data: FCSData, export: bool = False,
                           *, collect_only: bool = False):
    """
    Show a parameter dialog, then compute and plot the segmented correlation.
    Settings persist between calls within the same session.

    If *collect_only* is True the dialog gathers and returns the chosen
    parameter dict (the same keys ``_defaults`` uses, including
    ``points_per_decade`` and ``thin_factor``) without computing or plotting
    anything, and the file-specific warnings are skipped.  This is used by the
    multi-file batch/overlay path to ask for parameters once and then apply
    them to every selected file via :func:`compute_correlation_for`.  Returns
    None if the user cancels.
    """
    import tkinter as tk
    from tkinter import messagebox

    result_box: dict = {"params": None}

    dialog = tk.Toplevel()
    dialog.title("Correlation — options")
    dialog.resizable(False, True)
    dialog.grab_set()

    pad = dict(padx=12, pady=4)

    # ── Layout: pinned buttons + scrollable body ──────────────────────────────
    # The option list is long enough to overflow a short screen, so the body
    # scrolls.  The buttons are packed FIRST with side="bottom" so pack()
    # reserves the bottom slot for them before the canvas claims the rest --
    # they can never be scrolled away or clipped off-screen, whatever the
    # window height.  The buttons themselves are added further down, once
    # _on_compute exists; the frame is created here purely to claim the slot.
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(side="bottom", fill="x", pady=8)
    tk.Frame(dialog, height=1, bg="#c8c8c8").pack(side="bottom", fill="x")

    _canvas = tk.Canvas(dialog, borderwidth=0, highlightthickness=0)
    _vsb    = tk.Scrollbar(dialog, orient="vertical", command=_canvas.yview)
    _canvas.configure(yscrollcommand=_vsb.set)
    _vsb.pack(side="right", fill="y")
    _canvas.pack(side="left", fill="both", expand=True)

    body     = tk.Frame(_canvas)
    _body_id = _canvas.create_window((0, 0), window=body, anchor="nw")

    def _on_body_configure(_e=None):
        """Body changed size (e.g. a section was shown/hidden) — re-measure."""
        _canvas.configure(scrollregion=_canvas.bbox("all"))

    def _on_canvas_configure(e):
        """Keep the body exactly as wide as the canvas (no horizontal scroll)."""
        _canvas.itemconfigure(_body_id, width=e.width)

    body.bind("<Configure>", _on_body_configure)
    _canvas.bind("<Configure>", _on_canvas_configure)

    # Mouse wheel, bound only while the pointer is over this dialog.  bind_all
    # is global, so it is attached on <Enter> and removed on <Leave> rather
    # than left hanging on every other window in the app.
    def _on_mousewheel(e):
        if e.delta:                      # Windows / macOS
            _canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        elif e.num == 4:                 # X11 scroll up
            _canvas.yview_scroll(-1, "units")
        elif e.num == 5:                 # X11 scroll down
            _canvas.yview_scroll(1, "units")

    def _bind_wheel(_e=None):
        _canvas.bind_all("<MouseWheel>", _on_mousewheel)
        _canvas.bind_all("<Button-4>", _on_mousewheel)
        _canvas.bind_all("<Button-5>", _on_mousewheel)

    def _unbind_wheel(_e=None):
        _canvas.unbind_all("<MouseWheel>")
        _canvas.unbind_all("<Button-4>")
        _canvas.unbind_all("<Button-5>")

    _canvas.bind("<Enter>", _bind_wheel)
    _canvas.bind("<Leave>", _unbind_wheel)
    dialog.bind("<Destroy>", lambda e: _unbind_wheel())

    tk.Label(body, text="Correlation options",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    # ── Correlation type ──────────────────────────────────────────────────────
    type_frame = tk.LabelFrame(body, text="Type", padx=10, pady=4)
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
    range_frame = tk.LabelFrame(body, text="Lag range (ms)", padx=10, pady=6)
    range_frame.pack(fill="x", **pad)

    row = tk.Frame(range_frame)
    row.pack(fill="x")
    tk.Label(row, text="Min:", width=5, anchor="e").pack(side="left")
    min_var = tk.StringVar(value=str(_defaults["tau_min_ms"]))
    tk.Entry(row, textvariable=min_var, width=10).pack(side="left", padx=(4, 16))
    tk.Label(row, text="Max:", width=5, anchor="e").pack(side="left")
    max_var = tk.StringVar(value=str(_defaults["tau_max_ms"]))
    tk.Entry(row, textvariable=max_var, width=10).pack(side="left", padx=4)

    # ── Uncertainty (segment-and-average) ─────────────────────────────────────
    seg_frame = tk.LabelFrame(body, text="Uncertainty", padx=10, pady=4)
    seg_frame.pack(fill="x", **pad)

    segment_var = tk.BooleanVar(value=_defaults["segment"])
    seg_cb = tk.Checkbutton(
        seg_frame,
        text="Segment and average  (error bars from segment scatter)",
        variable=segment_var,
        anchor="w",
    )
    seg_cb.pack(fill="x")

    # Segment length (s) — only meaningful when segmenting.
    seg_len_row = tk.Frame(seg_frame)
    tk.Label(seg_len_row, text="Segment length:", anchor="w",
             width=14).pack(side="left")
    seg_len_var = tk.StringVar(value=str(_defaults["seg_duration_s"]))
    tk.Entry(seg_len_row, textvariable=seg_len_var, width=6).pack(side="left",
                                                                  padx=3)
    tk.Label(seg_len_row, text="s", anchor="w").pack(side="left")

    # Full-trace G: report G from the whole stream, segments only for G_std.
    full_trace_var = tk.BooleanVar(value=_defaults["full_trace_G"])
    full_trace_cb = tk.Checkbutton(
        seg_frame,
        text="Full trace G?  (avoids segment-boundary effects; 1 extra pass)",
        variable=full_trace_var,
        anchor="w",
    )

    # Info label: segment count / warnings; only visible when ticked
    seg_info_var = tk.StringVar(value="")
    seg_info_label = tk.Label(seg_frame, textvariable=seg_info_var,
                              font=("Helvetica", 9), fg="grey", anchor="w",
                              justify="left")

    def _update_seg_info(*_):
        """Recompute expected segment count and show/hide the sub-controls."""
        if not segment_var.get():
            seg_len_row.pack_forget()
            full_trace_cb.pack_forget()
            seg_info_label.pack_forget()
            return
        seg_len_row.pack(fill="x", pady=(3, 0))
        full_trace_cb.pack(fill="x")
        try:
            seg_dur = float(seg_len_var.get())
            if seg_dur <= 0:
                raise ValueError
            total_s = fcs_data.duration_s
            bounds  = segment_bounds(0.0, total_s, seg_dur)
            n       = len(bounds)
            last    = bounds[-1][1] - bounds[-1][0]

            parts = [f"  {n} segment{'s' if n != 1 else ''}"]
            if n > 1 and abs(last - seg_dur) > 1e-9:
                parts.append(f"last {last:.4g} s absorbs remainder")
            txt = "  ·  ".join(parts)

            if n == 1:
                txt += ("\n  ⚠ one segment only — no scatter to measure, "
                        "G_std will be blank")
            elif n < _MIN_SEGMENTS:
                txt += (f"\n  ⚠ fewer than {_MIN_SEGMENTS} segments — "
                        f"G_std unreliable")

            # A segment of duration D only supports lags well below D.
            tau_max_s = float(max_var.get()) * 1e-3
            if tau_max_s * _MIN_SEGMENT_FACTOR > seg_dur:
                usable = seg_dur / _MIN_SEGMENT_FACTOR
                txt += (f"\n  ⚠ τ max {tau_max_s*1e3:.3g} ms exceeds "
                        f"{usable*1e3:.3g} ms (1/{_MIN_SEGMENT_FACTOR} of a "
                        f"segment); long-lag error bars will be large")
            seg_info_var.set(txt)
        except (ValueError, tk.TclError):
            seg_info_var.set("")
        seg_info_label.pack(fill="x")

    segment_var.trace_add("write", _update_seg_info)
    seg_len_var.trace_add("write", _update_seg_info)
    max_var.trace_add("write", _update_seg_info)
    _update_seg_info()


    # ── Time gating ───────────────────────────────────────────────────────────
    gate_frame = tk.LabelFrame(body, text="Time gating", padx=10, pady=4)
    gate_frame.pack(fill="x", **pad)

    gate_var = tk.BooleanVar(value=_defaults["gate"])
    tk.Checkbutton(
        gate_frame,
        text="Apply microtime gate  (set interactively on histogram)",
        variable=gate_var,
        anchor="w",
    ).pack(fill="x")

    # ── Method ────────────────────────────────────────────────────────────────
    method_frame = tk.LabelFrame(body, text="Method", padx=10, pady=4)
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

    tk.Radiobutton(method_frame,
                   text="Multi-tau  (binned intensity; for ISS comparison)",
                   variable=method_var, value="multitau",
                   anchor="w").pack(fill="x")

    if _MULTIPLETAU:
        pkg_label = "Multi-tau  (multipletau package; base-2 grid)"
        pkg_state, pkg_fg = "normal", "black"
    else:
        pkg_label = "Multi-tau  (multipletau package — not installed)"
        pkg_state, pkg_fg = "disabled", "grey"
    tk.Radiobutton(method_frame, text=pkg_label,
                   variable=method_var, value="multitau_pkg",
                   anchor="w", state=pkg_state, fg=pkg_fg).pack(fill="x")

    # ── Multi-tau sub-controls ────────────────────────────────────────────────
    # Channels/octave applies to BOTH multi-tau backends.  The coarsening
    # factor applies only to the in-house one: the multipletau package is
    # hard-wired to base-2 octaves (measured successive-lag ratio 2.000), so
    # exposing a coarsen knob there would be a control that does nothing.
    mt_row = tk.Frame(method_frame)
    tk.Label(mt_row, text="channels/octave:").pack(side="left")
    mt_m_var = tk.StringVar(value=str(_defaults.get("mt_channels", 8)))
    tk.Entry(mt_row, textvariable=mt_m_var, width=4).pack(side="left", padx=(3, 10))

    mt_coarsen_row = tk.Frame(method_frame)
    tk.Label(mt_coarsen_row, text="coarsen ×:").pack(side="left")
    mt_coarsen_var = tk.StringVar(value=str(_defaults.get("mt_coarsen", 8)))
    tk.Entry(mt_coarsen_row, textvariable=mt_coarsen_var,
             width=4).pack(side="left", padx=3)
    tk.Label(mt_coarsen_row, text="(8 / 8 with τ min = 1 µs matches ISS)",
             font=("Helvetica", 8), fg="grey").pack(side="left", padx=8)

    mt_note_var = tk.StringVar(value="")
    mt_note = tk.Label(method_frame, textvariable=mt_note_var,
                       font=("Helvetica", 8), fg="grey", anchor="w",
                       justify="left")

    # ── Speed controls ────────────────────────────────────────────────────────
    # Only the per-bin and two-pointer estimators use a log-spaced tau_edges
    # grid and per-photon thinning; the multi-tau backends bin the intensity
    # onto their own octave grid and never see these.  So this whole frame is
    # shown only for those two methods (see _update_method_controls).
    speed_frame = tk.LabelFrame(body, text="Speed / resolution trade-off",
                                padx=10, pady=6)

    # Bins per decade
    ppd_row = tk.Frame(speed_frame)
    ppd_row.pack(fill="x", pady=(0, 3))
    tk.Label(ppd_row, text="Bins / decade:", anchor="w", width=16).pack(side="left")
    ppd_var = tk.StringVar(value=str(_defaults["points_per_decade"]))
    tk.Spinbox(ppd_row, from_=2, to=50, increment=1,
               textvariable=ppd_var, width=5).pack(side="left", padx=4)
    ppd_n_var = tk.StringVar(value="")
    tk.Label(ppd_row, textvariable=ppd_n_var,
             font=("Helvetica", 9), fg="grey").pack(side="left", padx=6)

    def _update_ppd_info(*_):
        try:
            tau_max_s_ = float(max_var.get()) * 1e-3
            tau_min_s_ = float(min_var.get()) * 1e-3
            ppd        = max(2, int(ppd_var.get()))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                edges_ = build_tau_edges(
                    tau_min_s_, tau_max_s_, ppd,
                    grid_period_s=fcs_data.macrotime_period_s)
            ppd_n_var.set(f"→ {len(edges_) - 1} bins total")
        except (ValueError, tk.TclError):
            ppd_n_var.set("")

    ppd_var.trace_add("write", _update_ppd_info)
    min_var.trace_add("write", _update_ppd_info)
    max_var.trace_add("write", _update_ppd_info)
    _update_ppd_info()

    # Photon thinning
    thin_row = tk.Frame(speed_frame)
    thin_row.pack(fill="x")
    tk.Label(thin_row, text="Keep 1 in:", anchor="w", width=16).pack(side="left")
    thin_var = tk.StringVar(value=str(_defaults["thin_factor"]))
    tk.Spinbox(thin_row, from_=1, to=100, increment=1,
               textvariable=thin_var, width=5).pack(side="left", padx=4)
    thin_info_var = tk.StringVar(value="")
    tk.Label(thin_row, textvariable=thin_info_var,
             font=("Helvetica", 9), fg="grey").pack(side="left", padx=6)

    def _update_thin_info(*_):
        try:
            k  = max(1, int(thin_var.get()))
            N  = max(len(fcs_data.ch1_times_s), len(fcs_data.ch2_times_s))
            Nk = (N + k - 1) // k
            if k == 1:
                thin_info_var.set(f"photons  ({N:,} total, no thinning)")
            else:
                thin_info_var.set(
                    f"photons  ({Nk:,} kept of {N:,};  "
                    f"SNR factor ×{1/k**0.5:.2f})"
                )
        except (ValueError, tk.TclError):
            thin_info_var.set("")

    thin_var.trace_add("write", _update_thin_info)
    _update_thin_info()


    # Info label: estimated computation time
    time_est_var = tk.StringVar(value="")
    time_est_label = tk.Label(method_frame, textvariable=time_est_var,
                              font=("Helvetica", 9), fg="grey", anchor="w")
    time_est_label.pack(fill="x")

    def _update_time_estimate(*_):
        """Recompute and display the estimated computation time."""
        try:
            tau_max_s  = float(max_var.get()) * 1e-3
            tau_min_s  = float(min_var.get()) * 1e-3
            ppd        = max(2, int(ppd_var.get()))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tau_edges_ = build_tau_edges(
                    tau_min_s, tau_max_s, ppd,
                    grid_period_s=fcs_data.macrotime_period_s)
            n_bins     = len(tau_edges_) - 1
            method     = method_var.get()
            if method == "wiener_khinchin":
                time_est_var.set("  (not yet implemented)")
                return

            N_raw    = max(len(fcs_data.ch1_times_s), len(fcs_data.ch2_times_s))
            k        = max(1, int(thin_var.get()))
            N        = (N_raw + k - 1) // k
            total_s  = fcs_data.duration_s
            seg_on   = segment_var.get()
            if seg_on:
                seg_dur = float(seg_len_var.get())
                n_segs  = len(segment_bounds(0.0, total_s, seg_dur))
                # full_trace_G costs one more pass over the whole stream
                if full_trace_var.get():
                    n_segs += 1
            else:
                n_segs = 1
            # Symmetrisation correlates both directions (the multi-tau
            # backends included -- this is real work, even though they report
            # only one progress step per segment).
            if type_var.get() == "cross":
                n_segs *= 2

            est = estimate_corr_time(
                N=N, n_bins=n_bins, method=method,
                n_segs=n_segs, tau_max_s=tau_max_s, total_s=total_s,
            )
            time_est_var.set(f"  Estimated time: {est}")
        except (ValueError, ZeroDivisionError, tk.TclError):
            time_est_var.set("")

    for _var in (method_var, min_var, max_var, segment_var, ppd_var, thin_var,
                 seg_len_var, full_trace_var, type_var):
        _var.trace_add("write", _update_time_estimate)
    _update_time_estimate()

    # Tail spacer: a stable anchor for re-packing sections that get hidden.
    # pack() appends to the END of the parent's order, so a section that is
    # hidden and later re-shown would jump to the bottom of the body.
    # `before=_tail` pins it back above this spacer.  btn_frame can no longer
    # serve as that anchor: it lives in `dialog`, while the sections live in
    # `body`, and pack(before=...) requires a common parent.
    _tail = tk.Frame(body)
    _tail.pack(fill="x")

    # ── Buttons ───────────────────────────────────────────────────────────────
    # btn_frame itself was created and pinned to the bottom at the top of this
    # function; only its contents are added here.

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

        try:
            ppd         = max(2, int(ppd_var.get()))
            thin_factor = max(1, int(thin_var.get()))
        except (ValueError, tk.TclError):
            messagebox.showerror("Invalid input",
                                 "Bins/decade and thinning factor must be integers.",
                                 parent=dialog)
            return

        corr_type = type_var.get()
        method    = method_var.get()
        segment   = segment_var.get()
        use_gate  = gate_var.get()
        is_multitau = method in ("multitau", "multitau_pkg")
        full_trace_G = full_trace_var.get()
        try:
            mt_m       = max(2, int(mt_m_var.get()))
            mt_coarsen = max(2, int(mt_coarsen_var.get()))
        except (ValueError, tk.TclError):
            mt_m, mt_coarsen = 8, 8

        try:
            seg_duration_s = float(seg_len_var.get())
        except (ValueError, tk.TclError):
            seg_duration_s = -1.0
        if seg_duration_s <= 0:
            if segment:
                messagebox.showerror("Invalid input",
                                     "Segment length must be a positive "
                                     "number of seconds.",
                                     parent=dialog)
                return
            # Not segmenting, so the value is unused — but it still gets
            # persisted into _defaults, so keep it sane rather than storing a
            # negative that would reappear next session.
            seg_duration_s = _SEGMENT_DURATION_S

        # Slow-path warning for two-pointer without numba (single-file only)
        if (not collect_only) and method == "twopointer" and not _NUMBA:
            N = max(len(fcs_data.ch1_deltas), len(fcs_data.ch2_deltas))
            N_eff = (N + thin_factor - 1) // thin_factor
            if N_eff > 5_000:
                if not messagebox.askyesno(
                    "Slow computation",
                    f"{N_eff:,} photons with two-pointer (pure Python) may take "
                    f"several minutes.\n\nProceed anyway?",
                    parent=dialog,
                ):
                    return

        # Check segment count and warn if low (only when segmentation enabled).
        # Skipped in collect_only mode: the batch path asks once, and the
        # per-file segment count is handled inside compute_correlation_for.
        tau_max_s  = tau_max_ms * 1e-3
        if (not collect_only) and segment:
            n_expected = len(segment_bounds(0.0, fcs_data.duration_s,
                                            seg_duration_s))
            if n_expected < _MIN_SEGMENTS:
                detail = (
                    "G_std cannot be estimated from a single segment and will "
                    "be left blank."
                    if n_expected == 1 else
                    f"Uncertainty estimates are unreliable with fewer than "
                    f"{_MIN_SEGMENTS} segments."
                )
                if not messagebox.askyesno(
                    "Few segments",
                    f"Only {n_expected} segment"
                    f"{'s' if n_expected != 1 else ''} can be formed from "
                    f"{fcs_data.duration_s:.4g} s of data at a "
                    f"{seg_duration_s:.4g} s segment length.\n\n"
                    f"{detail}\n\n"
                    f"Consider a shorter segment length or a longer "
                    f"dataset.\n\n"
                    f"Proceed anyway?",
                    parent=dialog,
                ):
                    return

            # A segment of duration D only supports lags well below D.
            if tau_max_s * _MIN_SEGMENT_FACTOR > seg_duration_s:
                usable_ms = (seg_duration_s / _MIN_SEGMENT_FACTOR) * 1e3
                if not messagebox.askyesno(
                    "τ max large for this segment length",
                    f"τ max = {tau_max_ms:.3g} ms exceeds "
                    f"{usable_ms:.3g} ms, which is 1/{_MIN_SEGMENT_FACTOR} of "
                    f"the {seg_duration_s:.4g} s segment length.\n\n"
                    f"G(τ) is still computed out to τ max, but the long-lag "
                    f"error bars will be large because each segment contains "
                    f"few independent samples at those lags.\n\n"
                    f"Proceed anyway?",
                    parent=dialog,
                ):
                    return

        # Persist
        _defaults["tau_min_ms"]        = tau_min_ms
        _defaults["tau_max_ms"]        = tau_max_ms
        _defaults["corr_type"]         = corr_type
        _defaults["method"]            = method
        _defaults["segment"]           = segment
        _defaults["gate"]              = use_gate
        _defaults["points_per_decade"] = ppd
        _defaults["thin_factor"]       = thin_factor
        _defaults["mt_channels"]       = mt_m
        _defaults["mt_coarsen"]        = mt_coarsen
        _defaults["seg_duration_s"]    = seg_duration_s
        _defaults["full_trace_G"]      = full_trace_G

        # ── Batch / collect-only mode ────────────────────────────────────────
        # When called by the multi-file batch path, just hand back the chosen
        # parameters; the caller applies them to every file via
        # compute_correlation_for().  No computation or plotting happens here.
        if collect_only:
            result_box["params"] = {
                "tau_min_ms":        tau_min_ms,
                "tau_max_ms":        tau_max_ms,
                "corr_type":         corr_type,
                "method":            method,
                "segment":           segment,
                "gate":              use_gate,
                "points_per_decade": ppd,
                "thin_factor":       thin_factor,
                "mt_channels":       mt_m,
                "mt_coarsen":        mt_coarsen,
                "seg_duration_s":    seg_duration_s,
                "full_trace_G":      full_trace_G,
                "mt_compress":       _defaults["mt_compress"],
            }
            dialog.destroy()
            return

        dialog.destroy()

        tau_min_s = tau_min_ms * 1e-3
        if is_multitau:
            tau_edges, n_bins = None, 0
        else:
            tau_edges = build_tau_edges(tau_min_s, tau_max_s, ppd,
                                        grid_period_s=fcs_data.macrotime_period_s)
            n_bins    = len(tau_edges) - 1

        # ── Photon stream selection (with optional gating) ───────────────────
        times_ch1 = fcs_data.ch1_times_s
        times_ch2 = fcs_data.ch2_times_s

        if use_gate:
            gate = fcs_lifetime.select_gate(fcs_data)
            if gate is None:
                return
            gate_min_ns, gate_max_ns = gate
            times_ch1 = apply_time_gate(
                times_ch1, fcs_data.ch1_micro_ns, gate_min_ns, gate_max_ns)
            times_ch2 = apply_time_gate(
                times_ch2, fcs_data.ch2_micro_ns, gate_min_ns, gate_max_ns)
            if len(times_ch1) < 10 or len(times_ch2) < 10:
                messagebox.showerror(
                    "Gate too narrow",
                    f"Gate {gate_min_ns:.2f}–{gate_max_ns:.2f} ns retains "
                    f"only {len(times_ch1):,} Ch1 and {len(times_ch2):,} Ch2 "
                    f"photons.\n\nWiden the gate and try again."
                )
                return
        else:
            gate_min_ns = gate_max_ns = None

        # ── Photon thinning ──────────────────────────────────────────────────
        if thin_factor > 1:
            times_ch1 = thin_photons(times_ch1, thin_factor)
            times_ch2 = thin_photons(times_ch2, thin_factor)

        # ── Compute total progress steps ─────────────────────────────────────
        # Must match the work actually done, or the bar saturates early and
        # sits at 100% while the computation continues:
        #   * segmenting  -> one pass per segment
        #   * full_trace_G -> one EXTRA pass over the whole stream
        #   * symmetric cross -> each pass runs both A->B and B->A
        N = max(len(times_ch1), len(times_ch2))
        if segment:
            n_segs = len(segment_bounds(0.0, fcs_data.duration_s,
                                        seg_duration_s))
        else:
            n_segs = 1

        # Passes over the data: one per segment, plus the full-trace pass.
        n_passes = n_segs + (1 if (segment and full_trace_G) else 0)

        # Symmetrisation correlates both directions, and both now report
        # progress — but only for the tau_edges-based estimators.  The
        # multi-tau backends report ONE step per segment regardless of
        # direction, so they must not be doubled.
        sym_mult = 2 if (corr_type == "cross" and not is_multitau) else 1

        if is_multitau:
            total_steps = n_passes
        elif method == "perbin":
            total_steps = n_passes * n_bins * sym_mult
        else:
            N_per_seg   = max(1, N // max(1, n_segs))
            total_steps = (n_passes * sym_mult
                           * max(1, (N_per_seg + _TP_CHUNK_SIZE - 1)
                                    // _TP_CHUNK_SIZE))

        # ── Progress window ──────────────────────────────────────────────────
        pw = _ProgressWindow(
            None,
            total_steps,
            title="Computing correlation…",
        )

        cancelled = False

        def _progress(completed: int, label: str):
            nonlocal cancelled
            if pw.cancelled():
                cancelled = True
                raise KeyboardInterrupt("User cancelled")
            pw.step(completed, label)

        try:
            if is_multitau:
                if corr_type == "cross":
                    _a, _b = times_ch1, times_ch2
                elif corr_type == "auto_ch1":
                    _a = _b = times_ch1
                else:
                    _a = _b = times_ch2
                if method == "multitau_pkg":
                    tau, G_mean, G_std, n_segs = compute_multitau_pkg(
                        _a, _b, tau_min_s, tau_max_s, m=mt_m,
                        segment=segment, duration_s=fcs_data.duration_s,
                        progress_cb=_progress,
                        seg_duration_s=seg_duration_s,
                        full_trace_G=full_trace_G,
                        symmetric=(corr_type == "cross"),
                        compress=_defaults["mt_compress"])
                else:
                    tau, G_mean, G_std, n_segs = compute_multitau(
                        _a, _b, tau_min_s, tau_max_s, m=mt_m,
                        coarsen=mt_coarsen,
                        segment=segment, duration_s=fcs_data.duration_s,
                        progress_cb=_progress,
                        seg_duration_s=seg_duration_s,
                        full_trace_G=full_trace_G,
                        symmetric=(corr_type == "cross"))
            elif corr_type == "cross":
                # Symmetric ½(G_AB + G_BA) — matches compute_correlation_for,
                # so a file correlated single vs. batch gives the same answer.
                tau, G_mean, G_std, n_segs = compute_crosscorr_symmetric(
                    times_ch1, times_ch2, tau_edges, method, segment,
                    progress_cb=_progress,
                    seg_duration_s=seg_duration_s,
                    full_trace_G=full_trace_G)
            elif corr_type == "auto_ch1":
                tau, G_mean, G_std, n_segs = compute_autocorr(
                    times_ch1, tau_edges, method, segment,
                    progress_cb=_progress,
                    seg_duration_s=seg_duration_s,
                    full_trace_G=full_trace_G)
            else:
                tau, G_mean, G_std, n_segs = compute_autocorr(
                    times_ch2, tau_edges, method, segment,
                    progress_cb=_progress,
                    seg_duration_s=seg_duration_s,
                    full_trace_G=full_trace_G)
        except KeyboardInterrupt:
            pw.close()
            messagebox.showinfo("Cancelled", "Correlation computation cancelled.")
            return
        except NotImplementedError as e:
            pw.close()
            messagebox.showerror("Not implemented", str(e))
            return
        except Exception as e:
            pw.close()
            messagebox.showerror("Computation error", str(e))
            return

        pw.close()


        _n1, _n2 = len(times_ch1), len(times_ch2)
        _starts = [t[0]  for t in (times_ch1, times_ch2) if len(t)]
        _stops  = [t[-1] for t in (times_ch1, times_ch2) if len(t)]
        _T = (max(_stops) - min(_starts)) if _starts else 0.0

        plot_correlation(
            tau, G_mean, G_std,
            corr_type=corr_type,
            fcs_data=fcs_data,
            tau_min_s=tau_min_s,
            tau_max_s=tau_max_s,
            n_segs=n_segs,
            method=method,
            gate_min_ns=gate_min_ns,
            gate_max_ns=gate_max_ns,
            export=export, n_used_ch1=_n1, n_used_ch2=_n2, T_used=_T,
            seg_duration_s=(seg_duration_s if segment else None),
            full_trace_G=(full_trace_G if segment else None),
            mt_channels=(mt_m if is_multitau else None),
            mt_coarsen=(mt_coarsen if method == "multitau" else None),
            mt_compress=(_defaults["mt_compress"]
                         if method == "multitau_pkg" else None),
        )

    tk.Button(btn_frame, text="Compute", width=12,
              command=_on_compute, pady=4).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel", width=10,
              command=dialog.destroy, pady=4).pack(side="left", padx=6)

    # ── Method-dependent control visibility ───────────────────────────────────
    # Defined last, so every widget it repositions already exists.  tkinter's
    # pack() appends to the END of the parent's packing order, so a widget that
    # is pack_forget()-ten and later re-packed would jump to the bottom of the
    # dialog; `before=` pins it back to its proper place.
    def _update_method_controls(*_):
        m = method_var.get()

        # Bins/decade and photon thinning drive the log-spaced tau_edges grid,
        # which only the per-bin and two-pointer estimators use.  The multi-tau
        # backends bin the intensity onto their own octave grid and never see
        # either control.
        if m in ("perbin", "twopointer"):
            speed_frame.pack(fill="x", before=_tail, **pad)
        else:
            speed_frame.pack_forget()

        # Channels/octave: both multi-tau backends.
        if m in ("multitau", "multitau_pkg"):
            mt_row.pack(fill="x", padx=(22, 0), before=time_est_label)
        else:
            mt_row.pack_forget()

        # Coarsening: in-house multi-tau only — multipletau is fixed base-2.
        if m == "multitau":
            mt_coarsen_row.pack(fill="x", padx=(22, 0), before=time_est_label)
        else:
            mt_coarsen_row.pack_forget()

        if m == "multitau_pkg":
            mt_note_var.set(
                "      base-2 octaves (no coarsening factor); an odd channel\n"
                "      count is rounded up to the next even number")
            mt_note.pack(fill="x", before=time_est_label)
        else:
            mt_note.pack_forget()

    method_var.trace_add("write", _update_method_controls)
    _update_method_controls()

    # ── Size the window to its content, clamped to the screen ────────────────
    # Guessing a fixed geometry is how the buttons ended up clipped: the option
    # list grows, and any hardcoded height is wrong on some screen.  Measure the
    # content instead, then cap at 85% of screen height so the window always
    # fits; whatever does not fit is reachable by scrolling, and the buttons are
    # pinned outside the scroll region regardless.
    dialog.update_idletasks()
    _w = max(380, body.winfo_reqwidth() + _vsb.winfo_reqwidth() + 6)
    _content_h = body.winfo_reqheight() + btn_frame.winfo_reqheight() + 14
    _h = min(_content_h, int(dialog.winfo_screenheight() * 0.85))
    dialog.geometry(f"{_w}x{_h}")
    # Never let the user drag it shorter than the buttons plus a usable strip.
    dialog.minsize(_w, min(_content_h, 320))
    _on_body_configure()

    dialog.wait_window()
    return result_box["params"]


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fcs_corr.py <file.fcs> [tau_min_ms] [tau_max_ms] [method]")
        print("  method: perbin (default) | twopointer | multitau | multitau_pkg")
        sys.exit(1)
    d    = read_fcs(sys.argv[1])
    tmin = float(sys.argv[2]) if len(sys.argv) > 2 else _defaults["tau_min_ms"]
    tmax = float(sys.argv[3]) if len(sys.argv) > 3 else _defaults["tau_max_ms"]
    meth = sys.argv[4]        if len(sys.argv) > 4 else "perbin"

    if meth == "multitau_pkg":
        tau, G_mean, G_std, n_segs = compute_multitau_pkg(
            d.ch1_times_s, d.ch2_times_s, tmin * 1e-3, tmax * 1e-3,
            m=_defaults["mt_channels"], duration_s=d.duration_s,
            symmetric=True)
    elif meth == "multitau":
        tau, G_mean, G_std, n_segs = compute_multitau(
            d.ch1_times_s, d.ch2_times_s, tmin * 1e-3, tmax * 1e-3,
            m=_defaults["mt_channels"], coarsen=_defaults["mt_coarsen"],
            duration_s=d.duration_s, symmetric=True)
    else:
        edges = build_tau_edges(tmin * 1e-3, tmax * 1e-3,
                                grid_period_s=d.macrotime_period_s)
        tau, G_mean, G_std, n_segs = compute_crosscorr_symmetric(
            d.ch1_times_s, d.ch2_times_s, edges, meth)
    print(f"Segments used: {n_segs}")
    plot_correlation(tau, G_mean, G_std, "cross", d,
                     tmin * 1e-3, tmax * 1e-3, n_segs, method=meth,
                     seg_duration_s=_defaults["seg_duration_s"],
                     full_trace_G=_defaults["full_trace_G"],
                     mt_channels=(_defaults["mt_channels"]
                                  if meth in ("multitau", "multitau_pkg")
                                  else None),
                     mt_coarsen=(_defaults["mt_coarsen"]
                                 if meth == "multitau" else None),
                     mt_compress=(_defaults["mt_compress"]
                                  if meth == "multitau_pkg" else None))
