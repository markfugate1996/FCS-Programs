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
    compute_correlation_for(fcs_data, params)       -> result dict | None
    plot_correlation_overlay(results, ...)          -> (fig, ax)
    run_correlation_dialog(fcs_data, collect_only=) -> params | None
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
    return np.logspace(log_min, log_max, n)


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


# ── Segmented computation (core) ──────────────────────────────────────────────

def compute_segmented(
    timesA_s: np.ndarray,
    timesB_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = False,
    progress_cb=None,
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
    progress_cb : callable(completed: int, label: str) | None
        Optional progress callback.  Called after each unit of work with:
          completed — number of steps done so far (out of total_steps)
          label     — human-readable status string
        The total number of steps is:
          • segment=False, perbin:      n_bins
          • segment=False, twopointer:  ceil(N / _TP_CHUNK_SIZE)
          • segment=True,  perbin:      n_segs × n_bins
          • segment=True,  twopointer:  n_segs × ceil(N_seg / _TP_CHUNK_SIZE)

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
        counts  = _correlate(segA, segB, tau_edges, method, progress_cb, 0)
        G_mean  = _normalize(counts, segA, segB, tau_edges)
        G_std   = np.full_like(G_mean, np.nan)
        return tau, G_mean, G_std, 1

    # Segmented path
    tau_max        = tau_edges[-1]
    n_bins         = len(tau_edges) - 1
    seg_duration_s = _MIN_SEGMENT_FACTOR * tau_max
    t_start = max(timesA_s[0], timesB_s[0])
    t_end   = min(timesA_s[-1], timesB_s[-1])
    total   = t_end - t_start
    n_segs  = max(1, int(total // seg_duration_s))

    # Pre-compute step size per segment for the progress offset
    if method == "perbin":
        steps_per_seg = n_bins
    else:
        # Approximate photons per segment (used to size chunk count)
        avg_N_seg = max(len(timesA_s), len(timesB_s)) // n_segs
        steps_per_seg = max(1, (avg_N_seg + _TP_CHUNK_SIZE - 1) // _TP_CHUNK_SIZE)

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

        # Wrap progress_cb to inject the segment label prefix
        seg_cb = None
        if progress_cb is not None:
            offset = k * steps_per_seg
            def seg_cb(completed, label, _k=k, _n=n_segs, _off=offset):
                progress_cb(
                    _off + completed,
                    f"Segment {_k + 1} / {_n}  —  {label}",
                )

        counts = _correlate(segA, segB, tau_edges, method, seg_cb, 0)
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
    progress_cb=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Directed cross-correlation G_AB(tau) with optional segmented uncertainty.

    Returns
    -------
    tau, G_mean, G_std, n_segments
    """
    return compute_segmented(
        timesA_s, timesB_s, tau_edges, method, segment, progress_cb)


def compute_autocorr(
    times_s: np.ndarray,
    tau_edges: np.ndarray,
    method: Method = "perbin",
    segment: bool = False,
    progress_cb=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Autocorrelation G(tau) with optional segmented uncertainty.

    Returns
    -------
    tau, G_mean, G_std, n_segments
    """
    return compute_segmented(
        times_s, times_s, tau_edges, method, segment, progress_cb)


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
        }
        if gate_min_ns is not None:
            meta["gate_min_ns"] = f"{gate_min_ns:.3f}"
            meta["gate_max_ns"] = f"{gate_max_ns:.3f}"
        fcs_export.safe_export(
            fcs_data, "correlation", cols, meta=meta, suffix=corr_type,
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
    """
    Write one file's plotted correlation curve to a CSV.

    Mirrors the inline export in :func:`plot_correlation` so single-file and
    batch/overlay exports use an identical column and metadata layout.
    """
    cols: dict = {
        "tau_s":  tau,
        "tau_ms": tau * 1e3,
        "G":      G_mean,
    }
    if np.isfinite(G_std).any():
        cols["G_std"] = G_std
    meta = {
        "type":       corr_type,
        "method":     method,
        "tau_min_s":  f"{tau_min_s:.6g}",
        "tau_max_s":  f"{tau_max_s:.6g}",
        "n_segments": n_segs,
    }
    if gate_min_ns is not None:
        meta["gate_min_ns"] = f"{gate_min_ns:.3f}"
        meta["gate_max_ns"] = f"{gate_max_ns:.3f}"
    fcs_export.safe_export(
        fcs_data, "correlation", cols, meta=meta, suffix=corr_type,
    )


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
    thin_factor = max(1, int(params.get("thin_factor", 1)))

    tau_min_s = float(params["tau_min_ms"]) * 1e-3
    tau_max_s = float(params["tau_max_ms"]) * 1e-3
    try:
        tau_edges = build_tau_edges(tau_min_s, tau_max_s, ppd)
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
    if method == "perbin":
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
        if corr_type == "cross":
            tau, G_mean, G_std, n_segs = compute_crosscorr(
                times_ch1, times_ch2, tau_edges, method, segment,
                progress_cb=progress_cb)
        elif corr_type == "auto_ch1":
            tau, G_mean, G_std, n_segs = compute_autocorr(
                times_ch1, tau_edges, method, segment, progress_cb=progress_cb)
        else:
            tau, G_mean, G_std, n_segs = compute_autocorr(
                times_ch2, tau_edges, method, segment, progress_cb=progress_cb)
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
    "tau_min_ms":       0.01,
    "tau_max_ms":       1000.0,
    "corr_type":        "cross",
    "method":           "perbin",
    "segment":          False,
    "gate":             False,
    "points_per_decade": _POINTS_PER_DECADE,
    "thin_factor":      1,
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
    dialog.geometry("340x640")
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

    # ── Speed controls ────────────────────────────────────────────────────────
    speed_frame = tk.LabelFrame(dialog, text="Speed / resolution trade-off",
                                padx=10, pady=6)
    speed_frame.pack(fill="x", **pad)

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
            edges_     = build_tau_edges(tau_min_s_, tau_max_s_, ppd)
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
            tau_edges_ = build_tau_edges(tau_min_s, tau_max_s, ppd)
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
                seg_dur = _MIN_SEGMENT_FACTOR * tau_max_s
                n_segs  = max(1, int(total_s // seg_dur))
            else:
                n_segs = 1

            est = estimate_corr_time(
                N=N, n_bins=n_bins, method=method,
                n_segs=n_segs, tau_max_s=tau_max_s, total_s=total_s,
            )
            time_est_var.set(f"  Estimated time: {est}")
        except (ValueError, ZeroDivisionError, tk.TclError):
            time_est_var.set("")

    for _var in (method_var, min_var, max_var, segment_var, ppd_var, thin_var):
        _var.trace_add("write", _update_time_estimate)
    _update_time_estimate()

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
        _defaults["tau_min_ms"]        = tau_min_ms
        _defaults["tau_max_ms"]        = tau_max_ms
        _defaults["corr_type"]         = corr_type
        _defaults["method"]            = method
        _defaults["segment"]           = segment
        _defaults["gate"]              = use_gate
        _defaults["points_per_decade"] = ppd
        _defaults["thin_factor"]       = thin_factor

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
            }
            dialog.destroy()
            return

        dialog.destroy()

        tau_min_s = tau_min_ms * 1e-3
        tau_edges = build_tau_edges(tau_min_s, tau_max_s, ppd)
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
        N = max(len(times_ch1), len(times_ch2))
        if segment:
            seg_dur = _MIN_SEGMENT_FACTOR * tau_max_s
            total_s = fcs_data.duration_s
            n_segs  = max(1, int(total_s // seg_dur))
        else:
            n_segs = 1

        if method == "perbin":
            total_steps = n_segs * n_bins
        else:
            N_per_seg   = max(1, N // n_segs)
            total_steps = n_segs * max(1, (N_per_seg + _TP_CHUNK_SIZE - 1)
                                          // _TP_CHUNK_SIZE)

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
            if corr_type == "cross":
                tau, G_mean, G_std, n_segs = compute_crosscorr(
                    times_ch1, times_ch2, tau_edges, method, segment,
                    progress_cb=_progress)
            elif corr_type == "auto_ch1":
                tau, G_mean, G_std, n_segs = compute_autocorr(
                    times_ch1, tau_edges, method, segment,
                    progress_cb=_progress)
            else:
                tau, G_mean, G_std, n_segs = compute_autocorr(
                    times_ch2, tau_edges, method, segment,
                    progress_cb=_progress)
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
