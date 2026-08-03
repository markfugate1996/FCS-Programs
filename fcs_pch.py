"""
fcs_pch.py
==========
Photon Counting Histogram (PCH) analysis for FCS data.

The PCH is the probability distribution p(k) of detecting exactly k photons
in a time bin of width T.  For a single diffusing species the shape is a
super-Poissonian distribution whose mean <k> and variance Var(k) encode:

    <k>       — proportional to concentration × brightness
    Var(k)    — for a Poisson process Var = <k>; excess variance
                  (Var/mean > 1) reflects photon bunching from
                  diffusion through the PSF and true molecular brightness

The ratio  ε = Var(k) / <k>  (the Mandel Q parameter + 1) and the
derived single-molecule brightness  ε_mol = (Var(k) − <k>) / <k>
are shown on the plot as annotations.

Public API
----------
    compute_pch_counts(times_s, bin_width_s)       -> (k, n_k, M, mean, var)
    compute_pch(times_s, bin_width_s)              -> (k, pk, mean, var)
    plot_pch(channels_data, bin_width_s, ...)      -> (fig, ax)
    run_pch_dialog(fcs_data)                       -> shows dialog + plot

Exported CSVs
-------------
A PCH export carries the RAW per-count frequencies ``n_<label>`` as well as
the normalised ``pk_<label>``, and records the number of sampled time bins M
and the observed count rate for every series in its header.  Those are exactly
the quantities :mod:`fcs_pch_fit` needs to weight a fit by the Poisson error of
each histogram bin (sigma = sqrt(n_k)) and to report the predicted/observed
count-rate ratio, so a saved histogram can be fitted with no .fcs file present
and gives numerically identical results to fitting the photon records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, NamedTuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fcs_plottools
from scipy.stats import poisson as _scipy_poisson

from fcs_reader import FCSData, read_fcs
import fcs_export

# ── Colours consistent with the rest of the suite ────────────────────────────

_CH_COLOUR = {1: "steelblue", 2: "tomato", "both": "mediumpurple"}

# ── Default bin widths to offer in the dialog ────────────────────────────────

_BIN_WIDTH_OPTIONS = {
    "10 µs":  10e-6,
    "50 µs":  50e-6,
    "100 µs": 100e-6,
    "500 µs": 500e-6,
    "1 ms":   1e-3,
    "5 ms":   5e-3,
    "10 ms":  10e-3,
}

_DEFAULT_BIN_WIDTH_LABEL = "100 µs"


# ── Core computation ──────────────────────────────────────────────────────────

def acquisition_window(fcs_data) -> Tuple[float, float, str]:
    """
    The interval over which the detectors were actually observing.

    Why this exists
    ---------------
    The PCH used to bin each channel from its OWN first photon to its OWN last
    photon.  That quietly discarded the observed-but-empty time at the head and
    tail of the acquisition -- time in which the detector was running and
    recorded nothing, which is a real measurement of k = 0 and belongs in the
    histogram.  Dropping it removes k = 0 bins only, so it biases p(0) DOWN and
    every other p(k) up, and it gave Ch1 and Ch2 different M for one shared
    acquisition.  Binning both channels over the true window fixes both.

    How the window is chosen
    ------------------------
    ``duration_s`` on the dataset is the acquisition length, so when every
    photon on both channels lies inside ``[0, duration_s]`` that is the window
    and the third return value is ``"acquisition"``.  If the photon times do
    not sit on that axis (an absolute clock, say) the fallback is the union of
    both channels' spans, tagged ``"photon_span_union"``: still not the true
    window, but at least one window shared by both channels rather than two.

    The tag is returned rather than inferred later because it is written into
    the export header -- a reader should be told which convention produced the
    k = 0 bin count instead of having to guess.
    """
    starts, ends = [], []
    for attr in ("ch1_times_s", "ch2_times_s"):
        t = getattr(fcs_data, attr, None)
        if t is not None and len(t):
            starts.append(float(t[0]))
            ends.append(float(t[-1]))
    if not starts:
        raise ValueError("No photons on either channel; cannot compute PCH.")
    t_first, t_last = min(starts), max(ends)

    dur = getattr(fcs_data, "duration_s", None)
    try:
        dur = float(dur)
    except (TypeError, ValueError):
        dur = None
    if dur is not None and np.isfinite(dur) and dur > 0:
        # Only trust [0, duration] if the photons really lie on that axis.
        # Asserting it blindly would invent empty leading bins that were never
        # observed -- the opposite error, and a worse one.
        if t_first >= 0.0 and t_last <= dur:
            return 0.0, dur, "acquisition"

    return t_first, t_last, "photon_span_union"


def compute_pch_counts(
    times_s: np.ndarray,
    bin_width_s: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    """
    Compute the photon counting histogram as RAW per-count frequencies.

    The window ``[t_start, t_end]`` is divided into non-overlapping bins of
    width bin_width_s and the histogram of counts-per-bin is the PCH.  Bins in
    which nothing was detected are counted as k = 0, including any at the head
    or tail of the acquisition: they are observations, not absent data.

    This is the primitive; :func:`compute_pch` normalises its output.  The raw
    frequencies and the sampled-bin count M are kept because they, not the
    probabilities, are what a weighted fit needs: p(k) alone cannot say how
    many bins went into each point, so it cannot supply a Poisson error bar.

    It is deliberately identical in construction to
    ``fcs_pch_fit.pch_counts`` -- same window, same M, same edges, same ddof --
    so a fit run from an exported histogram and one run from the photon records
    see exactly the same numbers.

    The trailing PARTIAL bin
    ------------------------
    If the window is not an exact multiple of the bin width, the leftover slice
    at the end is not binned, and photons inside it are not counted.  This is
    the one omission that is deliberate: a short bin is not an observation of
    the same quantity as a full one, and folding it in would add a point drawn
    from a narrower window and bias p(k) toward low k.  It is at most one bin
    wide -- under 0.001% of a typical acquisition -- and the amount discarded
    is reported by the caller rather than left silent.

    Parameters
    ----------
    times_s     : np.ndarray
        Sorted absolute photon arrival times in seconds.
    bin_width_s : float
        Bin width in seconds.
    t_start, t_end : float, optional
        The observation window.  Defaults to this channel's own first and last
        photon, which is the old behaviour and is retained only for callers
        with no dataset to ask; :func:`acquisition_window` supplies the honest
        window and every caller in this suite uses it.

    Returns
    -------
    k       : np.ndarray (int)
        Photon count values (0, 1, 2, …, k_max).
    n_k     : np.ndarray (float)
        Number of time bins holding exactly k photons  (sums to M).
    M       : int
        Number of sampled time bins.
    mean    : float
        Mean photons per bin  <k>.
    var     : float
        Variance of photons per bin  Var(k).
    """
    times_s = np.asarray(times_s, dtype=np.float64)
    if times_s.size == 0:
        raise ValueError("times_s is empty; cannot compute PCH.")

    t_start = float(times_s[0]) if t_start is None else float(t_start)
    t_end   = float(times_s[-1]) if t_end is None else float(t_end)
    if t_end <= t_start:
        raise ValueError(
            f"PCH window is empty: t_start={t_start:.6g} s is not before "
            f"t_end={t_end:.6g} s.")
    # A photon outside the window means the window is wrong for these data.
    # np.histogram would drop it without a word, so say so instead: silently
    # binning 90% of a channel is exactly the kind of quiet trimming this
    # window was introduced to stop.
    if times_s[0] < t_start or times_s[-1] > t_end:
        raise ValueError(
            f"Photon times span [{times_s[0]:.6g}, {times_s[-1]:.6g}] s, "
            f"outside the requested PCH window "
            f"[{t_start:.6g}, {t_end:.6g}] s.")

    n_bins = max(1, int((t_end - t_start) / bin_width_s))
    edges  = t_start + np.arange(n_bins + 1) * bin_width_s

    counts, _ = np.histogram(times_s, bins=edges)
    # np.histogram puts values equal to the last edge INTO the last bin, so a
    # photon landing exactly on it is kept rather than lost.

    mean = float(counts.mean())
    var  = float(counts.var(ddof=1)) if len(counts) > 1 else float("nan")

    k_max = int(counts.max())
    k     = np.arange(0, k_max + 1, dtype=int)
    n_k   = np.bincount(counts, minlength=k_max + 1).astype(float)

    return k, n_k, int(n_bins), mean, var


def window_tail(bin_width_s: float, t_start: float, t_end: float
                ) -> Tuple[float, int]:
    """
    Size of the unbinned trailing slice: ``(seconds, whole_bins_used)``.

    Reported in the export header so the discarded remainder is on the record.
    """
    span = float(t_end) - float(t_start)
    n_bins = max(1, int(span / bin_width_s))
    return span - n_bins * bin_width_s, n_bins


def compute_pch(
    times_s: np.ndarray,
    bin_width_s: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute the normalised photon counting histogram p(k).

    Thin wrapper over :func:`compute_pch_counts` kept for callers that only
    want probabilities.  ``pk = n_k / M`` exactly, since the frequencies sum
    to the number of sampled bins.

    Returns
    -------
    k, pk, mean, var  -- see :func:`compute_pch_counts`.
    """
    k, n_k, M, mean, var = compute_pch_counts(
        times_s, bin_width_s, t_start=t_start, t_end=t_end)
    return k, n_k / n_k.sum(), mean, var


# ── One plotted / exported series ─────────────────────────────────────────────

class PCHSeries(NamedTuple):
    """
    One PCH curve: what is plotted, plus what a later fit will need.

    ``channels_data`` maps a display label ('Ch1', 'Ch2', 'Ch1+Ch2') to one of
    these.  It was a bare 5-tuple ``(k, pk, mean, var, colour)``; the extra
    fields are the reason saved histograms became fittable, and a NamedTuple
    keeps every existing positional use valid while letting the exporter reach
    ``s.n_k`` and ``s.M`` by name.

    M is carried per series even though every series of one file now shares the
    same acquisition window and therefore the same M.  It is stored beside the
    counts it belongs to so that ``sum(n_k) == M`` is checkable per series on
    read, which is the check that catches a truncated or mis-padded column.
    """
    k:      np.ndarray      # count values 0..k_max
    pk:     np.ndarray      # probabilities, sum = 1
    mean:   float           # <k>
    var:    float           # Var(k), ddof=1
    colour: str             # plot colour
    n_k:    np.ndarray = None   # raw frequencies, sum = M
    M:      int = 0             # number of sampled time bins
    cps:    Optional[float] = None   # observed count rate for this series (Hz)
    window: Optional[Tuple[float, float]] = None   # (t_start, t_end) in s
    window_source: str = ""     # 'acquisition' | 'photon_span_union'


# ── Plotting ──────────────────────────────────────────────────────────────────

def _export_pch(
    fcs_data: FCSData,
    channels_data: Dict[str, "PCHSeries"],
    bin_width_s: float,
) -> None:
    """
    Write the plotted PCH series for one file to a CSV.

    Columns
    -------
    ``k`` then, per series, ``pk_<label>`` (probability) and ``n_<label>``
    (raw frequency).  The two are redundant -- ``pk = n_k / M`` -- and both are
    written anyway: p(k) is the plotted quantity and the one a human reads,
    n_k is the one a weighted fit needs.  Storing the raw counts is what makes
    the export fittable rather than merely viewable.

    Padding
    -------
    Each series has its own contiguous k = 0..k_last, and the table shares one
    k axis running to the longest.  Cells past a series' own k_last are NaN,
    not 0, matching ``fcs_lifetime._lifetime_export_columns``.  The distinction
    is load-bearing here: inside a series' range a zero is a real measurement
    ("no bin held exactly k photons"), while past its end nothing was sampled
    at all.  Zero-padding would blur the two and quietly hand the fitter extra
    data points whose Poisson weight sigma = sqrt(max(n,1)) = 1 is fictitious.
    NaN also round-trips, since fcs_export writes it as a blank cell and
    read_export maps a blank back to NaN.

    Metadata
    --------
    Per series: the sampled-bin count M, the observed count rate, the photon
    total, and the moments.  M and the count rate are not conveniences -- M
    sets the Poisson weights and the reduced chi-squared scale, and the count
    rate is what the fitter divides its predicted rate by to report
    ``predicted_over_observed``.  Neither is recoverable from p(k) with
    certainty, so both are recorded rather than inferred later.
    """
    if not channels_data:
        return
    k_max  = max(int(np.asarray(s.k)[-1]) for s in channels_data.values())
    k_full = np.arange(0, k_max + 1)
    cols: Dict[str, np.ndarray] = {"k": k_full.astype(float)}
    meta: Dict[str, object] = {
        "pch_schema":  "2",
        "bin_width_s": f"{bin_width_s:.10g}",
        "series":      "+".join(channels_data.keys()),
    }
    duration = getattr(fcs_data, "duration_s", None)
    if duration is not None:
        meta["duration_s"] = f"{float(duration):.10g}"

    # The window every series was binned over, and what was left unbinned.
    # A reader can then tell an honest k = 0 count from a trimmed one without
    # re-deriving anything, and can see exactly how much time went unused.
    _any = next(iter(channels_data.values()))
    if _any.window is not None:
        w0, w1 = _any.window
        tail_s, n_used = window_tail(bin_width_s, w0, w1)
        meta["window_start_s"]   = f"{w0:.10g}"
        meta["window_end_s"]     = f"{w1:.10g}"
        meta["window_source"]    = _any.window_source
        meta["window_unbinned_tail_s"] = f"{tail_s:.6g}"
        meta["note_window"] = (
            "all series binned over one shared window; empty time inside it "
            "is counted as k=0. Only the trailing partial bin is unbinned.")

    for label, s in channels_data.items():
        idx = np.asarray(s.k, dtype=int)

        pk_full = np.full(k_max + 1, np.nan, dtype=float)
        pk_full[idx] = s.pk
        cols[f"pk_{label}"] = pk_full

        if s.n_k is not None:
            n_full = np.full(k_max + 1, np.nan, dtype=float)
            n_full[idx] = np.asarray(s.n_k, dtype=float)
            cols[f"n_{label}"] = n_full
            meta[f"{label}_photons"] = f"{int(round(float(np.sum(np.asarray(s.n_k) * idx))))}"

        if s.M:
            meta[f"{label}_sampled_bins_M"] = f"{int(s.M)}"
        if s.cps is not None and np.isfinite(s.cps):
            meta[f"{label}_observed_cps"] = f"{float(s.cps):.10g}"
        meta[f"{label}_mean"]          = f"{s.mean:.10g}"
        meta[f"{label}_var"]           = f"{s.var:.10g}"
        meta[f"{label}_var_over_mean"] = (f"{s.var / s.mean:.10g}"
                                          if s.mean > 0 else "nan")

    ch_tag = "_".join(channels_data.keys())
    bw_tag = f"{bin_width_s * 1e6:.0f}us"
    fcs_export.safe_export(
        fcs_data, "pch", cols, meta=meta, suffix=f"{ch_tag}_{bw_tag}",
    )


def plot_pch(
    channels_data: Dict[str, "PCHSeries"],
    bin_width_s: float,
    fcs_data: FCSData,
    show: bool = True,
    export: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot one or more photon counting histograms.

    Parameters
    ----------
    channels_data : dict
        Keys are display labels (e.g. 'Ch1', 'Ch2', 'Ch1+Ch2').
        Values are PCHSeries records from _build_channels_data.
    bin_width_s : float
        Bin width used (shown in title).
    fcs_data : FCSData
        Source data object (for file name in title).
    show : bool
        Call plt.show() if True.

    Returns
    -------
    fig, ax
    """
    # ── Optional CSV export of the plotted data ───────────────────────────────
    if export:
        _export_pch(fcs_data, channels_data, bin_width_s)

    fig, ax = plt.subplots(figsize=(9, 5))

    annotation_lines: list[str] = []

    for label, _s in channels_data.items():
        k, pk, mean, var, colour = _s.k, _s.pk, _s.mean, _s.var, _s.colour
        # ── Bar chart of the PCH ──────────────────────────────────────────────
        ax.bar(k, pk, width=0.75, color=colour, alpha=0.55,
               label=label, zorder=2)
        ax.plot(k, pk, color=colour, linewidth=1.2,
                marker="o", markersize=3.5, zorder=3)

        # ── Overlaid Poisson reference ────────────────────────────────────────
        # A perfect Poisson process with the same mean, for visual comparison.
        k_ref = np.arange(0, max(k[-1] + 1, int(mean * 3 + 6 * mean**0.5 + 1)))
        pk_ref = _scipy_poisson.pmf(k_ref, mean)
        ax.plot(k_ref, pk_ref,
                color=colour, linewidth=1.0, linestyle="--", alpha=0.55,
                zorder=2, label=f"{label} Poisson (μ={mean:.2f})")

        # ── Annotation block per channel ──────────────────────────────────────
        mandel_q   = (var - mean) / mean if mean > 0 else float("nan")
        brightness = (var - mean) / mean if mean > 0 else float("nan")
        # ε_mol = (Var − <k>) / <k>   [photons per molecule per bin]
        # For a pure Poisson, Var = <k> so ε_mol = 0.
        annotation_lines.append(
            f"{label}:  <k> = {mean:.3f}   Var = {var:.3f}   "
            f"Var/<k> = {var/mean:.3f}   Q = {mandel_q:.3f}"
        )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_yscale("log")
    ax.set_xlabel("Photons per bin  k", fontsize=12)
    ax.set_ylabel("Probability  p(k)", fontsize=12)

    bw_str = _format_bin_width(bin_width_s)
    title    = f"Photon Counting Histogram — {fcs_data.filepath.name}"
    subtitle = f"Bin width: {bw_str}  ·  dashed = Poisson reference"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.0e}" if x < 1e-2 else f"{x:.3f}")
    )
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    # ── Legend ────────────────────────────────────────────────────────────────
    ax.legend(fontsize=9, framealpha=0.85, loc="upper right")

    # ── Statistics annotation (bottom of figure) ──────────────────────────────
    ann_text = "\n".join(annotation_lines)
    fig.text(
        0.5, 0.01, ann_text,
        ha="center", va="bottom",
        fontsize=8.5, color="#333333",
        family="monospace",
    )

    fig.tight_layout(rect=[0, 0.06 * len(annotation_lines), 1, 1])

    if show:
        #plt.show was static; now dynamic w/ fcs_plottools
        #plt.show()
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


def _format_bin_width(bin_width_s: float) -> str:
    if bin_width_s < 1e-3:
        return f"{bin_width_s * 1e6:.3g} µs"
    elif bin_width_s < 1:
        return f"{bin_width_s * 1e3:.3g} ms"
    return f"{bin_width_s:.3g} s"


# ── Per-file helpers (shared by dialog, batch, and overlay) ───────────────────

def _build_channels_data(
    fcs_data: FCSData,
    channel_choice: str,
    bin_width_s: float,
) -> Dict[str, "PCHSeries"]:
    """
    Compute the per-series ``{label: PCHSeries}`` dict for a single file given
    a channel choice ('ch1', 'ch2', 'both', or 'combined').

    Every series is binned over the SAME window -- the acquisition, not each
    channel's own first-to-last photon -- so Ch1 and Ch2 share one M, their
    bins line up in time, and observed empty time is counted as k = 0 on both.
    """
    t0, t1, wsrc = acquisition_window(fcs_data)

    def _series(times, colour, cps) -> PCHSeries:
        k, n_k, M, mean, var = compute_pch_counts(
            times, bin_width_s, t_start=t0, t_end=t1)
        return PCHSeries(k=k, pk=n_k / n_k.sum(), mean=mean, var=var,
                         colour=colour, n_k=n_k, M=M, cps=cps,
                         window=(t0, t1), window_source=wsrc)

    r1 = getattr(fcs_data, "count_rate_ch1_hz", None)
    r2 = getattr(fcs_data, "count_rate_ch2_hz", None)

    channels_data: Dict[str, PCHSeries] = {}
    if channel_choice in ("ch1", "both"):
        channels_data["Ch1"] = _series(fcs_data.ch1_times_s, _CH_COLOUR[1], r1)
    if channel_choice in ("ch2", "both"):
        channels_data["Ch2"] = _series(fcs_data.ch2_times_s, _CH_COLOUR[2], r2)
    if channel_choice == "combined":
        t = np.sort(np.concatenate([fcs_data.ch1_times_s, fcs_data.ch2_times_s]))
        both = (r1 + r2) if (r1 is not None and r2 is not None) else None
        channels_data["Ch1+Ch2"] = _series(t, _CH_COLOUR["both"], both)
    return channels_data


def plot_pch_single(
    fcs_data: FCSData,
    channel_choice: str,
    bin_width_s: float,
    show: bool = True,
    export: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Compute and plot the PCH for one file in its own figure."""
    channels_data = _build_channels_data(fcs_data, channel_choice, bin_width_s)
    return plot_pch(channels_data, bin_width_s, fcs_data, show=show, export=export)


# ── Overlay plotting (batch / combined) ───────────────────────────────────────

# Marker per series, so files (colour) and channels (marker) stay readable
# when several PCHs share one axes.
_SERIES_MARKER = {"Ch1": "o", "Ch2": "s", "Ch1+Ch2": "o"}


def plot_pch_overlay(
    datasets,
    channel_choice: str,
    bin_width_s: float,
    show: bool = True,
    export: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Overlay photon counting histograms from several files on one log-y axes.

    Each file is drawn in its own colour.  When ``channel_choice == "both"``
    Ch1 uses circle markers and Ch2 square markers within the file's colour.
    The per-channel Poisson reference curves drawn in the single-file plot are
    omitted here to keep a multi-file overlay readable.  When *export* is True
    each file's series are written to their own CSV exactly as in the
    single-file path.

    Parameters
    ----------
    datasets : sequence of FCSData
        Files to overlay (must be non-empty).
    channel_choice : str
        'ch1', 'ch2', 'both', or 'combined'; applied to every file.
    bin_width_s : float
        Bin width in seconds, shared by all files.

    Returns
    -------
    fig, ax
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("plot_pch_overlay requires at least one dataset.")

    colours = fcs_plottools.palette(len(datasets))
    fig, ax = plt.subplots(figsize=(9, 5))

    multi_channel = channel_choice == "both"
    for d, colour in zip(datasets, colours):
        channels_data = _build_channels_data(d, channel_choice, bin_width_s)
        first = True
        for label, _s in channels_data.items():
            ax.plot(
                _s.k, _s.pk, color=colour, linewidth=1.2,
                marker=_SERIES_MARKER.get(label, "o"), markersize=3.5,
                alpha=0.9,
                label=d.filepath.name if first else None,
            )
            first = False
        if export:
            _export_pch(d, channels_data, bin_width_s)

    ax.set_yscale("log")
    ax.set_xlabel("Photons per bin  k", fontsize=12)
    ax.set_ylabel("Probability  p(k)", fontsize=12)

    bw_str   = _format_bin_width(bin_width_s)
    ch_note  = "   ·   circles = Ch1   ·   squares = Ch2" if multi_channel else ""
    ax.set_title(
        f"PCH overlay — {len(datasets)} files  ·  bin width: {bw_str}{ch_note}",
        fontsize=10,
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.0e}" if x < 1e-2 else f"{x:.3f}")
    )
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
    ax.legend(fontsize=9, framealpha=0.85, loc="upper right", title="File")

    fig.tight_layout()
    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── Reopening an exported PCH ─────────────────────────────────────────────────

def _label_colour(label: str) -> str:
    """Plot colour for an exported series label ('Ch1', 'Ch2', 'Ch1+Ch2')."""
    if "+" in label:
        return _CH_COLOUR["both"]
    if label.startswith("Ch") and label[2:].isdigit():
        return _CH_COLOUR.get(int(label[2:]), _CH_COLOUR[1])
    return _CH_COLOUR[1]


class _RebuiltSource:
    """
    Stands in for the FCSData a rebuilt plot no longer has.

    Named after the ORIGINAL measurement rather than the CSV, so a reopened
    figure carries the same title as a freshly drawn one; the point of
    reopening is to get the result back, not to be reminded which file it was
    cached in.
    """
    def __init__(self, path, name):
        self.filepath = Path(name) if name else Path(path)


def rebuild_plot(meta, columns, show: bool = True, path=None):
    """
    Redraw an exported PCH as a live figure, for :mod:`fcs_plotopen`.

    Takes what :func:`fcs_export.read_export` returns and calls the SAME
    :func:`plot_pch` that drew the file in the first place, so a reopened
    figure is identical to a fresh one and there is no second rendering path to
    drift out of step.

    Rebuilding uses ``pk``, not the counts: p(k) is what was plotted, and it is
    present in every PCH export including ones written before the raw counts
    were added.  A histogram that cannot be FITTED can still be LOOKED at, and
    refusing to reopen it would be a needless loss -- the counts matter for
    Poisson weighting, which a plot does not do.
    """
    if "k" not in columns:
        raise ValueError(
            "This file has no 'k' column, so it is not a PCH export.")
    if "pk_fit" in columns or "counts_fit" in columns:
        raise ValueError(
            "This is a PCH fit curve, not a measured histogram.  Reopen the "
            "histogram written by the PCH plotting task instead.")

    k_all = np.asarray(columns["k"], dtype=np.float64)

    labels = [c[3:] for c in columns if c.startswith("pk_")]
    if not labels:
        raise ValueError(
            "This file has a 'k' column but no 'pk_<channel>' column, so "
            "there is no histogram in it to draw.")

    try:
        bin_width_s = float(meta.get("bin_width_s"))
    except (TypeError, ValueError):
        raise ValueError(
            "This PCH export does not record 'bin_width_s' in its header, so "
            "the histogram cannot be labelled with the bin width it used.")

    channels_data: Dict[str, PCHSeries] = {}
    for label in labels:
        pk = np.asarray(columns[f"pk_{label}"], dtype=np.float64)
        good = np.isfinite(pk)
        if not good.any():
            continue
        k_ser  = k_all[good].astype(int)
        pk_ser = pk[good]

        # Moments come from the header when it has them and are otherwise
        # recomputed from p(k).  Both are needed: the Poisson reference curve
        # and the Q annotation are drawn from <k> and Var(k).
        def _hdr(key, fallback):
            try:
                v = float(meta[f"{label}_{key}"])
                return v if np.isfinite(v) else fallback
            except (KeyError, TypeError, ValueError):
                return fallback
        norm  = pk_ser.sum()
        mean0 = float(np.sum(pk_ser * k_ser) / norm) if norm else float("nan")
        var0  = (float(np.sum(pk_ser * (k_ser - mean0) ** 2) / norm)
                 if norm else float("nan"))

        n_col = columns.get(f"n_{label}")
        n_k = M = None
        if n_col is not None:
            n_arr = np.asarray(n_col, dtype=np.float64)[good]
            if np.isfinite(n_arr).all():
                n_k = n_arr
                M = int(round(float(n_arr.sum())))

        channels_data[label] = PCHSeries(
            k=k_ser, pk=pk_ser,
            mean=_hdr("mean", mean0), var=_hdr("var", var0),
            colour=_label_colour(label),
            n_k=n_k, M=M or 0,
            cps=_hdr("observed_cps", None),
        )

    if not channels_data:
        raise ValueError("This PCH export contains no plottable series.")

    src = _RebuiltSource(path, meta.get("source file") or meta.get("source"))
    # export=False without exception: reopening is for looking at a result, and
    # re-exporting would overwrite the very file just read.
    return plot_pch(channels_data, bin_width_s, src, show=show, export=False)


# ── Dialog ────────────────────────────────────────────────────────────────────

_defaults: dict = {
    "channels":       "both",
    "bin_width_label": _DEFAULT_BIN_WIDTH_LABEL,
}


def run_pch_dialog(fcs_data: FCSData, export: bool = False):
    """
    Show a parameter dialog, then compute and plot the PCH.
    Settings persist between calls within the same session.
    """
    import tkinter as tk
    from tkinter import messagebox

    dialog = tk.Toplevel()
    dialog.title("PCH — options")
    dialog.geometry("320x310")
    dialog.resizable(False, False)
    dialog.grab_set()

    pad = dict(padx=12, pady=4)

    tk.Label(dialog, text="Photon Counting Histogram",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    # ── Channel selection ─────────────────────────────────────────────────────
    ch_frame = tk.LabelFrame(dialog, text="Channel(s)", padx=10, pady=4)
    ch_frame.pack(fill="x", **pad)

    ch_var = tk.StringVar(value=_defaults["channels"])
    n_ch1 = len(fcs_data.ch1_deltas)
    n_ch2 = len(fcs_data.ch2_deltas)
    for text, value in [
        (f"Ch1 only   ({n_ch1:,} photons)",                "ch1"),
        (f"Ch2 only   ({n_ch2:,} photons)",                "ch2"),
        (f"Both channels — overlay",                        "both"),
        (f"Both channels — combined (Ch1 + Ch2)",           "combined"),
    ]:
        tk.Radiobutton(ch_frame, text=text, variable=ch_var,
                       value=value, anchor="w").pack(fill="x")

    # ── Bin width ─────────────────────────────────────────────────────────────
    bw_frame = tk.LabelFrame(dialog, text="Bin width", padx=10, pady=6)
    bw_frame.pack(fill="x", **pad)

    bw_var = tk.StringVar(value=_defaults["bin_width_label"])

    bw_row = tk.Frame(bw_frame)
    bw_row.pack(fill="x")
    tk.Label(bw_row, text="Width:", width=6, anchor="e").pack(side="left")
    bw_menu = tk.OptionMenu(bw_row, bw_var, *_BIN_WIDTH_OPTIONS.keys())
    bw_menu.config(width=10)
    bw_menu.pack(side="left", padx=4)

    # Info: expected mean photons per bin given current selection
    info_var = tk.StringVar(value="")
    info_lbl = tk.Label(bw_frame, textvariable=info_var,
                        font=("Helvetica", 9), fg="grey", anchor="w")
    info_lbl.pack(fill="x")

    def _update_info(*_):
        bw = _BIN_WIDTH_OPTIONS.get(bw_var.get(), 100e-6)
        dur = fcs_data.duration_s
        mean_ch1 = fcs_data.count_rate_ch1_hz * bw
        mean_ch2 = fcs_data.count_rate_ch2_hz * bw
        n_bins   = int(dur / bw)
        info_var.set(
            f"  ~{n_bins:,} bins  ·  "
            f"<k> Ch1≈{mean_ch1:.2f}  Ch2≈{mean_ch2:.2f} photons/bin"
        )

    bw_var.trace_add("write", _update_info)
    _update_info()

    # ── Buttons ───────────────────────────────────────────────────────────────
    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=12)

    def _on_compute():
        channel_choice  = ch_var.get()
        bw_label        = bw_var.get()
        bin_width_s     = _BIN_WIDTH_OPTIONS[bw_label]

        # Persist
        _defaults["channels"]        = channel_choice
        _defaults["bin_width_label"] = bw_label

        dialog.destroy()

        # Build the dict of PCHSeries per series
        try:
            channels_data = _build_channels_data(
                fcs_data, channel_choice, bin_width_s)
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Computation error", str(e))
            return

        plot_pch(channels_data, bin_width_s, fcs_data, export=export)

    tk.Button(btn_frame, text="Compute", width=12,
              command=_on_compute, pady=4).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel", width=10,
              command=dialog.destroy, pady=4).pack(side="left", padx=6)

    dialog.wait_window()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fcs_pch.py <file.fcs> [bin_width_s] [ch1|ch2|both|combined]")
        print("  bin_width_s default: 100e-6 (100 µs)")
        sys.exit(1)

    d       = read_fcs(sys.argv[1])
    bw      = float(sys.argv[2]) if len(sys.argv) > 2 else 100e-6
    choice  = sys.argv[3] if len(sys.argv) > 3 else "both"

    # Use the shared builder rather than a second hand-rolled copy of it, so
    # the CLI cannot drift from the GUI path the way the old duplicate did.
    channels_data = _build_channels_data(d, choice, bw)

    plot_pch(channels_data, bw, d)
