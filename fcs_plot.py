"""
fcs_plot.py
===========
Plotting functions for FCS (Fluorescence Correlation Spectroscopy) data.

This module provides visualisation tools that work directly with the FCSData
objects returned by fcs_reader.py.  Each function can accept either a file
path or an already-loaded FCSData object, so you can use them in two ways:

    # Quick look — load and plot in one step
    plot_intensity("experiment.fcs")

    # If you have already loaded the data
    from fcs_reader import read_fcs
    d = read_fcs("experiment.fcs")
    plot_intensity(d)

Dependencies
------------
    pip install matplotlib

fcs_reader.py must be in the same folder (or on the Python path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fcs_plottools
import numpy as np

# Import the reader from the same directory
from fcs_reader import FCSData, read_fcs
import fcs_export


# ── Type alias ───────────────────────────────────────────────────────────────

# Functions accept either a file path or an already-loaded FCSData object.
DataOrPath = Union[FCSData, str, Path]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(data: DataOrPath) -> FCSData:
    """
    Accept either an FCSData object or a file path and return an FCSData.
    This lets every plotting function work with either input type.
    """
    if isinstance(data, FCSData):
        return data
    return read_fcs(data)


def _recorded_channels(d) -> tuple:
    """
    Which detector channels *d* actually RECORDED: (1,), (2,) or (1, 2).

    Reads the reader's ``channels`` attribute, defaulting to both for any
    object that predates it -- the same contract fcs_lifetime._resolve_channels
    and fcs_corr.supported_corr_types work from.

    This module needs it because ``bin_intensity`` cannot express the
    difference on its own: an absent channel has an empty photon array, and
    binning an empty array gives a trace of zeros -- which is also exactly what
    a dual-channel acquisition with a dead detector gives.  FCSData.channels is
    deliberate about keeping those two apart (whether a silent detector means a
    broken PMT or an empty sample is the user's judgement, not the parser's),
    and the intensity plot should not merge them back together by drawing a
    flat line at zero and calling it Ch2.
    """
    return tuple(getattr(d, "channels", (1, 2)))


def _choose_bin_width(duration_s: float) -> float:
    """
    Choose a sensible default bin width based on the measurement duration.

    The goal is a trace with enough time resolution to see meaningful
    fluctuations, but not so fine-grained that the plot becomes noisy and
    unreadable.  FCS measurements typically last from a few seconds to
    several minutes; we aim for around 500–2000 bins across the trace.

    Duration          Bin width
    --------          ---------
    < 10 s            1 ms
    10 s – 2 min      10 ms
    2 min – 20 min    100 ms
    > 20 min          1 s
    """
    if duration_s < 10:
        return 1e-3          # 1 ms
    elif duration_s < 120:
        return 10e-3         # 10 ms
    elif duration_s < 1200:
        return 100e-3        # 100 ms
    else:
        return 1.0           # 1 s


def _format_bin_width(bin_width_s: float) -> str:
    """Return a human-readable bin width string, e.g. '10 ms'."""
    if bin_width_s < 1e-3:
        return f"{bin_width_s * 1e6:.3g} µs"
    elif bin_width_s < 1:
        return f"{bin_width_s * 1e3:.3g} ms"
    else:
        return f"{bin_width_s:.3g} s"


# ── User-selectable bin widths (offered in the intensity dialogs) ─────────────

_BIN_WIDTH_OPTIONS = {
    "1 ms":    1e-3,
    "5 ms":    5e-3,
    "10 ms":   10e-3,
    "50 ms":   50e-3,
    "100 ms":  100e-3,
    "250 ms":  250e-3,
    "500 ms":  500e-3,
    "1 s":     1.0,
}

_DEFAULT_BIN_WIDTH_LABEL = "100 ms"


# Unit suffixes accepted when a bin width is TYPED rather than picked from the
# preset list.  Both micro signs are listed: U+00B5 MICRO SIGN is what Windows
# and most keyboards emit, U+03BC GREEK SMALL LETTER MU is what many fonts and
# copy-paste sources emit.  They are visually identical, so a user cannot tell
# which one they have typed and must not be punished for guessing wrong.
_BIN_WIDTH_UNITS = {
    "s":    1.0,
    "sec":  1.0,
    "ms":   1e-3,
    "us":   1e-6,
    "µs":   1e-6,
    "ns":   1e-9,
}


def parse_bin_width(text) -> float:
    """
    Parse a typed bin width into seconds.

    Accepts a number with an optional unit -- ``2.5 ms``, ``250 us``,
    ``250 µs``, ``1e-4 s`` -- or a bare number, which is read as SECONDS to
    match the units used by every ``bin_width_s`` argument in the suite.  Case
    and the space before the unit are both optional.

    The preset labels in :data:`_BIN_WIDTH_OPTIONS` are themselves valid input
    here ("100 ms", "1 s"), which is what lets the dialog keep one variable for
    both the menu and the entry instead of trying to keep two in step.

    Raises
    ------
    ValueError
        With a message written for the user, not the developer: it is shown
        verbatim in the dialog when parsing fails.
    """
    s = str(text).strip().replace("μ", "µ")
    if not s:
        raise ValueError("Enter a bin width, e.g. '2.5 ms' or '250 µs'.")

    # Peel off the longest matching unit suffix, longest first so that 'ms' is
    # never mistaken for 's'.  Peeling a known suffix rather than stripping
    # trailing letters keeps scientific notation intact: '1e-4s' must lose the
    # 's' and keep the 'e', which a naive rstrip of alphabetic characters
    # would not do.
    scale = 1.0
    low = s.lower()
    for suffix in sorted(_BIN_WIDTH_UNITS, key=len, reverse=True):
        if low.endswith(suffix) and len(low) > len(suffix):
            scale = _BIN_WIDTH_UNITS[suffix]
            s = s[: len(s) - len(suffix)]
            break

    try:
        value = float(s.strip())
    except ValueError:
        raise ValueError(
            f"Could not read {text!r} as a bin width.  Use a number with an "
            f"optional unit -- '2.5 ms', '250 us', '1e-4 s' -- or a bare "
            f"number, which is read as seconds."
        ) from None

    # float() accepts 'nan' and 'inf'; both would sail through a bare > 0 test
    # and only fail much later inside bin_intensity, where the error would no
    # longer point at the entry box that caused it.
    if not np.isfinite(value):
        raise ValueError("Bin width must be a finite number.")
    if value <= 0:
        raise ValueError(f"Bin width must be greater than zero (got {value:g}).")

    return value * scale


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw_intensity(ax, time_s, I1_cps, I2_cps,
                    mean_cps_ch1: float, mean_cps_ch2: float,
                    source_name: str, bin_width_s: float,
                    channels: tuple = (1, 2)) -> None:
    """
    Draw one intensity trace onto *ax*, given the already-binned arrays.

    Split out of :func:`plot_intensity` so that :func:`rebuild_plot` can reuse
    it.  Reopening an export must produce the same figure as the original run,
    and the only way to be sure of that is for both to go through one drawing
    routine -- a second copy would drift, and the drift would be invisible
    until someone compared a reopened figure with the PNG beside it.

    Everything here is presentation: the axis limits, the legend corner chosen
    to avoid the data, the mean-CPS rows folded into the legend box.  Nothing
    is computed from photon records, which is why a CSV is enough to redraw it.

    *channels* is the tuple of channels the acquisition RECORDED.  Only those
    are drawn, and only those contribute to the axis limits, the legend-corner
    choice and the mean-CPS rows.  It defaults to both, so a caller with no
    channel information behaves exactly as this function always has.
    """
    # One record per channel actually recorded, built once and reused for the
    # traces, the limits, the legend corner and the legend rows -- so a channel
    # can never be drawn but left out of the scaling, or scaled but not drawn.
    series = [
        (ch, trace, colour, mean)
        for ch, trace, colour, mean in (
            (1, I1_cps, "steelblue", mean_cps_ch1),
            (2, I2_cps, "tomato",    mean_cps_ch2),
        )
        if ch in channels
    ]
    if not series:
        raise ValueError(
            f"No recorded channels to draw (channels={tuple(channels)!r}); "
            f"there is nothing to plot for {source_name}.")

    for ch, trace, colour, _mean in series:
        ax.plot(time_s, trace, color=colour, linewidth=0.8,
                label=f"Ch{ch}", alpha=0.9)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Intensity (CPS)", fontsize=12)

    title = f"Intensity trace — {source_name}"
    subtitle = f"Bin width: {_format_bin_width(bin_width_s)}"
    if len(series) < 2:
        # Say it on the figure, not just by the absence of a line: a
        # single-channel trace and a dual-channel one with a dead detector look
        # identical once plotted, and the figure is what gets saved.
        subtitle += f"  ·  Ch{series[0][0]} only (single-channel acquisition)"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)

    # ── Y-axis limits ─────────────────────────────────────────────────────────
    # Lower limit is always zero.  Upper limit is 1.25× the 99th percentile
    # of the combined channels, so bright transient bursts don't compress the
    # rest of the trace toward the top of the plot.
    combined_cps = np.concatenate([t for _ch, t, _c, _m in series])
    y_upper = np.percentile(combined_cps, 99) * 1.25
    y_upper = max(y_upper, 1.0)   # guard against near-zero data
    ax.set_ylim(0, y_upper)

    # ── Combined legend + mean CPS annotation ────────────────────────────────
    # We fold the mean CPS values directly into the legend as text-only rows,
    # keeping everything in one box instead of two separate elements.
    #
    # matplotlib doesn't support plain text rows natively, so we use invisible
    # Patch handles (alpha=0) as spacers, with the label carrying the text.
    from matplotlib.patches import Patch

    # Determine the best corner: pick the quadrant with the lowest mean
    # intensity so the legend is least likely to overlap data.
    n = len(time_s)
    left_mean  = np.mean(np.concatenate([t[:n//2] for _ch, t, _c, _m in series]))
    right_mean = np.mean(np.concatenate([t[n//2:] for _ch, t, _c, _m in series]))
    upper_band = combined_cps[combined_cps > 0.6 * y_upper]
    use_top    = len(upper_band) < 0.15 * len(combined_cps)
    use_left   = left_mean <= right_mean

    x_loc = "left"  if use_left else "right"
    y_loc = "upper" if use_top  else "lower"
    loc   = f"{y_loc} {x_loc}"

    # Build legend handles: one real line per recorded channel + spacer + one
    # text row per recorded channel.  A "Mean Ch2: 0 CPS" row for a channel
    # that was never recorded would state a measurement that was not made.
    handles = [plt.Line2D([0], [0], color=colour, linewidth=1.5,
                          label=f"Ch{ch}")
               for ch, _t, colour, _m in series]
    handles.append(Patch(alpha=0, label="─────────────"))
    handles += [Patch(alpha=0, label=f"Mean Ch{ch}: {mean:,.0f} CPS")
                for ch, _t, _c, mean in series]

    ax.legend(
        handles=handles,
        loc=loc,
        fontsize=10,
        framealpha=0.8,
        handlelength=1.5,
        handletextpad=0.5,
    )

    ax.set_xlim(time_s[0], time_s[-1])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:,.0f}"    # comma-separated thousands in y axis
    ))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)


# ── Export ────────────────────────────────────────────────────────────────────

def _intensity_export(d, time_s, I1_counts, I2_counts,
                      I1_cps, I2_cps, bin_width_s, channels) -> None:
    """
    Write one file's binned intensity trace to a CSV.

    The schema is FIXED -- time_s, ch1_counts, ch2_counts, ch1_cps, ch2_cps,
    always, in that order, whatever the file recorded -- so every intensity
    export has the same header and can be read without inspecting it first.
    This is the contract fcs_lifetime._lifetime_export_columns follows for the
    decay exports.

    A channel the acquisition never recorded gets a column of NaN rather than
    the zeros bin_intensity returns for an empty photon list.  NaN is what
    fcs_export.csv_value writes as a blank cell and what read_export maps a
    blank cell back to, so it round-trips as "not measured"; a 0 would read
    back as a genuine measurement of no light, which is what a dead detector
    looks like and is a different claim.  Its ``chN_photons`` header field is
    written blank for the same reason -- the convention fcs_corr._cps_meta
    already follows.

    ``channels_recorded`` names the channels the file actually had, so a reader
    (rebuild_plot included) can tell a blank column that was never measured
    from one that is blank for some other reason.

    The single-file and overlay paths both call this.  They previously carried
    a copy of this block each, which is the arrangement that lets two writers
    of one format drift apart.
    """
    blank = np.full(len(time_s), np.nan, dtype=np.float64)

    def _col(ch, arr):
        if ch not in channels:
            return blank.copy()
        return np.asarray(arr, dtype=np.float64)

    def _photons(ch, deltas):
        return len(deltas) if ch in channels else ""

    fcs_export.safe_export(
        d, "intensity",
        {
            "time_s":     time_s,
            "ch1_counts": _col(1, I1_counts),
            "ch2_counts": _col(2, I2_counts),
            "ch1_cps":    _col(1, I1_cps),
            "ch2_cps":    _col(2, I2_cps),
        },
        meta={
            "bin_width_s": f"{bin_width_s:.6g}",
            "duration_s":  f"{d.duration_s:.6g}",
            "ch1_photons": _photons(1, d.ch1_deltas),
            "ch2_photons": _photons(2, d.ch2_deltas),
            "channels_recorded": "+".join(f"Ch{c}" for c in channels),
        },
    )


# ── Main plotting function ────────────────────────────────────────────────────

def plot_intensity(
    data: DataOrPath,
    bin_width_s: float = None,
    show: bool = True,
    export: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the photon intensity time trace for both channels.

    Ch1 and Ch2 are shown on the same axes.  The mean count rate (in counts
    per second, CPS) for each channel is displayed as an annotation on the
    plot.

    Only the channels the acquisition recorded are drawn.  A single-channel
    file plots its one trace and says so in the subtitle, rather than adding a
    flat line at zero for a detector that was never there.

    Parameters
    ----------
    data : FCSData, str, or Path
        An already-loaded FCSData object, or a path to a .fcs file.
    bin_width_s : float, optional
        Width of each time bin in seconds.  If not given, a sensible default
        is chosen automatically based on the measurement duration (see
        _choose_bin_width for the logic).
    show : bool
        If True (default), call plt.show() to open the plot window.
        Set to False if you want to further modify the figure before showing,
        or if you are embedding the figure elsewhere.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    """

    # ── Load data ─────────────────────────────────────────────────────────────
    d = _load(data)

    # Which channels this acquisition recorded.  Drives both the plot and the
    # export, so the figure and the CSV cannot disagree about what was measured.
    channels = _recorded_channels(d)

    # ── Choose bin width ──────────────────────────────────────────────────────
    if bin_width_s is None:
        bin_width_s = _choose_bin_width(d.duration_s)

    # ── Bin photon arrivals into intensity traces ──────────────────────────────
    # bin_intensity() returns counts per bin; divide by bin width to get CPS.
    time_s, I1_counts, I2_counts = d.bin_intensity(bin_width_s=bin_width_s)

    I1_cps = I1_counts / bin_width_s   # counts per second, Ch1
    I2_cps = I2_counts / bin_width_s   # counts per second, Ch2

    # ── Optional CSV export of the plotted data ───────────────────────────────
    if export:
        _intensity_export(d, time_s, I1_counts, I2_counts,
                          I1_cps, I2_cps, bin_width_s, channels)

    # ── Mean count rates for the annotation ───────────────────────────────────
    # Use the true mean count rate from the FCSData object (computed from the
    # total photon count and measurement duration) rather than the mean of the
    # binned trace, which can be slightly affected by edge effects.
    mean_cps_ch1 = d.count_rate_ch1_hz
    mean_cps_ch2 = d.count_rate_ch2_hz

    # ── Build the plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    _draw_intensity(ax, time_s, I1_cps, I2_cps,
                    mean_cps_ch1, mean_cps_ch2,
                    d.filepath.name, bin_width_s, channels=channels)
    fig.tight_layout()

    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax

# ── Reopening an exported trace ───────────────────────────────────────────────

def rebuild_plot(meta, columns, show: bool = True, path=None):
    """
    Redraw an exported intensity trace as a live figure, for fcs_plotopen.

    Calls the same :func:`_draw_intensity` that drew the file in the first
    place, so a reopened figure is identical to a fresh one.

    Recovering the legend
    ---------------------
    The legend's mean count rates are ``count_rate_chN_hz`` on the dataset,
    which is defined as photons / duration -- and the export header records
    both ``chN_photons`` and ``duration_s``, so the same numbers come back
    exactly rather than being approximated from the binned trace.  The trace
    mean is only a fallback, for an export written before those keys existed:
    it differs slightly because the last bin is usually a partial one.

    Recovering the channels
    -----------------------
    ``channels_recorded`` says which channels the acquisition had.  Exports
    written before that key existed fall back to whichever columns hold a
    finite value -- see :func:`_channels_from_meta`.

    Overlay exports need no special handling.  ``plot_intensity_overlay``
    writes one CSV per file, in the same schema as the single-file path, so a
    row of an overlay reopens as the single trace it is.
    """
    if "time_s" not in columns:
        raise ValueError(
            "This file has no 'time_s' column, so it is not an intensity "
            "export.")

    time_s = np.asarray(columns["time_s"], dtype=np.float64)
    if len(time_s) < 2:
        raise ValueError("This intensity export has fewer than two time bins.")

    bin_width_s = _meta_float(meta, "bin_width_s")
    if bin_width_s is None or not np.isfinite(bin_width_s) or bin_width_s <= 0:
        # Bins are uniform and the column holds their left edges, so the
        # spacing IS the bin width -- recoverable exactly, no guessing.
        bin_width_s = float(np.median(np.diff(time_s)))

    def _cps(ch: str) -> np.ndarray:
        if f"{ch}_cps" in columns:
            return np.asarray(columns[f"{ch}_cps"], dtype=np.float64)
        if f"{ch}_counts" in columns:
            return (np.asarray(columns[f"{ch}_counts"], dtype=np.float64)
                    / bin_width_s)
        raise ValueError(
            f"This intensity export has neither '{ch}_cps' nor "
            f"'{ch}_counts', so the {ch.upper()} trace cannot be drawn.")

    # The channel decision is made on the RAW columns, before nan_to_num: an
    # all-blank column is how a never-recorded channel reads back, and zeroing
    # it first would erase exactly the evidence needed to recognise it.
    raw1, raw2 = _cps("ch1"), _cps("ch2")
    channels = _channels_from_meta(meta, {1: raw1, 2: raw2})
    # Within a RECORDED channel a NaN is a gap in the trace, not an absent
    # channel, so it is still zeroed for drawing, exactly as before.
    I1_cps, I2_cps = np.nan_to_num(raw1), np.nan_to_num(raw2)

    duration = _meta_float(meta, "duration_s")

    def _mean(ch: str, trace: np.ndarray) -> float:
        photons = _meta_float(meta, f"{ch}_photons")
        if (photons is not None and duration and np.isfinite(duration)
                and duration > 0):
            return photons / duration
        return float(np.mean(trace)) if len(trace) else float("nan")

    source_name = (meta.get("source file") or meta.get("source")
                   or (Path(path).name if path else "intensity export"))

    fig, ax = plt.subplots(figsize=(10, 4))
    _draw_intensity(ax, time_s, I1_cps, I2_cps,
                    _mean("ch1", I1_cps), _mean("ch2", I2_cps),
                    source_name, bin_width_s, channels=channels)
    fig.tight_layout()
    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


def _channels_from_meta(meta: dict, traces: dict) -> tuple:
    """
    Which channels an intensity export actually recorded.

    Prefers the ``channels_recorded`` header.  Exports written before that key
    existed lack it and fall back to whichever columns hold a finite value:
    such a file wrote real zeros for an absent channel, so it will report both
    -- which is what it claimed at the time, and reproducing a file's own claim
    is the right behaviour when its convention cannot be recovered.  A column
    that is entirely non-finite is dropped either way; it cannot be a
    measurement.
    """
    raw = str(meta.get("channels_recorded") or "").strip()
    if raw:
        named = tuple(c for c in (1, 2) if f"Ch{c}" in raw)
        if named:
            return named
    present = tuple(c for c in (1, 2) if np.isfinite(traces[c]).any())
    return present or (1, 2)


def _meta_float(meta: dict, key: str):
    """Read one numeric header field, or None if absent/unparseable."""
    v = meta.get(key)
    if v is None:
        return None
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


# ── Overlay plotting (batch / combined) ───────────────────────────────────────

def plot_intensity_overlay(
    datasets,
    bin_width_s: float = None,
    show: bool = True,
    export: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay intensity time traces from several files on one set of axes.

    Each file is drawn in its own colour; Ch1 is solid and Ch2 dashed.
    Channels are resolved PER FILE, so a mixed selection overlays a
    single-channel file's one trace alongside a dual-channel file's two -- the
    convention fcs_lifetime.plot_lifetime_overlay follows, and for the same
    reason: each file is drawn and exported independently, so there is nothing
    to reconcile across the selection.  If
    *bin_width_s* is None each file uses its own duration-based default,
    otherwise the same width is forced for all files.  When *export* is True
    each file's binned trace is written to its own CSV exactly as in the
    single-file path.

    Parameters
    ----------
    datasets : sequence of FCSData
        Files to overlay (must be non-empty).
    bin_width_s : float, optional
        Common bin width in seconds, or None for per-file auto widths.

    Returns
    -------
    fig, ax
    """
    datasets = list(datasets)
    if not datasets:
        raise ValueError("plot_intensity_overlay requires at least one dataset.")

    colours = fcs_plottools.palette(len(datasets))
    fig, ax = plt.subplots(figsize=(10, 4.5))

    y_percentiles: list[float] = []
    t_max = 0.0
    for d, colour in zip(datasets, colours):
        bw = bin_width_s if bin_width_s else _choose_bin_width(d.duration_s)
        time_s, I1_counts, I2_counts = d.bin_intensity(bin_width_s=bw)
        I1_cps = I1_counts / bw
        I2_cps = I2_counts / bw

        file_channels = _recorded_channels(d)

        # Legend row: "file   <Ch1 CPS>, <Ch2 CPS>".  Use the true mean count
        # rates (total photons / duration) for consistency with the single-file
        # plot's legend, rather than the mean of the binned trace.  A channel
        # the file never recorded gets an em dash: a "0" there would read as a
        # detector that ran and saw nothing.
        def _rate(ch, hz):
            return f"{hz:,.0f}" if ch in file_channels else "—"

        cps_label = (f"{d.filepath.name}   "
                     f"{_rate(1, d.count_rate_ch1_hz)}, "
                     f"{_rate(2, d.count_rate_ch2_hz)}")

        # Ch1 solid, Ch2 dashed, as before.  The file's label rides on the
        # FIRST line drawn for it, whichever channel that turns out to be -- a
        # Ch2-only file has no Ch1 line to carry it, and its legend row must
        # not go missing along with it.
        drawn: list = []
        for ch, trace, alpha, style in ((1, I1_cps, 0.9,  "-"),
                                        (2, I2_cps, 0.55, "--")):
            if ch not in file_channels:
                continue
            ax.plot(time_s, trace, color=colour, linewidth=0.8, alpha=alpha,
                    linestyle=style,
                    label=cps_label if not drawn else None)
            drawn.append(trace)

        if len(time_s) and drawn:
            t_max = max(t_max, float(time_s[-1]))
            combined = np.concatenate(drawn)
            if len(combined):
                y_percentiles.append(float(np.percentile(combined, 99)))

        if export:
            _intensity_export(d, time_s, I1_counts, I2_counts,
                              I1_cps, I2_cps, bw, file_channels)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Intensity (CPS)", fontsize=12)
    bw_str = (f"bin width: {_format_bin_width(bin_width_s)}"
              if bin_width_s else "bin width: auto (per file)")
    ax.set_title(
        f"Intensity overlay — {len(datasets)} files  ·  {bw_str}\n"
        f"solid = Ch1   ·   dashed = Ch2",
        fontsize=11,
    )
    if t_max > 0:
        ax.set_xlim(0, t_max)
    y_upper = max(max(y_percentiles) * 1.25, 1.0) if y_percentiles else 1.0
    ax.set_ylim(0, y_upper)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=9, framealpha=0.8, loc="upper right",
              title="File   ·   CPS: Ch1, Ch2")

    fig.tight_layout()
    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── CLI: run directly to plot a file ─────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fcs_plot.py <file.fcs> [bin_width_s]")
        sys.exit(1)

    path = sys.argv[1]
    bw   = float(sys.argv[2]) if len(sys.argv) > 2 else None

    plot_intensity(path, bin_width_s=bw)
