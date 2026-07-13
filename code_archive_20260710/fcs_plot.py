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
        fcs_export.safe_export(
            d, "intensity",
            {
                "time_s":     time_s,
                "ch1_counts": I1_counts,
                "ch2_counts": I2_counts,
                "ch1_cps":    I1_cps,
                "ch2_cps":    I2_cps,
            },
            meta={
                "bin_width_s": f"{bin_width_s:.6g}",
                "duration_s":  f"{d.duration_s:.6g}",
                "ch1_photons": len(d.ch1_deltas),
                "ch2_photons": len(d.ch2_deltas),
            },
        )

    # ── Mean count rates for the annotation ───────────────────────────────────
    # Use the true mean count rate from the FCSData object (computed from the
    # total photon count and measurement duration) rather than the mean of the
    # binned trace, which can be slightly affected by edge effects.
    mean_cps_ch1 = d.count_rate_ch1_hz
    mean_cps_ch2 = d.count_rate_ch2_hz

    # ── Build the plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot both channels
    ax.plot(time_s, I1_cps, color="steelblue", linewidth=0.8,
            label="Ch1", alpha=0.9)
    ax.plot(time_s, I2_cps, color="tomato", linewidth=0.8,
            label="Ch2", alpha=0.9)

    # ── Axes labels and title ─────────────────────────────────────────────────
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Intensity (CPS)", fontsize=12)

    # Title: file name and bin width
    title = f"Intensity trace — {d.filepath.name}"
    subtitle = f"Bin width: {_format_bin_width(bin_width_s)}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)

    # ── Y-axis limits ─────────────────────────────────────────────────────────
    # Lower limit is always zero.  Upper limit is 1.25× the 99th percentile
    # of the combined channels, so bright transient bursts don't compress the
    # rest of the trace toward the top of the plot.
    combined_cps = np.concatenate([I1_cps, I2_cps])
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
    left_mean  = np.mean(np.concatenate([I1_cps[:n//2], I2_cps[:n//2]]))
    right_mean = np.mean(np.concatenate([I1_cps[n//2:], I2_cps[n//2:]]))
    upper_band = combined_cps[combined_cps > 0.6 * y_upper]
    use_top    = len(upper_band) < 0.15 * len(combined_cps)
    use_left   = left_mean <= right_mean

    x_loc = "left"  if use_left else "right"
    y_loc = "upper" if use_top  else "lower"
    loc   = f"{y_loc} {x_loc}"

    # Build legend handles: two real channel lines + spacer + two text rows
    ch1_handle     = plt.Line2D([0], [0], color="steelblue", linewidth=1.5, label="Ch1")
    ch2_handle     = plt.Line2D([0], [0], color="tomato",    linewidth=1.5, label="Ch2")
    spacer         = Patch(alpha=0, label="─────────────")
    mean_ch1_label = Patch(alpha=0, label=f"Mean Ch1: {mean_cps_ch1:,.0f} CPS")
    mean_ch2_label = Patch(alpha=0, label=f"Mean Ch2: {mean_cps_ch2:,.0f} CPS")

    ax.legend(
        handles=[ch1_handle, ch2_handle, spacer, mean_ch1_label, mean_ch2_label],
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

    fig.tight_layout()

    if show:
        #plt.show was the static plot; now dynamic w/ fcs_plottools
        #plt.show()
        fcs_plottools.show_figure(fig, ax)

    return fig, ax


# ── Overlay plotting (batch / combined) ───────────────────────────────────────

def plot_intensity_overlay(
    datasets,
    bin_width_s: float = None,
    show: bool = True,
    export: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Overlay intensity time traces from several files on one set of axes.

    Each file is drawn in its own colour; Ch1 is solid and Ch2 dashed.  If
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

        # Legend row: "file   <Ch1 CPS>, <Ch2 CPS>".  Use the true mean count
        # rates (total photons / duration) for consistency with the single-file
        # plot's legend, rather than the mean of the binned trace.
        cps_label = (f"{d.filepath.name}   "
                     f"{d.count_rate_ch1_hz:,.0f}, {d.count_rate_ch2_hz:,.0f}")
        ax.plot(time_s, I1_cps, color=colour, linewidth=0.8, alpha=0.9,
                label=cps_label)                  # Ch1 — solid, carries file label
        ax.plot(time_s, I2_cps, color=colour, linewidth=0.8, alpha=0.55,
                linestyle="--")                    # Ch2 — dashed, no extra legend row

        if len(time_s):
            t_max = max(t_max, float(time_s[-1]))
            combined = np.concatenate([I1_cps, I2_cps])
            if len(combined):
                y_percentiles.append(float(np.percentile(combined, 99)))

        if export:
            fcs_export.safe_export(
                d, "intensity",
                {
                    "time_s":     time_s,
                    "ch1_counts": I1_counts,
                    "ch2_counts": I2_counts,
                    "ch1_cps":    I1_cps,
                    "ch2_cps":    I2_cps,
                },
                meta={
                    "bin_width_s": f"{bw:.6g}",
                    "duration_s":  f"{d.duration_s:.6g}",
                    "ch1_photons": len(d.ch1_deltas),
                    "ch2_photons": len(d.ch2_deltas),
                },
            )

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
