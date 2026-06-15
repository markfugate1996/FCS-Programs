"""
fcs_lifetime.py
===============
Photon arrival histogram (TCSPC decay) plotting for FCS data.

Plots the microtime histogram for one or both channels on a log-y scale,
which is the standard way to inspect TCSPC decays before lifetime fitting.

Usage
-----
    # Quick look from a file path
    plot_lifetime("experiment.fcs")

    # From an already-loaded FCSData object
    from fcs_reader import read_fcs
    d = read_fcs("experiment.fcs")
    plot_lifetime(d)

    # Coarsen to 256 bins and show only Ch1
    plot_lifetime(d, n_bins=256, channels=(1,))

    # Suppress the plot window (e.g. to save to disk)
    fig, ax = plot_lifetime(d, show=False)
    fig.savefig("decay.png", dpi=150)

    # Interactive gate selection (called from fcs_corr.py)
    gate_min_ns, gate_max_ns = select_gate(d)

Dependencies
------------
    pip install matplotlib

fcs_reader.py must be in the same folder (or on the Python path).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fcs_plottools
import numpy as np

from fcs_reader import FCSData, read_fcs
import fcs_export

# ── Type alias ────────────────────────────────────────────────────────────────

DataOrPath = Union[FCSData, str, Path]

# Colours consistent with fcs_plot.py
_CH_COLOUR = {1: "steelblue", 2: "tomato"}

# Sensible default bin counts (must divide 4096 evenly for clean binning)
_VALID_N_BINS = [64, 128, 256, 512, 1024, 2048, 4096]

# Gate line appearance
_GATE_COLOUR   = "#e6820a"   # warm orange — visible on both blue and red dots
_GATE_LW       = 1.5
_GATE_SHADE    = "#e6820a"
_GATE_SHADE_ALPHA = 0.10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(data: DataOrPath) -> FCSData:
    if isinstance(data, FCSData):
        return data
    return read_fcs(data)


def _format_n_bins(n_bins: int, n_raw: int = 4096) -> str:
    """Human-readable description of the binning."""
    factor = n_raw // n_bins
    if factor == 1:
        return f"{n_bins} bins (native resolution)"
    return f"{n_bins} bins ({factor}× rebinned)"


def _default_gate(d: FCSData, n_bins: int = 256) -> Tuple[float, float]:
    """
    Compute default gate boundaries from the lifetime histogram.

    Uses the combined Ch1+Ch2 histogram to find the peak, then sets:
      gate_min = time of peak + 1 ns   (just past the IRF-dominated rise)
      gate_max = gate_min + 20 ns      (covers the fluorescence decay)

    Both values are clamped to [0, laser_period_ns].

    Parameters
    ----------
    d      : FCSData
    n_bins : int — histogram resolution used for peak finding (coarser
             than display is fine; 256 bins is fast and smooth enough)

    Returns
    -------
    gate_min_ns, gate_max_ns : float
    """
    # Use combined counts from both channels to find the peak robustly
    bin_times, counts1 = d.lifetime_histogram(channel=1, n_bins=n_bins)
    _,          counts2 = d.lifetime_histogram(channel=2, n_bins=n_bins)
    combined    = counts1.astype(np.float64) + counts2.astype(np.float64)

    peak_idx    = int(np.argmax(combined))
    peak_ns     = float(bin_times[peak_idx])

    gate_min_ns = peak_ns + 1.0
    gate_max_ns = gate_min_ns + 20.0

    # Clamp to valid range
    period      = d.laser_period_ns
    gate_min_ns = float(np.clip(gate_min_ns, 0.0, period))
    gate_max_ns = float(np.clip(gate_max_ns, gate_min_ns + 0.1, period))

    return gate_min_ns, gate_max_ns


# ── Main plotting function ────────────────────────────────────────────────────

def plot_lifetime(
    data: DataOrPath,
    n_bins: int = 4096,
    channels: tuple[int, ...] = (1, 2),
    show: bool = True,
    export: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the TCSPC photon arrival histogram (lifetime decay) for one or both
    channels on a log-y scale.

    Parameters
    ----------
    data : FCSData, str, or Path
    n_bins : int
        Histogram bins (must be in _VALID_N_BINS).
    channels : tuple of int
        Channels to plot: (1,), (2,), or (1, 2).
    show : bool
        Call plt.show() if True.

    Returns
    -------
    fig, ax
    """
    if n_bins not in _VALID_N_BINS:
        raise ValueError(f"n_bins must be one of {_VALID_N_BINS}; got {n_bins}.")
    if not channels or not all(c in (1, 2) for c in channels):
        raise ValueError("channels must be a non-empty tuple containing 1, 2, or both.")

    d   = _load(data)
    fig, ax = plt.subplots(figsize=(9, 4))
    _draw_histogram(ax, d, n_bins, channels)
    fig.tight_layout()

    # ── Optional CSV export of the plotted data ───────────────────────────────
    # The time axis is identical across channels (same n_bins / range), so a
    # single time column is shared and each plotted channel adds a counts column.
    if export:
        time_ns = None
        columns: dict[str, np.ndarray] = {}
        for ch in channels:
            bin_times_ns, counts = d.lifetime_histogram(channel=ch, n_bins=n_bins)
            if time_ns is None:
                time_ns = bin_times_ns
            columns[f"ch{ch}_counts"] = counts
        if time_ns is not None:
            fcs_export.safe_export(
                d, "lifetime",
                {"time_ns": time_ns, **columns},
                meta={
                    "n_bins":         n_bins,
                    "bin_width_ps":   f"{d.laser_period_ns / n_bins * 1000:.4f}",
                    "laser_period_ns": f"{d.laser_period_ns:.4f}",
                    "channels":       "+".join(f"Ch{c}" for c in channels),
                },
                suffix=f"{n_bins}bins",
            )

    if show:
        #plt.show was static; now dynamic w/ fcs_plottools
        #plt.show()
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── Interactive gate selection ────────────────────────────────────────────────

def select_gate(
    data: DataOrPath,
    n_bins: int = 512,
    channels: tuple[int, ...] = (1, 2),
    initial_gate: Optional[Tuple[float, float]] = None,
    title: str = "Set time gate",
    gate_label: str = "Gate",
    confirm_text: str = "Confirm gate and compute",
) -> Optional[Tuple[float, float]]:
    """
    Show the TCSPC histogram with two draggable gate lines and wait for the
    user to confirm or cancel.

    The gate lines can be moved by clicking and dragging.  Exact values can
    also be typed into the entry boxes below the plot.  A shaded region
    between the lines shows the accepted window.

    Parameters
    ----------
    data : FCSData, str, or Path
    n_bins : int
        Histogram resolution for the gate view.  512 is a good balance of
        speed and detail.
    channels : tuple of int
        Which channels to draw.
    initial_gate : (gate_min_ns, gate_max_ns) or None
        Starting gate positions.  If None, defaults are computed from the
        histogram peak (peak + 1 ns, peak + 21 ns).

    Returns
    -------
    (gate_min_ns, gate_max_ns) if the user confirmed, or None if cancelled.
    """
    import tkinter as tk
    from tkinter import messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    d = _load(data)

    if initial_gate is None:
        gate_min_ns, gate_max_ns = _default_gate(d)
    else:
        gate_min_ns, gate_max_ns = float(initial_gate[0]), float(initial_gate[1])

    period = d.laser_period_ns

    # ── Result container ──────────────────────────────────────────────────────
    # Using a list so the nested callbacks can write to it
    result: list[Optional[Tuple[float, float]]] = [None]

    # ── Tkinter window ────────────────────────────────────────────────────────
    win = tk.Toplevel()
    win.title(title)
    win.resizable(True, True)

    # ── Matplotlib figure embedded in Tk ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    _draw_histogram(ax, d, n_bins, channels)

    # Shade the accepted region and draw the two gate lines
    shade = ax.axvspan(gate_min_ns, gate_max_ns,
                       color=_GATE_SHADE, alpha=_GATE_SHADE_ALPHA, zorder=1)
    line_lo = ax.axvline(gate_min_ns, color=_GATE_COLOUR,
                         lw=_GATE_LW, ls="--", zorder=5, label=gate_label)
    line_hi = ax.axvline(gate_max_ns, color=_GATE_COLOUR,
                         lw=_GATE_LW, ls="--", zorder=5)

    # Gate label on the upper x-axis (text annotation that moves with lines)
    gate_text = ax.text(
        0.5, 1.01, "",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=9, color=_GATE_COLOUR,
    )

    def _update_gate_text():
        lo = float(line_lo.get_xdata()[0])
        hi = float(line_hi.get_xdata()[0])
        gate_text.set_text(f"{gate_label}: {lo:.2f} – {hi:.2f} ns")

    _update_gate_text()
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

    toolbar = NavigationToolbar2Tk(canvas, win)
    toolbar.update()

    # ── Drag logic ────────────────────────────────────────────────────────────
    # Track which line (if any) is being dragged.
    _drag_state: dict = {"line": None, "active": False}
    _PICK_RADIUS_NS = period * 0.015   # click within ~1.5% of period to grab

    def _on_press(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        lo = float(line_lo.get_xdata()[0])
        hi = float(line_hi.get_xdata()[0])
        dist_lo = abs(event.xdata - lo)
        dist_hi = abs(event.xdata - hi)
        if min(dist_lo, dist_hi) > _PICK_RADIUS_NS:
            return
        _drag_state["line"] = line_lo if dist_lo <= dist_hi else line_hi
        _drag_state["active"] = True

    def _on_release(event):
        if _drag_state["active"]:
            _drag_state["active"] = False
            _drag_state["line"]   = None
            _sync_entries_from_lines()

    def _on_motion(event):
        if not _drag_state["active"] or event.inaxes is not ax:
            return
        if event.xdata is None:
            return
        x = float(np.clip(event.xdata, 0.0, period))
        lo = float(line_lo.get_xdata()[0])
        hi = float(line_hi.get_xdata()[0])

        dragging = _drag_state["line"]
        if dragging is line_lo:
            # Don't let lower gate cross upper
            x = min(x, hi - 0.05)
            line_lo.set_xdata([x, x])
        else:
            x = max(x, lo + 0.05)
            line_hi.set_xdata([x, x])

        _update_shade()
        _update_gate_text()
        canvas.draw_idle()

    def _update_shade():
        lo = float(line_lo.get_xdata()[0])
        hi = float(line_hi.get_xdata()[0])
        # Remove the old span and redraw — avoids Polygon.set_xy() whose
        # calling convention changed across matplotlib versions.
        nonlocal shade
        shade.remove()
        shade = ax.axvspan(lo, hi,
                           color=_GATE_SHADE, alpha=_GATE_SHADE_ALPHA,
                           zorder=1)

    canvas.mpl_connect("button_press_event",   _on_press)
    canvas.mpl_connect("button_release_event", _on_release)
    canvas.mpl_connect("motion_notify_event",  _on_motion)

    # ── Entry boxes ───────────────────────────────────────────────────────────
    ctrl_frame = tk.Frame(win)
    ctrl_frame.pack(fill="x", padx=12, pady=6)

    tk.Label(ctrl_frame, text="Gate min (ns):", anchor="e",
             width=14).pack(side="left")
    lo_var = tk.StringVar(value=f"{gate_min_ns:.2f}")
    lo_entry = tk.Entry(ctrl_frame, textvariable=lo_var, width=8)
    lo_entry.pack(side="left", padx=(2, 16))

    tk.Label(ctrl_frame, text="Gate max (ns):", anchor="e",
             width=14).pack(side="left")
    hi_var = tk.StringVar(value=f"{gate_max_ns:.2f}")
    hi_entry = tk.Entry(ctrl_frame, textvariable=hi_var, width=8)
    hi_entry.pack(side="left", padx=2)

    def _sync_lines_from_entries(*_):
        """Parse entry boxes and move lines to match."""
        try:
            lo = float(lo_var.get())
            hi = float(hi_var.get())
        except ValueError:
            return
        lo = float(np.clip(lo, 0.0, period))
        hi = float(np.clip(hi, 0.0, period))
        if lo >= hi:
            return
        line_lo.set_xdata([lo, lo])
        line_hi.set_xdata([hi, hi])
        _update_shade()
        _update_gate_text()
        canvas.draw_idle()

    def _sync_entries_from_lines():
        """Update entry boxes to match current line positions."""
        lo = float(line_lo.get_xdata()[0])
        hi = float(line_hi.get_xdata()[0])
        lo_var.set(f"{lo:.2f}")
        hi_var.set(f"{hi:.2f}")

    lo_entry.bind("<Return>", _sync_lines_from_entries)
    lo_entry.bind("<FocusOut>", _sync_lines_from_entries)
    hi_entry.bind("<Return>", _sync_lines_from_entries)
    hi_entry.bind("<FocusOut>", _sync_lines_from_entries)

    # ── Confirm / Cancel buttons ──────────────────────────────────────────────
    btn_frame = tk.Frame(win)
    btn_frame.pack(pady=(0, 10))

    def _confirm():
        lo = float(line_lo.get_xdata()[0])
        hi = float(line_hi.get_xdata()[0])
        if lo >= hi:
            messagebox.showerror("Invalid gate",
                                 "Gate min must be less than gate max.",
                                 parent=win)
            return
        result[0] = (lo, hi)
        plt.close(fig)
        win.destroy()

    def _cancel():
        result[0] = None
        plt.close(fig)
        win.destroy()

    tk.Button(btn_frame, text=confirm_text,
              command=_confirm, width=26, pady=4).pack(side="left", padx=6)
    tk.Button(btn_frame, text="Cancel",
              command=_cancel, width=10, pady=4).pack(side="left", padx=6)

    win.protocol("WM_DELETE_WINDOW", _cancel)
    win.grab_set()
    win.wait_window()

    return result[0]


# ── Shared histogram drawing ──────────────────────────────────────────────────

def _draw_histogram(
    ax: plt.Axes,
    d: FCSData,
    n_bins: int,
    channels: tuple[int, ...],
) -> list[int]:
    """
    Draw the TCSPC decay histogram onto ax.  Returns list of peak counts.
    Used by both plot_lifetime() and select_gate() to avoid duplication.
    """
    peak_counts: list[int] = []

    for ch in channels:
        bin_times_ns, counts = d.lifetime_histogram(channel=ch, n_bins=n_bins)
        if counts.sum() == 0:
            continue
        total = int(counts.sum())
        peak  = int(counts.max())
        peak_counts.append(peak)
        ax.plot(
            bin_times_ns, counts,
            linestyle="none", marker=".", markersize=2.5,
            color=_CH_COLOUR[ch], alpha=0.9,
            label=f"Ch{ch}  ({total:,} photons)",
        )

    ax.set_yscale("log")
    if peak_counts:
        all_counts = np.concatenate([
            d.lifetime_histogram(channel=ch, n_bins=n_bins)[1]
            for ch in channels
        ])
        nonzero = all_counts[all_counts > 0]
        y_min = nonzero.min() * 0.5 if len(nonzero) else 0.5
        ax.set_ylim(y_min, max(peak_counts) * 1.5)

    ax.set_xlim(0, d.laser_period_ns)
    ax.set_xlabel("Arrival time within laser cycle (ns)", fontsize=12)
    ax.set_ylabel("Photon counts", fontsize=12)

    title    = f"TCSPC decay — {d.filepath.name}"
    subtitle = (
        f"{_format_n_bins(n_bins)}  ·  "
        f"bin width: {d.laser_period_ns / n_bins * 1000:.1f} ps  ·  "
        f"laser period: {d.laser_period_ns:.4f} ns"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.85)

    return peak_counts


# ── CLI: run directly to plot a file ─────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fcs_lifetime.py <file.fcs> [n_bins]")
        print(f"  n_bins choices: {_VALID_N_BINS}  (default: 4096)")
        sys.exit(1)

    path   = sys.argv[1]
    n_bins = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    plot_lifetime(path, n_bins=n_bins)
