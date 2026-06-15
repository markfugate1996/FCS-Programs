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
    compute_pch(times_s, bin_width_s)              -> (k, pk, mean, var)
    plot_pch(channels_data, bin_width_s, ...)      -> (fig, ax)
    run_pch_dialog(fcs_data)                       -> shows dialog + plot
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

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

def compute_pch(
    times_s: np.ndarray,
    bin_width_s: float,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute the photon counting histogram from a photon arrival time array.

    The arrival times are binned into non-overlapping windows of width
    bin_width_s.  The histogram of counts-per-bin is the PCH.

    Parameters
    ----------
    times_s     : np.ndarray
        Sorted absolute photon arrival times in seconds.
    bin_width_s : float
        Bin width in seconds.

    Returns
    -------
    k       : np.ndarray (int)
        Photon count values (0, 1, 2, …, k_max).
    pk      : np.ndarray (float)
        Probability of each count value (normalised to sum = 1).
    mean    : float
        Mean photons per bin  <k>.
    var     : float
        Variance of photons per bin  Var(k).
    """
    if len(times_s) == 0:
        raise ValueError("times_s is empty; cannot compute PCH.")

    t_start = times_s[0]
    t_end   = times_s[-1]
    n_bins  = max(1, int((t_end - t_start) / bin_width_s))
    edges   = t_start + np.arange(n_bins + 1) * bin_width_s

    counts, _ = np.histogram(times_s, bins=edges)

    mean = float(counts.mean())
    var  = float(counts.var(ddof=1))

    k_max = int(counts.max())
    k     = np.arange(0, k_max + 1, dtype=int)
    pk    = np.bincount(counts, minlength=k_max + 1).astype(float)
    pk   /= pk.sum()   # normalise to probability

    return k, pk, mean, var


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pch(
    channels_data: Dict[str, Tuple[np.ndarray, np.ndarray, float, float, str]],
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
        Values are (k, pk, mean, var, colour) tuples from compute_pch.
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
    # Each series has its own contiguous k = 0..k_last.  We pad every series'
    # p(k) onto a common k axis (0..max) with zeros so they share one table.
    if export and channels_data:
        k_max  = max(int(np.asarray(k)[-1]) for (k, *_rest) in channels_data.values())
        k_full = np.arange(0, k_max + 1)
        cols: Dict[str, np.ndarray] = {"k": k_full}
        meta: Dict[str, object] = {"bin_width_s": f"{bin_width_s:.6g}"}
        for label, (k, pk, mean, var, _colour) in channels_data.items():
            pk_full = np.zeros(k_max + 1, dtype=float)
            pk_full[np.asarray(k, dtype=int)] = pk
            cols[f"pk_{label}"] = pk_full
            meta[f"{label}_mean"]          = f"{mean:.6g}"
            meta[f"{label}_var"]           = f"{var:.6g}"
            meta[f"{label}_var_over_mean"] = f"{var / mean:.6g}" if mean > 0 else "nan"
        ch_tag = "_".join(channels_data.keys())
        bw_tag = f"{bin_width_s * 1e6:.0f}us"
        fcs_export.safe_export(
            fcs_data, "pch", cols, meta=meta, suffix=f"{ch_tag}_{bw_tag}",
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    annotation_lines: list[str] = []

    for label, (k, pk, mean, var, colour) in channels_data.items():
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

        # Build the dict of (k, pk, mean, var, colour) per series
        channels_data: Dict[str, Tuple] = {}

        try:
            if channel_choice in ("ch1", "both"):
                t = fcs_data.ch1_times_s
                k, pk, mean, var = compute_pch(t, bin_width_s)
                channels_data["Ch1"] = (k, pk, mean, var, _CH_COLOUR[1])

            if channel_choice in ("ch2", "both"):
                t = fcs_data.ch2_times_s
                k, pk, mean, var = compute_pch(t, bin_width_s)
                channels_data["Ch2"] = (k, pk, mean, var, _CH_COLOUR[2])

            if channel_choice == "combined":
                # Merge both channels into a single arrival-time array
                t = np.sort(np.concatenate([
                    fcs_data.ch1_times_s,
                    fcs_data.ch2_times_s,
                ]))
                k, pk, mean, var = compute_pch(t, bin_width_s)
                channels_data["Ch1+Ch2"] = (k, pk, mean, var, _CH_COLOUR["both"])

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

    channels_data: Dict[str, Tuple] = {}
    if choice in ("ch1", "both"):
        k, pk, mean, var = compute_pch(d.ch1_times_s, bw)
        channels_data["Ch1"] = (k, pk, mean, var, _CH_COLOUR[1])
    if choice in ("ch2", "both"):
        k, pk, mean, var = compute_pch(d.ch2_times_s, bw)
        channels_data["Ch2"] = (k, pk, mean, var, _CH_COLOUR[2])
    if choice == "combined":
        t = np.sort(np.concatenate([d.ch1_times_s, d.ch2_times_s]))
        k, pk, mean, var = compute_pch(t, bw)
        channels_data["Ch1+Ch2"] = (k, pk, mean, var, _CH_COLOUR["both"])

    plot_pch(channels_data, bw, d)
