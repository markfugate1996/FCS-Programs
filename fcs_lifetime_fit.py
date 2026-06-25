"""
fcs_lifetime_fit.py
===================
Fit TCSPC lifetime decays (microtime histograms) to exponential tail models.

This is the lifetime counterpart of fcs_fit.py (which fits correlation curves).
It is launched from the "Model Data" task in the main window, via the Lifetime
button in fcs_fit.run_model_dialog.

Workflow (launched from the main window)
----------------------------------------
    1. Pick the channel, histogram resolution, and fit window
       (peak → end by default; the window can be set visually on the
       histogram, reusing fcs_lifetime.select_gate).      -> _lifetime_data_dialog
    2. Choose a model (single- or two-exponential).         -> _select_lifetime_model_dialog
    3. Set guesses / bounds / fixed flags, then fit.        -> _lifetime_setup_dialog
    4. Fit, plot data + fit + residuals, write a report and curve CSV
       to a 'fits' folder beside the source .fcs file.

Tail fit, not reconvolution
---------------------------
The model is a sum of exponentials plus a constant background.  The data prep
(prepare_decay) drops the first and last microtime bins — these are time-tagger
catch-all artifacts, not fluorescence — and restricts the fit to the chosen
window.  Counts are Poisson-distributed, so the fit is weighted by σ = √counts
and the reduced χ² is meaningful.

The numerical core (prepare_decay, auto_guess_lifetime, fit_lifetime) has no GUI
dependency and can be reused for batch fitting.

Models come from fcs_models.LIFETIME_MODELS.

Dependencies
------------
    pip install numpy scipy matplotlib
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import fcs_plottools
from scipy.optimize import curve_fit

import fcs_models
from fcs_models import FCSModel
from fcs_reader import FCSData
import fcs_lifetime


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_decay(
    t_ns: np.ndarray,
    counts: np.ndarray,
    fit_start_ns: float,
    fit_end_ns: float,
    drop_edges: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the fit window from a TCSPC histogram and return the data to fit.

    The first and last histogram bins are time-tagger catch-all artifacts
    (untimed / clamped records), so by default they are dropped before the
    window is applied.  The window is then [fit_start_ns, fit_end_ns]
    inclusive, in absolute microtime (ns).

    Returns
    -------
    t_ns_win : np.ndarray   absolute bin times within the window (ns)
    counts_win : np.ndarray  counts within the window
    t_rel : np.ndarray       t_ns_win re-zeroed to the window start (ns); this
                             is what the model is evaluated on, so the amplitude
                             refers to the window start.
    """
    t_ns   = np.asarray(t_ns, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if t_ns.shape != counts.shape:
        raise ValueError("t_ns and counts must have the same shape.")
    if fit_end_ns <= fit_start_ns:
        raise ValueError("fit_end_ns must be greater than fit_start_ns.")

    keep = np.ones(len(counts), dtype=bool)
    if drop_edges and len(counts) >= 2:
        keep[0] = False
        keep[-1] = False
    keep &= (t_ns >= fit_start_ns) & (t_ns <= fit_end_ns)

    t_win = t_ns[keep]
    c_win = counts[keep]
    if len(t_win) < 3:
        raise ValueError(
            "Fit window contains fewer than 3 bins after dropping edge bins. "
            "Widen the window or use more histogram bins."
        )
    t_rel = t_win - t_win[0]
    return t_win, c_win, t_rel


def default_window(d: FCSData, channel: int, n_bins: int) -> Tuple[float, float]:
    """
    Default fit window: from the decay peak to the last usable bin.

    The peak is found on the edge-excluded histogram, so the bin-0 spike never
    wins.  The window end is the last bin's time (the final-bin artifact is
    dropped separately by prepare_decay).
    """
    t_ns, counts = d.lifetime_histogram(channel=channel, n_bins=n_bins)
    core = counts.astype(np.float64).copy()
    if len(core) >= 2:
        core[0] = 0.0
        core[-1] = 0.0
    peak_ns = float(t_ns[int(np.argmax(core))])
    end_ns  = float(t_ns[-2]) if len(t_ns) >= 2 else float(t_ns[-1])
    if end_ns <= peak_ns:
        end_ns = float(t_ns[-1])
    return peak_ns, end_ns


# ── Initial-guess heuristics ──────────────────────────────────────────────────

def auto_guess_lifetime(
    model: FCSModel,
    t_rel: np.ndarray,
    counts: np.ndarray,
) -> Dict[str, float]:
    """
    Sensible starting guesses from the windowed decay.

    Background from the median of the last 10% of the window; amplitude from the
    first bin above background; lifetime from the 1/e crossing.  For the
    two-exponential model the amplitude is split fast/slow and the single-decay
    lifetime estimate is spread around to seed the two components.
    """
    g = model.defaults()
    c = np.asarray(counts, dtype=np.float64)
    t = np.asarray(t_rel, dtype=np.float64)
    if len(c) < 3:
        return g

    n_tail = max(3, len(c) // 10)
    offset = float(np.median(c[-n_tail:]))
    A0 = max(float(c[0]) - offset, 1.0)

    below = np.where((c - offset) < (A0 / np.e))[0]
    tau0 = float(t[below[0]]) if len(below) else float(t[-1] / 2.0)
    tau0 = max(tau0, 1e-2)

    names = set(model.param_names())
    if "offset" in names:
        g["offset"] = max(offset, 0.0)
    if {"A", "tau"} <= names:
        g["A"] = A0
        g["tau"] = tau0
    if {"A1", "tau1", "A2", "tau2"} <= names:
        g["A1"] = 0.6 * A0
        g["tau1"] = max(tau0 * 0.4, 1e-2)
        g["A2"] = 0.4 * A0
        g["tau2"] = tau0 * 1.8
    return g


# ── Fit core ──────────────────────────────────────────────────────────────────

def fit_lifetime(
    model: FCSModel,
    t_ns: np.ndarray,
    counts: np.ndarray,
    fit_start_ns: float,
    fit_end_ns: float,
    guesses: Dict[str, float],
    lowers: Dict[str, float],
    uppers: Dict[str, float],
    fixed: Dict[str, bool],
    weighted: bool = True,
    drop_edges: bool = True,
    channel: Optional[int] = None,
    n_bins: Optional[int] = None,
    maxfev: int = 20000,
) -> dict:
    """
    Weighted least-squares tail fit of ``model`` to a TCSPC decay, honouring
    per-parameter bounds and "fixed" flags.

    The model is evaluated on time re-zeroed to the window start.  When
    ``weighted`` is True (the default) the fit uses Poisson errors
    σ = √max(counts, 1) and parameter errors are absolute, so the reduced χ²
    is meaningful.

    Returns a result dict mirroring fcs_fit.fit_correlation (values, 1σ errors,
    masked data, fit curve, residuals, R², χ², reduced χ²), plus lifetime
    extras: the absolute and relative time axes, the channel / n_bins / window,
    and — for the two-exponential model — the amplitude-weighted mean lifetime.
    """
    names = model.param_names()

    t_win, c_win, t_rel = prepare_decay(
        t_ns, counts, fit_start_ns, fit_end_ns, drop_edges=drop_edges)

    s = np.sqrt(np.maximum(c_win, 1.0)) if weighted else None

    free = [n for n in names if not fixed.get(n, False)]
    if not free:
        raise ValueError("At least one parameter must be free (not fixed).")
    fixed_vals = {n: guesses[n] for n in names if fixed.get(n, False)}

    def _model_free(tt, *free_vals):
        allv = dict(fixed_vals)
        for n, v in zip(free, free_vals):
            allv[n] = v
        return model.func(tt, **{n: allv[n] for n in names})

    p0 = [guesses[n] for n in free]
    lb = [lowers[n] for n in free]
    ub = [uppers[n] for n in free]
    # Nudge guesses inside the bounds so curve_fit doesn't reject p0.
    p0 = [min(max(p, lo), hi) for p, lo, hi in zip(p0, lb, ub)]

    popt, pcov = curve_fit(
        _model_free, t_rel, c_win, p0=p0, bounds=(lb, ub),
        sigma=s, absolute_sigma=(s is not None), maxfev=maxfev,
    )
    perr = np.sqrt(np.diag(pcov))

    values = dict(fixed_vals)
    errors = {n: 0.0 for n in fixed_vals}        # fixed params have no error
    for n, v, e in zip(free, popt, perr):
        values[n] = float(v)
        errors[n] = float(e)

    yfit  = model.func(t_rel, **{n: values[n] for n in names})
    resid = c_win - yfit

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((c_win - c_win.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof    = len(c_win) - len(free)
    if s is not None and dof > 0:
        chi2     = float(np.sum((resid / s) ** 2))
        red_chi2 = chi2 / dof
    else:
        chi2 = red_chi2 = float("nan")

    # Derived: amplitude-weighted mean lifetime for the two-exponential model.
    tau_mean = float("nan")
    if {"A1", "tau1", "A2", "tau2"} <= set(names):
        a1, t1 = values["A1"], values["tau1"]
        a2, t2 = values["A2"], values["tau2"]
        denom = a1 + a2
        tau_mean = (a1 * t1 + a2 * t2) / denom if denom > 0 else float("nan")

    return {
        "model": model, "names": names, "free": free,
        "values": values, "errors": errors,
        "t_ns": t_win, "t_rel": t_rel, "counts": c_win,
        "fit": yfit, "resid": resid, "sigma": s,
        "guesses": dict(guesses), "lowers": dict(lowers),
        "uppers": dict(uppers), "fixed": dict(fixed),
        "r2": r2, "chi2": chi2, "red_chi2": red_chi2,
        "ss_res": ss_res, "n_points": len(c_win), "dof": dof,
        "weighted": s is not None,
        "channel": channel, "n_bins": n_bins,
        "fit_start_ns": float(fit_start_ns), "fit_end_ns": float(fit_end_ns),
        "drop_edges": drop_edges, "tau_mean": tau_mean,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_lifetime_fit(
    result: dict,
    source_name: str,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot the decay (log-y), the fitted curve, and the residuals."""
    model = result["model"]
    t_ns  = result["t_ns"]
    t_rel = result["t_rel"]

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )

    # Data + smooth fit curve (log-y)
    ax.semilogy(t_ns, np.maximum(result["counts"], 0.1), linestyle="none",
                marker=".", markersize=3, color="black", label="data")
    t_dense_rel = np.linspace(t_rel.min(), t_rel.max(), 600)
    y_dense = model.func(t_dense_rel, **{n: result["values"][n] for n in result["names"]})
    ax.semilogy(t_dense_rel + t_ns[0], np.maximum(y_dense, 0.1),
                color="tomato", linewidth=1.6, label="fit")
    ax.set_ylabel("Photon counts", fontsize=12)

    # Parameter summary box
    lines = []
    for n in result["names"]:
        val = result["values"][n]
        err = result["errors"][n]
        unit = next((p.unit for p in model.params if p.name == n), "")
        tag = "  (fixed)" if result["fixed"].get(n) else f" ± {err:.3g}"
        lines.append(f"{n} = {val:.4g}{tag} {unit}".rstrip())
    if np.isfinite(result.get("tau_mean", float("nan"))):
        lines.append(f"⟨τ⟩ = {result['tau_mean']:.4g} ns")
    gof = (f"red. χ² = {result['red_chi2']:.3g}"
           if result["weighted"] else f"R² = {result['r2']:.4f}")
    lines.append(gof)
    ax.text(0.98, 0.95, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    ax.legend(loc="lower left", fontsize=10, framealpha=0.85)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    ch = result.get("channel")
    ch_str = f"Ch{ch}  ·  " if ch else ""
    title = f"Lifetime fit — {model.name}"
    subtitle = (f"{source_name}  ·  {ch_str}{result['n_points']} points  ·  "
                f"window {result['fit_start_ns']:.2f}–{result['fit_end_ns']:.2f} ns"
                f"  ·  {model.formula}")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    # Residuals (weighted residuals if a Poisson σ was used)
    res = result["resid"]
    if result["weighted"]:
        res = res / result["sigma"]
        axr.set_ylabel("resid/σ", fontsize=10)
    else:
        axr.set_ylabel("resid", fontsize=10)
    axr.plot(t_ns, res, linestyle="none", marker=".",
             markersize=3, color="steelblue")
    axr.axhline(0, color="grey", linewidth=0.8)
    axr.set_xlabel("Arrival time within laser cycle (ns)", fontsize=12)
    axr.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    axr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2g}"))

    if show:
        fcs_plottools.show_figure(fig, np.array([ax, axr]))
    return fig, np.array([ax, axr])


# ── Export ────────────────────────────────────────────────────────────────────

def _fits_dir(source_path: Path) -> Path:
    """
    Return (creating if needed) a 'fits' folder beside the original data file.
    Mirrors fcs_fit._fits_dir so lifetime and correlation fits land together.
    """
    base = source_path.parent
    if base.name.lower() == "analysis":
        base = base.parent
    out = base / "fits"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fmt(x: float) -> str:
    if x == np.inf:
        return "inf"
    if x == -np.inf:
        return "-inf"
    return f"{x:.4g}"


def export_lifetime_fit(result: dict, source_path: str | Path) -> Tuple[Path, Path]:
    """
    Write a human-readable .txt report plus a .csv of the fitted curve to the
    'fits' folder.  Returns (report_path, curve_path).
    """
    source_path = Path(source_path)
    model = result["model"]
    out_dir = _fits_dir(source_path)
    ch = result.get("channel")
    ch_tag = f"_ch{ch}" if ch else ""
    stem = f"{source_path.stem}_lifetimefit{ch_tag}_{model.key}"

    report_path = out_dir / f"{stem}.txt"
    curve_path  = out_dir / f"{stem}_curve.csv"

    # ── Report ────────────────────────────────────────────────────────────────
    L: list[str] = []
    L.append("Lifetime fit report")
    L.append("=" * 60)
    L.append(f"source     : {source_path.name}")
    L.append(f"model      : {model.name}  [{model.key}]")
    L.append(f"formula    : {model.formula}")
    L.append(f"fitted     : {datetime.now().isoformat(timespec='seconds')}")
    if ch is not None:
        L.append(f"channel    : Ch{ch}")
    if result.get("n_bins") is not None:
        L.append(f"histogram  : {result['n_bins']} bins")
    L.append(f"fit window : {result['fit_start_ns']:.4f} – {result['fit_end_ns']:.4f} ns"
             f"  (first/last bins {'dropped' if result['drop_edges'] else 'kept'})")
    L.append(f"points     : {result['n_points']}   "
             f"free params : {len(result['free'])}   dof : {result['dof']}")
    L.append(f"weighted   : {'yes (Poisson σ = √counts)' if result['weighted'] else 'no'}")
    L.append("")
    L.append("Parameters")
    L.append("-" * 60)
    L.append(f"{'name':<8}{'value':>14}{'std err':>14}  {'unit':<7} "
             f"{'bounds':>22}  fixed")
    for n in result["names"]:
        p_unit = next((p.unit for p in model.params if p.name == n), "")
        val = result["values"][n]
        err = result["errors"][n]
        lo  = result["lowers"][n]
        hi  = result["uppers"][n]
        is_fixed = result["fixed"].get(n, False)
        err_str = f"{'—':>14}" if is_fixed else f"{err:>14.6g}"
        bnd = f"[{_fmt(lo)}, {_fmt(hi)}]"
        L.append(f"{n:<8}{val:>14.6g}{err_str}  {p_unit:<7} "
                 f"{bnd:>22}  {'yes' if is_fixed else 'no'}")
    if np.isfinite(result.get("tau_mean", float("nan"))):
        L.append("")
        L.append(f"derived    : amplitude-weighted mean lifetime "
                 f"⟨τ⟩ = {result['tau_mean']:.6g} ns")
    L.append("")
    L.append("Goodness of fit")
    L.append("-" * 60)
    L.append(f"  SS_res     : {result['ss_res']:.6g}")
    L.append(f"  R^2        : {result['r2']:.6f}")
    if result["weighted"]:
        L.append(f"  chi^2      : {result['chi2']:.6g}")
        L.append(f"  red. chi^2 : {result['red_chi2']:.6g}")
    L.append("")
    report_path.write_text("\n".join(L), encoding="utf-8")

    # ── Curve CSV ─────────────────────────────────────────────────────────────
    cols = {
        "t_ns":     result["t_ns"],
        "t_rel_ns": result["t_rel"],
        "counts":   result["counts"],
        "fit":      result["fit"],
        "residual": result["resid"],
    }
    if result["weighted"]:
        cols["sigma"] = result["sigma"]
    names = list(cols.keys())
    with curve_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# Lifetime fit curve — {model.key}\n")
        fh.write(f"# source : {source_path.name}\n")
        if ch is not None:
            fh.write(f"# channel : Ch{ch}\n")
        fh.write(f"# exported : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(",".join(names) + "\n")
        for row in zip(*(cols[n] for n in names)):
            fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

    print(f"[lifetime fit] wrote {report_path}")
    print(f"[lifetime fit] wrote {curve_path}")
    return report_path, curve_path


def _parse_bound(text: str, default: float) -> float:
    """Parse a bound entry, accepting blank, 'inf', '-inf'."""
    t = text.strip().lower()
    if t in ("", "inf", "+inf", "infinity"):
        return np.inf if t != "" else default
    if t in ("-inf", "-infinity"):
        return -np.inf
    return float(t)


# ── GUI: entry point and dialogs ──────────────────────────────────────────────

# Remembers the most recently used fit window (ns), so fitting a family of
# datasets over the same range doesn't require re-typing it each time.  Module-
# level → persists across files for the session.
_last_fit_window: dict = {"start": None, "end": None}

def run_lifetime_fit_dialog(fcs_data: FCSData, parent=None):
    """
    Full GUI flow for lifetime modelling: choose channel / resolution / window,
    pick a model, set guesses and bounds, then fit, plot and export.
    """
    def _after_data(channel, n_bins, t_ns, counts, fit_start, fit_end):
        def _after_model(model: FCSModel):
            _lifetime_setup_dialog(parent, fcs_data, model, channel, n_bins,
                                   t_ns, counts, fit_start, fit_end)
        _select_lifetime_model_dialog(parent, _after_model)

    _lifetime_data_dialog(parent, fcs_data, _after_data)

def _lifetime_data_dialog(parent, fcs_data, on_done):
    """Screen 1 — channel, histogram resolution, and fit window.

    Accepts photon data (FCSData) or an already-binned lifetime decay
    (fcs_ifx.LifetimeData).  For a decay there is a single curve and a fixed
    native resolution, so the channel and bin controls are hidden.
    """
    import tkinter as tk
    from tkinter import messagebox

    is_lt = getattr(fcs_data, "kind", None) == "lifetime_decay"

    # .ifx is already binned → fit at native resolution (n_bins=None); .fcs uses
    # the chosen histogram resolution.
    def _nbins():
        return None if is_lt else bin_var.get()

    win = tk.Toplevel(parent)
    win.title("Lifetime fit — data")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="Lifetime fit — select data",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    tk.Label(win, text=f"File: {fcs_data.filepath.name}",
             font=("Helvetica", 9), fg="grey").pack()

    body = tk.Frame(win, padx=14, pady=8)
    body.pack(fill="x")

    # Channel — photon data has Ch1/Ch2; an .ifx decay is a single curve.
    ch_var = tk.IntVar(value=1)
    tk.Label(body, text="Channel:", anchor="w").grid(row=0, column=0, sticky="w", pady=3)
    if is_lt:
        tk.Label(body, text="decay (.ifx)", anchor="w").grid(row=0, column=1, sticky="w")
    else:
        ch_frame = tk.Frame(body)
        ch_frame.grid(row=0, column=1, sticky="w")
        tk.Radiobutton(ch_frame, text="Ch1", variable=ch_var, value=1).pack(side="left")
        tk.Radiobutton(ch_frame, text="Ch2", variable=ch_var, value=2).pack(side="left")

    # Bins — only meaningful for photon data; an .ifx decay is already binned.
    bin_var = tk.IntVar(value=4096)
    if not is_lt:
        tk.Label(body, text="Histogram bins:", anchor="w").grid(row=1, column=0, sticky="w", pady=3)
        tk.OptionMenu(body, bin_var, *fcs_lifetime._VALID_N_BINS).grid(row=1, column=1, sticky="w")

    # Window entries
    tk.Label(body, text="Fit start (ns):", anchor="w").grid(row=2, column=0, sticky="w", pady=3)
    start_var = tk.StringVar()
    tk.Entry(body, textvariable=start_var, width=12).grid(row=2, column=1, sticky="w")
    tk.Label(body, text="Fit end (ns):", anchor="w").grid(row=3, column=0, sticky="w", pady=3)
    end_var = tk.StringVar()
    tk.Entry(body, textvariable=end_var, width=12).grid(row=3, column=1, sticky="w")

    def _set_default_window(*_):
        try:
            lo, hi = default_window(fcs_data, ch_var.get(), _nbins())
            start_var.set(f"{lo:.2f}")
            end_var.set(f"{hi:.2f}")
        except Exception:
            pass

    # Pre-fill with the last-used window if we have one; otherwise compute the
    # per-file default (peak → end).
    if _last_fit_window["start"] is not None and _last_fit_window["end"] is not None:
        start_var.set(f"{_last_fit_window['start']:.2f}")
        end_var.set(f"{_last_fit_window['end']:.2f}")
    else:
        _set_default_window()
    ch_var.trace_add("write", _set_default_window)
    bin_var.trace_add("write", _set_default_window)

    def _current_window():
        try:
            return (float(start_var.get()), float(end_var.get()))
        except ValueError:
            return None

    def _pick_on_hist():
        gate = fcs_lifetime.select_gate(
            fcs_data, channels=(ch_var.get(),),
            initial_gate=_current_window(),
            title="Set lifetime fit window",
            gate_label="Fit window",
            confirm_text="Use this window",
        )
        if gate is not None:
            start_var.set(f"{gate[0]:.2f}")
            end_var.set(f"{gate[1]:.2f}")

    tk.Button(win, text="Pick window on histogram…", command=_pick_on_hist,
              pady=3).pack(pady=(0, 4))

    tk.Label(win, text="The first and last bins (edge artifacts)\n"
                       "are always excluded from the fit.",
             font=("Helvetica", 9), fg="grey", justify="center").pack()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        win_range = _current_window()
        if win_range is None:
            messagebox.showerror("Invalid window",
                                 "Fit start and end must be numbers.", parent=win)
            return
        lo, hi = win_range
        if hi <= lo:
            messagebox.showerror("Invalid window",
                                 "Fit end must be greater than fit start.", parent=win)
            return
        # Remember this window so the next dataset opens pre-filled with it.
        _last_fit_window["start"], _last_fit_window["end"] = lo, hi
        channel = ch_var.get()
        
        n_bins  = _nbins()
        t_ns, counts = fcs_data.lifetime_histogram(channel=channel, n_bins=n_bins)
        win.destroy()
        on_done(channel, n_bins, t_ns, counts, lo, hi)

    tk.Button(btns, text="Next →", width=12, command=_next, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


def _select_lifetime_model_dialog(parent, on_choose):
    """Screen 2 — choose a lifetime model from the registry."""
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Lifetime fit — select model")
    win.geometry("525x380")
    win.minsize(440, 340)
    win.resizable(True, True)
    win.grab_set()

    tk.Label(win, text="Select a lifetime model",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    models = fcs_models.list_lifetime_models()
    key_var = tk.StringVar(value=models[0].key)

    list_frame = tk.LabelFrame(win, text="Models", padx=10, pady=6)
    list_frame.pack(fill="x", padx=12, pady=(0, 6))
    for m in models:
        tk.Radiobutton(list_frame, text=m.name, variable=key_var,
                       value=m.key, anchor="w").pack(fill="x")

    desc = tk.Text(win, height=8, wrap="word", font=("Courier", 9),
                   bg="#f7f7f7", relief="flat", padx=8, pady=6)
    desc.pack(fill="both", expand=True, padx=12, pady=(0, 6))

    def _refresh_desc(*_):
        m = fcs_models.get_lifetime_model(key_var.get())
        desc.config(state="normal")
        desc.delete("1.0", tk.END)
        desc.insert(tk.END, m.description)
        desc.config(state="disabled")

    key_var.trace_add("write", _refresh_desc)
    _refresh_desc()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        m = fcs_models.get_lifetime_model(key_var.get())
        win.destroy()
        on_choose(m)

    tk.Button(btns, text="Next →", width=12, command=_next, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


def _lifetime_setup_dialog(parent, fcs_data, model, channel, n_bins,
                           t_ns, counts, fit_start, fit_end):
    """Screen 3 — initial guesses, bounds and fixed flags, then fit."""
    import tkinter as tk
    from tkinter import messagebox

    win = tk.Toplevel(parent)
    win.title(f"Lifetime fit — {model.name}")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text=model.name, font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text=f"{fcs_data.filepath.name}  ·  Ch{channel}  ·  "
                       f"window {fit_start:.2f}–{fit_end:.2f} ns",
             font=("Helvetica", 9), fg="grey").pack()
    tk.Label(win, text=model.formula, font=("Courier", 9), fg="#444").pack(pady=(0, 6))

    # Auto guesses from the windowed data
    try:
        _t, _c, _trel = prepare_decay(t_ns, counts, fit_start, fit_end)
        guesses0 = auto_guess_lifetime(model, _trel, _c)
    except Exception:
        guesses0 = model.defaults()

    table = tk.Frame(win, padx=12, pady=4)
    table.pack(fill="x")

    headers = ["Parameter", "Guess", "Lower", "Upper", "Fix"]
    for c, h in enumerate(headers):
        tk.Label(table, text=h, font=("Helvetica", 10, "bold")).grid(
            row=0, column=c, padx=4, pady=(0, 4))

    guess_vars: Dict[str, tk.StringVar] = {}
    lower_vars: Dict[str, tk.StringVar] = {}
    upper_vars: Dict[str, tk.StringVar] = {}
    fixed_vars: Dict[str, tk.BooleanVar] = {}

    for r, p in enumerate(model.params, start=1):
        label = f"{p.name}" + (f" ({p.unit})" if p.unit else "")
        tk.Label(table, text=label, anchor="w", width=12).grid(
            row=r, column=0, sticky="w", padx=4, pady=2)

        gv = tk.StringVar(value=f"{guesses0.get(p.name, p.default):.6g}")
        lv = tk.StringVar(value=_fmt(p.lower))
        uv = tk.StringVar(value=_fmt(p.upper))
        fv = tk.BooleanVar(value=p.fixed)

        tk.Entry(table, textvariable=gv, width=12).grid(row=r, column=1, padx=4)
        tk.Entry(table, textvariable=lv, width=10).grid(row=r, column=2, padx=4)
        tk.Entry(table, textvariable=uv, width=10).grid(row=r, column=3, padx=4)
        tk.Checkbutton(table, variable=fv).grid(row=r, column=4, padx=4)

        guess_vars[p.name] = gv
        lower_vars[p.name] = lv
        upper_vars[p.name] = uv
        fixed_vars[p.name] = fv

    # Poisson weighting toggle (on by default for counts)
    weight_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        win, text="Weight by Poisson σ = √counts  (recommended)",
        variable=weight_var, anchor="w",
    ).pack(fill="x", padx=12, pady=(6, 0))

    btns = tk.Frame(win)
    btns.pack(pady=10)

    def _do_fit():
        try:
            guesses = {n: float(guess_vars[n].get()) for n in guess_vars}
            lowers  = {n: _parse_bound(lower_vars[n].get(), -np.inf) for n in lower_vars}
            uppers  = {n: _parse_bound(upper_vars[n].get(),  np.inf) for n in upper_vars}
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Guesses and bounds must be numbers "
                                 "(use 'inf' / '-inf' for open bounds).",
                                 parent=win)
            return
        fixed = {n: fixed_vars[n].get() for n in fixed_vars}

        for n in guesses:
            if lowers[n] >= uppers[n]:
                messagebox.showerror("Invalid bounds",
                                     f"For '{n}', lower must be < upper.",
                                     parent=win)
                return

        try:
            result = fit_lifetime(
                model, t_ns, counts, fit_start, fit_end,
                guesses, lowers, uppers, fixed,
                weighted=weight_var.get(),
                channel=channel, n_bins=n_bins,
            )
        except Exception as e:
            messagebox.showerror("Fit failed", str(e), parent=win)
            return

        win.destroy()

        report_path, _curve = export_lifetime_fit(result, fcs_data.filepath)
        fig, _axes = plot_lifetime_fit(result, fcs_data.filepath.name, show=False)
        try:
            fig.savefig(report_path.with_suffix(".png"), dpi=150)
        except Exception as e:
            print(f"[lifetime fit] could not save figure: {e}")

        summary = "\n".join(
            f"{n} = {result['values'][n]:.4g}"
            + ("" if result['fixed'].get(n) else f" ± {result['errors'][n]:.2g}")
            for n in result["names"]
        )
        if np.isfinite(result.get("tau_mean", float("nan"))):
            summary += f"\n⟨τ⟩ = {result['tau_mean']:.4g} ns"
        gof = (f"red. χ² = {result['red_chi2']:.3g}"
               if result["weighted"] else f"R² = {result['r2']:.4f}")
        messagebox.showinfo(
            "Lifetime fit complete",
            f"{model.name}  (Ch{channel})\n\n{summary}\n\n{gof}\n\n"
            f"Results saved to:\n{report_path.parent}",
            parent=parent,
        )
        fcs_plottools.show_figure(fig, _axes)

    tk.Button(btns, text="Fit", width=12, command=_do_fit, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from fcs_reader import read_fcs

    if len(sys.argv) < 2:
        print("Usage: python fcs_lifetime_fit.py <file.fcs> [model_key] [channel]")
        print("Available lifetime models:")
        for m in fcs_models.list_lifetime_models():
            print(f"  {m.key:<16} {m.name}")
        sys.exit(1)

    path = Path(sys.argv[1])
    key  = sys.argv[2] if len(sys.argv) > 2 else fcs_models.list_lifetime_models()[0].key
    ch   = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = fcs_models.get_lifetime_model(key)

    d = read_fcs(path)
    t_ns, counts = d.lifetime_histogram(channel=ch, n_bins=4096)
    lo, hi = default_window(d, ch, 4096)
    t_, c_, trel = prepare_decay(t_ns, counts, lo, hi)
    guesses = auto_guess_lifetime(model, trel, c_)
    lowers = {p.name: p.lower for p in model.params}
    uppers = {p.name: p.upper for p in model.params}
    fixed  = {p.name: p.fixed for p in model.params}

    result = fit_lifetime(model, t_ns, counts, lo, hi,
                          guesses, lowers, uppers, fixed,
                          channel=ch, n_bins=4096)
    export_lifetime_fit(result, path)
    plot_lifetime_fit(result, path.name)
