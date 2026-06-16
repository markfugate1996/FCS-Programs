"""
fcs_pch_fit.py
==============
Fit Photon Counting Histograms (PCH) to the 3D-Gaussian single- and two-species
models of Chen, Müller, So & Gratton (Biophys J 1999, 77:553).

This is the PCH counterpart of fcs_fit.py (correlation) and fcs_lifetime_fit.py
(lifetime).  It is launched from the "Model Data" task in the main window, via
the PCH button in fcs_fit.run_model_dialog.

Workflow (launched from the main window)
----------------------------------------
    1. Pick the channel and bin width.                 -> _pch_data_dialog
    2. Choose a model (single- or two-species).        -> _select_pch_model_dialog
    3. Set guesses / bounds / fixed flags, then fit.   -> _pch_setup_dialog
    4. Fit, plot data + fit + residuals, write a report and curve CSV
       to a 'fits' folder beside the source .fcs file.

Like the lifetime fitter, the PCH fitter works from the FCSData object (not from
a saved CSV): it bins the photon stream itself, which keeps the raw per-count
frequencies and the number of sampled bins M.  M is needed to weight the fit by
the Poisson error of each histogram bin (σ = √counts), which is what makes the
reduced χ² meaningful — the same statistics family as the correlation and
lifetime fitters.

The numerical core (pch_counts, auto_guess_pch, fit_pch) has no GUI dependency.
The PCH model kernel itself lives in fcs_models (PCH_MODELS).

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
from scipy.stats import poisson as _poisson

import fcs_models
from fcs_models import FCSModel, _PCH_GAMMA2
from fcs_reader import FCSData
import fcs_pch


# ── Data preparation ──────────────────────────────────────────────────────────

def pch_counts(
    times_s: np.ndarray,
    bin_width_s: float,
) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    """
    Build the photon counting histogram as RAW per-count frequencies.

    Like fcs_pch.compute_pch, but returns the unnormalised frequencies n_k (the
    number of time bins containing exactly k photons) and the number of sampled
    bins M, which the fit needs for Poisson weighting.

    Returns
    -------
    k    : np.ndarray (int)    count values 0..k_max
    n_k  : np.ndarray (float)  number of time bins with exactly k photons (Σ = M)
    M    : int                 number of sampled time bins
    mean : float               mean photons per bin  <k>
    var  : float               variance of photons per bin  Var(k)
    """
    times_s = np.asarray(times_s, dtype=np.float64)
    if times_s.size == 0:
        raise ValueError("times_s is empty; cannot compute PCH.")

    t_start, t_end = times_s[0], times_s[-1]
    M = max(1, int((t_end - t_start) / bin_width_s))
    edges = t_start + np.arange(M + 1) * bin_width_s
    counts, _ = np.histogram(times_s, bins=edges)

    mean = float(counts.mean())
    var  = float(counts.var(ddof=1)) if len(counts) > 1 else float("nan")

    k_max = int(counts.max())
    k   = np.arange(0, k_max + 1, dtype=int)
    n_k = np.bincount(counts, minlength=k_max + 1).astype(float)
    return k, n_k, int(M), mean, var


def channel_times(fcs_data: FCSData, channel: str) -> np.ndarray:
    """Photon arrival times (s) for 'ch1', 'ch2', or 'combined'."""
    if channel == "ch1":
        return fcs_data.ch1_times_s
    if channel == "ch2":
        return fcs_data.ch2_times_s
    if channel == "combined":
        return np.sort(np.concatenate([fcs_data.ch1_times_s, fcs_data.ch2_times_s]))
    raise ValueError(f"Unknown channel '{channel}' (use ch1, ch2, or combined).")


def channel_cps(fcs_data: FCSData, channel: str) -> float:
    """Observed count rate (counts/s) for the chosen channel, from the trace."""
    if channel == "ch1":
        return float(fcs_data.count_rate_ch1_hz)
    if channel == "ch2":
        return float(fcs_data.count_rate_ch2_hz)
    if channel == "combined":
        return float(fcs_data.count_rate_ch1_hz + fcs_data.count_rate_ch2_hz)
    raise ValueError(f"Unknown channel '{channel}'.")


# ── Initial-guess heuristics ──────────────────────────────────────────────────

def auto_guess_pch(model: FCSModel, mean: float, var: float) -> Dict[str, float]:
    """
    Method-of-moments starting guesses.

    From the data: ε ≈ Q/γ₂ with Q = Var/<k> − 1, then N ≈ <k>/ε.  For the
    two-species model the single-species estimate is split into a brighter and a
    dimmer component to seed the brightness separation.
    """
    g = model.defaults()
    if not (np.isfinite(mean) and np.isfinite(var)) or mean <= 0:
        return g

    Q = var / mean - 1.0
    eps0 = Q / _PCH_GAMMA2
    # Guard: near-Poisson data gives tiny/negative Q; fall back to a small ε.
    eps0 = float(np.clip(eps0, 1e-3, 1e3))
    N0 = max(mean / eps0, 1e-3)

    names = set(model.param_names())
    if {"N", "epsilon"} <= names:
        g["N"] = N0
        g["epsilon"] = eps0
    if {"N1", "epsilon1", "N2", "epsilon2"} <= names:
        # Brighter + dimmer split sharing the total mean N0·eps0 = <k>.
        g["epsilon1"] = eps0 * 1.8
        g["epsilon2"] = max(eps0 * 0.5, 1e-3)
        g["N1"] = 0.5 * N0
        g["N2"] = 0.5 * N0
    return g


# ── Fit core ──────────────────────────────────────────────────────────────────

def fit_pch(
    model: FCSModel,
    k: np.ndarray,
    n_k: np.ndarray,
    M: int,
    guesses: Dict[str, float],
    lowers: Dict[str, float],
    uppers: Dict[str, float],
    fixed: Dict[str, bool],
    weighted: bool = True,
    channel: Optional[str] = None,
    bin_width_s: Optional[float] = None,
    mean: Optional[float] = None,
    var: Optional[float] = None,
    observed_cps: Optional[float] = None,
    maxfev: int = 40000,
) -> dict:
    """
    Weighted least-squares fit of a PCH ``model`` to the measured histogram.

    The model returns probabilities Π(k); the fit compares expected counts
    M·Π(k) with the observed frequencies n_k.  With ``weighted`` True (default)
    each bin carries its Poisson error σ = √max(n_k, 1), so the reduced χ² is
    meaningful.

    Returns a result dict mirroring fcs_lifetime_fit.fit_lifetime (values, 1σ
    errors, data, fit, residuals, R², χ², reduced χ²), plus PCH extras: the
    probability vectors, M, channel / bin width, the measured moments, and —
    for the two-species model — the number fraction f1.
    """
    names = model.param_names()
    k   = np.asarray(k)
    n_k = np.asarray(n_k, dtype=np.float64)
    M   = int(M)

    s = np.sqrt(np.maximum(n_k, 1.0)) if weighted else None

    free = [n for n in names if not fixed.get(n, False)]
    if not free:
        raise ValueError("At least one parameter must be free (not fixed).")
    fixed_vals = {n: guesses[n] for n in names if fixed.get(n, False)}

    def _counts_free(kk, *free_vals):
        allv = dict(fixed_vals)
        for n, v in zip(free, free_vals):
            allv[n] = v
        Pi = model.func(kk, **{n: allv[n] for n in names})
        return M * Pi

    p0 = [guesses[n] for n in free]
    lb = [lowers[n] for n in free]
    ub = [uppers[n] for n in free]
    p0 = [min(max(p, lo), hi) for p, lo, hi in zip(p0, lb, ub)]

    popt, pcov = curve_fit(
        _counts_free, k, n_k, p0=p0, bounds=(lb, ub),
        sigma=s, absolute_sigma=(s is not None), maxfev=maxfev,
    )
    perr = np.sqrt(np.diag(pcov))

    values = dict(fixed_vals)
    errors = {n: 0.0 for n in fixed_vals}
    for n, v, e in zip(free, popt, perr):
        values[n] = float(v)
        errors[n] = float(e)

    pk_fit = model.func(k, **{n: values[n] for n in names})
    counts_fit = M * pk_fit
    resid = n_k - counts_fit

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((n_k - n_k.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof    = len(k) - len(free)
    if s is not None and dof > 0:
        chi2     = float(np.sum((resid / s) ** 2))
        red_chi2 = chi2 / dof
    else:
        chi2 = red_chi2 = float("nan")

    # Derived quantities
    derived: Dict[str, float] = {}
    species_params = []   # (N_i, eps_i) for the count-rate link
    if {"N", "epsilon"} <= set(names):
        derived["<k>_model"] = values["N"] * values["epsilon"]
        species_params = [(values["N"], values["epsilon"])]
    if {"N1", "epsilon1", "N2", "epsilon2"} <= set(names):
        N1, N2 = values["N1"], values["N2"]
        tot = N1 + N2
        derived["f1"] = N1 / tot if tot > 0 else float("nan")
        derived["<k>_model"] = (N1 * values["epsilon1"] + N2 * values["epsilon2"])
        species_params = [(N1, values["epsilon1"]), (N2, values["epsilon2"])]

    # ── Link to the intensity-trace count rate ────────────────────────────────
    # epsilon is counts per molecule per bin; eta = epsilon / T is the molecular
    # brightness in counts per second (bin-width independent), and the model's
    # total rate is N*eta = (N*epsilon)/T, which should reproduce the channel's
    # measured CPS from the intensity trace.
    if bin_width_s and species_params:
        T = float(bin_width_s)
        if len(species_params) == 1:
            derived["eta_cps_per_molecule"] = species_params[0][1] / T
        else:
            for i, (_Ni, ei) in enumerate(species_params, start=1):
                derived[f"eta{i}_cps_per_molecule"] = ei / T
        derived["predicted_cps"] = sum(Ni * ei for Ni, ei in species_params) / T
        if observed_cps is not None and np.isfinite(observed_cps):
            derived["observed_cps"] = float(observed_cps)
            pred = derived["predicted_cps"]
            if pred > 0:
                derived["predicted_over_observed"] = pred / float(observed_cps)

    return {
        "model": model, "names": names, "free": free,
        "values": values, "errors": errors,
        "k": k, "n_k": n_k, "pk": n_k / M,
        "counts_fit": counts_fit, "pk_fit": pk_fit,
        "resid": resid, "sigma": s,
        "guesses": dict(guesses), "lowers": dict(lowers),
        "uppers": dict(uppers), "fixed": dict(fixed),
        "r2": r2, "chi2": chi2, "red_chi2": red_chi2,
        "ss_res": ss_res, "n_points": len(k), "dof": dof,
        "weighted": s is not None,
        "M": M, "channel": channel, "bin_width_s": bin_width_s,
        "mean": mean, "var": var,
        "Q": (var / mean - 1.0) if (mean and np.isfinite(var)) else float("nan"),
        "derived": derived,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_pch_fit(result: dict, source_name: str, show: bool = True
                 ) -> Tuple[plt.Figure, np.ndarray]:
    """Plot the PCH (log-y), the fitted distribution, and the residuals."""
    model = result["model"]
    k     = result["k"]
    pk    = result["pk"]
    pk_fit = result["pk_fit"]

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )

    floor = 0.5 / result["M"]
    ax.bar(k, np.maximum(pk, floor), width=0.7, color="steelblue", alpha=0.5,
           label="data", zorder=2)
    ax.plot(k, np.maximum(pk_fit, floor), color="tomato", linewidth=1.4,
            marker="o", markersize=3.5, zorder=3, label="fit")
    # Poisson reference at the measured mean
    if result.get("mean"):
        pois = _poisson.pmf(k, result["mean"])
        ax.plot(k, np.maximum(pois, floor), color="grey", linewidth=1.0,
                linestyle="--", alpha=0.7, zorder=2,
                label=f"Poisson (μ={result['mean']:.3g})")
    ax.set_yscale("log")
    ax.set_ylabel("Probability  p(k)", fontsize=12)

    lines = []
    for n in result["names"]:
        val = result["values"][n]
        err = result["errors"][n]
        unit = next((p.unit for p in model.params if p.name == n), "")
        tag = "  (fixed)" if result["fixed"].get(n) else f" ± {err:.3g}"
        lines.append(f"{n} = {val:.4g}{tag} {unit}".rstrip())
    if "f1" in result["derived"]:
        lines.append(f"f1 = {result['derived']['f1']:.3g}")
    d = result["derived"]
    if "eta_cps_per_molecule" in d:
        lines.append(f"η = {d['eta_cps_per_molecule']/1e3:.3g} kHz/mol")
    elif "eta1_cps_per_molecule" in d:
        lines.append(f"η1,η2 = {d['eta1_cps_per_molecule']/1e3:.3g}, "
                     f"{d['eta2_cps_per_molecule']/1e3:.3g} kHz/mol")
    lines.append(f"<k> = {result['mean']:.4g}   Q = {result['Q']:.3g}")
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
    ch_str = f"{ch}  ·  " if ch else ""
    bw = result.get("bin_width_s")
    bw_str = f"bin {fcs_pch._format_bin_width(bw)}  ·  " if bw else ""
    title = f"PCH fit — {model.name}"
    subtitle = (f"{source_name}  ·  {ch_str}{bw_str}"
                f"{result['M']:,} bins  ·  {result['n_points']} count values")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    res = result["resid"]
    if result["weighted"]:
        res = res / result["sigma"]
        axr.set_ylabel("resid/σ", fontsize=10)
    else:
        axr.set_ylabel("resid", fontsize=10)
    axr.plot(k, res, linestyle="none", marker="o", markersize=4, color="steelblue")
    axr.axhline(0, color="grey", linewidth=0.8)
    axr.set_xlabel("Photons per bin  k", fontsize=12)
    axr.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    axr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2g}"))

    if show:
        fcs_plottools.show_figure(fig, np.array([ax, axr]))
    return fig, np.array([ax, axr])


# ── Export ────────────────────────────────────────────────────────────────────

def _fits_dir(source_path: Path) -> Path:
    """Return (creating if needed) a 'fits' folder beside the source file."""
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


def export_pch_fit(result: dict, source_path: str | Path) -> Tuple[Path, Path]:
    """Write a .txt report and a .csv of the fitted PCH to the 'fits' folder."""
    source_path = Path(source_path)
    model = result["model"]
    out_dir = _fits_dir(source_path)
    ch = result.get("channel")
    ch_tag = f"_{ch}" if ch else ""
    stem = f"{source_path.stem}_pchfit{ch_tag}_{model.key}"

    report_path = out_dir / f"{stem}.txt"
    curve_path  = out_dir / f"{stem}_curve.csv"

    L: list[str] = []
    L.append("PCH fit report")
    L.append("=" * 60)
    L.append(f"source     : {source_path.name}")
    L.append(f"model      : {model.name}  [{model.key}]")
    L.append(f"formula    : {model.formula}")
    L.append(f"fitted     : {datetime.now().isoformat(timespec='seconds')}")
    if ch is not None:
        L.append(f"channel    : {ch}")
    if result.get("bin_width_s"):
        L.append(f"bin width  : {result['bin_width_s']:.6g} s")
    L.append(f"sampled bins (M) : {result['M']:,}")
    L.append(f"data moments     : <k> = {result['mean']:.6g}   "
             f"Var = {result['var']:.6g}   Q = {result['Q']:.6g}")
    L.append(f"points     : {result['n_points']}   "
             f"free params : {len(result['free'])}   dof : {result['dof']}")
    L.append(f"weighted   : {'yes (Poisson σ = √counts)' if result['weighted'] else 'no'}")
    L.append("")
    L.append("Parameters")
    L.append("-" * 60)
    L.append(f"{'name':<10}{'value':>14}{'std err':>14}  {'unit':<13} fixed")
    for n in result["names"]:
        p_unit = next((p.unit for p in model.params if p.name == n), "")
        val = result["values"][n]
        err = result["errors"][n]
        is_fixed = result["fixed"].get(n, False)
        err_str = f"{'—':>14}" if is_fixed else f"{err:>14.6g}"
        L.append(f"{n:<10}{val:>14.6g}{err_str}  {p_unit:<13} "
                 f"{'yes' if is_fixed else 'no'}")
    if result["derived"]:
        L.append("")
        L.append("Derived")
        L.append("-" * 60)
        _unit = {
            "<k>_model": "photons/bin",
            "f1": "number fraction N1/(N1+N2)",
            "eta_cps_per_molecule": "counts/s per molecule  (ε/T)",
            "eta1_cps_per_molecule": "counts/s per molecule  (ε1/T)",
            "eta2_cps_per_molecule": "counts/s per molecule  (ε2/T)",
            "predicted_cps": "counts/s  (N·ε/T, molecular only)",
            "observed_cps": "counts/s  (intensity trace)",
            "predicted_over_observed": "ratio (1.0 = fully accounted for)",
        }
        for key, val in result["derived"].items():
            tag = _unit.get(key, "")
            L.append(f"  {key:<24} : {val:>12.6g}   {tag}".rstrip())
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

    cols = {
        "k":          result["k"].astype(float),
        "pk_data":    result["pk"],
        "pk_fit":     result["pk_fit"],
        "counts_data": result["n_k"],
        "counts_fit": result["counts_fit"],
        "residual":   result["resid"],
    }
    if result["weighted"]:
        cols["sigma"] = result["sigma"]
    names = list(cols.keys())
    with curve_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# PCH fit curve — {model.key}\n")
        fh.write(f"# source : {source_path.name}\n")
        if ch is not None:
            fh.write(f"# channel : {ch}\n")
        if result.get("bin_width_s"):
            fh.write(f"# bin_width_s : {result['bin_width_s']:.6g}\n")
        fh.write(f"# sampled_bins_M : {result['M']}\n")
        fh.write(f"# exported : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(",".join(names) + "\n")
        for row in zip(*(cols[n] for n in names)):
            fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

    print(f"[pch fit] wrote {report_path}")
    print(f"[pch fit] wrote {curve_path}")
    return report_path, curve_path


def _parse_bound(text: str, default: float) -> float:
    t = text.strip().lower()
    if t in ("", "inf", "+inf", "infinity"):
        return np.inf if t != "" else default
    if t in ("-inf", "-infinity"):
        return -np.inf
    return float(t)


# ── GUI: entry point and dialogs ──────────────────────────────────────────────

def run_pch_fit_dialog(fcs_data: FCSData, parent=None):
    """Full GUI flow: choose channel / bin width, pick a model, fit, plot, export."""
    def _after_data(channel, bin_width_s, k, n_k, M, mean, var):
        def _after_model(model: FCSModel):
            _pch_setup_dialog(parent, fcs_data, model, channel, bin_width_s,
                              k, n_k, M, mean, var)
        _select_pch_model_dialog(parent, _after_model)

    _pch_data_dialog(parent, fcs_data, _after_data)


def _pch_data_dialog(parent, fcs_data: FCSData, on_done):
    """Screen 1 — channel and bin width."""
    import tkinter as tk
    from tkinter import messagebox

    win = tk.Toplevel(parent)
    win.title("PCH fit — data")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="PCH fit — select data",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    tk.Label(win, text=f"File: {fcs_data.filepath.name}",
             font=("Helvetica", 9), fg="grey").pack()

    ch_frame = tk.LabelFrame(win, text="Channel", padx=10, pady=4)
    ch_frame.pack(fill="x", padx=14, pady=6)
    ch_var = tk.StringVar(value="ch1")
    n1, n2 = len(fcs_data.ch1_deltas), len(fcs_data.ch2_deltas)
    for text, val in [(f"Ch1   ({n1:,} photons)", "ch1"),
                      (f"Ch2   ({n2:,} photons)", "ch2"),
                      ("Combined (Ch1 + Ch2)", "combined")]:
        tk.Radiobutton(ch_frame, text=text, variable=ch_var, value=val,
                       anchor="w").pack(fill="x")

    bw_frame = tk.LabelFrame(win, text="Bin width", padx=10, pady=6)
    bw_frame.pack(fill="x", padx=14, pady=(0, 6))
    bw_var = tk.StringVar(value=fcs_pch._DEFAULT_BIN_WIDTH_LABEL)
    tk.OptionMenu(bw_frame, bw_var, *fcs_pch._BIN_WIDTH_OPTIONS.keys()).pack(anchor="w")

    info_var = tk.StringVar(value="")
    tk.Label(bw_frame, textvariable=info_var, font=("Helvetica", 9),
             fg="grey", anchor="w").pack(fill="x")

    def _update_info(*_):
        bw = fcs_pch._BIN_WIDTH_OPTIONS.get(bw_var.get(), 100e-6)
        r1 = fcs_data.count_rate_ch1_hz * bw
        r2 = fcs_data.count_rate_ch2_hz * bw
        nb = int(fcs_data.duration_s / bw)
        info_var.set(f"  ~{nb:,} bins  ·  <k> Ch1≈{r1:.3g}  Ch2≈{r2:.3g} /bin")
    bw_var.trace_add("write", _update_info)
    _update_info()

    btns = tk.Frame(win)
    btns.pack(pady=10)

    def _next():
        channel = ch_var.get()
        bw = fcs_pch._BIN_WIDTH_OPTIONS[bw_var.get()]
        try:
            t = channel_times(fcs_data, channel)
            k, n_k, M, mean, var = pch_counts(t, bw)
        except Exception as e:
            messagebox.showerror("PCH error", str(e), parent=win)
            return
        if len(k) < 3:
            messagebox.showerror(
                "Too few count values",
                "This histogram has fewer than 3 distinct photon-count values; "
                "the counts per bin are too low to fit. Use a larger bin width "
                "or a brighter / higher-rate dataset.",
                parent=win)
            return
        win.destroy()
        on_done(channel, bw, k, n_k, M, mean, var)

    tk.Button(btns, text="Next →", width=12, command=_next, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)
    win.wait_window()


def _select_pch_model_dialog(parent, on_choose):
    """Screen 2 — choose a PCH model from the registry."""
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("PCH fit — select model")
    win.geometry("525x380")
    win.minsize(440, 340)
    win.resizable(True, True)
    win.grab_set()

    tk.Label(win, text="Select a PCH model",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    models = fcs_models.list_pch_models()
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
        m = fcs_models.get_pch_model(key_var.get())
        desc.config(state="normal")
        desc.delete("1.0", tk.END)
        desc.insert(tk.END, m.description)
        desc.config(state="disabled")
    key_var.trace_add("write", _refresh_desc)
    _refresh_desc()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        m = fcs_models.get_pch_model(key_var.get())
        win.destroy()
        on_choose(m)

    tk.Button(btns, text="Next →", width=12, command=_next, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)
    win.wait_window()


def _pch_setup_dialog(parent, fcs_data, model, channel, bin_width_s,
                      k, n_k, M, mean, var):
    """Screen 3 — initial guesses, bounds and fixed flags, then fit."""
    import tkinter as tk
    from tkinter import messagebox

    win = tk.Toplevel(parent)
    win.title(f"PCH fit — {model.name}")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text=model.name, font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text=f"{fcs_data.filepath.name}  ·  {channel}  ·  "
                       f"bin {fcs_pch._format_bin_width(bin_width_s)}  ·  "
                       f"<k>={mean:.3g}  Q={var/mean-1:.3g}",
             font=("Helvetica", 9), fg="grey").pack()
    tk.Label(win, text=model.formula, font=("Courier", 9), fg="#444").pack(pady=(0, 6))

    guesses0 = auto_guess_pch(model, mean, var)

    table = tk.Frame(win, padx=12, pady=4)
    table.pack(fill="x")
    for c, h in enumerate(["Parameter", "Guess", "Lower", "Upper", "Fix"]):
        tk.Label(table, text=h, font=("Helvetica", 10, "bold")).grid(
            row=0, column=c, padx=4, pady=(0, 4))

    guess_vars, lower_vars, upper_vars, fixed_vars = {}, {}, {}, {}
    for r, p in enumerate(model.params, start=1):
        label = f"{p.name}" + (f" ({p.unit})" if p.unit else "")
        tk.Label(table, text=label, anchor="w", width=16).grid(
            row=r, column=0, sticky="w", padx=4, pady=2)
        gv = tk.StringVar(value=f"{guesses0.get(p.name, p.default):.6g}")
        lv = tk.StringVar(value=_fmt(p.lower))
        uv = tk.StringVar(value=_fmt(p.upper))
        fv = tk.BooleanVar(value=p.fixed)
        tk.Entry(table, textvariable=gv, width=12).grid(row=r, column=1, padx=4)
        tk.Entry(table, textvariable=lv, width=10).grid(row=r, column=2, padx=4)
        tk.Entry(table, textvariable=uv, width=10).grid(row=r, column=3, padx=4)
        tk.Checkbutton(table, variable=fv).grid(row=r, column=4, padx=4)
        guess_vars[p.name] = gv; lower_vars[p.name] = lv
        upper_vars[p.name] = uv; fixed_vars[p.name] = fv

    weight_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text="Weight by Poisson σ = √counts  (recommended)",
                   variable=weight_var, anchor="w").pack(fill="x", padx=12, pady=(6, 0))

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
                                 "(use 'inf' / '-inf' for open bounds).", parent=win)
            return
        fixed = {n: fixed_vars[n].get() for n in fixed_vars}
        for n in guesses:
            if lowers[n] >= uppers[n]:
                messagebox.showerror("Invalid bounds",
                                     f"For '{n}', lower must be < upper.", parent=win)
                return
        try:
            result = fit_pch(model, k, n_k, M, guesses, lowers, uppers, fixed,
                             weighted=weight_var.get(), channel=channel,
                             bin_width_s=bin_width_s, mean=mean, var=var,
                             observed_cps=channel_cps(fcs_data, channel))
        except Exception as e:
            messagebox.showerror("Fit failed", str(e), parent=win)
            return

        win.destroy()
        report_path, _curve = export_pch_fit(result, fcs_data.filepath)
        fig, _axes = plot_pch_fit(result, fcs_data.filepath.name, show=False)
        try:
            fig.savefig(report_path.with_suffix(".png"), dpi=150)
        except Exception as e:
            print(f"[pch fit] could not save figure: {e}")

        summary = "\n".join(
            f"{n} = {result['values'][n]:.4g}"
            + ("" if result['fixed'].get(n) else f" ± {result['errors'][n]:.2g}")
            for n in result["names"]
        )
        if "f1" in result["derived"]:
            summary += f"\nf1 = {result['derived']['f1']:.3g}"
        gof = (f"red. χ² = {result['red_chi2']:.3g}"
               if result["weighted"] else f"R² = {result['r2']:.4f}")
        messagebox.showinfo(
            "PCH fit complete",
            f"{model.name}  ({channel})\n\n{summary}\n\n{gof}\n\n"
            f"Results saved to:\n{report_path.parent}", parent=parent)
        fcs_plottools.show_figure(fig, _axes)

    tk.Button(btns, text="Fit", width=12, command=_do_fit, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)
    win.wait_window()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from fcs_reader import read_fcs

    if len(sys.argv) < 2:
        print("Usage: python fcs_pch_fit.py <file.fcs> [model_key] [ch1|ch2|combined] [bin_width_s]")
        print("Available PCH models:")
        for m in fcs_models.list_pch_models():
            print(f"  {m.key:<14} {m.name}")
        sys.exit(1)

    path = Path(sys.argv[1])
    key  = sys.argv[2] if len(sys.argv) > 2 else fcs_models.list_pch_models()[0].key
    ch   = sys.argv[3] if len(sys.argv) > 3 else "ch1"
    bw   = float(sys.argv[4]) if len(sys.argv) > 4 else 100e-6
    model = fcs_models.get_pch_model(key)

    d = read_fcs(path)
    k, n_k, M, mean, var = pch_counts(channel_times(d, ch), bw)
    guesses = auto_guess_pch(model, mean, var)
    lowers = {p.name: p.lower for p in model.params}
    uppers = {p.name: p.upper for p in model.params}
    fixed  = {p.name: p.fixed for p in model.params}
    result = fit_pch(model, k, n_k, M, guesses, lowers, uppers, fixed,
                     channel=ch, bin_width_s=bw, mean=mean, var=var,
                     observed_cps=channel_cps(d, ch))
    export_pch_fit(result, path)
    plot_pch_fit(result, path.name)
