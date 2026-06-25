"""
fcs_lifetime_recon.py
=====================
Fit time-domain fluorescence decays (from ISS ``.ifx`` files, read by
:mod:`fcs_ifx`) to multi-exponential models by **iterative reconvolution**
with the measured IRF.

This is the reconvolution branch of  Model Data ▸ Lifetime ▸ Reconvolution
for lifetime-decay datasets.  It is the companion of :mod:`fcs_lifetime_fit`,
which fits the *tail* of a decay (no IRF) and is the default/common path; this
module is reached only when the IRF is used.  The reconvolution core is
deliberately source-agnostic — ``fit_lifetime`` takes ``(t, decay, irf)``
arrays, not a :class:`fcs_ifx.LifetimeData` — so the same engine can later be
driven from ``.fcs`` microtime histograms plus a separately measured IRF.

Workflow (launched from the main window)
----------------------------------------
    1. The active dataset is a lifetime decay (a LifetimeData).
    2. Choose a model               -> _select_lifetime_model_dialog
    3. Set guesses / bounds / fixed -> _lifetime_setup_dialog
    4. Fit, plot data + IRF + fit + weighted residuals, and write a report
       and fitted-curve CSV to a 'fits' folder beside the source file.

The numerical core (reconvolve, auto_guess_lifetime, fit_lifetime) has no GUI
dependency and can be reused for batch fitting.

Model
-----
    measured(t) = [ IRF(t) ⊗ Σ aᵢ·exp(−t/τᵢ) ] + bg

with an optional sub-bin IRF time ``shift``.  Counts are weighted by Poisson
statistics (σ = √N) for the χ² minimisation, the standard for TCSPC.

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
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

import fcs_plottools
import fcs_lifetime_models
from fcs_lifetime_models import LifetimeModel
from fcs_ifx import LifetimeData


# ── IRF preparation ───────────────────────────────────────────────────────────

def clean_irf(t_ns: np.ndarray, irf: np.ndarray) -> np.ndarray:
    """
    Remove the IRF's flat dark-count baseline so it is a clean impulse response.

    A measured IRF usually sits on a small constant background (detector dark
    counts).  Left in, that baseline is spread by the convolution into a
    uniform floor under the whole model decay — a large artefact for
    high-amplitude decays.  Here the pre-pulse level (bins before the IRF rises
    above 5 % of its peak) is estimated and subtracted, with negatives clipped
    to zero.
    """
    r = np.asarray(irf, dtype=np.float64).copy()
    if r.size == 0 or r.max() <= 0:
        return r
    peak = int(np.argmax(r))
    thr = 0.05 * r.max()
    pre = np.where(r[:peak] < thr)[0]
    base = float(np.median(r[pre])) if pre.size > 5 else float(np.percentile(r, 10))
    return np.clip(r - base, 0.0, None)


# ── Reconvolution forward model ───────────────────────────────────────────────

def _normalised_irf(
    t_ns: np.ndarray,
    irf: np.ndarray,
    shift_ns: float,
    has_irf: bool,
    peak_idx: int,
) -> np.ndarray:
    """
    Return a unit-sum IRF on the data grid, optionally time-shifted.

    When no usable IRF is present, a unit impulse placed at the decay peak is
    used instead — this degrades reconvolution gracefully into a tail fit.
    """
    if not has_irf:
        delta = np.zeros_like(t_ns)
        delta[max(0, min(peak_idx, len(delta) - 1))] = 1.0
        return delta

    r = np.asarray(irf, dtype=np.float64)
    if shift_ns:
        # Positive shift moves the IRF later in time.
        r = np.interp(t_ns - shift_ns, t_ns, irf, left=0.0, right=0.0)
    total = r.sum()
    if total <= 0:
        r = np.asarray(irf, dtype=np.float64)
        total = r.sum()
    return r / total if total > 0 else r


def reconvolve(
    t_ns: np.ndarray,
    irf: np.ndarray,
    model: LifetimeModel,
    params: Dict[str, float],
    *,
    has_irf: bool = True,
    peak_idx: int = 0,
) -> np.ndarray:
    """
    Evaluate the reconvolution forward model on the data grid.

        measured(t) = [ IRF_shifted ⊗ Σ aᵢ·exp(−t/τᵢ) ] + bg

    Parameters
    ----------
    t_ns : array
        Time axis (ns), assumed uniform.
    irf : array
        Instrument response on the same grid.
    model : LifetimeModel
    params : dict
        Values for every model parameter (amplitudes, lifetimes, bg, shift).
    has_irf : bool
        If False, a delta-IRF at ``peak_idx`` is used (tail fit).
    peak_idx : int
        Index of the decay peak, used only for the no-IRF fallback.

    Returns
    -------
    model_counts : array, same length as ``t_ns``.
    """
    n = len(t_ns)
    t0 = t_ns - t_ns[0]                     # decay origin at the first bin
    ideal = model.ideal(t0, **params)       # Σ aᵢ exp(−t/τᵢ)
    irf_n = _normalised_irf(t_ns, irf, float(params.get("shift", 0.0)),
                            has_irf, peak_idx)
    conv = fftconvolve(ideal, irf_n)[:n]
    conv = np.clip(conv, 0.0, None)
    return conv + float(params.get("bg", 0.0))


# ── Initial-guess heuristics ──────────────────────────────────────────────────

def _estimate_tail_tau(t_ns: np.ndarray, decay: np.ndarray,
                       bg: float, peak_idx: int) -> float:
    """
    Rough lifetime estimate from a log-linear fit of the decay tail (from a
    little after the peak down to near background).
    """
    y = decay - bg
    start = min(peak_idx + 3, len(t_ns) - 4)
    # Use points that are safely above background and positive.
    floor = max(1.0, 0.02 * float(np.nanmax(y)))
    mask = np.zeros_like(t_ns, dtype=bool)
    mask[start:] = True
    mask &= np.isfinite(y) & (y > floor)
    if mask.sum() < 4:
        return 2.0
    tt = t_ns[mask]
    ln = np.log(y[mask])
    slope = np.polyfit(tt, ln, 1)[0]
    if slope >= 0 or not np.isfinite(slope):
        return 2.0
    tau = -1.0 / slope
    return float(np.clip(tau, 0.05, 100.0))


def auto_guess_lifetime(
    model: LifetimeModel,
    t_ns: np.ndarray,
    decay: np.ndarray,
    peak_idx: int,
) -> Dict[str, float]:
    """
    Produce sensible starting guesses for a lifetime model from the data:
    background from the pre-pulse baseline, a tail-slope lifetime estimate,
    and amplitudes from the peak height (split across components).
    """
    guess = model.defaults()

    n = len(t_ns)
    head = max(5, n // 40)
    bg = float(np.median(decay[:head])) if n else 0.0
    bg = max(bg, 0.0)

    peak = float(np.nanmax(decay)) if n else 1.0
    amp_total = max(peak - bg, 1.0) * 1.2     # mild over-estimate vs broadened peak

    tau_est = _estimate_tail_tau(t_ns, decay, bg, peak_idx)

    # Spread component lifetimes around the single-exponential estimate.
    spread = {1: [1.0], 2: [0.5, 1.6], 3: [0.4, 1.0, 2.5]}[model.n_exp]
    fracs  = {1: [1.0], 2: [0.6, 0.4], 3: [0.5, 0.3, 0.2]}[model.n_exp]

    for i, (an, tn) in enumerate(zip(model.amp_names, model.tau_names)):
        guess[an] = amp_total * fracs[i]
        guess[tn] = float(np.clip(tau_est * spread[i], 1e-2, 1e2))

    if "bg" in guess:
        guess["bg"] = bg
    if "shift" in guess:
        guess["shift"] = 0.0
    return guess


# ── Fit core ──────────────────────────────────────────────────────────────────

def fit_lifetime(
    model: LifetimeModel,
    t_ns: np.ndarray,
    decay: np.ndarray,
    irf: np.ndarray,
    guesses: Dict[str, float],
    lowers: Dict[str, float],
    uppers: Dict[str, float],
    fixed: Dict[str, bool],
    *,
    weighted: bool = True,
    has_irf: bool = True,
    fit_start_ns: Optional[float] = None,
    fit_end_ns: Optional[float] = None,
    max_nfev: int = 20000,
) -> dict:
    """
    Iterative-reconvolution least-squares fit of ``model`` to ``decay``.

    Fixed parameters are held at their guess and excluded from the optimisation.
    When ``weighted`` is True (recommended for TCSPC), residuals are scaled by
    Poisson σ = √max(N, 1).

    ``fit_start_ns`` / ``fit_end_ns`` restrict the time window over which the
    residuals (and the goodness-of-fit statistics) are evaluated.  The forward
    model is always convolved over the full axis — only the cost function is
    windowed — so excluding, e.g., a noisy far tail or the pre-pulse region
    does not corrupt the convolution.  ``None`` means "use the full range".

    Returns a result dict with fitted values, 1σ errors, the data, the fitted
    curve, residuals and goodness-of-fit statistics, plus derived lifetime
    quantities (fractional contributions and mean lifetimes).
    """
    names = model.param_names()
    t = np.asarray(t_ns, dtype=np.float64)
    y = np.asarray(decay, dtype=np.float64)
    r = np.asarray(irf, dtype=np.float64)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    r = r[mask] if r.shape == mask.shape else r
    if len(t) < len(names) + 2:
        raise ValueError("Too few finite data points to fit.")

    # Strip the IRF's dark-count baseline so reconvolution is artefact-free.
    irf = clean_irf(t, r) if has_irf else r

    peak_idx = int(np.argmax(y))

    # Fit window: which bins contribute to the cost / statistics.
    fit_mask = np.ones_like(t, dtype=bool)
    if fit_start_ns is not None:
        fit_mask &= t >= float(fit_start_ns)
    if fit_end_ns is not None:
        fit_mask &= t <= float(fit_end_ns)
    if fit_mask.sum() < len(names) + 2:
        raise ValueError("Fit window is too narrow for the number of parameters.")

    # Poisson weights (σ = √N, floored at 1 count).
    if weighted:
        sigma = np.sqrt(np.clip(y, 1.0, None))
    else:
        sigma = np.ones_like(y)

    free = [n for n in names if not fixed.get(n, False)]
    if not free:
        raise ValueError("At least one parameter must be free (not fixed).")
    fixed_vals = {n: guesses[n] for n in names if fixed.get(n, False)}

    def _params_from(free_vals) -> Dict[str, float]:
        allv = dict(fixed_vals)
        for n, v in zip(free, free_vals):
            allv[n] = v
        return allv

    def _residuals(free_vals):
        p = _params_from(free_vals)
        model_y = reconvolve(t, irf, model, p, has_irf=has_irf, peak_idx=peak_idx)
        return ((model_y - y) / sigma)[fit_mask]

    p0 = np.array([guesses[n] for n in free], dtype=np.float64)
    lb = np.array([lowers[n] for n in free], dtype=np.float64)
    ub = np.array([uppers[n] for n in free], dtype=np.float64)
    p0 = np.clip(p0, lb, ub)

    sol = least_squares(
        _residuals, p0, bounds=(lb, ub),
        method="trf", x_scale="jac", max_nfev=max_nfev,
    )

    # 1σ errors from the Jacobian (Gauss–Newton covariance approximation).
    values = dict(fixed_vals)
    errors = {n: 0.0 for n in fixed_vals}
    perr = _jacobian_errors(sol, int(fit_mask.sum()), len(free))
    for n, v, e in zip(free, sol.x, perr):
        values[n] = float(v)
        errors[n] = float(e)

    yfit = reconvolve(t, irf, model, values, has_irf=has_irf, peak_idx=peak_idx)
    resid = y - yfit
    wresid = resid / sigma

    # Goodness-of-fit is evaluated over the fit window only.
    rw = resid[fit_mask]
    yw = y[fit_mask]
    ss_res = float(np.sum(rw ** 2))
    ss_tot = float(np.sum((yw - yw.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = int(fit_mask.sum()) - len(free)
    chi2 = float(np.sum(wresid[fit_mask] ** 2))
    red_chi2 = chi2 / dof if dof > 0 else float("nan")

    derived = _derived_lifetimes(model, values)

    win = (
        float(t[fit_mask][0]) if fit_mask.any() else float(t[0]),
        float(t[fit_mask][-1]) if fit_mask.any() else float(t[-1]),
    )

    return {
        "model": model, "names": names, "free": free,
        "values": values, "errors": errors,
        "t_ns": t, "decay": y, "fit": yfit, "irf": np.asarray(irf)[mask] if np.asarray(irf).shape == mask.shape else np.asarray(irf),
        "resid": resid, "wresid": wresid, "sigma": sigma,
        "fit_mask": fit_mask, "fit_window_ns": win,
        "guesses": dict(guesses), "lowers": dict(lowers),
        "uppers": dict(uppers), "fixed": dict(fixed),
        "r2": r2, "chi2": chi2, "red_chi2": red_chi2,
        "ss_res": ss_res, "n_points": int(fit_mask.sum()), "dof": dof,
        "weighted": weighted, "has_irf": has_irf,
        "derived": derived,
    }


def _jacobian_errors(sol, n_points: int, n_free: int) -> np.ndarray:
    """1σ parameter errors from the least_squares Jacobian at the solution."""
    try:
        J = sol.jac
        # Reduced χ² of the (already weighted) residuals scales the covariance.
        dof = max(n_points - n_free, 1)
        resid = sol.fun
        s_sq = float(np.sum(resid ** 2)) / dof
        JTJ = J.T @ J
        cov = np.linalg.pinv(JTJ) * s_sq
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        return np.full(n_free, np.nan)


def _derived_lifetimes(model: LifetimeModel, values: Dict[str, float]) -> dict:
    """
    Compute fractional intensity contributions and mean lifetimes from the
    fitted amplitudes/lifetimes.

        fᵢ = aᵢτᵢ / Σ aⱼτⱼ                 fractional intensity
        ⟨τ⟩_amp = Σ aᵢτᵢ / Σ aᵢ           amplitude-weighted mean
        ⟨τ⟩_int = Σ aᵢτᵢ² / Σ aᵢτᵢ        intensity-weighted mean
    """
    amps = np.array([values[a] for a in model.amp_names], dtype=np.float64)
    taus = np.array([values[t] for t in model.tau_names], dtype=np.float64)
    at = amps * taus
    sum_a = float(np.sum(amps))
    sum_at = float(np.sum(at))
    fracs = (at / sum_at) if sum_at > 0 else np.full_like(at, np.nan)
    tau_amp = (sum_at / sum_a) if sum_a > 0 else float("nan")
    tau_int = (float(np.sum(amps * taus ** 2)) / sum_at) if sum_at > 0 else float("nan")
    return {
        "fractions": {model.tau_names[i]: float(fracs[i]) for i in range(len(taus))},
        "tau_mean_amp": tau_amp,
        "tau_mean_int": tau_int,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_lifetime_fit(
    result: dict,
    source_name: str,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot the decay, IRF and fitted curve (log-y) over a weighted-residuals
    panel — the standard TCSPC fit view.
    """
    model = result["model"]
    t = result["t_ns"]
    y = result["decay"]
    yfit = result["fit"]
    irf = result["irf"]

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )

    # Data
    ax.plot(t, y, linestyle="none", marker=".", markersize=2.5,
            color="black", alpha=0.8, label="decay")
    # IRF, scaled to the decay peak for visual comparison
    if result["has_irf"] and np.nanmax(irf) > 0:
        irf_scaled = irf * (np.nanmax(y) / np.nanmax(irf))
        ax.plot(t, irf_scaled, color="0.6", linewidth=1.0,
                label="IRF (scaled)", zorder=1)
    # Fit
    ax.plot(t, yfit, color="tomato", linewidth=1.6, label="fit")

    ax.set_yscale("log")
    nz = y[y > 0]
    ax.set_ylim((nz.min() * 0.5) if len(nz) else 0.5, np.nanmax(y) * 1.6)
    ax.set_ylabel("Photon counts", fontsize=12)

    # Parameter / results box
    lines = []
    for i, (an, tn) in enumerate(zip(model.amp_names, model.tau_names)):
        tv, te = result["values"][tn], result["errors"][tn]
        tag = "  (fixed)" if result["fixed"].get(tn) else f" ± {te:.3g}"
        frac = result["derived"]["fractions"].get(tn, float("nan"))
        lines.append(f"τ{i+1} = {tv:.3g}{tag} ns   (f={frac:.2f})")
    if model.n_exp > 1:
        lines.append(f"⟨τ⟩amp = {result['derived']['tau_mean_amp']:.3g} ns")
    bgv = result["values"].get("bg", 0.0)
    lines.append(f"bg = {bgv:.3g} cnt")
    if not result["fixed"].get("shift", True):
        lines.append(f"shift = {result['values'].get('shift', 0.0):.3g} ns")
    lines.append(f"red. χ² = {result['red_chi2']:.3g}")
    ax.text(0.98, 0.95, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    ax.legend(loc="lower left", fontsize=10, framealpha=0.85)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.3)

    title = f"Lifetime fit — {model.name}"
    subtitle = f"{source_name}  ·  {result['n_points']} points  ·  {model.formula}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    # Weighted residuals
    axr.plot(t, result["wresid"], linestyle="none", marker=".",
             markersize=2.5, color="steelblue")
    axr.axhline(0, color="grey", linewidth=0.8)
    axr.set_ylabel("resid/σ", fontsize=10)
    axr.set_xlabel("Time (ns)", fontsize=12)
    axr.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)

    # Shade any region excluded from the fit window so it's clearly not fitted.
    fit_mask = result.get("fit_mask")
    if fit_mask is not None and not bool(np.all(fit_mask)):
        w0, w1 = result.get("fit_window_ns", (t[0], t[-1]))
        for a in (ax, axr):
            if w0 > t[0]:
                a.axvspan(t[0], w0, color="0.85", alpha=0.5, zorder=0)
            if w1 < t[-1]:
                a.axvspan(w1, t[-1], color="0.85", alpha=0.5, zorder=0)
        ax.axvline(w0, color="0.5", lw=0.8, ls=":")
        ax.axvline(w1, color="0.5", lw=0.8, ls=":")

    if show:
        fcs_plottools.show_figure(fig, np.array([ax, axr]))
    return fig, np.array([ax, axr])


# ── Export ────────────────────────────────────────────────────────────────────

def _fits_dir(source_path: Path) -> Path:
    """Return (creating if needed) a 'fits' folder beside the source file."""
    base = source_path.parent
    if base.name.lower() in ("analysis", "fits"):
        base = base.parent
    out = base / "fits"
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_lifetime_fit(result: dict, source_path: str | Path) -> Tuple[Path, Path]:
    """
    Write a human-readable .txt report plus a .csv of the fitted curve.

    Returns ``(report_path, curve_path)``.
    """
    source_path = Path(source_path)
    model = result["model"]
    out_dir = _fits_dir(source_path)
    stem = f"{source_path.stem}_lifetimefit_{model.key}"
    report_path = out_dir / f"{stem}.txt"
    curve_path = out_dir / f"{stem}_curve.csv"

    der = result["derived"]
    L: list[str] = []
    L.append("Lifetime fit report")
    L.append("=" * 60)
    L.append(f"source     : {source_path.name}")
    L.append(f"model      : {model.name}  [{model.key}]")
    L.append(f"formula    : {model.formula}")
    L.append(f"fitted     : {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"points     : {result['n_points']}   "
             f"free params : {len(result['free'])}   dof : {result['dof']}")
    L.append(f"weighting  : {'Poisson σ=√N' if result['weighted'] else 'none (unweighted)'}")
    L.append(f"IRF        : {'measured (reconvolution)' if result['has_irf'] else 'none (tail fit)'}")
    win = result.get("fit_window_ns")
    if win is not None:
        L.append(f"fit window : {win[0]:.3f} – {win[1]:.3f} ns "
                 f"({result['n_points']} of {len(result['t_ns'])} bins)")
    L.append("")
    L.append("Parameters")
    L.append("-" * 60)
    L.append(f"{'name':<8}{'value':>14}{'std err':>14}  {'unit':<5} fixed")
    for n in result["names"]:
        p_unit = next((p.unit for p in model.params if p.name == n), "")
        val = result["values"][n]
        err = result["errors"][n]
        is_fixed = result["fixed"].get(n, False)
        err_str = f"{'—':>14}" if is_fixed else f"{err:>14.6g}"
        L.append(f"{n:<8}{val:>14.6g}{err_str}  {p_unit:<5} {'yes' if is_fixed else 'no'}")
    L.append("")
    L.append("Derived")
    L.append("-" * 60)
    for tn in model.tau_names:
        L.append(f"  fractional intensity f({tn}) : {der['fractions'].get(tn, float('nan')):.4f}")
    if model.n_exp > 1:
        L.append(f"  amplitude-weighted <tau>   : {der['tau_mean_amp']:.4f} ns")
        L.append(f"  intensity-weighted <tau>   : {der['tau_mean_int']:.4f} ns")
    L.append("")
    L.append("Goodness of fit")
    L.append("-" * 60)
    L.append(f"  SS_res     : {result['ss_res']:.6g}")
    L.append(f"  R^2        : {result['r2']:.6f}")
    L.append(f"  chi^2      : {result['chi2']:.6g}")
    L.append(f"  red. chi^2 : {result['red_chi2']:.6g}")
    L.append("")
    report_path.write_text("\n".join(L), encoding="utf-8")

    # Curve CSV
    cols = {
        "time_ns":   result["t_ns"],
        "decay":     result["decay"],
        "fit":       result["fit"],
        "irf":       result["irf"],
        "residual":  result["resid"],
        "resid_sigma": result["wresid"],
    }
    names = list(cols.keys())
    with curve_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# Lifetime fit curve — {model.key}\n")
        fh.write(f"# source : {source_path.name}\n")
        fh.write(f"# exported : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(",".join(names) + "\n")
        for row in zip(*(cols[n] for n in names)):
            fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

    print(f"[lifetime] wrote {report_path}")
    print(f"[lifetime] wrote {curve_path}")
    return report_path, curve_path


# ── GUI ───────────────────────────────────────────────────────────────────────

def run_reconv_fit_dialog(data: LifetimeData, parent=None):
    """
    Full GUI flow for IRF-reconvolution lifetime modelling: choose a model, set
    up the fit, then fit, plot and export.  ``data`` must be a
    :class:`fcs_ifx.LifetimeData`.
    """
    import tkinter as tk
    from tkinter import messagebox

    if getattr(data, "kind", None) != getattr(__import__("fcs_ifx"), "LIFETIME_KIND", "lifetime_decay"):
        messagebox.showinfo(
            "Lifetime fitting",
            "Lifetime fitting works on time-domain decay files (.ifx).\n"
            "The active file is not a lifetime decay dataset.",
            parent=parent,
        )
        return

    def _after_model(model: LifetimeModel):
        _lifetime_setup_dialog(parent, model, data)

    _select_lifetime_model_dialog(parent, _after_model)


def _select_lifetime_model_dialog(parent, on_choose):
    """Screen 1 — choose a lifetime model from the registry."""
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Lifetime fit — select model")
    win.geometry("540x420")
    win.minsize(460, 380)
    win.resizable(True, True)
    win.grab_set()

    tk.Label(win, text="Select a lifetime model",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    models = fcs_lifetime_models.list_models()
    key_var = tk.StringVar(value=models[0].key)

    list_frame = tk.LabelFrame(win, text="Models", padx=10, pady=6)
    list_frame.pack(fill="x", padx=12, pady=(0, 6))
    for m in models:
        tk.Radiobutton(list_frame, text=m.name, variable=key_var,
                       value=m.key, anchor="w").pack(fill="x")

    desc = tk.Text(win, height=10, wrap="word", font=("Courier", 9),
                   bg="#f7f7f7", relief="flat", padx=8, pady=6)
    desc.pack(fill="both", expand=True, padx=12, pady=(0, 6))

    def _refresh_desc(*_):
        m = fcs_lifetime_models.get_model(key_var.get())
        desc.config(state="normal")
        desc.delete("1.0", tk.END)
        desc.insert(tk.END, m.description)
        desc.config(state="disabled")

    key_var.trace_add("write", _refresh_desc)
    _refresh_desc()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        m = fcs_lifetime_models.get_model(key_var.get())
        win.destroy()
        on_choose(m)

    tk.Button(btns, text="Next →", width=12, command=_next, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


def _lifetime_setup_dialog(parent, model: LifetimeModel, data: LifetimeData):
    """Screen 2 — initial guesses, bounds and fixed flags, then fit."""
    import tkinter as tk
    from tkinter import messagebox

    t_ns, decay, irf = data.decay_curve()
    peak_idx = int(np.argmax(decay))
    guesses0 = auto_guess_lifetime(model, t_ns, decay, peak_idx)

    win = tk.Toplevel(parent)
    win.title(f"Lifetime fit — {model.name}")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text=model.name, font=("Helvetica", 12, "bold"),
             pady=6).pack()
    tk.Label(win, text=f"Data: {data.filepath.name}", font=("Helvetica", 9),
             fg="grey").pack()
    tk.Label(win, text=model.formula, font=("Courier", 9), fg="#444").pack(pady=(0, 4))
    irf_note = ("IRF: measured (iterative reconvolution)" if data.has_irf
                else "IRF: none in file — tail fit from the decay peak")
    tk.Label(win, text=irf_note, font=("Helvetica", 9),
             fg=("#2a7" if data.has_irf else "#a60")).pack(pady=(0, 6))

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

    def _fmt(x: float) -> str:
        if x == np.inf:
            return "inf"
        if x == -np.inf:
            return "-inf"
        return f"{x:.4g}"

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

    weight_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        win, text="Weight by Poisson noise (σ = √N)  — recommended for TCSPC",
        variable=weight_var, anchor="w",
    ).pack(fill="x", padx=12, pady=(6, 0))

    # Fit window (ns) — restricts which bins enter the cost / statistics.
    twin = tk.Frame(win, padx=12, pady=2)
    twin.pack(fill="x")
    tk.Label(twin, text="Fit window (ns):", font=("Helvetica", 10),
             anchor="w").grid(row=0, column=0, sticky="w")
    tk.Label(twin, text="start", font=("Helvetica", 9), fg="grey").grid(
        row=0, column=1, padx=(8, 2))
    start_var = tk.StringVar(value=f"{t_ns[0]:.3g}")
    tk.Entry(twin, textvariable=start_var, width=8).grid(row=0, column=2)
    tk.Label(twin, text="end", font=("Helvetica", 9), fg="grey").grid(
        row=0, column=3, padx=(8, 2))
    end_var = tk.StringVar(value=f"{t_ns[-1]:.3g}")
    tk.Entry(twin, textvariable=end_var, width=8).grid(row=0, column=4)
    tk.Label(
        win,
        text="Leave at the full range for the whole curve.  Narrow the window "
             "to exclude\na noisy far tail or pre-pulse region (e.g. a "
             "scatter-contaminated IRF tail).",
        font=("Helvetica", 8), fg="grey", justify="left",
    ).pack(fill="x", padx=12, pady=(0, 2))

    btns = tk.Frame(win)
    btns.pack(pady=10)

    def _parse_bound(text: str, default: float) -> float:
        s = text.strip().lower()
        if s in ("", "inf", "+inf", "infinity"):
            return np.inf if s != "" else default
        if s in ("-inf", "-infinity"):
            return -np.inf
        return float(s)

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

        # Fit window: blank or full-range entries mean "no restriction".
        def _win_val(text, fallback):
            s = text.strip()
            if s == "":
                return None
            try:
                return float(s)
            except ValueError:
                return fallback
        start_ns = _win_val(start_var.get(), None)
        end_ns = _win_val(end_var.get(), None)
        if start_ns is not None and start_ns <= t_ns[0]:
            start_ns = None
        if end_ns is not None and end_ns >= t_ns[-1]:
            end_ns = None

        try:
            result = fit_lifetime(
                model, t_ns, decay, irf,
                guesses, lowers, uppers, fixed,
                weighted=weight_var.get(), has_irf=data.has_irf,
                fit_start_ns=start_ns, fit_end_ns=end_ns,
            )
        except Exception as e:
            messagebox.showerror("Fit failed", str(e), parent=win)
            return

        win.destroy()

        report_path, _curve = export_lifetime_fit(result, data.filepath)
        fig, _axes = plot_lifetime_fit(result, data.filepath.name, show=False)
        try:
            fig.savefig(report_path.with_suffix(".png"), dpi=150)
        except Exception as e:
            print(f"[lifetime] could not save figure: {e}")

        # Summary
        slines = []
        for i, tn in enumerate(model.tau_names):
            tv = result["values"][tn]
            te = result["errors"][tn]
            tag = "" if result["fixed"].get(tn) else f" ± {te:.2g}"
            slines.append(f"τ{i+1} = {tv:.3g}{tag} ns")
        if model.n_exp > 1:
            slines.append(f"⟨τ⟩amp = {result['derived']['tau_mean_amp']:.3g} ns")
        summary = "\n".join(slines)
        messagebox.showinfo(
            "Lifetime fit complete",
            f"{model.name}\n\n{summary}\n\n"
            f"red. χ² = {result['red_chi2']:.3g}\n\n"
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
    from fcs_ifx import read_ifx

    if len(sys.argv) < 2:
        print("Usage: python fcs_lifetime_recon.py <file.ifx> [model_key] "
              "[start_ns] [end_ns]")
        print("Models:")
        for m in fcs_lifetime_models.list_models():
            print(f"  {m.key:<18} {m.name}")
        sys.exit(1)

    d = read_ifx(sys.argv[1])
    key = sys.argv[2] if len(sys.argv) > 2 else fcs_lifetime_models.list_models()[0].key
    model = fcs_lifetime_models.get_model(key)
    start_ns = float(sys.argv[3]) if len(sys.argv) > 3 else None
    end_ns = float(sys.argv[4]) if len(sys.argv) > 4 else None

    t_ns, decay, irf = d.decay_curve()
    peak_idx = int(np.argmax(decay))
    guesses = auto_guess_lifetime(model, t_ns, decay, peak_idx)
    lowers = {p.name: p.lower for p in model.params}
    uppers = {p.name: p.upper for p in model.params}
    fixed  = {p.name: p.fixed for p in model.params}

    result = fit_lifetime(model, t_ns, decay, irf, guesses, lowers, uppers,
                          fixed, weighted=True, has_irf=d.has_irf,
                          fit_start_ns=start_ns, fit_end_ns=end_ns)
    export_lifetime_fit(result, d.filepath)
    print("\nFitted lifetimes:")
    for i, tn in enumerate(model.tau_names):
        print(f"  tau{i+1} = {result['values'][tn]:.4g} ns  "
              f"(f={result['derived']['fractions'][tn]:.3f})")
    if model.n_exp > 1:
        print(f"  <tau>_amp = {result['derived']['tau_mean_amp']:.4g} ns")
    print(f"  red chi2 = {result['red_chi2']:.4g}   R2 = {result['r2']:.5f}")
