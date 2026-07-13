"""
fcs_calib.py
============
Effective-volume calibration from a global-fit parameter table.

Idea
----
After a global fit, each dataset has a mean occupancy <N> = 1/G0 (read from the
amplitude, so it does NOT depend on the poorly-determined structure parameter
kappa).  Given a *known* concentration C for each dataset, plot <N> against C
and fit a straight line through the origin:

    <N> = s · C          s = N_A · V_eff   (molecules per concentration unit)

The calibration constant is reported as alpha, defined by

    C = alpha · <N>      alpha = 1 / s

so an unknown experimental concentration can later be read directly from its
fitted <N>, with no dependence on kappa or the absolute optical geometry.

This is the first instance of the general "fit parameter vs external variable"
tool; for now it is specialised to <N>-vs-concentration volume calibration.

Input is the *_params.csv written by fcs_fit.export_global_fit (it carries the
N and N_err columns).

Dependencies
------------
    pip install numpy matplotlib
"""

#compatibility feature ... maybe not needed
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import fcs_plottools

# <N> = f(unit) · C · Veff[µm³].  (Veff in µm³ = 1e-15 L, N_A = 6.022e23 /mol)
_CONC_FACTOR = {
    "M":  6.022e8,
    "mM": 6.022e5,
    "uM": 602.2,
    "µM": 602.2,
    "nM": 0.6022,
    "pM": 6.022e-4,
}


# ── Parameter-table loading ───────────────────────────────────────────────────

def load_params_csv(path) -> Tuple[List[dict], Dict[str, str]]:
    """
    Read a *_params.csv parameter table.  Returns (rows, meta) where each row
    is a dict keyed by column name (numeric columns parsed to float, 'dataset'
    kept as a string) and meta is the parsed ``# key : value`` header block.
    """
    path = Path(path)
    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    meta: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                body = line[1:].strip()
                if ":" in body:
                    k, v = body.split(":", 1)
                    meta[k.strip()] = v.strip()
                continue
            if not line.strip():
                continue
            cells = [c.strip() for c in line.rstrip("\n").split(",")]
            if header is None:
                header = cells
                continue
            rows.append(cells)

    if not header:
        raise ValueError(f"No header row found in {path.name}.")

    out: List[dict] = []
    for r in rows:
        d: dict = {}
        for k, v in zip(header, r):
            if k == "dataset":
                d[k] = v
            else:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = np.nan
        out.append(d)
    return out, meta


# ── Line fits ─────────────────────────────────────────────────────────────────

def fit_through_origin(x, y, yerr=None) -> Tuple[float, float, float]:
    """
    Weighted least-squares fit of y = s·x (no intercept).
    Returns (slope, slope_err, r2).  Weights are 1/yerr² when yerr is given.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if yerr is not None:
        yerr = np.asarray(yerr, float)
        w = np.where(yerr > 0, 1.0 / np.square(yerr), np.nan)
    else:
        w = np.ones_like(x)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 1:
        raise ValueError("No valid points to fit.")

    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)
    s = Sxy / Sxx

    if yerr is not None:
        s_err = float(np.sqrt(1.0 / Sxx))
    else:
        resid = y - s * x
        dof = max(1, len(x) - 1)
        s2 = np.sum(resid ** 2) / dof
        s_err = float(np.sqrt(s2 / Sxx))

    resid = y - s * x
    ss_res = np.sum(w * resid ** 2)
    ss_tot = np.sum(w * y * y)          # about zero, for a through-origin model
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return float(s), s_err, r2


def fit_with_intercept(x, y, yerr=None) -> Tuple[float, float]:
    """Diagnostic free-intercept fit. Returns (slope, intercept)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if yerr is not None:
        yerr = np.asarray(yerr, float)
        w = np.where(yerr > 0, 1.0 / np.square(yerr), np.nan)
    else:
        w = np.ones_like(x)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 2:
        return float("nan"), float("nan")
    Sw = np.sum(w); Sx = np.sum(w * x); Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x); Sxy = np.sum(w * x * y)
    den = Sw * Sxx - Sx * Sx
    if den == 0:
        return float("nan"), float("nan")
    slope = (Sw * Sxy - Sx * Sy) / den
    intercept = (Sxx * Sy - Sx * Sxy) / den
    return float(slope), float(intercept)


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(
    names: List[str],
    N: np.ndarray,
    N_err: Optional[np.ndarray],
    conc: np.ndarray,
    unit: str = "nM",
    use_weights: bool = True,
    corrected: bool = False,
) -> dict:
    """
    Fit <N> = s·C through the origin and derive alpha (C = alpha·<N>) and,
    when the unit is recognised, the effective volume V_eff.

    ``corrected`` records whether the supplied <N> is background-corrected
    (for labelling only).
    """
    N = np.asarray(N, float)
    conc = np.asarray(conc, float)
    have_err = (N_err is not None and np.all(np.isfinite(N_err)) and np.all(N_err > 0))
    yerr = N_err if (have_err and use_weights) else None

    s, s_err, r2 = fit_through_origin(conc, N, yerr)
    slope_free, intercept = fit_with_intercept(conc, N, yerr)

    alpha = 1.0 / s
    alpha_err = s_err / (s * s)

    veff = None
    f = _CONC_FACTOR.get(unit)
    if f:
        veff = s / f                    # µm³

    return {
        "names": names, "N": N, "N_err": N_err, "conc": conc, "unit": unit,
        "slope": s, "slope_err": s_err, "r2": r2,
        "alpha": alpha, "alpha_err": alpha_err,
        "intercept": intercept, "slope_free": slope_free,
        "veff_um3": veff, "weighted": yerr is not None,
        "corrected": corrected,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_calibration(result: dict, show: bool = True):
    conc = result["conc"]
    N = result["N"]
    N_err = result["N_err"]
    unit = result["unit"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")

    if N_err is not None and np.all(np.isfinite(N_err)):
        ax.errorbar(conc, N, yerr=N_err, fmt="o", color="steelblue",
                    capsize=3, markersize=5, label="data", zorder=3)
    else:
        ax.plot(conc, N, "o", color="steelblue", markersize=5,
                label="data", zorder=3)

    xline = np.linspace(0, conc.max() * 1.05, 100)
    ax.plot(xline, result["slope"] * xline, color="tomato", linewidth=1.6,
            label="fit ⟨N⟩ = s·C", zorder=2)
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)

    box = [
        f"s     = {result['slope']:.4g} ± {result['slope_err']:.2g}  ⟨N⟩/{unit}",
        f"α     = {result['alpha']:.4g} ± {result['alpha_err']:.2g}  {unit}/molecule",
    ]
    if result["veff_um3"] is not None:
        box.append(f"V_eff = {result['veff_um3']:.4g} µm³")
    box.append(f"R²    = {result['r2']:.5f}")
    box.append(f"intercept (free) = {result['intercept']:.3g}")
    ax.text(0.03, 0.97, "\n".join(box), transform=ax.transAxes,
            ha="left", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    ax.set_xlabel(f"Known concentration C ({unit})", fontsize=12)
    ax.set_ylabel("Mean occupancy ⟨N⟩"
                  + ("  (bg-corrected)" if result.get("corrected") else " = 1/G0"),
                  fontsize=12)
    ax.set_title("Effective-volume calibration  ·  C = α·⟨N⟩", fontsize=11)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    if show:
        #plt.show was static; now dynamic w/ fcs_plottools
        #plt.show()
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── Export ────────────────────────────────────────────────────────────────────

def _calib_dir(source_path: Path) -> Path:
    base = source_path.parent
    if base.name.lower() == "fits":
        base = base.parent
    out = base / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    return out


def export_calibration(result: dict, source_path) -> Tuple[Path, Path]:
    """Write a calibration report (.txt) and the plotted points (.csv)."""
    source_path = Path(source_path)
    out_dir = _calib_dir(source_path)
    stem = f"{source_path.stem}_calibration"
    report_path = out_dir / f"{stem}.txt"
    points_path = out_dir / f"{stem}_points.csv"

    unit = result["unit"]
    L = []
    L.append("FCS effective-volume calibration")
    L.append("=" * 60)
    L.append(f"source     : {source_path.name}")
    L.append(f"fitted     : {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"model      : <N> = s·C through the origin")
    L.append(f"<N> source : {'background-corrected (N_corr)' if result.get('corrected') else 'raw (1/G0)'}")
    L.append(f"weighted   : {'yes (1/N_err²)' if result['weighted'] else 'no'}")
    L.append("")
    L.append(f"  slope s        = {result['slope']:.6g} ± {result['slope_err']:.3g}"
             f"  (molecules per {unit})")
    L.append(f"  alpha          = {result['alpha']:.6g} ± {result['alpha_err']:.3g}"
             f"  ({unit} per molecule)   [C = alpha · <N>]")
    if result["veff_um3"] is not None:
        L.append(f"  V_eff          = {result['veff_um3']:.6g} µm³")
    L.append(f"  R^2            = {result['r2']:.6f}")
    L.append(f"  free intercept = {result['intercept']:.6g}   "
             f"(should be ≈ 0; large values hint at background)")
    L.append("")
    L.append("Points")
    L.append("-" * 60)
    L.append(f"{'dataset':<24}{unit:>10}{'<N>':>12}{'<N>_err':>12}")
    for i, nm in enumerate(result["names"]):
        ne = (result["N_err"][i] if result["N_err"] is not None else float("nan"))
        L.append(f"{nm:<24}{result['conc'][i]:>10.4g}{result['N'][i]:>12.4g}{ne:>12.4g}")
    report_path.write_text("\n".join(L), encoding="utf-8")

    with points_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# FCS volume calibration points\n")
        fh.write(f"# unit : {unit}\n")
        fh.write(f"# C = alpha * <N> ; alpha = {result['alpha']:.6g} {unit}/molecule\n")
        fh.write("dataset,concentration,N,N_err,N_fit\n")
        for i, nm in enumerate(result["names"]):
            ne = (result["N_err"][i] if result["N_err"] is not None else float("nan"))
            nfit = result["slope"] * result["conc"][i]
            fh.write(f"{nm},{result['conc'][i]:.10g},{result['N'][i]:.10g},"
                     f"{ne:.10g},{nfit:.10g}\n")

    print(f"[calib] wrote {report_path}")
    print(f"[calib] wrote {points_path}")
    return report_path, points_path


# ── GUI ───────────────────────────────────────────────────────────────────────

def run_calibration_dialog(parent=None, init_dir=None):
    """
    Pick a *_params.csv, enter the known concentration for each dataset, then
    fit <N> vs C, plot, and export the calibration.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    init_dir = Path(init_dir) if init_dir else Path.cwd()
    csv_path = filedialog.askopenfilename(
        title="Select a fit parameter table (*_params.csv)",
        initialdir=str(init_dir),
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        parent=parent,
    )
    if not csv_path:
        return
    csv_path = Path(csv_path)

    try:
        rows, _meta = load_params_csv(csv_path)
    except Exception as e:
        messagebox.showerror("Could not read table", str(e), parent=parent)
        return
    if not rows:
        messagebox.showerror("Empty table", "No data rows found.", parent=parent)
        return
    if "N" not in rows[0]:
        messagebox.showerror(
            "No ⟨N⟩ column",
            "This table has no 'N' column.  Re-export the global fit so the "
            "parameter table includes ⟨N⟩ = 1/G0.",
            parent=parent)
        return

    # Prefer background-corrected occupancy when the table provides it.
    use_corr = ("N_corr" in rows[0]
                and np.all(np.isfinite([r.get("N_corr", np.nan) for r in rows])))
    ncol  = "N_corr" if use_corr else "N"
    necol = "N_corr_err" if use_corr else "N_err"

    names = [r["dataset"] for r in rows]
    N     = np.array([r.get(ncol, np.nan) for r in rows], float)
    N_err = np.array([r.get(necol, np.nan) for r in rows], float)
    #has_err = np.all(np.isfinite(N_err)) and np.all(N_err > 0)
    has_err = bool(np.all(np.isfinite(N_err)) and np.all(N_err > 0))
    win = tk.Toplevel(parent)
    win.title("Volume calibration — ⟨N⟩ vs concentration")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="Enter the known concentration for each dataset",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text=f"Source: {csv_path.name}",
             font=("Helvetica", 9), fg="grey").pack()
    tk.Label(win,
             text=("Using background-corrected ⟨N⟩ (N_corr)" if use_corr
                   else "Using raw ⟨N⟩ = 1/G0"),
             font=("Helvetica", 9),
             fg=("seagreen" if use_corr else "grey")).pack()

    top = tk.Frame(win, padx=12, pady=4)
    top.pack(fill="x")
    tk.Label(top, text="Concentration unit:", anchor="e").pack(side="left")
    unit_var = tk.StringVar(value="nM")
    tk.OptionMenu(top, unit_var, "pM", "nM", "µM", "uM", "mM", "M").pack(side="left", padx=6)

    table = tk.Frame(win, padx=12, pady=4)
    table.pack(fill="x")
    tk.Label(table, text="dataset", font=("Helvetica", 10, "bold")).grid(
        row=0, column=0, sticky="w", padx=4, pady=(0, 4))
    tk.Label(table, text="⟨N⟩", font=("Helvetica", 10, "bold")).grid(
        row=0, column=1, padx=4)
    tk.Label(table, text="concentration", font=("Helvetica", 10, "bold")).grid(
        row=0, column=2, padx=4)

    conc_vars = []
    for r, nm in enumerate(names, start=1):
        tk.Label(table, text=nm, anchor="w", width=22,
                 font=("Courier", 9)).grid(row=r, column=0, sticky="w", padx=4)
        nstr = f"{N[r-1]:.3g}" + (f" ± {N_err[r-1]:.2g}" if has_err else "")
        tk.Label(table, text=nstr, anchor="e", width=14, fg="grey").grid(
            row=r, column=1, padx=4)
        cv = tk.StringVar(value="")
        tk.Entry(table, textvariable=cv, width=12).grid(row=r, column=2, padx=4)
        conc_vars.append(cv)

    weight_var = tk.BooleanVar(value=has_err)
    tk.Checkbutton(
        win,
        text="Weight by 1/⟨N⟩_err²" if has_err
        else "Weight — unavailable (no N_err column)",
        variable=weight_var, anchor="w",
        state="normal" if has_err else "disabled",
    ).pack(fill="x", padx=12, pady=(4, 0))

    btns = tk.Frame(win)
    btns.pack(pady=10)

    def _do_fit():
        try:
            conc = np.array([float(cv.get()) for cv in conc_vars], float)
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Enter a numeric concentration for every dataset.",
                                 parent=win)
            return
        if np.any(conc < 0):
            messagebox.showerror("Invalid input",
                                 "Concentrations must be ≥ 0.", parent=win)
            return
        unit = unit_var.get()
        try:
            result = calibrate(names, N, N_err if has_err else None, conc,
                               unit=unit, use_weights=weight_var.get(),
                               corrected=use_corr)
        except Exception as e:
            messagebox.showerror("Calibration failed", str(e), parent=win)
            return

        win.destroy()
        report_path, _pts = export_calibration(result, csv_path)
        fig, _ax = plot_calibration(result, show=False)
        try:
            fig.savefig(report_path.with_suffix(".png"), dpi=150)
        except Exception as e:
            print(f"[calib] could not save figure: {e}")

        msg = (f"C = α·⟨N⟩\n\n"
               f"α = {result['alpha']:.4g} ± {result['alpha_err']:.2g} {unit}/molecule\n"
               f"s = {result['slope']:.4g} ⟨N⟩/{unit}\n")
        if result["veff_um3"] is not None:
            msg += f"V_eff = {result['veff_um3']:.4g} µm³\n"
        msg += f"R² = {result['r2']:.5f}\n\nSaved to:\n{report_path.parent}"
        messagebox.showinfo("Calibration complete", msg, parent=parent)
        #plt.show was the static version; fcs_plottools is dynamic
        #plt.show()
        fcs_plottools.show_figure(fig, _ax)

    tk.Button(btns, text="Fit calibration", width=16, command=_do_fit,
              pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(side="left", padx=6)

    win.wait_window()


# ── Cmd Line Interface ───────────────────────────────────────────────────────────────────────
# ──If the __name__ is "__main__" that means we are running from command line──────────────────

if __name__ == "__main__":
    import sys
    #So, the cmd line arguments are stored in "sys.argv"
    if len(sys.argv) < 3:
        print("Usage: python fcs_calib.py <params.csv> <c1,c2,...> [unit]")
        print("  concentrations: one per dataset row, comma-separated")
        sys.exit(1)
    rows, _ = load_params_csv(sys.argv[1])
    names = [r["dataset"] for r in rows]
    N = np.array([r.get("N", np.nan) for r in rows], float)
    N_err = np.array([r.get("N_err", np.nan) for r in rows], float)
    conc = np.array([float(x) for x in sys.argv[2].split(",")], float)
    unit = sys.argv[3] if len(sys.argv) > 3 else "nM"
    res = calibrate(names, N, N_err if np.all(np.isfinite(N_err)) else None,
                    conc, unit=unit)
    export_calibration(res, Path(sys.argv[1]))
    plot_calibration(res)
