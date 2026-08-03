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

The PCH fitter bins the photon stream itself, which keeps the raw per-count
frequencies and the number of sampled bins M.  M is needed to weight the fit by
the Poisson error of each histogram bin (σ = √counts), which is what makes the
reduced χ² meaningful — the same statistics family as the correlation and
lifetime fitters.

Binning runs over the acquisition window from fcs_pch.acquisition_window, not
over each channel's own first-to-last photon.  Time in which the detector was
running and saw nothing is a measurement of k = 0 and is binned as such; the
older photon-span window silently dropped those bins, which biased p(0) low and
gave the two channels different M for one shared acquisition.

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
import fcs_export
import fcs_fisher

from fcs_fitcommon import (
    fits_dir as _fits_dir,
    new_fit_dir as _new_fit_dir,
    fmt_bound as _fmt,
    parse_bound as _parse_bound,
    write_params_table as _write_params_table,
)

# ── Data preparation ──────────────────────────────────────────────────────────

def pch_counts(
    times_s: np.ndarray,
    bin_width_s: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, int, float, float]:
    """
    Build the photon counting histogram as RAW per-count frequencies.

    Like fcs_pch.compute_pch, but returns the unnormalised frequencies n_k (the
    number of time bins containing exactly k photons) and the number of sampled
    bins M, which the fit needs for Poisson weighting.

    This is now a thin alias for :func:`fcs_pch.compute_pch_counts`.  The two
    were separate copies of the same arithmetic, which is precisely the
    arrangement that lets a fit and the export of the same data drift apart;
    one definition cannot.  Pass ``t_start`` / ``t_end`` -- normally from
    :func:`fcs_pch.acquisition_window` -- so that observed empty time at the
    head and tail of the acquisition is binned honestly as k = 0 instead of
    being trimmed off with the window.

    Returns
    -------
    k    : np.ndarray (int)    count values 0..k_max
    n_k  : np.ndarray (float)  number of time bins with exactly k photons (Σ = M)
    M    : int                 number of sampled time bins
    mean : float               mean photons per bin  <k>
    var  : float               variance of photons per bin  Var(k)
    """
    return fcs_pch.compute_pch_counts(
        times_s, bin_width_s, t_start=t_start, t_end=t_end)


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


# ── Reading an exported PCH histogram ─────────────────────────────────────────

def load_pch_csv(path) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    Read a PCH exported by the "PCH" plotting task.

    The PCH counterpart of :func:`fcs_fit.load_correlation_csv` and
    :func:`fcs_lifetime_fit.load_lifetime_csv`, and it exists for the same
    reason: once a histogram has been exported you should be able to fit it
    without carrying the .fcs photon records around.

    What makes a PCH export fittable
    --------------------------------
    Probabilities alone are not enough.  A weighted fit compares M*Pi(k) with
    the observed frequencies n_k and weights each bin by its Poisson error
    sigma = sqrt(n_k), so it needs the RAW counts and the number of sampled
    time bins M.  Neither can be recovered from p(k) with certainty, so both
    are read from the file and never inferred: an export that predates the
    ``n_<label>`` columns is rejected with an explanation rather than fitted on
    a reconstructed M that might be wrong.

    Returns
    -------
    series : dict of label -> dict
        One entry per exported series ('Ch1', 'Ch2', 'Ch1+Ch2'), each holding
        ``k``, ``n_k``, ``M``, ``mean``, ``var``, ``cps``, ``bin_width_s``.
    meta : dict
        The header ``# key : value`` fields.

    Raises
    ------
    ValueError
        If the file is not a fittable PCH export, with a message that says what
        was found instead.
    """
    path = Path(path)
    meta, columns = fcs_export.read_export(path)

    if "k" not in columns:
        raise ValueError(
            f"{path.name} has no 'k' column - is this a PCH export?")

    # A fit CURVE export also has a k column, so name that case specifically:
    # picking one out of a fits/ folder is an easy mistake from a browser.
    if "pk_fit" in columns or "counts_fit" in columns:
        raise ValueError(
            f"{path.name} looks like a PCH FIT curve, not a measured "
            f"histogram.  Fit the histogram written by the 'PCH' plotting "
            f"task instead."
        )

    k_all = np.asarray(columns["k"], dtype=np.float64)

    labels = [c[2:] for c in columns if c.startswith("n_")]
    if not labels:
        pk_labels = [c[3:] for c in columns if c.startswith("pk_")]
        if pk_labels:
            raise ValueError(
                f"{path.name} carries probabilities p(k) but not the raw "
                f"counts, so it cannot be fitted: the Poisson weight of each "
                f"bin, and the reduced chi-squared that depends on it, need "
                f"the frequencies n_k and the sampled-bin count M.\n\n"
                f"Re-export this file's PCH from the 'PCH' task to write a "
                f"fittable histogram."
            )
        raise ValueError(
            f"{path.name} has a k column but no 'n_<channel>' counts column; "
            f"this does not look like a PCH export.")

    bw = _meta_float(meta, "bin_width_s")
    if bw is None or not np.isfinite(bw) or bw <= 0:
        raise ValueError(
            f"{path.name} does not record a usable 'bin_width_s' in its "
            f"header.  The bin width sets the brightness scale (epsilon is "
            f"counts per molecule per bin), so a fit without it would report "
            f"a number with no unit behind it.")

    series: Dict[str, dict] = {}
    for label in labels:
        n_col = np.asarray(columns[f"n_{label}"], dtype=np.float64)
        good  = np.isfinite(n_col)
        if not good.any():
            continue
        k_ser = k_all[good].astype(int)
        n_ser = n_col[good]

        # The exporter pads only at the TAIL, so a series' own range must run
        # 0,1,2,... without gaps.  A gap means the file was edited or merged;
        # fitting it would silently drop count values from the histogram.
        if k_ser[0] != 0 or not np.array_equal(k_ser, np.arange(len(k_ser))):
            raise ValueError(
                f"{path.name}: the '{label}' counts do not cover a contiguous "
                f"k = 0,1,2,... range (found {k_ser[0]}..{k_ser[-1]} with "
                f"{len(k_ser)} values).  A PCH must be complete from k = 0 to "
                f"be fitted.")

        M = _meta_float(meta, f"{label}_sampled_bins_M")
        if M is None or not np.isfinite(M) or M <= 0:
            raise ValueError(
                f"{path.name} does not record '{label}_sampled_bins_M'.  M is "
                f"the number of time bins sampled; it sets every Poisson "
                f"weight and the reduced chi-squared scale, and it cannot be "
                f"recovered from p(k) with certainty.  Re-export this file's "
                f"PCH to record it.")
        M = int(round(M))

        # sum(n_k) == M is the checksum that catches a truncated or mis-padded
        # column.  Off by one bin is not a rounding artefact here: the counts
        # are integers and M is written as an integer.
        total = float(n_ser.sum())
        if abs(total - M) > 0.5:
            raise ValueError(
                f"{path.name}: the '{label}' counts sum to {total:.0f} but the "
                f"header records M = {M} sampled bins.  The histogram is "
                f"incomplete or the header does not match it; refusing to fit "
                f"rather than weight the points by a wrong M.")

        # Moments are RECOMPUTED from the counts rather than read from the
        # header.  n_k and M are integers, so the recomputation is exact to
        # double precision, whereas the header is rounded to 10 significant
        # figures on the way out -- reading it back would hand the fit a <k>
        # differing from the photon-record value in the last few digits and
        # make an otherwise identical fit look non-reproducible.
        #
        # The header value is still used, as a checksum: if it disagrees by
        # more than its own rounding, the header is not describing these
        # counts and something is wrong with the file.
        mean = float(np.sum(n_ser * k_ser) / M)
        var  = (float(np.sum(n_ser * (k_ser - mean) ** 2) / (M - 1))
                if M > 1 else float("nan"))
        for name, got, key in (("mean", mean, f"{label}_mean"),
                               ("variance", var, f"{label}_var")):
            hdr = _meta_float(meta, key)
            if (hdr is not None and np.isfinite(hdr) and np.isfinite(got)
                    and abs(hdr - got) > 1e-6 * max(abs(got), 1e-30)):
                raise ValueError(
                    f"{path.name}: the '{label}' counts give a {name} of "
                    f"{got:.10g}, but the header records {hdr:.10g}.  The "
                    f"header does not describe these counts; refusing to fit."
                )

        series[label] = {
            "k":           k_ser,
            "n_k":         n_ser,
            "M":           M,
            "mean":        mean,
            "var":         var,
            "cps":         _meta_float(meta, f"{label}_observed_cps"),
            "bin_width_s": float(bw),
        }

    if not series:
        raise ValueError(
            f"{path.name} has counts columns but none of them hold data.")

    return series, meta


def _meta_float(meta: dict, key: str) -> Optional[float]:
    """Read one numeric header field, or None if absent/unparseable."""
    v = meta.get(key)
    if v is None:
        return None
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


class LoadedPCH:
    """
    A PCH read back from an exported CSV.

    Deliberately thin.  Unlike a loaded lifetime decay -- which has to
    impersonate FCSData well enough for the interactive gate picker to work --
    a PCH fit needs only a name to put on the plot, a path to anchor its output
    to, and the per-series arrays.  There is nothing to rebin and nothing to
    re-window: the histogram in the file IS the fit's input.
    """

    kind = "loaded_pch"

    def __init__(self, path, series: Dict[str, dict], meta: Dict[str, str]):
        self.filepath = Path(path)
        self.series   = dict(series)
        self.meta     = dict(meta)
        self.params   = dict(meta)

    @property
    def labels(self) -> list:
        """Exported series names, Ch1 / Ch2 first and combined last."""
        order = {"Ch1": 0, "Ch2": 1, "Ch1+Ch2": 2}
        return sorted(self.series, key=lambda L: (order.get(L, 3), L))

    @property
    def source_name(self) -> str:
        """Name of the .fcs the histogram was originally measured from."""
        return (self.meta.get("source file")
                or self.meta.get("source")
                or self.filepath.name)

    @property
    def bin_width_s(self) -> float:
        return float(next(iter(self.series.values()))["bin_width_s"])

    def get(self, label: str) -> dict:
        if label not in self.series:
            raise ValueError(
                f"{self.filepath.name} holds {', '.join(self.labels)}; "
                f"there is no '{label}' series in it.")
        return self.series[label]

    def describe(self, label: str) -> str:
        """One-line summary for a picker row."""
        s = self.get(label)
        q = (s["var"] / s["mean"] - 1.0) if s["mean"] else float("nan")
        return (f"{label:<8} <k>={s['mean']:.3g}  Q={q:.3g}  "
                f"M={s['M']:,}  k=0..{s['k'][-1]}")


def load_pch_object(path) -> LoadedPCH:
    """Read an exported PCH and wrap it as a :class:`LoadedPCH`."""
    series, meta = load_pch_csv(path)
    return LoadedPCH(path, series, meta)


def discover_pch_csvs(folder) -> list:
    """
    Return the CSVs in *folder* that parse as fittable PCH exports.

    Matches on content rather than filename, exactly as
    fcs_fit._discover_correlation_csvs and
    fcs_lifetime_fit.discover_lifetime_csvs do, so a renamed file is still
    found and a correlation or lifetime export in the same analysis folder is
    still skipped.
    """
    folder = Path(folder)
    found = []
    if folder.exists():
        for p in sorted(folder.glob("*.csv")):
            try:
                load_pch_csv(p)
                found.append(p)
            except Exception:
                continue
    return found


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
        "pcov": pcov,
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

def export_pch_fit(result: dict, source_path: str | Path
                   ) -> Tuple[Path, Path, Path]:
    """
    Write a fit's report (.txt), fitted curve (.csv) and parameter table
    (.csv + .xlsx) to the 'fits' folder.

    Returns (report_path, curve_path, params_path).

    The parameter table is the artefact the other three fitters have always
    written and this one did not: a machine-readable row of value/_err pairs
    that can be collected across fits, plus the .xlsx mirror for reading.  The
    .txt report is for a human reading ONE fit; a table is what you need when
    comparing twenty, and re-typing numbers out of report files is how
    transcription errors get into a figure.
    """
    source_path = Path(source_path)
    model = result["model"]

    out_dir = _new_fit_dir(source_path)
    ch = result.get("channel")
    report_path = out_dir / "pch_fit_report.txt"
    curve_path  = out_dir / "pch_fit_curve.csv"
    params_path = out_dir / "pch_fit_params.csv"



    L: list[str] = []
    L.append("PCH fit report")
    L.append("=" * 60)
    L.append(f"source     : {source_path.name}")
    if result.get("origin") and result["origin"] != source_path.name:
        # Fitting a saved histogram: name the .fcs the counts really came from,
        # so the report identifies the measurement and not just the file that
        # happened to be on disk.
        L.append(f"measured from : {result['origin']}")
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
    L.extend(fcs_fisher.format_report_lines(
        fcs_fisher.fisher_from_covariance(
            result.get("pcov"), result["free"], result["weighted"])))
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

    # ── Parameter table (.csv + .xlsx) ────────────────────────────────────────
    # Column order mirrors the lifetime and correlation tables: identifying
    # columns, then every model parameter as a value/_err pair, then derived
    # quantities, then goodness of fit.  A reader who knows one of these tables
    # knows all of them.
    p_header: list = ["source", "channel", "model", "bin_width_s",
                      "sampled_bins_M"]
    for nm in result["names"]:
        p_header += [nm, f"{nm}_err"]
    derived_keys = list(result["derived"].keys())
    p_header += derived_keys
    p_header += ["mean_k", "var_k", "Q", "weighted", "r2", "chi2", "red_chi2",
                 "n_points", "dof"]

    row: list = [
        source_path.name,
        (str(ch) if ch is not None else ""),
        model.key,
        float(result["bin_width_s"]) if result.get("bin_width_s") else float("nan"),
        int(result["M"]),
    ]
    for nm in result["names"]:
        row.append(float(result["values"][nm]))
        # A fixed parameter has no fitted error.  NaN rather than 0, so the
        # cell reads back as "not estimated" instead of "estimated to be
        # exactly zero" -- params_cell renders it blank.
        row.append(float("nan") if result["fixed"].get(nm)
                   else float(result["errors"][nm]))
    for kname in derived_keys:
        row.append(float(result["derived"][kname]))
    row += [
        float(result["mean"]), float(result["var"]), float(result["Q"]),
        "yes" if result["weighted"] else "no",
        float(result["r2"]),
        float(result["chi2"]) if result["weighted"] else float("nan"),
        float(result["red_chi2"]) if result["weighted"] else float("nan"),
        int(result["n_points"]), int(result["dof"]),
    ]

    comments: list = []
    comments.append("PCH fit - parameter table")
    comments.append(f"source : {source_path.name}")
    if result.get("origin") and result["origin"] != source_path.name:
        comments.append(f"measured_from : {result['origin']}")
    comments.append(f"model : {model.name} [{model.key}]")
    comments.append(f"formula : {model.formula}")
    comments.append(f"exported : {datetime.now().isoformat(timespec='seconds')}")
    comments.append("weighted : " + ("yes (Poisson sigma = sqrt(counts))"
                                     if result["weighted"] else "no"))
    fixed_names = [nm for nm in result["names"] if result["fixed"].get(nm)]
    comments.append(f"fixed : {', '.join(fixed_names) if fixed_names else '(none)'}")
    # Distinct keys: fcs_export.read_export parses "# key : value" into a dict,
    # so repeated keys overwrite each other and only the last would survive.
    comments.append("units_epsilon : epsilon is counts per molecule per BIN; "
                    "eta = epsilon / bin_width_s is counts per second and is "
                    "the bin-width-independent brightness")
    comments.append("note_fixed : blank *_err means the parameter was held fixed")
    comments.append("note_Q : Q = Var/<k> - 1, the excess of the counting "
                    "variance over Poisson; Q = 0 means no brightness "
                    "information is present")

    _write_params_table(params_path, comments, p_header, [row],
                        log_tag="pch fit")

    return report_path, curve_path, params_path

# ── GUI: entry point and dialogs ──────────────────────────────────────────────

def run_pch_fit_dialog(fcs_data: Optional[FCSData] = None, parent=None):
    """
    Full GUI flow: choose a source, then a model, then fit, plot and export.

    Two sources are offered.  Photon records give the full choice of channel
    and bin width, because the histogram is built on the spot.  A saved PCH CSV
    fixes both -- they were decided when it was exported -- so that screen is
    skipped and only the series is picked.  With no photon data loaded the
    chooser is bypassed straight to the CSV browser, since there is nothing
    else to fit.
    """
    def _from_photons():
        def _after_data(channel, bin_width_s, k, n_k, M, mean, var):
            def _after_model(model: FCSModel):
                _pch_setup_dialog(parent, fcs_data, model, channel, bin_width_s,
                                  k, n_k, M, mean, var,
                                  observed_cps=channel_cps(fcs_data, channel))
            _select_pch_model_dialog(parent, _after_model)
        _pch_data_dialog(parent, fcs_data, _after_data)

    def _from_csv():
        def _after_series(loaded: "LoadedPCH", label: str):
            s = loaded.get(label)
            def _after_model(model: FCSModel):
                _pch_setup_dialog(parent, loaded, model, label,
                                  s["bin_width_s"], s["k"], s["n_k"], s["M"],
                                  s["mean"], s["var"], observed_cps=s["cps"])
            _select_pch_model_dialog(parent, _after_model)
        _pch_csv_dialog(parent, fcs_data, _after_series)

    if fcs_data is None:
        _from_csv()
        return
    _pch_source_dialog(parent, fcs_data, _from_photons, _from_csv)


def _pch_source_dialog(parent, fcs_data, on_photons, on_csv):
    """Screen 0 — fit the active file's photon records, or a saved histogram."""
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("PCH fit — source")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="PCH fit — choose the data",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    body = tk.Frame(win, padx=16, pady=4)
    body.pack(fill="x")

    def _go(fn):
        win.destroy()
        fn()

    tk.Button(body, text="Active file — photon records", width=32, pady=6,
              command=lambda: _go(on_photons)).pack(pady=3)
    tk.Label(body, text=f"{fcs_data.filepath.name}\n"
                        f"choose channel and bin width, histogram built now",
             font=("Helvetica", 8), fg="grey", justify="center").pack()

    tk.Button(body, text="Saved PCH histogram (CSV)…", width=32, pady=6,
              command=lambda: _go(on_csv)).pack(pady=(10, 3))
    tk.Label(body, text="channel and bin width are fixed by the export",
             font=("Helvetica", 8), fg="grey").pack()

    tk.Button(win, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(pady=10)
    win.wait_window()


# Folder the CSV browser last worked in, remembered for the session.
_last_pch_browse_dir: Optional[str] = None


def _default_pch_dir(fcs_data=None) -> Optional[Path]:
    """
    Folder the PCH CSV browser should open on.

    Same preference order as fcs_fit._default_dataset_dir: the active file's
    'analysis' folder (where PCH exports are written), then beside it, then
    the folder last browsed this session, then None.
    """
    if fcs_data is not None and getattr(fcs_data, "filepath", None) is not None:
        start = Path(fcs_data.filepath).parent
        analysis = start / "analysis"
        return analysis if analysis.exists() else start
    if _last_pch_browse_dir:
        prev = Path(_last_pch_browse_dir)
        if prev.exists():
            return prev
    return None


def _pch_csv_dialog(parent, fcs_data, on_done):
    """Screen 1b — pick a saved PCH CSV, then one series from it."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    global _last_pch_browse_dir

    init_dir = _default_pch_dir(fcs_data)
    paths = discover_pch_csvs(init_dir) if init_dir else []

    win = tk.Toplevel(parent)
    win.title("PCH fit — saved histogram")
    win.geometry("620x460")
    win.minsize(520, 380)
    win.grab_set()

    tk.Label(win, text="Select a saved PCH histogram",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text="Only exports carrying raw counts and the sampled-bin "
                       "count M can be fitted; others are not listed.",
             font=("Helvetica", 9), fg="grey", wraplength=580,
             justify="left").pack()

    folder_var = tk.StringVar()

    def _update_folder():
        folder_var.set(f"Folder: {init_dir}" if init_dir
                       else "Folder: none chosen yet")
    _update_folder()
    tk.Label(win, textvariable=folder_var, font=("Courier", 8), fg="grey",
             wraplength=580, justify="left").pack()

    lb_frame = tk.Frame(win)
    lb_frame.pack(fill="both", expand=True, padx=12, pady=6)
    scroll = tk.Scrollbar(lb_frame, orient="vertical")
    listbox = tk.Listbox(lb_frame, yscrollcommand=scroll.set,
                         activestyle="none", font=("Courier", 9),
                         exportselection=False)
    scroll.config(command=listbox.yview)
    scroll.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    # Series of the highlighted file.  A combined Ch1+Ch2 histogram cannot be
    # rebuilt from the two marginals, so it is offered only when it was itself
    # exported; fitting is one series at a time either way.
    ser_frame = tk.LabelFrame(win, text="Series in this file", padx=10, pady=4)
    ser_frame.pack(fill="x", padx=12, pady=(0, 4))
    ser_var = tk.StringVar(value="")
    ser_holder = tk.Frame(ser_frame)
    ser_holder.pack(fill="x")

    info = tk.StringVar(value="")
    tk.Label(win, textvariable=info, font=("Courier", 8), fg="grey",
             wraplength=580, justify="left").pack(padx=12, anchor="w")

    loaded_cache: dict = {}

    def _current_loaded():
        sel = listbox.curselection()
        if not sel:
            return None
        p = paths[sel[0]]
        if p not in loaded_cache:
            try:
                loaded_cache[p] = load_pch_object(p)
            except Exception as e:
                loaded_cache[p] = e
        got = loaded_cache[p]
        return None if isinstance(got, Exception) else got

    def _refresh_series(*_):
        for w in ser_holder.winfo_children():
            w.destroy()
        loaded = _current_loaded()
        if loaded is None:
            ser_var.set("")
            info.set("")
            return
        for lab in loaded.labels:
            tk.Radiobutton(ser_holder, text=loaded.describe(lab),
                           variable=ser_var, value=lab, anchor="w",
                           font=("Courier", 9)).pack(fill="x")
        if ser_var.get() not in loaded.labels:
            ser_var.set(loaded.labels[0])
        info.set(f"from {loaded.source_name}  ·  bin "
                 f"{fcs_pch._format_bin_width(loaded.bin_width_s)}  ·  "
                 f"window {loaded.meta.get('window_source', 'unrecorded')}")

    listbox.bind("<<ListboxSelect>>", _refresh_series)

    def _populate(select=0):
        listbox.delete(0, tk.END)
        for p in paths:
            listbox.insert(tk.END, p.name)
        if paths:
            listbox.selection_set(min(select, len(paths) - 1))
        _refresh_series()

    def _browse_folder():
        nonlocal init_dir, paths
        chosen = filedialog.askdirectory(
            title="Choose a folder of PCH exports",
            initialdir=str(init_dir) if init_dir else "",
            mustexist=True, parent=win)
        if not chosen:
            return
        folder = Path(chosen)
        found = discover_pch_csvs(folder)
        if not found:
            messagebox.showinfo(
                "No fittable PCH exports",
                f"Nothing in '{folder.name}' parsed as a PCH histogram with "
                f"raw counts.\n\nThese are the CSVs written by the 'PCH' task "
                f"with 'Export plotted data to CSV' ticked.",
                parent=win)
            return
        init_dir, paths = folder, found
        _update_folder()
        _populate()

    def _add_files():
        nonlocal init_dir, paths
        new = filedialog.askopenfilenames(
            title="Add PCH export files",
            initialdir=str(init_dir) if init_dir else "",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            parent=win)
        if not new:
            return
        errors, added = [], 0
        for n in new:
            p = Path(n)
            try:
                load_pch_csv(p)
            except Exception as e:
                errors.append(f"{p.name}: {e}")
                continue
            if p not in paths:
                paths.append(p)
                added += 1
        init_dir = Path(new[0]).parent
        _update_folder()
        _populate(select=len(paths) - 1 if added else 0)
        if errors:
            messagebox.showerror("Some files could not be read",
                                 "\n\n".join(errors), parent=win)

    bar = tk.Frame(win)
    bar.pack(fill="x", padx=12)
    tk.Button(bar, text="Add files…", command=_add_files, pady=3).pack(side="left")
    tk.Button(bar, text="Browse folder…", command=_browse_folder,
              pady=3).pack(side="left", padx=6)

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        loaded = _current_loaded()
        if loaded is None:
            sel = listbox.curselection()
            if sel and isinstance(loaded_cache.get(paths[sel[0]]), Exception):
                messagebox.showerror(
                    "Cannot read this file",
                    str(loaded_cache[paths[sel[0]]]), parent=win)
            else:
                messagebox.showinfo("Nothing selected",
                                    "Select a PCH export first.", parent=win)
            return
        label = ser_var.get()
        if label not in loaded.series:
            messagebox.showinfo("No series selected",
                                "Select which series to fit.", parent=win)
            return
        s = loaded.get(label)
        if len(s["k"]) < 3:
            messagebox.showerror(
                "Too few count values",
                "This histogram has fewer than 3 distinct photon-count "
                "values; the counts per bin are too low to fit.  Re-export "
                "with a larger bin width.", parent=win)
            return
        if init_dir:
            _last_pch_browse_dir = str(init_dir)
        win.destroy()
        on_done(loaded, label)

    tk.Button(btns, text="Next →", width=12, command=_next,
              pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(side="left", padx=6)

    _populate()
    win.wait_window()


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
            # Bin over the acquisition window, not this channel's first-to-last
            # photon, so leading/trailing empty time counts as k = 0.
            t0, t1, _wsrc = fcs_pch.acquisition_window(fcs_data)
            k, n_k, M, mean, var = pch_counts(t, bw, t_start=t0, t_end=t1)
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


def _pch_setup_dialog(parent, source, model, channel, bin_width_s,
                      k, n_k, M, mean, var, observed_cps=None):
    """Screen 3 — initial guesses, bounds and fixed flags, then fit.

    ``source`` is whatever the histogram came from -- an FCSData or a
    LoadedPCH.  Only ``.filepath`` is used, for the plot title and to anchor
    the output folder, so the two are interchangeable here.  The observed count
    rate is passed in rather than looked up, because a loaded CSV reads it from
    its header while photon data computes it from the trace.
    """
    import tkinter as tk
    from tkinter import messagebox

    win = tk.Toplevel(parent)
    win.title(f"PCH fit — {model.name}")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text=model.name, font=("Helvetica", 12, "bold"), pady=6).pack()
    _origin = getattr(source, "source_name", None)
    _name = source.filepath.name
    if _origin and _origin != _name:
        _name = f"{_name}  (from {_origin})"
    tk.Label(win, text=f"{_name}  ·  {channel}  ·  "
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
                             observed_cps=observed_cps)
        except Exception as e:
            messagebox.showerror("Fit failed", str(e), parent=win)
            return

        win.destroy()
        result["origin"] = getattr(source, "source_name", None)
        report_path, _curve, _params = export_pch_fit(result, source.filepath)
        fig, _axes = plot_pch_fit(result, source.filepath.name, show=False)
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
    _t0, _t1, _wsrc = fcs_pch.acquisition_window(d)
    k, n_k, M, mean, var = pch_counts(channel_times(d, ch), bw,
                                      t_start=_t0, t_end=_t1)
    guesses = auto_guess_pch(model, mean, var)
    lowers = {p.name: p.lower for p in model.params}
    uppers = {p.name: p.upper for p in model.params}
    fixed  = {p.name: p.fixed for p in model.params}
    result = fit_pch(model, k, n_k, M, guesses, lowers, uppers, fixed,
                     channel=ch, bin_width_s=bw, mean=mean, var=var,
                     observed_cps=channel_cps(d, ch))
    export_pch_fit(result, path)
    plot_pch_fit(result, path.name)
