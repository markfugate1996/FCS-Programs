"""
fcs_fit.py
==========
Fit saved correlation curves to an FCS model.

Workflow (launched from the main window)
----------------------------------------
    1. Pick correlation CSV(s)  (these are the files written by the "Export
       plotted data to CSV" option).  No .fcs file is needed: a correlation CSV
       carries its own metadata, so saved curves can be modelled on their own —
       useful because the .fcs originals are far larger to carry around.  When
       there *is* an active .fcs, its 'analysis' folder is offered as the
       default starting folder; otherwise the last folder browsed this session.
    2. Choose a model               -> _select_model_dialog
    3. Set guesses / bounds / fixed -> _global_setup_dialog
    4. Fit, plot data + fit + residuals, and write results to a 'fits' folder.

The numerical core (load_correlation_csv, auto_guess) has no
GUI dependency and can be reused for batch / multi-dataset fitting later.

Models come from fcs_models.MODELS — see that file to add or edit models.

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
from scipy.optimize import least_squares

import fcs_models
from fcs_models import FCSModel
import fcs_export
from fcs_reader import read_fcs, FCSData
import fcs_lifetime
import fcs_lifetime_fit
import fcs_pch_fit
import fcs_fisher
import fcs_noise

from fcs_fitcommon import (
    fits_dir as _fits_dir,
    new_fit_dir as _new_fit_dir,
    fmt_bound as _fmt,
    parse_bound as _parse_bound,
)

# ── Session state ─────────────────────────────────────────────────────────────
# The dataset-selection dialog remembers, for the rest of the session, the list
# a user curated: the manual row order and the rows they removed.  All three are
# committed only by "Next →" (so Cancel discards list edits, matching the way
# the reorder buttons behave) and are wiped by the dialog's "Reset list" button.
# Paths are stored as absolute strings.
#
# Membership is otherwise still decided by fresh discovery on each open, so a
# correlation CSV written since the last visit shows up on its own.
_last_dataset_order: list[str] = []       # row order, included rows and not
_last_dataset_removed: set[str] = set()   # rows removed from the list by hand
_last_browse_dir: Optional[str] = None    # folder the dialog last worked in

# ── CSV loading ───────────────────────────────────────────────────────────────

def load_correlation_csv(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray], Dict[str, str]]:
    """
    Read a correlation export written by fcs_corr / fcs_export.

    Skips ``#`` comment lines (parsing ``# key : value`` lines into a metadata
    dict), reads the column header, and returns the lag axis in seconds, the
    correlation G, and (if present) the per-segment standard deviation G_std.

    Returns
    -------
    tau_s : np.ndarray
    G     : np.ndarray
    G_std : np.ndarray or None
    columns : dict of every column found, by name
    meta  : dict of header ``# key : value`` fields (e.g. 'source file', 'type')
    """
    # Parsing lives in fcs_export.read_export -- the same reader the writer
    # is paired with, so the format is defined in exactly one place.  This
    # function keeps its own signature and its FCS-specific validation.
    path = Path(path)
    meta, columns = fcs_export.read_export(path)

    if "tau_s" in columns:
        tau_s = columns["tau_s"]
    elif "tau_ms" in columns:
        tau_s = columns["tau_ms"] * 1e-3
    else:
        raise ValueError(
            f"{path.name} has no 'tau_s' or 'tau_ms' column — "
            f"is this a correlation export?"
        )

    if "G" not in columns:
        raise ValueError(
            f"{path.name} has no 'G' column — is this a correlation export?"
        )
    G = columns["G"]
    G_std = columns.get("G_std")

    return tau_s, G, G_std, columns, meta


def _cps_from_meta(meta: Optional[dict]) -> Optional[dict]:
    """
    Derive measurement CPS (and acquisition time / photon counts) from a
    correlation CSV's header meta.

    The single "fit CPS" follows the correlation type: for an autocorrelation
    it is that channel's rate; for a cross-correlation it is the average of the
    two channels (not a fitted quantity, but a convenient brightness summary).

    Uncertainties
    -------------
    Each rate gets a companion ``*_err`` from Poisson counting statistics,

        sigma_R = sqrt(N) / T        so   sigma_R / R = 1 / sqrt(N)

    The cross-correlation "fit CPS" is the GEOMETRIC mean sqrt(R1.R2), not the
    arithmetic one, so its error propagates as

        sigma_g / g = 0.5 * sqrt(1/N1 + 1/N2)

    which reduces to sqrt(N1 + N2) / (2T) -- exactly what the arithmetic mean
    would give.  The two conventions coincide for Poisson errors, since
    sqrt(N1.N2) . sqrt(1/N1 + 1/N2) = sqrt(N1 + N2), so this column does not
    depend on which mean the fit CPS uses.

    Treat these as a SHOT-NOISE FLOOR rather than the full uncertainty on the
    rate.  A fluorescing sample fluctuates by more than Poisson -- the excess
    is exactly the bunching that G(tau) measures -- so the true variance is

        Var(N) = <N> * (1 + 2.<I>.integral G(tau) dtau)

    At low amplitude the two agree closely (G0 = 0.1 at 10 kcps gives only a
    1.25x underestimate), but at G0 ~ 1 and 100 kcps the real spread is about
    7.6x larger than sqrt(N)/T.  The quoted error is therefore a lower bound,
    and a useful one for spotting a mis-set acquisition time or a dead
    channel, but it is not a substitute for repeat measurements.

    Returns a dict with floats (None where a field is absent), or None when the
    header carries no CPS at all — e.g. a correlation file written before this
    field existed — so callers can simply skip the section.
    """
    if not meta:
        return None

    def _num(key):
        v = meta.get(key)
        if v in (None, ""):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    cps1 = _num("cps_ch1")
    cps2 = _num("cps_ch2")
    if cps1 is None and cps2 is None:
        return None   # not a CPS-aware correlation export

    ctype = (meta.get("type") or "cross").strip()
    if ctype == "auto_ch1":
        cps_fit, label = cps1, "Ch1"
    elif ctype == "auto_ch2":
        cps_fit, label = cps2, "Ch2"
    else:   # cross (or unknown) → geometric mean, so R² = R1·R2 exactly,
            # matching the correlator's rateA·rateB normalisation
        present = [c for c in (cps1, cps2) if c is not None]
        if cps1 and cps2:
            cps_fit = float(np.sqrt(cps1 * cps2))
            label = "sqrt(Ch1·Ch2)"
        else:
            cps_fit = present[0] if present else None
            label = "Ch1" if cps1 else ("Ch2" if cps2 else "n/a")
            
    T  = _num("acquisition_time_s")
    n1 = _num("n_photons_ch1")
    n2 = _num("n_photons_ch2")

    def _rate_err(n_ph):
        """Poisson counting error on a rate: sqrt(N)/T."""
        if n_ph is None or T is None or T <= 0 or n_ph < 0:
            return None
        return float(np.sqrt(n_ph) / T)

    err1 = _rate_err(n1)
    err2 = _rate_err(n2)

    # Error on the reported "fit CPS", following whichever combination was
    # used above.
    if ctype == "auto_ch1":
        err_fit = err1
    elif ctype == "auto_ch2":
        err_fit = err2
    elif cps1 and cps2 and n1 and n2 and cps_fit is not None:
        # Geometric mean: sigma_g/g = 0.5*sqrt(1/N1 + 1/N2)
        err_fit = float(cps_fit * 0.5 * np.sqrt(1.0 / n1 + 1.0 / n2))
    else:
        err_fit = err1 if cps1 else (err2 if cps2 else None)

    return {
        "cps_ch1":       cps1,
        "cps_ch2":       cps2,
        "cps_fit":       cps_fit,
        "cps_label":     label,
        "cps_ch1_err":   err1,
        "cps_ch2_err":   err2,
        "cps_fit_err":   err_fit,
        "acq_time_s":    T,
        "n_photons_ch1": n1,
        "n_photons_ch2": n2,
    }


def _fmt_cps(x: Optional[float]) -> str:
    """Format a CPS / photon count for a report (thousands-separated)."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:,.0f}"


def _fmt_cps_err(x: Optional[float], err: Optional[float]) -> str:
    """
    Format a rate with its uncertainty.

    Rates below 1000 cps keep a decimal place: at 10 cps a whole-number
    format would render the value and its error identically useless
    ("10 ± 0").
    """
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    dec = 0 if abs(x) >= 1000 else 2
    base = f"{x:,.{dec}f}"
    if err is None or not np.isfinite(err):
        return base
    return f"{base} ± {err:,.{dec}f}"


def _bg_corr_from_meta(meta: Optional[dict]) -> Optional[Tuple[float, float]]:
    """
    Read the Koppel amplitude correction (kappa, sigma_kappa) from a
    correlation CSV header, as written by fcs_corr.

    kappa = bg_correction_factor is the AMPLITUDE multiplier:
    G0_corr = G0 * kappa (kappa >= 1).  Occupancy is the reciprocal:
    N_corr = N / kappa.  Returns None when the header has no factor (a CSV
    exported before this field existed), so the correction is simply skipped.
    """
    if not meta:
        return None
    v = meta.get("bg_correction_factor")
    if v in (None, ""):
        return None
    try:
        kappa = float(v)
    except (TypeError, ValueError):
        return None
    if not (kappa > 0):
        return None
    e = meta.get("bg_correction_factor_std")
    try:
        kappa_err = float(e) if e not in (None, "") else 0.0
    except (TypeError, ValueError):
        kappa_err = 0.0
    return kappa, kappa_err


# ── Initial-guess heuristics ──────────────────────────────────────────────────

# Pair-starved cutoff.
#
# G(tau) = C.n/(Sa.Sb) - 1 with C the number of photon PAIRS at that lag, so
# G = -1 is the hard floor of the estimator: it is what a lag returns when no
# pair was observed at all.  Such a channel is CENSORED, not measured -- the
# true G could be anything below the detection limit.
#
# Two things make these channels poisonous to a fit rather than merely
# uninformative.  Every segment returns the same floored value, so the
# segment-to-segment scatter is zero and G_std collapses; the point then
# carries an enormous 1/sigma^2 weight.  On a real 10 cps dataset this
# produced weights of 1e6 against 0.05 for the legitimate points, and the fit
# returned a negative amplitude and a negative diffusion time.
#
# The cutoff sits at -0.9 rather than exactly -1 because the observed floor is
# not exactly -1: normalising by the MEASURED trace mean carries a finite
# sample bias of order 1/N, which lifts the floor to -(1 - (1/Sa + 1/Sb)/2).
# On the dataset above that put the floor at -0.998.  For the floor to rise
# above -0.9 a channel would need fewer than about ten photons in the entire
# acquisition.
#
# The only physical process that drives G below -0.9 is photon antibunching,
# which lives at nanosecond lags -- four decades below the microsecond tau_min
# this suite works at -- so a false positive is not a practical concern.
_PAIR_STARVED_CUTOFF = -0.9


def auto_guess(model: FCSModel, tau_s: np.ndarray, G: np.ndarray) -> Dict[str, float]:
    """
    Produce sensible starting guesses from the data.

    Generic models fall back to their declared defaults; for the standard
    diffusion parameters (G0, tau_D, offset) the amplitude is read from the
    short-lag plateau and the diffusion time from the half-amplitude crossing.
    """
    guess = model.defaults()

    t = np.asarray(tau_s, dtype=np.float64)
    y = np.asarray(G, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(y) & (t > 0)
    # Pair-starved channels cluster at SHORT lag, which is exactly where the
    # amplitude guess is read from, so leaving them in would start the fit
    # from an amplitude of about -1.  Dropped here unconditionally: this only
    # sets a starting point, never the fitted result.
    m &= y > _PAIR_STARVED_CUTOFF
    t, y = t[m], y[m]
    if len(t) < 3:
        return guess

    order = np.argsort(t)
    t, y = t[order], y[order]

    # Amplitude from the first few (shortest-lag) finite points.
    n_head = max(3, len(y) // 20)
    amp = float(np.median(y[:n_head]))

    if "offset" in guess:
        guess["offset"] = 0.0
    if "G0" in guess:
        guess["G0"] = max(amp, 1e-9)

    # Diffusion time: first lag where G falls below half amplitude.
    if "tau_D" in guess:
        off    = guess.get("offset", 0.0)
        amp0   = guess.get("G0", amp)
        target = off + 0.5 * amp0
        below  = np.where(y < target)[0]
        if len(below):
            guess["tau_D"] = float(t[below[0]])
        else:
            guess["tau_D"] = float(np.sqrt(t[0] * t[-1]))   # geometric midpoint

    return guess



# ── Export ────────────────────────────────────────────────────────────────────

def _write_params_xlsx(path: Path, comments: list, header: list,
                        rows: list) -> Optional[Path]:
    """
    Mirror the parameter table as a real .xlsx for viewing in Excel.

    Delegates to fcs_export.write_table_xlsx so the fit and the calibration
    produce identically formatted spreadsheets from one implementation; the
    .csv remains the machine-readable copy.
    """
    return fcs_export.write_table_xlsx(
        path, comments, header, rows,
        sheet_title="fit parameters", log_tag="globalfit")


# ── Global / linked fitting ───────────────────────────────────────────────────

def combined_guess(model: FCSModel, datasets: list) -> Dict[str, float]:
    """
    A shared starting guess across datasets: the median of each dataset's own
    auto-guess (falls back to model defaults where data are unusable).
    """
    g = model.defaults()
    per = [auto_guess(model, ds["tau"], ds["G"]) for ds in datasets]
    for n in g:
        vals = [p[n] for p in per if np.isfinite(p.get(n, np.nan))]
        if vals:
            g[n] = float(np.median(vals))
    return g


def fit_global(
    model: FCSModel,
    datasets: list,
    linked: Dict[str, bool],
    guesses: Dict[str, float],
    lowers: Dict[str, float],
    uppers: Dict[str, float],
    fixed: Dict[str, bool],
    weighted: bool = False,
    maxfev: int = 20000,
    drop_pair_starved: bool = True,
    pair_starved_cutoff: float = _PAIR_STARVED_CUTOFF,
) -> dict:
    """
    Global least-squares fit of one model to several correlation datasets.

    Each model parameter is one of:
      * fixed   — held at its guess for every dataset (no free variable);
      * linked  — a single shared free variable used by every dataset;
      * unlinked— one independent free variable per dataset.

    ``datasets`` is a list of dicts with keys ``name``, ``tau``, ``G`` and
    optionally ``sigma``.  Weighting is applied only if ``weighted`` is True
    *and* every included dataset carries a usable sigma.

    When *drop_pair_starved* is True (the default) any lag whose G lies at or
    below *pair_starved_cutoff* is discarded before fitting.  See
    ``_PAIR_STARVED_CUTOFF`` for why those channels are censored rather than
    measured, and why their uncertainties are not to be trusted.  The count
    discarded per dataset is returned in the result as ``n_pair_starved`` so
    the report can state it rather than silently using fewer points.

    Returns a result dict with global goodness-of-fit plus a per-dataset
    breakdown (values, 1σ errors, fit curve, residuals, R²).
    """
    names = model.param_names()

    # ── Mask each dataset to finite, positive-lag points ─────────────────────
    prepped = []
    starved_counts: Dict[str, int] = {}
    for ds in datasets:
        t = np.asarray(ds["tau"], dtype=np.float64)
        y = np.asarray(ds["G"],   dtype=np.float64)
        m = np.isfinite(t) & np.isfinite(y) & (t > 0)
        s = None
        if weighted and ds.get("sigma") is not None:
            s = np.asarray(ds["sigma"], dtype=np.float64)
            m &= np.isfinite(s) & (s > 0)

        n_starved = 0
        if drop_pair_starved:
            starved = m & (y <= pair_starved_cutoff)
            n_starved = int(np.count_nonzero(starved))
            m &= ~starved
        starved_counts[ds["name"]] = n_starved

        t, y = t[m], y[m]
        s = s[m] if s is not None else None
        if len(t) < 2:
            if n_starved:
                raise ValueError(
                    f"Dataset '{ds['name']}' has too few usable points to fit: "
                    f"{n_starved} lag channel"
                    f"{'s' if n_starved != 1 else ''} were discarded as "
                    f"pair-starved (G <= {pair_starved_cutoff:g}, meaning no "
                    f"photon pairs were detected at those lags) and fewer than "
                    f"two remain.\n\n"
                    f"This usually means the count rate is too low for the "
                    f"chosen lag range, not that the fit settings are wrong."
                )
            raise ValueError(
                f"Dataset '{ds['name']}' has too few finite points to fit.")
        prepped.append({"name": ds["name"], "tau": t, "G": y, "sigma": s,
                        "meta": ds.get("meta", {}),
                        "n_pair_starved": n_starved})

    D = len(prepped)
    if D == 0:
        raise ValueError("No datasets selected.")
    use_weights = weighted and all(p["sigma"] is not None for p in prepped)

    # ── Lay out the free-parameter vector ────────────────────────────────────
    free_spec: list = []          # (param_name, dataset_index | None)
    for p in names:
        if fixed.get(p, False):
            continue
        if linked.get(p, False):
            free_spec.append((p, None))
        else:
            free_spec.extend((p, di) for di in range(D))
    if not free_spec:
        raise ValueError("At least one parameter must be free (not fixed).")
    idx = {spec: i for i, spec in enumerate(free_spec)}
    fixed_value = {p: guesses[p] for p in names if fixed.get(p, False)}

    def value_of(theta, p, di):
        if fixed.get(p, False):
            return fixed_value[p]
        if linked.get(p, False):
            return theta[idx[(p, None)]]
        return theta[idx[(p, di)]]

    def residuals(theta):
        chunks = []
        for di, pp in enumerate(prepped):
            vals = {p: value_of(theta, p, di) for p in names}
            mvals = model.func(pp["tau"], **vals)
            r = pp["G"] - mvals
            if use_weights:
                r = r / pp["sigma"]
            chunks.append(r)
        return np.concatenate(chunks)

    theta0, lb, ub = [], [], []
    for (p, _di) in free_spec:
        theta0.append(guesses[p]); lb.append(lowers[p]); ub.append(uppers[p])
    theta0 = [min(max(v, lo), hi) for v, lo, hi in zip(theta0, lb, ub)]

    sol = least_squares(residuals, theta0, bounds=(lb, ub), max_nfev=maxfev)

    # ── Covariance / parameter errors ────────────────────────────────────────
    n_obs = int(sum(len(pp["tau"]) for pp in prepped))
    n_par = len(free_spec)
    dof = n_obs - n_par
    cov = None
    try:
        JtJ = sol.jac.T @ sol.jac
        cov = np.linalg.pinv(JtJ)
        if not use_weights and dof > 0:
            cov = cov * (2.0 * sol.cost / dof)   # scale by residual variance
        perr_free = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    except Exception:
        perr_free = np.full(n_par, np.nan)

    def err_of(p, di):
        if fixed.get(p, False):
            return 0.0
        if linked.get(p, False):
            return float(perr_free[idx[(p, None)]])
        return float(perr_free[idx[(p, di)]])

    # ── Per-dataset breakdown ────────────────────────────────────────────────
    theta = sol.x
    per_dataset = []
    for di, pp in enumerate(prepped):
        vals = {p: value_of(theta, p, di) for p in names}
        errs = {p: err_of(p, di) for p in names}
        mvals = model.func(pp["tau"], **vals)
        resid = pp["G"] - mvals
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((pp["G"] - pp["G"].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        per_dataset.append({
            "name": pp["name"], "tau": pp["tau"], "G": pp["G"],
            "Gfit": mvals, "resid": resid, "sigma": pp["sigma"],
            "values": vals, "errors": errs, "r2": r2,
            "ss_res": ss_res, "n_points": len(pp["tau"]),
            "n_pair_starved": pp.get("n_pair_starved", 0),
            "meta": pp.get("meta", {}),
        })

    full_res = residuals(theta)
    ss_res_tot = float(np.sum(full_res ** 2))
    if use_weights and dof > 0:
        chi2, red_chi2 = ss_res_tot, ss_res_tot / dof
    else:
        chi2 = red_chi2 = float("nan")

    #this builds labels for the free param vector
    free_labels = [
        p if di is None else f"{p}[{prepped[di]['name']}]"
        for (p, di) in free_spec
    ]

    return {
        "model": model, "names": names, "datasets": per_dataset,
        "linked": dict(linked), "fixed": dict(fixed),
        "lowers": dict(lowers), "uppers": dict(uppers), "guesses": dict(guesses),
        "weighted": use_weights, "n_datasets": D,
        "drop_pair_starved": bool(drop_pair_starved),
        "pair_starved_cutoff": float(pair_starved_cutoff),
        "dof": dof, "n_free": n_par, "n_obs": n_obs,
        "chi2": chi2, "red_chi2": red_chi2, "ss_res": ss_res_tot,
        "success": bool(sol.success), "message": str(sol.message),
        "cov": cov, "free_labels": free_labels,
    }


def plot_global_fit(result: dict, show: bool = True):
    """Overlay every dataset's data + fit, with a shared residual panel."""
    model = result["model"]
    dsets = result["datasets"]
    cmap = plt.cm.tab10 if len(dsets) <= 10 else plt.cm.viridis
    colors = [cmap(i / max(1, len(dsets) - 1)) if cmap is plt.cm.viridis
              else cmap(i % 10) for i in range(len(dsets))]

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(9.5, 6),
        gridspec_kw={"height_ratios": [3, 1]}, layout="constrained",
    )

    for ds, c in zip(dsets, colors):
        tau_ms = ds["tau"] * 1e3
        ax.semilogx(tau_ms, ds["G"], linestyle="none", marker=".",
                    markersize=3.5, color=c, alpha=0.8,
                    label=f"_{ds['name']}  (data)")
        # A rebuilt fit supplies its own curve (see rebuild_plot): the model
        # function may be unavailable, or re-evaluating it may not reproduce
        # what was actually fitted.  A live fit has no "dense" key and the
        # model is evaluated as usual.
        dense = ds.get("dense")
        if dense is not None:
            t_dense, g_dense = dense
        else:
            t_dense = np.logspace(np.log10(ds["tau"].min()),
                                  np.log10(ds["tau"].max()), 400)
            g_dense = model.func(
                t_dense, **{n: ds["values"][n] for n in result["names"]})
        ax.semilogx(t_dense * 1e3, g_dense, color=c, linewidth=1.4,
                    label=ds["name"])
        res = ds["resid"] / ds["sigma"] if result["weighted"] else ds["resid"]
        axr.semilogx(tau_ms, res, linestyle="none", marker=".",
                     markersize=2.5, color=c, alpha=0.8,
                     label=f"_{ds['name']}  (residual)")

    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--")
    ax.set_ylabel("G(τ)", fontsize=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    # Linked-parameter summary box (linked values are shared, so read ds 0)
    ref = dsets[0].get("values", {})
    referr = dsets[0].get("errors", {})
    # A rebuilt fit whose parameter table could not be found has no values to
    # report, so those entries are simply omitted rather than raising.
    linked_names = [n for n in result["names"]
                    if result["linked"].get(n) and n in ref]
    box = []
    for n in linked_names:
        unit = next((p.unit for p in model.params if p.name == n), "")
        tag = ("(fixed)" if result["fixed"].get(n)
               else (f"± {referr[n]:.2g}" if n in referr else ""))
        box.append(f"{n} = {ref[n]:.4g} {tag} {unit}".rstrip())
    if result["weighted"] and result.get("red_chi2") is not None:
        box.append(f"red. χ² = {result['red_chi2']:.3g}")
    elif not result["weighted"] and result.get("ss_res") is not None:
        box.append(
            f"global R² = {1 - result['ss_res'] / _grand_ss_tot(dsets):.4f}")
    if result.get("rebuilt_note"):
        box.append(result["rebuilt_note"])
    if box:
        ax.text(0.98, 0.95, "linked:\n" + "\n".join(box),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    # One entry per dataset, so a 30-dataset global fit produced a legend
    # 2.4x taller than the axes (497 px against 208 px measured) which
    # constrained layout then shrank the plot to accommodate.  Size it from
    # the entry count instead; above the threshold it is built but hidden and
    # the plot-controls Legend section can switch it back on.
    fcs_plottools.adaptive_legend(ax, base_fontsize=8, loc="lower left")
    ax.set_title(
        f"Global FCS fit — {model.name}\n"
        f"{result['n_datasets']} datasets  ·  {result['n_free']} free params  ·  "
        f"{model.formula}",
        fontsize=10,
    )

    axr.axhline(0, color="grey", linewidth=0.8)
    axr.set_ylabel("resid/σ" if result["weighted"] else "resid", fontsize=10)
    axr.set_xlabel("Lag time τ (ms)", fontsize=12)
    axr.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    axr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2g}"))

    if show:
        #plot.show was static; fcs_plottools is dynamic
        #plt.show()
        fcs_plottools.show_figure(fig, np.array([ax, axr]))
    return fig, np.array([ax, axr])


class _StoredModel:
    """
    Stand-in for an FCSModel when the original cannot be reconstructed.

    Carries only what plot_global_fit reads for labelling.  Its func() raises:
    a rebuilt fit that reaches this class must supply its own curve, and a
    silent wrong curve would be far worse than a loud failure.
    """

    def __init__(self, name: str, formula: str = "", params=()):
        self.key = ""
        self.name = name
        self.formula = formula
        self.params = list(params)

    def func(self, *_a, **_k):
        raise RuntimeError(
            "This fit was rebuilt from a CSV and its model could not be "
            "reconstructed, so the model function is unavailable."
        )


def _read_param_table(path: Path):
    """
    Read the sibling globalfit_params.csv, if it is there.

    Returns (values_by_dataset, errors_by_dataset, columns) -- all empty when
    the file is missing or unreadable.  A missing parameter table is not an
    error: it only means the linked-parameter box cannot be filled in and the
    fit curve cannot be re-evaluated densely.
    """
    empty: tuple = ({}, {}, [])
    if path is None:
        return empty

    # The four files of one fit share a stem and differ only by suffix, so the
    # parameter table is found by swapping "_curves.csv" for "_params.csv".
    # Fall back to the old fixed name, then to any *_params.csv in the folder,
    # so fits exported before filenames were timestamped still reopen.
    cp = Path(path)
    candidates = []
    if cp.name.endswith("_curves.csv"):
        candidates.append(cp.with_name(cp.name[:-len("_curves.csv")]
                                       + "_params.csv"))
    candidates.append(cp.with_name("globalfit_params.csv"))
    try:
        candidates.extend(sorted(cp.parent.glob("*_params.csv")))
    except Exception:
        pass

    pp = next((c for c in candidates if c.is_file()), None)
    if pp is None:
        return empty
    try:
        _meta, cols = fcs_export.read_export(pp)
    except Exception:
        return empty
    if "dataset" not in cols:
        return empty

    names = [c for c in cols if f"{c}_err" in cols]
    values: Dict[str, dict] = {}
    errors: Dict[str, dict] = {}
    for i, raw in enumerate(cols["dataset"]):
        ds = str(raw)
        values[ds] = {n: float(cols[n][i]) for n in names}
        errors[ds] = {n: float(cols[f"{n}_err"][i]) for n in names}
    return values, errors, names


def rebuild_plot(meta: dict, columns: dict, show: bool = True, path=None):
    """
    Rebuild a global-fit figure from an exported globalfit_curves.csv.

    The curves file holds the data, the fitted curve and the residuals at the
    lags that were actually fitted.  Two further things are recovered when
    available, and both degrade gracefully:

    * The linked-parameter box needs per-dataset values, which live in the
      sibling ``globalfit_params.csv``.  Without it the box is omitted.
    * The smooth fit line is drawn by re-evaluating the model densely, which
      needs both the model key (recorded in the header) and those parameter
      values.  Without either, the stored fit is plotted at its own lags.
      On a semilog axis with 20+ points per decade that is visually
      indistinguishable -- under a tenth of a typical error bar.

    Re-evaluating the model is VERIFIED before it is used.  fcs_models can
    change: correct a factor, alter a convention, and re-evaluating a fit
    saved beforehand would silently draw a curve that is not the one that was
    fitted.  The stored G_fit column is the ground truth, so the
    reconstruction is checked against it at the data lags and discarded if it
    disagrees, with the reason shown on the plot.

    Parameters
    ----------
    meta, columns : from fcs_export.read_export
    show : passed through to plot_global_fit
    path : the curves CSV itself, used to find the sibling parameter table

    Returns
    -------
    fig, axes
    """
    if "dataset" not in columns:
        raise ValueError(
            "No 'dataset' column: this does not look like a global-fit "
            "curves export."
        )
    if "tau_s" in columns:
        tau_all = np.asarray(columns["tau_s"], dtype=float)
    elif "tau_ms" in columns:
        tau_all = np.asarray(columns["tau_ms"], dtype=float) * 1e-3
    else:
        raise ValueError("No 'tau_s' or 'tau_ms' column in this export.")
    for need in ("G_data", "G_fit"):
        if need not in columns:
            raise ValueError(f"No '{need}' column in this export.")

    G_all    = np.asarray(columns["G_data"], dtype=float)
    Gfit_all = np.asarray(columns["G_fit"], dtype=float)
    res_all  = (np.asarray(columns["residual"], dtype=float)
                if "residual" in columns else G_all - Gfit_all)
    sig_all  = (np.asarray(columns["sigma"], dtype=float)
                if "sigma" in columns else None)

    weighted = (meta.get("weighted", "").strip().lower() == "yes"
                if "weighted" in meta else sig_all is not None)

    def _name_list(key):
        raw = meta.get(key, "")
        return [x.strip() for x in raw.split(",") if x.strip()]

    linked_names = _name_list("linked")
    fixed_names  = _name_list("fixed")

    values_by_ds, errors_by_ds, table_names = _read_param_table(path)

    model = None
    model_key = meta.get("model_key", "").strip()
    if model_key:
        try:
            model = fcs_models.get_model(model_key)
        except Exception:
            model = None

    if model is not None:
        names = [p.name for p in model.params]
    else:
        names = table_names or sorted(set(linked_names) | set(fixed_names))

    # ── Split the long format back into datasets, preserving file order ──────
    order: list = []
    index: Dict[str, list] = {}
    for i, raw in enumerate(columns["dataset"]):
        ds = str(raw)
        if ds not in index:
            index[ds] = []
            order.append(ds)
        index[ds].append(i)

    notes: list = []
    dsets: list = []
    for ds in order:
        idx = np.asarray(index[ds], dtype=int)
        tau_i, G_i, Gfit_i = tau_all[idx], G_all[idx], Gfit_all[idx]
        vals = values_by_ds.get(ds, {})
        errs = errors_by_ds.get(ds, {})

        dense = None
        if model is not None and vals and all(n in vals for n in names):
            try:
                check = model.func(tau_i, **{n: vals[n] for n in names})
                scale = float(np.nanmax(np.abs(Gfit_i))) or 1.0
                if np.allclose(check, Gfit_i, rtol=1e-4, atol=1e-6 * scale):
                    t_dense = np.logspace(np.log10(tau_i.min()),
                                          np.log10(tau_i.max()), 400)
                    dense = (t_dense,
                             model.func(t_dense, **{n: vals[n] for n in names}))
                else:
                    worst = float(np.nanmax(np.abs(check - Gfit_i)))
                    notes.append(
                        f"model '{model_key}' no longer reproduces the stored "
                        f"fit (max Δ {worst:.3g}); showing the stored curve"
                    )
            except Exception as exc:      # noqa: BLE001 — fall back, don't fail
                notes.append(f"could not re-evaluate '{model_key}' ({exc})")

        if dense is None:
            dense = (tau_i, Gfit_i)

        dsets.append({
            "name": ds,
            "tau": tau_i,
            "G": G_i,
            "Gfit": Gfit_i,
            "resid": res_all[idx],
            "sigma": (sig_all[idx] if sig_all is not None
                      else np.ones(idx.size)),
            "values": vals,
            "errors": errs,
            "n_points": int(idx.size),
            "dense": dense,
        })

    if model is None:
        model = _StoredModel(meta.get("model", model_key or "unknown model"))
        if model_key:
            notes.append(f"model '{model_key}' is not in fcs_models")

    ss_res = float(sum(float(np.sum(d["resid"] ** 2)) for d in dsets))

    note = ""
    if notes:
        # One line, even when several datasets report the same thing.
        note = "⚠ " + sorted(set(notes))[0]

    result = {
        "model": model,
        "datasets": dsets,
        "names": names,
        "linked": {n: True for n in linked_names},
        "fixed": {n: True for n in fixed_names},
        "weighted": weighted,
        "red_chi2": _to_float(meta.get("red_chi2")),
        "ss_res": ss_res,
        "n_datasets": _to_int(meta.get("n_datasets"), len(dsets)),
        "n_free": _to_int(meta.get("n_free"), 0),
        "rebuilt_note": note,
    }
    return plot_global_fit(result, show=show)


def _to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _grand_ss_tot(dsets) -> float:
    allG = np.concatenate([d["G"] for d in dsets])
    return float(np.sum((allG - allG.mean()) ** 2)) or float("nan")


def _fit_file_stem(name_override: Optional[str] = None,
                   when: Optional[datetime] = None) -> str:
    """
    Build the shared stem for one fit's output files.

    Every fit used to write globalfit_report.txt / _curves.csv / _params.csv
    under exactly those names, distinguished only by their folder.  Excel
    refuses to hold two open workbooks with the same filename regardless of
    path, so comparing two fits meant renaming one by hand first.  A
    timestamp makes the names unique, and an optional label makes them
    meaningful.
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    slug = _slug_name(name_override)
    return f"{slug}_{stamp}" if slug else f"globalfit_{stamp}"


def _slug_name(text: Optional[str]) -> str:
    """Reduce a user-supplied label to a filesystem-safe token."""
    if not text:
        return ""
    out = []
    for ch in str(text).strip():
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:60]


def export_global_fit(result: dict, out_source: str | Path,
                      name_override: Optional[str] = None,
                      ) -> Tuple[Path, Path, Path]:
    """
    Write a global-fit report (.txt) and a combined long-format curve CSV.

    ``out_source`` is only used to locate the output folder (its 'fits'
    sibling, via fcs_fitcommon.new_fit_dir).  It may be any file that sits in
    the right place — the caller passes the first selected correlation CSV, so
    no .fcs file is required.

    Filenames carry a timestamp, and optionally *name_override*, so that two
    fits can be open in Excel at once.  The four files of one fit share a stem
    and differ only in their suffix, which is what lets the reopener find a
    curves file's matching parameter table.
    """
    out_source = Path(out_source)
    model = result["model"]
    out_dir = _new_fit_dir(out_source)
    stem = _fit_file_stem(name_override)
    report_path = out_dir / f"{stem}_report.txt"
    curve_path  = out_dir / f"{stem}_curves.csv"


    L: list[str] = []
    L.append("FCS global fit report")
    L.append("=" * 64)
    L.append(f"model      : {model.name}  [{model.key}]")
    L.append(f"formula    : {model.formula}")
    L.append(f"fitted     : {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"datasets   : {result['n_datasets']}")
    L.append(f"free params: {result['n_free']}   observations: {result['n_obs']}   "
             f"dof: {result['dof']}")
    L.append(f"weighted   : {'yes (σ from G_std)' if result['weighted'] else 'no'}")
    _starved = sum(d.get("n_pair_starved", 0) for d in result["datasets"])
    if result.get("drop_pair_starved"):
        L.append(f"pair-starved: {_starved} lag channel"
                 f"{'s' if _starved != 1 else ''} discarded "
                 f"(G <= {result.get('pair_starved_cutoff', -0.9):g})")
    else:
        L.append("pair-starved: guard DISABLED — censored channels, if any, "
                 "were fitted")
    L.append(f"converged  : {result['success']}  ({result['message']})")
    if result["weighted"]:
        L.append(f"red. chi^2 : {result['red_chi2']:.6g}")
    L.append("")
    L.append("Parameter linking")
    L.append("-" * 64)
    for n in result["names"]:
        state = ("fixed" if result["fixed"].get(n)
                 else "linked" if result["linked"].get(n) else "per-dataset")
        L.append(f"  {n:<8} : {state:<11} "
                 f"bounds [{_fmt(result['lowers'][n])}, {_fmt(result['uppers'][n])}]")
    L.append("")

    linked_names = [n for n in result["names"] if result["linked"].get(n)]
    if linked_names:
        L.append("Linked / shared parameters")
        L.append("-" * 64)
        ref, referr = result["datasets"][0]["values"], result["datasets"][0]["errors"]
        for n in linked_names:
            unit = next((p.unit for p in model.params if p.name == n), "")
            err = "—" if result["fixed"].get(n) else f"{referr[n]:.6g}"
            L.append(f"  {n:<8} = {ref[n]:.6g}  ± {err}  {unit}".rstrip())
        L.append("")

    L.append("Per-dataset results")
    L.append("-" * 64)
    for ds in result["datasets"]:
        _ns = ds.get("n_pair_starved", 0)
        _drop = f"   pair-starved dropped: {_ns}" if _ns else ""
        L.append(f"[{ds['name']}]   points: {ds['n_points']}   "
                 f"R² = {ds['r2']:.5f}{_drop}")
        cps = _cps_from_meta(ds.get("meta"))
        if cps is not None:
            L.append(f"    CPS      : Ch1 {_fmt_cps_err(cps['cps_ch1'], cps.get('cps_ch1_err'))}   "
                     f"Ch2 {_fmt_cps_err(cps['cps_ch2'], cps.get('cps_ch2_err'))}   "
                     f"{cps['cps_label']} "
                     f"{_fmt_cps_err(cps['cps_fit'], cps.get('cps_fit_err'))}")
            L.append("               (± is the Poisson floor sqrt(N)/T; a "
                     "fluctuating sample exceeds it)")
        for n in result["names"]:
            if result["linked"].get(n):
                continue  # already reported above
            unit = next((p.unit for p in model.params if p.name == n), "")
            err = "—" if result["fixed"].get(n) else f"{ds['errors'][n]:.6g}"
            L.append(f"    {n:<8} = {ds['values'][n]:.6g}  ± {err}  {unit}".rstrip())
        L.append("")
    
    corr = [(ds["name"], _bg_corr_from_meta(ds.get("meta")))
            for ds in result["datasets"]]
    if ("G0" in result["names"]) and any(c is not None for _n, c in corr):
        L.append("Background correction (Koppel amplitude factor)")
        L.append("-" * 64)
        L.append(f"  {'dataset':<24}{'factor':>12}{'err':>12}")
        for name, c in corr:
            if c is None:
                L.append(f"  {name:<24}{'n/a':>12}")
            else:
                L.append(f"  {name:<24}{c[0]:>12.4f}{c[1]:>12.4f}")
        L.append("  G0_corr = G0 x factor ;  N_corr = N / factor  "
                 "(factor + error from the correlation CSV)")
        L.append("")
    
    L.extend(fcs_fisher.format_report_lines(
        fcs_fisher.fisher_from_covariance(
            result.get("cov"), result["free_labels"], result["weighted"])))
    L.append("")
    
    L.append("Information (analytical noise covariance)")
    L.append("-" * 64)
    for ds in result["datasets"]:
        L.append(f"[{ds['name']}]")
        _c = _cps_from_meta(ds.get("meta"))
        _T = (_c or {}).get("acq_time_s")
        _R = (_c or {}).get("cps_fit")
        if not _T or not _R:
            L.append(f"  not available — need CPS and acquisition time in the "
                     f"correlation header (got cps={_R!r}, T={_T!r})")
        else:
            try:
                nc = fcs_noise.noise_covariance(
                    model, ds["values"], ds["tau"],
                    T=float(_T), count_rate=float(_R))
                free_names = [p for p in result["names"]
                              if not result["fixed"].get(p, False)]
                info = fcs_noise.fisher_information(
                    model, ds["values"], ds["tau"], nc.sigma, free_names)
                L.extend(fcs_noise.information_report_lines(info))
            except Exception as e:
                L.append(f"  not available — {type(e).__name__}: {e}")
        L.append("")

    report_path.write_text("\n".join(L), encoding="utf-8")

    # Combined long-format curve CSV
    weighted = result["weighted"]
    header = ["dataset", "tau_s", "tau_ms", "G_data", "G_fit", "residual"]
    if weighted:
        header.append("sigma")
    linked_names = [n for n in result["names"] if result["linked"].get(n)]
    fixed_names  = [n for n in result["names"] if result["fixed"].get(n)]
    with curve_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# FCS global fit curves — {model.key}\n")
        # Machine-readable header, so this file can be reopened as a live plot
        # (see fcs_plotopen).  The banner above uses an em-dash and no colon,
        # so a "key : value" parser cannot read it.  Which parameters were
        # linked and which were fixed is recorded here because it is not
        # recoverable from the numbers alone: a linked parameter merely has
        # equal values across datasets, which can also happen by coincidence.
        fh.write("# analysis   : global fit curves\n")
        fh.write(f"# model_key  : {model.key}\n")
        fh.write(f"# model      : {model.name}\n")
        fh.write(f"# weighted   : {'yes' if weighted else 'no'}\n")
        fh.write(f"# n_datasets : {result['n_datasets']}\n")
        fh.write(f"# n_free     : {result['n_free']}\n")
        fh.write(f"# linked     : {', '.join(linked_names)}\n")
        fh.write(f"# fixed      : {', '.join(fixed_names)}\n")
        if weighted:
            fh.write(f"# red_chi2   : {result['red_chi2']:.10g}\n")
        else:
            fh.write(f"# r2_global  : "
                     f"{1 - result['ss_res'] / _grand_ss_tot(result['datasets']):.10g}\n")
        fh.write(f"# exported : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(",".join(header) + "\n")
        for ds in result["datasets"]:
            for i in range(ds["n_points"]):
                row = [ds["name"],
                       f"{ds['tau'][i]:.10g}", f"{ds['tau'][i]*1e3:.10g}",
                       f"{ds['G'][i]:.10g}", f"{ds['Gfit'][i]:.10g}",
                       f"{ds['resid'][i]:.10g}"]
                if weighted:
                    row.append(f"{ds['sigma'][i]:.10g}")
                fh.write(",".join(row) + "\n")

    print(f"[globalfit] wrote {report_path}")
    print(f"[globalfit] wrote {curve_path}")

    # ── Wide parameter table (one row per dataset) for spreadsheets ───────────
    # Designed for downstream "parameter vs variable" plots (e.g. tau_D vs
    # concentration): each dataset is a row, each parameter a value+error pair.
    # ── Wide parameter table (one row per dataset) for spreadsheets ───────────
    # Calibrate Volume reads the .csv; the .xlsx is the convenient view.
    params_path = out_dir / f"{stem}_params.csv"
    xlsx_path   = out_dir / f"{stem}_params.xlsx"
    linked_names = [n for n in result["names"] if result["linked"].get(n)]
    fixed_names  = [n for n in result["names"] if result["fixed"].get(n)]

    p_header = ["dataset"]
    for n in result["names"]:
        p_header += [n, f"{n}_err"]
    has_N = "G0" in result["names"]
    if has_N:
        p_header += ["N", "N_err"]                 # <N> = 1/G0, propagated error
    
    has_bg = has_N and any(_bg_corr_from_meta(ds.get("meta")) is not None
                           for ds in result["datasets"])
    if has_bg:
        p_header += ["bg_correction_factor", "bg_correction_factor_err",
                     "G0_corr", "G0_corr_err", "N_corr", "N_corr_err"]
                     
    has_cps = any(_cps_from_meta(ds.get("meta")) is not None
                  for ds in result["datasets"])
    if has_cps:
        p_header += ["cps_ch1", "cps_ch1_err",
                     "cps_ch2", "cps_ch2_err",
                     "cps_fit", "cps_fit_err", "acq_time_s"]
    p_header += ["r2", "n_points"]

    # Build native-typed rows ONCE, then render to both .csv and .xlsx.
    data_rows: list[list] = []
    for ds in result["datasets"]:
        row: list = [ds["name"]]
        for n in result["names"]:
            row.append(float(ds["values"][n]))
            row.append(float(ds["errors"][n]))
        N = N_err = float("nan")
        if has_N:
            g0  = ds["values"]["G0"]
            g0e = ds["errors"]["G0"]
            if g0 > 0:
                N     = 1.0 / g0
                N_err = g0e / (g0 * g0)             # σ_N = σ_G0 / G0²
            row.append(float(N))
            row.append(float(N_err))
        
        if has_bg:
            corr = _bg_corr_from_meta(ds.get("meta"))
            if corr is not None and g0 > 0:
                kappa, kappa_err = corr
                rel = float(np.sqrt((g0e / g0) ** 2 + (kappa_err / kappa) ** 2))
                g0_corr = g0 * kappa
                N_corr  = N / kappa
                row += [float(kappa), float(kappa_err),
                        float(g0_corr), float(g0_corr * rel),
                        float(N_corr),  float(N_corr * rel)]
            else:
                row += [float("nan")] * 6
                
        if has_cps:
            cps = _cps_from_meta(ds.get("meta")) or {}
            for key in ("cps_ch1", "cps_ch1_err",
                        "cps_ch2", "cps_ch2_err",
                        "cps_fit", "cps_fit_err", "acq_time_s"):
                v = cps.get(key)
                row.append(float(v) if v is not None else float("nan"))
        row.append(float(ds["r2"]))
        row.append(int(ds["n_points"]))
        data_rows.append(row)

    # Comment/header block (no em-dash → no mojibake; Calibrate Volume ignores
    # these '#' lines except for the '# key : value' metadata it parses).
    comments: list[str] = []
    comments.append("FCS global fit - parameter table")
    comments.append(f"model : {model.name} [{model.key}]")
    comments.append(f"exported : {datetime.now().isoformat(timespec='seconds')}")
    comments.append(f"weighted : {'yes' if result['weighted'] else 'no'}")
    comments.append(f"linked : {', '.join(linked_names) if linked_names else '(none)'}")
    comments.append(f"fixed : {', '.join(fixed_names) if fixed_names else '(none)'}")
    if result["weighted"]:
        comments.append(f"global_red_chi2 : {result['red_chi2']:.6g}")
    comments.append("units : tau_D in seconds")
    if has_N:
        comments.append("note : N = 1/G0 (geometric factor 1); N_err = G0_err / G0^2")

    if has_bg:
        comments.append("note : G0_corr = G0 * bg_correction_factor "
                        "(Koppel amplitude correction, from the correlation CSV)")
        comments.append("note : N_corr = N / bg_correction_factor ; "
                        "*_corr_err combine fit and factor errors in quadrature")

    if has_cps:
        comments.append("note : cps_* are mean count rates (Hz) from the correlation "
                        "headers; cps_fit = that channel (auto) or the Ch1/Ch2 average (cross)")

    def _csv_cell(v) -> str:
        return f"{v:.10g}" if isinstance(v, float) else str(v)

    with params_path.open("w", encoding="utf-8", newline="") as fh:
        for line in comments:
            fh.write(f"# {line}\n")
        fh.write(",".join(p_header) + "\n")
        for row in data_rows:
            fh.write(",".join(_csv_cell(v) for v in row) + "\n")
    print(f"[globalfit] wrote {params_path}")

    _write_params_xlsx(xlsx_path, comments, p_header, data_rows)

    return report_path, curve_path, params_path

# ── GUI workflow ──────────────────────────────────────────────────────────────

def run_model_dialog(fcs_data=None, parent=None, workspace_order=None):
    """
    Top-level entry: choose which data type to model, then dispatch.

    Correlation runs the fit workflow implemented in this module; Lifetime and
    PCH dispatch to run_lifetime_fit_dialog / run_pch_fit_dialog in the
    fcs_lifetime_fit and fcs_pch_fit modules respectively.

    ``fcs_data`` may be None.  Correlation modelling reads saved CSVs and needs
    no photon records, so it stays available; Lifetime and PCH do need them and
    are disabled in that case.  When ``fcs_data`` is given it only supplies a
    starting folder for the file browser — the curves themselves are whatever
    the user picks.

    ``workspace_order`` is an optional list of source .fcs file names in
    workspace order; when given, discovered correlation datasets are listed
    and reported in that order instead of alphabetically.
    """
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Model data")
    win.geometry("320x300")
    win.resizable(False, False)
    win.grab_set()

    have_photons = fcs_data is not None

    tk.Label(win, text="Model data",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    tk.Label(win, text="Select the data type you want to fit.",
             font=("Helvetica", 9), fg="grey").pack()

    btns = tk.Frame(win, padx=20, pady=12)
    btns.pack(fill="both", expand=True)

    def _correlation():
        win.destroy()
        run_global_fit_dialog(fcs_data, parent=parent,
                              workspace_order=workspace_order)

    def _lifetime():
        win.destroy()
        fcs_lifetime_fit.run_lifetime_fit_dialog(fcs_data, parent=parent)

    def _pch():
        win.destroy()
        fcs_pch_fit.run_pch_fit_dialog(fcs_data, parent=parent)

    tk.Button(btns, text="Correlation", width=26, pady=6,
              command=_correlation).pack(pady=4)
    tk.Button(btns, text="Lifetime", width=26, pady=6,
              command=_lifetime,
              state=("normal" if have_photons else "disabled")).pack(pady=4)
    tk.Button(btns, text="PCH", width=26, pady=6,
              command=_pch,
              state=("normal" if have_photons else "disabled")).pack(pady=4)

    if not have_photons:
        tk.Label(win, text="Lifetime and PCH need the photon records in an\n"
                           ".fcs file — add one to the workspace to use them.",
                 font=("Helvetica", 9), fg="grey", justify="center").pack()

    tk.Button(win, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(pady=(0, 10))

    win.wait_window()


def _default_dataset_dir(fcs_data=None) -> Optional[Path]:
    """
    Folder the dataset browser should open on.

    Preference order: the active .fcs file's 'analysis' folder (where its
    correlation exports are written), then the folder beside it, then whatever
    folder was last used in this session, then None — meaning "no default",
    which leaves the dialog's list empty until the user browses.
    """
    if fcs_data is not None:
        start = Path(fcs_data.filepath).parent
        analysis = start / "analysis"
        return analysis if analysis.exists() else start
    if _last_browse_dir:
        prev = Path(_last_browse_dir)
        if prev.exists():
            return prev
    return None


def run_global_fit_dialog(fcs_data=None, parent=None, workspace_order=None):
    """
    Entry point for correlation modelling: select datasets, choose a model,
    set linking / guesses / bounds, then fit, plot and export.

    Works for one dataset (a plain single-curve fit) or many (global fit with
    linked parameters).  ``workspace_order`` (source .fcs file names in
    workspace order) orders the dataset list and the output rows to match.

    ``fcs_data`` is optional and used only to pick the browser's starting
    folder; the datasets are whatever CSVs the user selects.  Fit output is
    anchored to the first selected CSV (fcs_fitcommon.fits_dir maps
    ``<data>/analysis/x.csv`` to the sibling ``<data>/fits/``), which is the
    same destination the .fcs path used to give, and works with no .fcs at all.
    """
    init_dir = _default_dataset_dir(fcs_data)

    def _after_datasets(loaded):
        def _after_model(model):
            _global_setup_dialog(parent, model, loaded, loaded[0]["path"])
        _select_model_dialog(parent, _after_model)

    _select_datasets_dialog(parent, init_dir, _after_datasets,
                            order=workspace_order)


def _discover_correlation_csvs(folder: Path) -> list:
    """
    Return CSVs in ``folder`` that parse as correlation exports (have a lag
    axis and a G column), regardless of filename.  Robust to custom names.
    """
    folder = Path(folder)
    found = []
    if folder.exists():
        for p in sorted(folder.glob("*.csv")):
            try:
                load_correlation_csv(p)
                found.append(p)
            except Exception:
                continue
    return found


def _source_name_of_csv(path: Path) -> str:
    """Return the originating .fcs file name recorded in a correlation CSV header."""
    try:
        _, _, _, _, meta = load_correlation_csv(path)
        return (meta.get("source file") or meta.get("source") or "").strip()
    except Exception:
        return ""


def _order_paths_by_workspace(paths: list, order_names: list) -> list:
    """
    Sort correlation CSV *paths* to match the workspace file order.

    Each CSV is matched to its source .fcs (via the 'source file' header) and
    ordered by that file's position in *order_names*.  Datasets whose source
    is not in the workspace (e.g. files added by hand) are kept after the
    workspace ones, ordered alphabetically.
    """
    order_index = {name: i for i, name in enumerate(order_names)}
    src = {p: _source_name_of_csv(p) for p in paths}
    return sorted(
        paths,
        key=lambda p: (order_index.get(src[p], len(order_names)), p.name.lower()),
    )


def _apply_remembered_order(paths: list) -> list:
    """
    Re-apply the manual row order confirmed earlier in this session.

    Paths that were present last time keep their hand-made positions; anything
    new (a CSV written since, or a file just added) follows afterwards in the
    order it was passed in.  Returns *paths* unchanged when nothing has been
    remembered yet, so a fresh session still gets the workspace ordering.
    """
    if not _last_dataset_order:
        return list(paths)
    rank = {s: i for i, s in enumerate(_last_dataset_order)}
    n = len(rank)
    decorated = [(rank.get(str(p), n), i, p) for i, p in enumerate(paths)]
    return [p for _, _, p in sorted(decorated, key=lambda t: (t[0], t[1]))]


def _select_datasets_dialog(parent, init_dir, on_done, order=None):
    """
    Screen — build the list of correlation CSVs to fit.

    The list is the user's to curate: seeded by discovering *init_dir* (may be
    None, in which case it starts empty), then extended by browsing for files
    or folders anywhere, pruned with Remove, and ordered with Move ↑ / ↓.  The
    arrangement and the removals last for the session; "Reset list" undoes both
    and re-discovers the current folder.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    init_dir = Path(init_dir) if init_dir else None
    # Session-local working copy of the removals — committed only by Next →.
    removed = set(_last_dataset_removed)

    path_list = _discover_correlation_csvs(init_dir) if init_dir else []
    path_list = [p for p in path_list if str(p) not in removed]
    if order:
        path_list = _order_paths_by_workspace(path_list, order)
    # A manual arrangement made earlier this session wins over the workspace
    # ordering; on the first visit this is a no-op.
    path_list = _apply_remembered_order(path_list)

    win = tk.Toplevel(parent)
    win.title("Select correlation datasets")
    win.geometry("640x520")
    win.minsize(560, 440)
    win.grab_set()

    tk.Label(win, text="Select datasets to include in the fit",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text="Selected (highlighted) rows are included.  "
                       "Ctrl-click toggles one row · Shift-click selects a range · "
                       "click empty space below the list to clear.\n"
                       "Row order sets the dataset order in the fit and its "
                       "output — use Move ↑ / Move ↓ to change it.",
             font=("Helvetica", 9), fg="grey",
             wraplength=600, justify="left").pack()

    folder_var = tk.StringVar(value="")

    def _update_folder():
        folder_var.set(f"Folder: {init_dir}" if init_dir
                       else "Folder: none chosen yet")

    tk.Label(win, textvariable=folder_var, font=("Courier", 8), fg="grey",
             wraplength=600, justify="left").pack()

    lb_frame = tk.Frame(win)
    lb_frame.pack(fill="both", expand=True, padx=12, pady=6)
    scroll = tk.Scrollbar(lb_frame, orient="vertical")
    listbox = tk.Listbox(lb_frame, selectmode="extended",
                         yscrollcommand=scroll.set, activestyle="none",
                         font=("Courier", 9))
    scroll.config(command=listbox.yview)
    scroll.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    info = tk.StringVar(value="")

    def _populate(select_all=True, keep=None, keep_indices=None):
        """Refill the listbox from *path_list*.

        Selection is restored either by path (*keep*) or by row index
        (*keep_indices*, used by the reorder buttons, which know exactly which
        rows moved and are unambiguous even if a path appears twice).
        """
        keep = keep or set()
        keep_indices = set(keep_indices or ())
        listbox.delete(0, tk.END)
        for i, p in enumerate(path_list):
            listbox.insert(tk.END, p.name)
            if select_all or p in keep or i in keep_indices:
                listbox.selection_set(i)
        _update_info()

    def _update_info(*_):
        if not path_list:
            info.set("Nothing listed — use 'Add files…' or 'Browse folder…'.")
        else:
            info.set(f"{len(listbox.curselection())} of {len(path_list)} included")

    def _selected_paths() -> set:
        return {path_list[i] for i in listbox.curselection()}

    def _click_deselect(event):
        """File-explorer feel: clicking the empty space below the rows clears."""
        idx = listbox.nearest(event.y)
        bbox = listbox.bbox(idx) if idx >= 0 else None
        if bbox is None or event.y > bbox[1] + bbox[3]:
            listbox.selection_clear(0, tk.END)
            _update_info()
            return "break"          # swallow it, so Tk does not re-select a row
        # On a real row: fall through to Tk's native extended-select handling,
        # which is what provides plain-click / Ctrl-click / Shift-click.

    def _select_all(*_):
        listbox.selection_set(0, tk.END)
        _update_info()
        return "break"

    def _select_none(*_):
        listbox.selection_clear(0, tk.END)
        _update_info()

    def _move(delta):
        """Shift the selected row(s) up (delta=-1) or down (delta=+1) by one.

        A multi-row selection moves as a block and keeps its internal order;
        rows that would run off the end, or push into another selected row,
        simply stay put.
        """
        sel = sorted(listbox.curselection())
        if not sel:
            return
        n = len(path_list)
        sel_set = set(sel)
        items = [{"path": p, "sel": i in sel_set} for i, p in enumerate(path_list)]
        # Move the leading edge of the block first, so rows never leapfrog.
        seq = [items[i] for i in (sel if delta < 0 else reversed(sel))]
        pos = {id(it): k for k, it in enumerate(items)}
        for it in seq:
            cur = pos[id(it)]
            tgt = cur + delta
            if tgt < 0 or tgt >= n or items[tgt]["sel"]:
                continue
            items[cur], items[tgt] = items[tgt], items[cur]
            pos[id(items[cur])] = cur
            pos[id(items[tgt])] = tgt
        path_list[:] = [it["path"] for it in items]
        moved = [k for k, it in enumerate(items) if it["sel"]]
        _populate(select_all=False, keep_indices=moved)
        if moved:
            listbox.see(min(moved))
            listbox.see(max(moved))

    def _reset_list():
        """Undo this session's curation: re-discover the folder, drop removals.

        Only local state is touched — the remembered order/removals are
        overwritten wholesale by Next →, so Cancel after a Reset still leaves
        the previous session state intact.  Files added by hand from other
        folders are dropped too: this is a full reset to "what is in the
        current folder", not a partial one.
        """
        if not messagebox.askyesno(
                "Reset list",
                "Rebuild the list from the current folder?\n\n"
                "This undoes the manual order and restores removed rows.\n"
                "Files added by hand from other folders are dropped.\n"
                "(No file on disk is affected.)",
                parent=win):
            return
        removed.clear()
        base = _discover_correlation_csvs(init_dir) if init_dir else []
        if order:
            base = _order_paths_by_workspace(base, order)
        path_list[:] = base
        _populate(select_all=True)

    listbox.bind("<<ListboxSelect>>", _update_info)
    listbox.bind("<Button-1>", _click_deselect)
    listbox.bind("<Control-a>", _select_all)
    listbox.bind("<Control-A>", _select_all)
    _populate(select_all=True)

    def _browse_dir_str() -> str:
        """initialdir for the file/folder browsers ('' lets Tk pick the cwd)."""
        return str(init_dir) if init_dir else ""

    def _extend(new_paths, keep):
        """Append paths that are not already listed; un-remove any re-added."""
        added = 0
        for p in new_paths:
            removed.discard(str(p))     # an explicit re-add overrides a Remove
            if p not in path_list:
                path_list.append(p)
                keep.add(p)
                added += 1
        return added

    def _add_files():
        nonlocal init_dir
        new = filedialog.askopenfilenames(
            title="Add correlation CSV files",
            initialdir=_browse_dir_str(),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"),
                       ("Correlation CSV", "*correlation*.csv"),
                       ("All files", "*.*")],
            parent=win)
        if not new:
            return
        keep = _selected_paths()
        added = _extend([Path(n) for n in new], keep)
        init_dir = Path(new[0]).parent
        _update_folder()
        _populate(select_all=False, keep=keep)
        if added == 0:
            messagebox.showinfo("Already listed",
                                "Every file picked is already in the list.",
                                parent=win)

    def _browse_folder():
        """Point the dialog at any folder of saved correlations."""
        nonlocal init_dir
        chosen = filedialog.askdirectory(
            title="Choose a folder of correlation CSVs",
            initialdir=_browse_dir_str(), mustexist=True, parent=win)
        if not chosen:
            return
        folder = Path(chosen)
        found = _discover_correlation_csvs(folder)
        if not found:
            messagebox.showinfo(
                "No correlation CSVs",
                f"Nothing in '{folder.name}' parsed as a correlation export.\n\n"
                "These are the CSVs written by 'Export plotted data to CSV'; "
                "they need a lag axis and a G column.",
                parent=win)
            return
        keep = _selected_paths()
        added = _extend(found, keep)
        init_dir = folder
        _update_folder()
        _populate(select_all=False, keep=keep)
        if added == 0:
            messagebox.showinfo("Already listed",
                                f"All {len(found)} correlation CSVs in "
                                f"'{folder.name}' are already in the list.",
                                parent=win)

    def _remove_files():
        """Drop the selected row(s) from the list (never from disk).

        Selection does double duty here — it marks both what gets fitted and
        what Remove acts on — so the rows that stay keep whatever inclusion
        state they had; if that would leave nothing included, everything left
        is re-included rather than handing back an empty list.
        """
        sel = sorted(listbox.curselection(), reverse=True)
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Select the row(s) to remove from the list first.",
                                parent=win)
            return
        keep = _selected_paths() - {path_list[i] for i in sel}
        for i in sel:
            removed.add(str(path_list[i]))
            del path_list[i]
        if path_list and not keep:
            _populate(select_all=True)
        else:
            _populate(select_all=False, keep=keep)

    # ── List management ──────────────────────────────────────────────────────
    lst = tk.Frame(win)
    lst.pack(fill="x", padx=12, pady=(0, 2))
    tk.Label(lst, text="List:", font=("Helvetica", 8), fg="grey",
             width=7, anchor="w").pack(side="left")
    tk.Button(lst, text="Add files…", command=_add_files,
              width=11, pady=3).pack(side="left", padx=(0, 4))
    tk.Button(lst, text="Browse folder…", command=_browse_folder,
              width=13, pady=3).pack(side="left", padx=4)
    tk.Button(lst, text="Remove", command=_remove_files,
              fg="tomato", width=9, pady=3).pack(side="left", padx=4)
    tk.Button(lst, text="Reset list", command=_reset_list,
              width=10, pady=3).pack(side="left", padx=4)

    # ── Selection / reorder tools ────────────────────────────────────────────
    tools = tk.Frame(win)
    tools.pack(fill="x", padx=12, pady=(0, 2))
    tk.Label(tools, text="Include:", font=("Helvetica", 8), fg="grey",
             width=7, anchor="w").pack(side="left")
    tk.Button(tools, text="Select all", command=_select_all,
              width=9, pady=3).pack(side="left", padx=(0, 4))
    tk.Button(tools, text="Select none", command=_select_none,
              width=9, pady=3).pack(side="left", padx=4)
    tk.Label(tools, text="Order:", font=("Helvetica", 8), fg="grey",
             anchor="w").pack(side="left", padx=(14, 2))
    tk.Button(tools, text="Move ↑", command=lambda: _move(-1),
              width=8, pady=3).pack(side="left", padx=4)
    tk.Button(tools, text="Move ↓", command=lambda: _move(+1),
              width=8, pady=3).pack(side="left", padx=4)

    _update_folder()
    _update_info()

    tk.Label(win, textvariable=info, font=("Helvetica", 9), fg="grey").pack()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        global _last_dataset_order, _last_dataset_removed, _last_browse_dir
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("No datasets",
                                "Select at least one dataset (click a row to "
                                "highlight it).",
                                parent=win)
            return
        # Commit the curated list: the arrangement of every row (included or
        # not, so excluded datasets keep their place if re-included later), the
        # rows removed by hand, and the folder to come back to next time.
        _last_dataset_order = [str(p) for p in path_list]
        _last_dataset_removed = set(removed)
        if init_dir:
            _last_browse_dir = str(init_dir)
        chosen = [path_list[i] for i in sel]
        loaded, errors = [], []
        for p in chosen:
            try:
                tau, G, Gstd, _cols, meta = load_correlation_csv(p)
                loaded.append({"name": p.stem, "path": p, "meta": meta,
                               "tau": tau, "G": G, "sigma": Gstd})
            except Exception as e:
                errors.append(f"{p.name}: {e}")
        if errors:
            messagebox.showerror("Some files could not be read",
                                 "\n".join(errors), parent=win)
        if not loaded:
            return
        win.destroy()
        on_done(loaded)

    tk.Button(btns, text="Next →", command=_next,
              width=12, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", command=win.destroy,
              width=10, pady=4).pack(side="left", padx=6)

    win.wait_window()


def _global_setup_dialog(parent, model, datasets, out_source):
    """Screen — per-parameter link / guess / bounds / fix, then fit."""
    import tkinter as tk
    from tkinter import messagebox

    D = len(datasets)
    all_have_sigma = all(
        ds["sigma"] is not None and np.isfinite(ds["sigma"]).any()
        for ds in datasets
    )

    win = tk.Toplevel(parent)
    win.title(f"Global fit — {model.name}")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text=f"Global fit — {model.name}",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text=f"{D} dataset{'s' if D != 1 else ''}  ·  {model.formula}",
             font=("Helvetica", 9), fg="grey").pack(pady=(0, 6))

    guesses0 = combined_guess(model, datasets)

    table = tk.Frame(win, padx=12, pady=4)
    table.pack(fill="x")
    for c, h in enumerate(["Parameter", "Link", "Guess", "Lower", "Upper", "Fix"]):
        tk.Label(table, text=h, font=("Helvetica", 10, "bold")).grid(
            row=0, column=c, padx=4, pady=(0, 4))

    link_vars: Dict[str, tk.BooleanVar] = {}
    guess_vars: Dict[str, tk.StringVar] = {}
    lower_vars: Dict[str, tk.StringVar] = {}
    upper_vars: Dict[str, tk.StringVar] = {}
    fixed_vars: Dict[str, tk.BooleanVar] = {}

    for r, p in enumerate(model.params, start=1):
        label = p.name + (f" ({p.unit})" if p.unit else "")
        tk.Label(table, text=label, anchor="w", width=12).grid(
            row=r, column=0, sticky="w", padx=4, pady=2)

        lkv = tk.BooleanVar(value=p.link_default if D > 1 else False)
        gv  = tk.StringVar(value=f"{guesses0.get(p.name, p.default):.6g}")
        lv  = tk.StringVar(value=_fmt(p.lower))
        uv  = tk.StringVar(value=_fmt(p.upper))
        fv  = tk.BooleanVar(value=p.fixed)

        tk.Checkbutton(table, variable=lkv,
                       state="normal" if D > 1 else "disabled").grid(row=r, column=1, padx=4)
        tk.Entry(table, textvariable=gv, width=12).grid(row=r, column=2, padx=4)
        tk.Entry(table, textvariable=lv, width=10).grid(row=r, column=3, padx=4)
        tk.Entry(table, textvariable=uv, width=10).grid(row=r, column=4, padx=4)
        tk.Checkbutton(table, variable=fv).grid(row=r, column=5, padx=4)

        link_vars[p.name]  = lkv
        guess_vars[p.name] = gv
        lower_vars[p.name] = lv
        upper_vars[p.name] = uv
        fixed_vars[p.name] = fv

    note = ("Linked = one shared value across all datasets; "
            "unlinked = an independent value per dataset.")
    if D == 1:
        note = "Linking applies with 2+ datasets (only one selected)."
    tk.Label(win, text=note, font=("Helvetica", 9), fg="grey",
             wraplength=440, justify="left").pack(fill="x", padx=12, pady=(4, 0))

    weight_var = tk.BooleanVar(value=all_have_sigma)
    tk.Checkbutton(
        win,
        text="Weight by σ (G_std)" if all_have_sigma
        else "Weight by σ — unavailable (a dataset lacks G_std)",
        variable=weight_var, anchor="w",
        state="normal" if all_have_sigma else "disabled",
    ).pack(fill="x", padx=12, pady=(6, 0))

    # How many channels this would actually remove, so the effect of the
    # option is visible before the fit rather than discovered afterwards.
    _n_starved = 0
    for _ds in datasets:
        _y = np.asarray(_ds["G"], dtype=np.float64)
        _n_starved += int(np.count_nonzero(
            np.isfinite(_y) & (_y <= _PAIR_STARVED_CUTOFF)))

    starved_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        win,
        text=(f"Drop pair-starved bins (G ≤ {_PAIR_STARVED_CUTOFF:g}) "
              f"— {_n_starved} found"),
        variable=starved_var, anchor="w",
    ).pack(fill="x", padx=12, pady=(2, 0))
    if _n_starved:
        tk.Label(
            win,
            text=("      those lags recorded no photon pairs, so their G is a "
                  "floor value\n      and their σ is spuriously small"),
            font=("Helvetica", 8), fg="grey", anchor="w", justify="left",
        ).pack(fill="x", padx=12)

    # Optional label folded into the output filenames, so a set of fits can
    # be told apart in a file listing (and in Excel's window list) without
    # opening them.
    name_row = tk.Frame(win)
    name_row.pack(fill="x", padx=12, pady=(8, 0))
    tk.Label(name_row, text="Save as:", anchor="w",
             font=("Helvetica", 9)).pack(side="left")
    name_var = tk.StringVar(value="")
    tk.Entry(name_row, textvariable=name_var,
             font=("Helvetica", 9)).pack(side="left", fill="x", expand=True,
                                         padx=(4, 0))
    tk.Label(
        win,
        text=("      optional label; a timestamp is always appended, e.g.\n"
              "      «myLabel»_20260723_142530_params.csv"),
        font=("Helvetica", 8), fg="grey", anchor="w", justify="left",
    ).pack(fill="x", padx=12)

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
                                 "(use 'inf' / '-inf').", parent=win)
            return
        linked = {n: (link_vars[n].get() if D > 1 else False) for n in link_vars}
        fixed  = {n: fixed_vars[n].get() for n in fixed_vars}
        for n in guesses:
            if lowers[n] >= uppers[n]:
                messagebox.showerror("Invalid bounds",
                                     f"For '{n}', lower must be < upper.", parent=win)
                return
        
        weighted = all_have_sigma and weight_var.get()

        try:
            result = fit_global(model, datasets, linked, guesses,
                                lowers, uppers, fixed, weighted=weighted,
                                drop_pair_starved=starved_var.get())
        except Exception as e:
            messagebox.showerror("Fit failed", str(e), parent=win)
            return

        # Read the label before the window goes away.
        _name = name_var.get().strip()
        win.destroy()
        report_path, _curve, _params = export_global_fit(
            result, out_source, name_override=_name)
        
        
        fig, _axes = plot_global_fit(result, show=False)
        # Vector by default: see fcs_plottools._FIGURE_FORMAT.  save_figure
        # swallows its own failures, so a figure that cannot be written never
        # costs us the report and tables already on disk.
        fcs_plottools.save_figure(fig, report_path.with_suffix(""))

        linked_lines = [
            f"{n} = {result['datasets'][0]['values'][n]:.4g} (linked)"
            for n in result["names"] if result["linked"].get(n)
        ]
        gof = (f"red. χ² = {result['red_chi2']:.3g}"
               if result["weighted"] else f"{result['n_datasets']} datasets")
        messagebox.showinfo(
            "Global fit complete",
            f"{model.name}\n\n"
            + ("\n".join(linked_lines) if linked_lines else "(no linked parameters)")
            + f"\n\n{gof}\n\nResults saved to:\n{report_path.parent}",
            parent=parent,
        )
        #plot.show was static; fcs_plottools is dynamic
        #plt.show()
        #fcs_plottools.show_figure(fig, np.array([ax, axr]))
        fcs_plottools.show_figure(fig, _axes)
        
    tk.Button(btns, text="Fit", width=12, command=_do_fit, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


def _select_model_dialog(parent, on_choose):
    """Screen 1 — choose a model from the registry (extensible)."""
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Fit — select model")
    win.geometry("525x400")
    win.minsize(440,360)
    win.resizable(True, True)
    win.grab_set()

    tk.Label(win, text="Select a fit model",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    models = fcs_models.list_models()
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
        m = fcs_models.get_model(key_var.get())
        desc.config(state="normal")
        desc.delete("1.0", tk.END)
        desc.insert(tk.END, m.description)
        desc.config(state="disabled")

    key_var.trace_add("write", _refresh_desc)
    _refresh_desc()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        m = fcs_models.get_model(key_var.get())
        win.destroy()
        on_choose(m)

    tk.Button(btns, text="Next →", width=12, command=_next, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fcs_fit.py <correlation.csv> [model_key]")
        print("Available models:")
        for m in fcs_models.list_models():
            print(f"  {m.key:<22} {m.name}")
        sys.exit(1)

    path = Path(sys.argv[1])
    key  = sys.argv[2] if len(sys.argv) > 2 else fcs_models.list_models()[0].key
    model = fcs_models.get_model(key)

    tau_s, G, G_std, _cols, _meta = load_correlation_csv(path)
    guesses = auto_guess(model, tau_s, G)
    lowers = {p.name: p.lower for p in model.params}
    uppers = {p.name: p.upper for p in model.params}
    fixed  = {p.name: p.fixed for p in model.params}
    linked = {p.name: False for p in model.params}          # single dataset: nothing to link

    weighted = G_std is not None and np.isfinite(G_std).any()
    datasets = [{"name": path.stem, "path": path, "meta": _meta or {},
                 "tau": tau_s, "G": G, "sigma": G_std}]

    result = fit_global(model, datasets, linked, guesses,
                        lowers, uppers, fixed, weighted=weighted)
    export_global_fit(result, path)
    plot_global_fit(result)
