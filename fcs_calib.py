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

try:
    from scipy.stats import chi2 as _chi2dist
    _HAVE_SCIPY = True
except Exception:                       # scipy is a suite dependency; degrade
    _HAVE_SCIPY = False                 # gracefully (GOF metric just disabled)

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


def load_concentrations(path, names) -> Tuple[np.ndarray, Optional[str], str]:
    """
    Read known concentrations for a calibration from a file and align them to
    the dataset ``names`` shown in the dialog.

    Accepted layouts (delimiter may be comma, tab, or whitespace):
      * a table with a ``dataset`` column and a concentration column — mapped by
        dataset name, so order does not matter and extra/missing rows are OK;
      * a single column of numbers (or ``name value`` pairs), one per dataset in
        the displayed order.

    Lines starting with '#' are comments; a '# unit : X' line, if present, sets
    the returned unit (when it names a known unit).

    Returns (conc, unit, note): ``conc`` is a float array aligned to ``names``
    (NaN where unknown), ``unit`` is a recognised unit string or None, and
    ``note`` summarises how the mapping went.
    """
    path = Path(path)
    meta: Dict[str, str] = {}
    rows: List[List[str]] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                body = line.strip().lstrip("#").strip()
                if ":" in body:
                    k, v = body.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
                continue
            if not line.strip():
                continue
            if "," in line:
                cells = [c.strip() for c in line.rstrip("\n").split(",")]
            elif "\t" in line:
                cells = [c.strip() for c in line.rstrip("\n").split("\t")]
            else:
                cells = line.split()
            rows.append(cells)

    if not rows:
        raise ValueError(f"No data found in {path.name}.")

    unit = meta.get("unit")
    if unit is not None and unit not in _CONC_FACTOR:
        unit = None

    def _num(s):
        try:
            return float(s)
        except (ValueError, TypeError):
            return float("nan")

    conc = np.full(len(names), np.nan, float)
    lower0 = [c.lower() for c in rows[0]]

    if "dataset" in lower0:                      # map by dataset name
        header = lower0
        d_idx = header.index("dataset")
        conc_names = ("concentration", "conc", "c", "known", "value",
                      "nm", "µm", "um", "mm", "pm", "m")
        c_idx = next((header.index(h) for h in conc_names if h in header), None)
        if c_idx is None:
            c_idx = next((k for k in range(len(header)) if k != d_idx), None)
        if c_idx is None:
            raise ValueError("File has a 'dataset' column but no value column.")
        mapping = {r[d_idx].strip(): _num(r[c_idx])
                   for r in rows[1:] if len(r) > max(d_idx, c_idx)}
        matched = 0
        for i, nm in enumerate(names):
            if nm in mapping:
                conc[i] = mapping[nm]
                matched += 1
        note = f"matched {matched}/{len(names)} datasets by name"
        n_missing = sum(1 for nm in names if nm not in mapping)
        if n_missing:
            note += f"; {n_missing} left blank"
    else:                                        # positional (displayed order)
        vals = []
        for r in rows:
            nums = [x for x in (_num(c) for c in r) if np.isfinite(x)]
            vals.append(nums[-1] if nums else float("nan"))
        for i in range(min(len(names), len(vals))):
            conc[i] = vals[i]
        note = f"loaded {len(vals)} value(s) by position (need {len(names)})"

    return conc, unit, note


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


def _through_origin_stats(x, y, yerr=None) -> dict:
    """
    Weighted through-origin fit with full goodness-of-fit statistics for one
    candidate window.  Returns a dict with:

        n, slope, s_err_int, s_err_ext, chi2, dof, red_chi2, gof_Q, Sxx

    ``s_err_int`` is the statistical (internal) slope error assuming the given
    sigma are correct absolute errors; ``s_err_ext`` is the Birge-scaled
    (external) error = s_err_int * sqrt(chi2_red).  When ``yerr`` is None the
    weights are uniform and the scale is taken from the residual scatter, so
    red_chi2 is 1 by construction and gof_Q is undefined.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    weighted = yerr is not None
    if weighted:
        yerr = np.asarray(yerr, float)
        w = np.where(yerr > 0, 1.0 / np.square(yerr), np.nan)
    else:
        w = np.ones_like(x)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    n = int(x.size)
    out = {"n": n, "slope": float("nan"), "s_err_int": float("nan"),
           "s_err_ext": float("nan"), "chi2": float("nan"),
           "dof": max(0, n - 1), "red_chi2": float("nan"),
           "gof_Q": float("nan"), "Sxx": float("nan")}
    if n < 1:
        return out
    Sxx = float(np.sum(w * x * x))
    Sxy = float(np.sum(w * x * y))
    if Sxx <= 0:
        return out
    s = Sxy / Sxx
    resid = y - s * x
    dof = max(1, n - 1)
    if weighted:
        chi2 = float(np.sum(w * resid * resid))
        red = chi2 / dof
        s_err_int = float(np.sqrt(1.0 / Sxx))
        s_err_ext = float(np.sqrt(max(red, 0.0) / Sxx))
        Q = (float(_chi2dist.sf(chi2, dof))
             if (_HAVE_SCIPY and dof > 0) else float("nan"))
    else:
        ssr = float(np.sum(resid * resid))          # unweighted
        sigma2 = ssr / dof                          # residual variance
        chi2 = ssr
        red = 1.0                                   # by construction
        s_err_int = s_err_ext = float(np.sqrt(sigma2 / Sxx))
        Q = float("nan")
    out.update({"slope": float(s), "s_err_int": s_err_int,
                "s_err_ext": s_err_ext, "chi2": chi2, "dof": dof,
                "red_chi2": red, "gof_Q": Q, "Sxx": Sxx})
    return out


def select_linear_subset(conc, N, N_err, use_weights: bool = True,
                         metric: str = "veff_err", n_min: int = 3,
                         alpha: float = 0.05) -> dict:
    """
    Choose the contiguous concentration window that best represents the linear
    calibration regime.

    Points are ordered by concentration; every contiguous window of at least
    ``n_min`` points is fit to <N> = s.C through the origin and scored by
    ``metric``:

      "veff_err" : minimise the *relative* external uncertainty of the
                   calibration constant,  sqrt(chi2_red / Sxx) / |s|.
                   Rewards more / higher-leverage points (larger Sxx), penalises
                   non-linear points via chi2_red, needs no tuning constant, and
                   is invariant to a global rescaling of the error bars.

      "gof"      : take the LARGEST window whose reduced chi-square is
                   statistically acceptable (survival Q >= alpha), tie-broken by
                   the largest Q.  Falls back to the window closest to
                   chi2_red = 1 if nothing passes.  Requires real error bars.

    Returns a dict:
        used     : boolean mask in the ORIGINAL point order (True = keep)
        order    : original indices sorted by concentration
        table    : per-window stat dicts (for the report / plot)
        best     : the chosen window record (or None)
        metric, n_min, alpha, fallback, weighted
    """
    conc = np.asarray(conc, float)
    N = np.asarray(N, float)
    have_err = (N_err is not None and np.all(np.isfinite(N_err))
                and np.all(np.asarray(N_err) > 0))
    weighted = bool(have_err and use_weights)
    N_err_arr = np.asarray(N_err, float) if have_err else None

    valid = np.isfinite(conc) & np.isfinite(N)
    idx_valid = np.nonzero(valid)[0]
    m = idx_valid.size
    n_min = max(2, int(n_min))

    table: List[dict] = []
    result = {"used": valid.copy(), "order": idx_valid, "table": table,
              "metric": metric, "n_min": n_min, "alpha": float(alpha),
              "best": None, "fallback": False, "weighted": weighted}

    if m < max(n_min, 2):               # too few points to trim: keep them all
        return result
    if metric == "gof" and not weighted:
        metric = "veff_err"             # GOF needs real error bars
        result["metric"] = metric

    order = idx_valid[np.argsort(conc[idx_valid])]
    for i in range(m):
        for j in range(i + n_min - 1, m):
            win = order[i:j + 1]
            yee = (N_err_arr[win] if weighted else None)
            st = _through_origin_stats(conc[win], N[win], yee)
            if not np.isfinite(st["slope"]) or st["slope"] <= 0:
                continue
            table.append({
                "i": int(i), "j": int(j), "n": int(win.size), "idx": win.copy(),
                "c_lo": float(conc[win].min()), "c_hi": float(conc[win].max()),
                "slope": st["slope"], "red_chi2": st["red_chi2"],
                "gof_Q": st["gof_Q"], "s_err_ext": st["s_err_ext"],
                "veff_err_rel": st["s_err_ext"] / st["slope"],
            })

    if not table:
        return result

    if metric == "gof":
        accepted = [r for r in table
                    if np.isfinite(r["gof_Q"]) and r["gof_Q"] >= alpha]
        if accepted:
            best = max(accepted, key=lambda r: (r["n"], r["gof_Q"]))
        else:
            result["fallback"] = True
            best = min(table, key=lambda r: abs(r["red_chi2"] - 1.0))
    else:                               # "veff_err"
        best = min(table, key=lambda r: r["veff_err_rel"])

    used = np.zeros_like(valid, dtype=bool)
    used[best["idx"]] = True
    result["used"] = used
    result["best"] = best
    return result


def _combine_replicates(Nvals, sig) -> dict:
    """
    Combine one group of replicate occupancies with a weighted mean and the
    two-sided external ("Birge") error.

    With within-replicate variances σ_i² and weights w_i = 1/σ_i²:

        ybar     = Σ w_i y_i / Σ w_i                   (weighted mean)
        Q        = Σ w_i (y_i - ybar)^2                (Cochran's Q)
        chi2_red = Q / (n - 1)
        σ_int    = sqrt(1 / Σ w_i)                     (internal error of mean)
        σ_ext    = σ_int * sqrt(chi2_red)              (external / Birge error)
        N        = ybar
        σ_N      = max(σ_ext, s/sqrt(n))               (floored at the raw SEM)

    The external error rescales the internal error by the observed scatter: it
    INFLATES the bar when replicates disagree more than their stated errors
    (chi2_red > 1) and DEFLATES it when they agree better (chi2_red < 1).  The
    deflation is the point of this stopgap: while the correlator is noisy the
    fit-reported N_err is far larger than the replicate reproducibility, so the
    external error tracks the reproducibility instead of the inflated bar.

    This supersedes a one-sided random-effects (DerSimonian-Laird) error, which
    can only inflate and so cannot correct over-large bars.  Caveats: on small n
    the Birge factor is noisy (hence the SEM floor); a singleton keeps its own
    error; with no usable intrinsic errors the plain mean ± s/sqrt(n) is used.

    Returns a dict: N, N_err, sd, sig_int, sig_ext, chi2_red, Q, n.
    """
    Nvals = np.asarray(Nvals, float)
    n = int(Nvals.size)
    sig = (np.asarray(sig, float) if sig is not None else np.full(n, np.nan))
    fin = np.isfinite(sig) & (sig > 0)
    sd = float(np.std(Nvals, ddof=1)) if n >= 2 else 0.0
    nanf = float("nan")

    if n == 1:
        s0 = float(sig[0]) if fin[0] else nanf
        return {"N": float(Nvals[0]), "N_err": s0, "sd": 0.0,
                "sig_int": s0, "sig_ext": s0, "chi2_red": nanf, "Q": nanf, "n": 1}

    if fin.all():
        w = 1.0 / sig ** 2
        Sw = float(np.sum(w))
        ybar = float(np.sum(w * Nvals) / Sw)
        Q = float(np.sum(w * (Nvals - ybar) ** 2))
        chi2_red = Q / (n - 1)
        sig_int = float(np.sqrt(1.0 / Sw))
        sig_ext = sig_int * float(np.sqrt(chi2_red))
        sem = sd / np.sqrt(n)
        sig_N = max(sig_ext, sem)                 # never below the raw SEM
        if not (sig_N > 0):                       # degenerate: identical reps
            sig_N = sig_int
        return {"N": ybar, "N_err": float(sig_N), "sd": sd,
                "sig_int": sig_int, "sig_ext": float(sig_ext),
                "chi2_red": float(chi2_red), "Q": Q, "n": n}

    # No usable intrinsic errors: plain mean +/- SEM.
    return {"N": float(np.mean(Nvals)),
            "N_err": (float(sd / np.sqrt(n)) if n >= 2 else nanf),
            "sd": sd, "sig_int": nanf, "sig_ext": nanf,
            "chi2_red": nanf, "Q": nanf, "n": n}


def collapse_replicates(names, N, N_err, conc, rtol: float = 1e-6,
                        atol: float = 0.0) -> tuple:
    """
    Merge points that share the same known concentration into one aggregate
    point (via :func:`_combine_replicates`), so replicate measurements used to
    test precision are not treated as independent concentrations by the
    linear-range finder or the weighted fit.

    Two concentrations are treated as equal when
    |c_i - c_j| <= atol + rtol*max(|c_i|, |c_j|).  Singletons pass through
    unchanged.  Groups are returned in first-appearance (workspace) order.

    Returns (names2, N2, N_err2, conc2, groups), where each groups entry is a
    dict: {conc, members, N, N_err, sd, sig_int, sig_ext, chi2_red, Q, n}.
    """
    N = np.asarray(N, float)
    conc = np.asarray(conc, float)
    have_err = N_err is not None
    sig_all = (np.asarray(N_err, float) if have_err
               else np.full(N.shape, np.nan))

    # Cluster by concentration on a sorted axis (single-linkage within tol),
    # then restore first-appearance order via the earliest member index.
    order = np.argsort(conc, kind="stable")
    clusters: List[List[int]] = []
    for k in order:
        if clusters and not np.isnan(conc[clusters[-1][-1]]) \
                and not np.isnan(conc[k]):
            ref = conc[clusters[-1][-1]]
            if abs(conc[k] - ref) <= atol + rtol * max(abs(conc[k]), abs(ref)):
                clusters[-1].append(k)
                continue
        clusters.append([k])
    clusters.sort(key=lambda cl: min(cl))

    names2: List[str] = []
    N2: List[float] = []
    Ne2: List[float] = []
    conc2: List[float] = []
    groups: List[dict] = []
    for cl in clusters:
        idx = np.array(sorted(cl), int)
        n = int(idx.size)
        g = _combine_replicates(N[idx], sig_all[idx])
        members = [names[i] for i in idx]
        label = members[0] if n == 1 else f"{members[0]} (+{n - 1} rep)"
        c_agg = float(np.mean(conc[idx]))
        names2.append(label)
        N2.append(g["N"])
        Ne2.append(g["N_err"])
        conc2.append(c_agg)
        groups.append({"conc": c_agg, "members": members, **g})

    N_err_out = np.array(Ne2, float) if have_err else None
    return (names2, np.array(N2, float), N_err_out,
            np.array(conc2, float), groups)


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
    collapse: bool = True,
    collapse_rtol: float = 1e-6,
    select: str = "auto",
    select_metric: str = "veff_err",
    n_min: int = 3,
    alpha_gof: float = 0.05,
) -> dict:
    """
    Fit <N> = s·C through the origin and derive alpha (C = alpha·<N>) and,
    when the unit is recognised, the effective volume V_eff.

    ``corrected`` records whether the supplied <N> is background-corrected
    (for labelling only).
    """
    n_raw = len(names)
    N = np.asarray(N, float)
    conc = np.asarray(conc, float)

    # ── Collapse replicate concentrations ─────────────────────────────────────
    groups = None
    if collapse:
        names, N, N_err, conc, groups = collapse_replicates(
            names, N, N_err, conc, rtol=collapse_rtol)

    have_err = (N_err is not None and np.all(np.isfinite(N_err)) and np.all(N_err > 0))

    # ── Linear-range selection ────────────────────────────────────────────────
    valid = np.isfinite(conc) & np.isfinite(N)
    sel = None
    if select == "auto":
        sel = select_linear_subset(
            conc, N, (N_err if have_err else None),
            use_weights=use_weights, metric=select_metric,
            n_min=n_min, alpha=alpha_gof,
        )
        used = np.asarray(sel["used"], bool)
    else:
        used = valid.copy()
    if used.sum() < 1:                  # never fit on nothing
        used = valid.copy()

    xu  = conc[used]
    yu  = N[used]
    yeru = (N_err[used] if (have_err and use_weights) else None)

    s, s_err, r2 = fit_through_origin(xu, yu, yeru)
    slope_free, intercept = fit_with_intercept(xu, yu, yeru)
    st = _through_origin_stats(xu, yu, yeru)     # red_chi2 / GOF on the subset

    alpha = 1.0 / s
    alpha_err = s_err / (s * s)

    veff = None
    f = _CONC_FACTOR.get(unit)
    if f:
        veff = s / f                    # µm³

    excluded = [names[i] for i in range(len(names)) if not used[i]]

    return {
        "names": names, "N": N, "N_err": N_err, "conc": conc, "unit": unit,
        "used": used, "excluded": excluded,
        "n_used": int(used.sum()), "n_total": int(valid.sum()),
        "slope": s, "slope_err": s_err, "r2": r2,
        "alpha": alpha, "alpha_err": alpha_err,
        "intercept": intercept, "slope_free": slope_free,
        "veff_um3": veff, "weighted": yeru is not None,
        "corrected": corrected,
        "groups": groups, "collapsed": bool(collapse), "n_raw": int(n_raw),
        "red_chi2": st["red_chi2"], "gof_Q": st["gof_Q"],
        "slope_err_ext": st["s_err_ext"],
        "select": select,
        "select_metric": (sel["metric"] if sel else None),
        "selection": sel,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_calibration(result: dict, show: bool = True):
    conc = result["conc"]
    N = result["N"]
    N_err = result["N_err"]
    unit = result["unit"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5), layout="constrained")

    used = np.asarray(result.get("used", np.ones(len(conc), bool)), bool)
    excl = (~used) & np.isfinite(conc) & np.isfinite(N)
    have_err = N_err is not None and np.all(np.isfinite(N_err))

    # Shade the trimmed low-/high-C regions.
    if excl.any() and used.any():
        lo, hi = float(conc[used].min()), float(conc[used].max())
        if conc[excl].min() < lo:
            ax.axvspan(0, lo, color="0.9", alpha=0.5, zorder=0)
        if conc[excl].max() > hi:
            ax.axvspan(hi, conc.max() * 1.05, color="0.9", alpha=0.5, zorder=0)

    def _pts(mask, **kw):
        if not mask.any():
            return
        fmt = kw.pop("fmt", "o")
        if have_err:
            ax.errorbar(conc[mask], N[mask], yerr=N_err[mask], fmt=fmt,
                        capsize=3, markersize=5, zorder=3, **kw)
        else:
            ax.plot(conc[mask], N[mask], fmt, markersize=5, zorder=3, **kw)

    _pts(used, color="steelblue",
         label=("data (fit)" if excl.any() else "data"))
    _pts(excl, color="0.6", markerfacecolor="none", label="excluded")

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
    if np.isfinite(result.get("red_chi2", float("nan"))):
        qline = (f"   (Q={result['gof_Q']:.2g})"
                 if np.isfinite(result.get("gof_Q", float("nan"))) else "")
        box.append(f"χ²_red = {result['red_chi2']:.3g}{qline}")
    if result.get("n_used") is not None:
        box.append(f"points used = {result['n_used']}/{result['n_total']}")
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
    if np.isfinite(result.get("red_chi2", float("nan"))):
        qtxt = (f"   (GOF Q = {result['gof_Q']:.3g})"
                if np.isfinite(result.get("gof_Q", float("nan"))) else "")
        L.append(f"  reduced χ²     = {result['red_chi2']:.4g}{qtxt}")
    sel = result.get("selection")
    if result.get("select") == "auto" and sel is not None:
        L.append("")
        L.append("Linear-range selection")
        L.append("-" * 60)
        L.append(f"metric     : {sel.get('metric')}"
                 + ("   [fallback: no window met the GOF threshold]"
                    if sel.get("fallback") else ""))
        L.append(f"n_min      : {sel.get('n_min')}    "
                 f"points used: {result['n_used']}/{result['n_total']}")
        if result["excluded"]:
            L.append(f"excluded   : {', '.join(result['excluded'])}")
        tbl = sel.get("table") or []
        if tbl:
            best = sel.get("best")
            L.append("")
            L.append(f"    {'n':>2}  {'C range':>18}  {'slope':>9}  "
                     f"{'chi2_red':>9}  {'GOF Q':>8}  {'relErr(s)':>10}")
            for r in sorted(tbl, key=lambda r: (r["i"], r["j"])):
                star = " *" if (best is not None and r["i"] == best["i"]
                                and r["j"] == best["j"]) else "  "
                q = r["gof_Q"]
                qs = f"{q:.3g}" if np.isfinite(q) else "—"
                L.append(f"{star}  {r['n']:>2}  "
                         f"{r['c_lo']:>8.3g}–{r['c_hi']:<9.3g}  "
                         f"{r['slope']:>9.4g}  {r['red_chi2']:>9.3g}  "
                         f"{qs:>8}  {r['veff_err_rel']:>10.3g}")
            L.append("  (* = selected window)")
    groups = result.get("groups")
    if result.get("collapsed") and groups and any(g["n"] > 1 for g in groups):
        L.append("")
        L.append("Replicate aggregation  (weighted mean, external/Birge error)")
        L.append("-" * 60)
        L.append(f"{result.get('n_raw', '?')} points collapsed to {len(groups)} "
                 f"by shared concentration")
        for g in groups:
            if g["n"] <= 1:
                continue
            L.append(f"  {g['conc']:.4g} {unit}   n={g['n']}   "
                     f"<N> = {g['N']:.4g} ± {g['N_err']:.4g}")
            if np.isfinite(g.get("chi2_red", float("nan"))):
                L.append(f"      Q = {g['Q']:.3g} (dof {g['n'] - 1}), "
                         f"χ²_red = {g['chi2_red']:.3g};  "
                         f"err {g['sig_int']:.4g} → {g['N_err']:.4g}  "
                         f"(raw SD {g['sd']:.4g})")
            else:
                L.append(f"      plain mean ± SEM (no intrinsic errors), "
                         f"raw SD = {g['sd']:.4g}")
            L.append(f"      members: {', '.join(g['members'])}")
    L.append("")
    L.append("Points")
    L.append("-" * 60)
    L.append(f"{'dataset':<24}{unit:>10}{'<N>':>12}{'<N>_err':>12}")
    for i, nm in enumerate(result["names"]):
        ne = (result["N_err"][i] if result["N_err"] is not None else float("nan"))
        L.append(f"{nm:<24}{result['conc'][i]:>10.4g}{result['N'][i]:>12.4g}{ne:>12.4g}")
    report_path.write_text("\n".join(L), encoding="utf-8")

    used = result.get("used")
    with points_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write("# FCS volume calibration points\n")
        fh.write(f"# unit : {unit}\n")
        fh.write(f"# C = alpha * <N> ; alpha = {result['alpha']:.6g} {unit}/molecule\n")
        fh.write(f"# fit uses only rows with used=1 "
                 f"({result.get('n_used', '?')}/{result.get('n_total', '?')} points)\n")
        groups = result.get("groups")
        fh.write("dataset,concentration,N,N_err,N_fit,used,n_rep\n")
        for i, nm in enumerate(result["names"]):
            ne = (result["N_err"][i] if result["N_err"] is not None else float("nan"))
            nfit = result["slope"] * result["conc"][i]
            u = 1 if (used is None or bool(used[i])) else 0
            nr = groups[i]["n"] if groups else 1
            fh.write(f"{nm},{result['conc'][i]:.10g},{result['N'][i]:.10g},"
                     f"{ne:.10g},{nfit:.10g},{u},{nr}\n")

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

    imp = tk.Frame(win, padx=12)
    imp.pack(fill="x", pady=(2, 0))

    def _import_conc():
        fpath = filedialog.askopenfilename(
            title="Import known concentrations",
            initialdir=str(csv_path.parent),
            filetypes=[("CSV / text", "*.csv *.txt *.tsv"), ("All files", "*.*")],
            parent=win,
        )
        if not fpath:
            return
        try:
            vals, funit, note = load_concentrations(fpath, names)
        except Exception as e:
            messagebox.showerror("Import failed", str(e), parent=win)
            return
        for cv, v in zip(conc_vars, vals):
            cv.set("" if not np.isfinite(v) else f"{v:g}")
        if funit in ("pM", "nM", "µM", "uM", "mM", "M"):
            unit_var.set(funit)
        messagebox.showinfo("Concentrations imported", note, parent=win)

    tk.Button(imp, text="Import concentrations from file…",
              command=_import_conc).pack(side="left")
    tk.Label(imp, text="  (dataset,concentration  —  or one value per row)",
             font=("Helvetica", 8), fg="grey").pack(side="left")

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

    collapse_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text="Collapse repeated concentrations",
                   variable=collapse_var, anchor="w").pack(fill="x", padx=12, pady=(4, 0))

    sel_frame = tk.LabelFrame(win, text="Linear-range selection", padx=12, pady=6)
    sel_frame.pack(fill="x", padx=12, pady=(8, 0))
    auto_var = tk.BooleanVar(value=True)
    tk.Checkbutton(sel_frame, text="Auto-select the linear range",
                   variable=auto_var, anchor="w").grid(
        row=0, column=0, columnspan=4, sticky="w")
    tk.Label(sel_frame, text="metric:").grid(row=1, column=0, sticky="e", pady=(4, 0))
    metric_var = tk.StringVar(value="veff_err")
    tk.OptionMenu(sel_frame, metric_var, "veff_err", "gof").grid(
        row=1, column=1, sticky="w", padx=(4, 12), pady=(4, 0))
    tk.Label(sel_frame, text="min points:").grid(row=1, column=2, sticky="e", pady=(4, 0))
    nmin_var = tk.StringVar(value="3")
    tk.Entry(sel_frame, textvariable=nmin_var, width=5).grid(
        row=1, column=3, sticky="w", padx=4, pady=(4, 0))

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
            n_min = int(float(nmin_var.get()))
        except ValueError:
            n_min = 3
        try:
            result = calibrate(names, N, N_err if has_err else None, conc,
                               unit=unit, use_weights=weight_var.get(),
                               corrected=use_corr,
                               collapse=collapse_var.get(),
                               select=("auto" if auto_var.get() else "none"),
                               select_metric=metric_var.get(), n_min=n_min)
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
        msg += f"R² = {result['r2']:.5f}\n"
        if result.get("collapsed") and result.get("n_raw") != result.get("n_total"):
            msg += (f"replicates collapsed: {result['n_raw']} → "
                    f"{result['n_total']} points\n")
        if result.get("select") == "auto":
            msg += (f"points used: {result['n_used']}/{result['n_total']}"
                    f"  (metric: {result.get('select_metric')})\n")
            if result["excluded"]:
                msg += f"excluded: {', '.join(result['excluded'])}\n"
        msg += f"\nSaved to:\n{report_path.parent}"
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
        print("Usage: python fcs_calib.py <params.csv> <c1,c2,...|conc_file> [unit]")
        print("  concentrations: one per dataset row, comma-separated, or a file")
        sys.exit(1)
    rows, _ = load_params_csv(sys.argv[1])
    names = [r["dataset"] for r in rows]
    N = np.array([r.get("N", np.nan) for r in rows], float)
    N_err = np.array([r.get("N_err", np.nan) for r in rows], float)
    arg2 = sys.argv[2]
    unit = sys.argv[3] if len(sys.argv) > 3 else "nM"
    if Path(arg2).exists():
        conc, funit, note = load_concentrations(arg2, names)
        print(f"[calib] {note}")
        if funit and len(sys.argv) <= 3:
            unit = funit
    else:
        conc = np.array([float(x) for x in arg2.split(",")], float)
    res = calibrate(names, N, N_err if np.all(np.isfinite(N_err)) else None,
                    conc, unit=unit)
    export_calibration(res, Path(sys.argv[1]))
    plot_calibration(res)
