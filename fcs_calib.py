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
import fcs_export

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
    # Parsing lives in fcs_export.read_export so the header format is defined
    # once.  read_export returns columns; this function's callers expect one
    # dict per row, so the orientation is flipped here.
    path = Path(path)
    meta, columns = fcs_export.read_export(path)
    if not columns:
        raise ValueError(f"No header row found in {path.name}.")

    n_rows = len(next(iter(columns.values())))
    out: List[dict] = []
    for i in range(n_rows):
        d: dict = {}
        for k, arr in columns.items():
            v = arr[i]
            if k == "dataset":
                d[k] = str(v)
            else:
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
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


# ── Linear-range criteria ─────────────────────────────────────────────────────

# A point lies "on the line" when its relative deviation from the fitted
# through-origin model is within tolerance.  The two points at the ENDS of a
# candidate window get a looser tolerance than the interior: roll-off sets in
# gradually, so insisting on interior-grade agreement at the very edges
# shortens the window for no physical reason.
_MAX_REL_DEV      = 0.15        # interior points
_MAX_REL_DEV_EDGE = 0.20        # the first and last point of the window

# Sentinel shown in the error-column dropdown when the calibration is to
# run unweighted.
_NO_ERR_COL = "(none — unweighted)"

# Residual panel: half-height as a multiple of the edge tolerance.  1.5 keeps
# the +/-15% and +/-20% guides comfortably inside the frame while leaving room
# to see how far a marginal point sits beyond them.
_RESID_SPAN_FACTOR = 1.5

# Use log axes once the concentration series spans at least this many decades.
_LOG_SCALE_DECADES = 10.0


def _window_linearity(x, y, yerr, max_dev, max_dev_edge):
    """
    Fit y = s.x through the origin over one window and score its linearity.

    Returns (slope, rel_dev, score):

      rel_dev : |y - s.x| / (s.x) for each point -- deviation FROM THE FITTED
                LINE, not the point's own error bar.  A point can be measured
                very precisely and still sit far off the line; that is exactly
                what leaving the linear regime looks like, and it is what this
                criterion is meant to catch.
      score   : max over points of rel_dev / (that point's allowance).
                score <= 1 means every point is within tolerance.  When
                nothing passes, the value itself ranks windows by how far the
                worst point overshoots, so "least bad" needs no second metric.

    A non-positive or non-finite slope scores infinite: the model is unusable.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    st = _through_origin_stats(x, y, yerr)
    s_ = st["slope"]
    if not np.isfinite(s_) or s_ <= 0:
        return s_, np.full(x.size, np.inf), np.inf

    fit = s_ * x
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(y - fit) / np.abs(fit)
    rel = np.where(np.isfinite(rel), rel, np.inf)

    allow = np.full(x.size, float(max_dev))
    allow[0] = allow[-1] = float(max_dev_edge)   # window ends are the limits
    score = float(np.max(rel / allow))
    return s_, rel, score


def select_linear_subset(conc, N, N_err, use_weights: bool = True,
                         n_min: int = 3,
                         max_rel_dev: float = _MAX_REL_DEV,
                         max_rel_dev_edge: float = _MAX_REL_DEV_EDGE) -> dict:
    """
    Choose the contiguous concentration window that represents the linear
    calibration regime.

    Criterion
    ---------
    Points are ordered by concentration.  Every contiguous window of at least
    ``n_min`` points is fit to <N> = s.C through the origin, and the window is
    accepted when no point deviates from that line by more than
    ``max_rel_dev`` (default 15%), with ``max_rel_dev_edge`` (default 20%)
    allowed at the window's own first and last point.

    Among accepted windows the LARGEST is chosen; ties go to the window at
    lower concentration.  If no window is accepted, the one whose worst point
    overshoots its allowance by the least is used and ``fallback`` is set, so
    the caller can say so rather than presenting a bad range as a good one.

    Every contiguous window is enumerated rather than trimmed greedily from
    the ends.  The fit changes each time a point is dropped, so a point that
    failed can pass afterwards; a greedy trim can stop early and miss the
    largest valid window.  At the ~10 points a calibration typically has this
    is a few dozen fits.

    This replaces the earlier "veff_err" / "gof" metrics, both of which keyed
    off the ABSOLUTE scale of the error bars.  Those bars come from a fit
    covariance and are routinely optimistic several-fold, which made the
    selection swing between keeping everything and trimming perfectly linear
    points.  Deviation from the line does not depend on that scale at all.

    Returns a dict:
        used     : boolean mask in the ORIGINAL point order (True = keep)
        order    : original indices sorted by concentration
        table    : per-window stat dicts (for the report / plot)
        best     : the chosen window record (or None)
        metric, n_min, max_rel_dev, max_rel_dev_edge, fallback, weighted
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
              "metric": "rel_dev", "n_min": n_min,
              "max_rel_dev": float(max_rel_dev),
              "max_rel_dev_edge": float(max_rel_dev_edge),
              "best": None, "fallback": False, "weighted": weighted}

    if m < max(n_min, 2):               # too few points to trim: keep them all
        return result

    order = idx_valid[np.argsort(conc[idx_valid])]
    for i in range(m):
        for j in range(i + n_min - 1, m):
            win = order[i:j + 1]
            yee = (N_err_arr[win] if weighted else None)
            slope, rel, score = _window_linearity(
                conc[win], N[win], yee, max_rel_dev, max_rel_dev_edge)
            if not np.isfinite(slope) or slope <= 0:
                continue
            st = _through_origin_stats(conc[win], N[win], yee)
            table.append({
                "i": int(i), "j": int(j), "n": int(win.size), "idx": win.copy(),
                "c_lo": float(conc[win].min()), "c_hi": float(conc[win].max()),
                "slope": slope, "red_chi2": st["red_chi2"],
                "gof_Q": st["gof_Q"], "s_err_ext": st["s_err_ext"],
                "veff_err_rel": (st["s_err_ext"] / slope if slope else np.nan),
                "max_rel_dev": float(np.max(rel)) if rel.size else np.nan,
                "score": float(score),
                "passes": bool(score <= 1.0),
            })

    if not table:
        return result

    passing = [r for r in table if r["passes"]]
    if passing:
        # Most points; ties to the window sitting at lower concentration.
        best = max(passing, key=lambda r: (r["n"], -r["c_lo"]))
    else:
        # Least bad: smallest overshoot, then most points, then lower conc.
        result["fallback"] = True
        best = min(table, key=lambda r: (r["score"], -r["n"], r["c_lo"]))

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
    n_min: int = 3,
    max_rel_dev: float = _MAX_REL_DEV,
    max_rel_dev_edge: float = _MAX_REL_DEV_EDGE,
    y_label: str = "⟨N⟩",
    y_is_occupancy: bool = True,
) -> dict:
    """
    Fit Y = s·C through the origin and derive alpha (C = alpha·Y).

    Y is normally the mean occupancy <N>, in which case the slope also gives
    the effective volume V_eff.  Any other column from the fit parameter table
    may be supplied instead -- tauD against viscosity, brightness against
    labelling ratio, and so on -- and the straight-line calibration is done the
    same way.

    ``y_is_occupancy`` MUST be False for any Y that is not a molecule count.
    V_eff is derived as s / f(unit), an operation that silently assumes the
    slope carries units of molecules per concentration unit; applied to, say, a
    diffusion time it would return a confident number of cubic micrometres that
    means nothing at all.  When the flag is False, V_eff is left as None and
    every V_eff-specific label is suppressed.

    ``y_label`` is used for axis labels, the report, and the CSV header.
    ``corrected`` records whether a supplied <N> is background-corrected
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
            use_weights=use_weights, n_min=n_min,
            max_rel_dev=max_rel_dev, max_rel_dev_edge=max_rel_dev_edge,
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

    # V_eff only means anything when the slope is molecules per concentration
    # unit.  See the docstring: deriving it from an arbitrary Y would produce
    # an authoritative-looking volume with no physical content.
    veff = None
    if y_is_occupancy:
        f = _CONC_FACTOR.get(unit)
        if f:
            veff = s / f                # µm³

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
        "y_label": y_label, "y_is_occupancy": bool(y_is_occupancy),
        "groups": groups, "collapsed": bool(collapse), "n_raw": int(n_raw),
        "red_chi2": st["red_chi2"], "gof_Q": st["gof_Q"],
        "slope_err_ext": st["s_err_ext"],
        "select": select,
        "select_metric": (sel["metric"] if sel else None),
        "max_rel_dev": float(max_rel_dev),
        "max_rel_dev_edge": float(max_rel_dev_edge),
        "selection": sel,
    }


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_calibration(result: dict, show: bool = True):
    conc = result["conc"]
    N = result["N"]
    N_err = result["N_err"]
    unit = result["unit"]
    ylab = result.get("y_label") or "⟨N⟩"
    is_occ = result.get("y_is_occupancy", True)

    # Two panels sharing the concentration axis: the calibration itself, and
    # the relative deviation of each point from the fitted line.  The lower
    # panel is what the linear-range selection actually thresholds on, so
    # plotting it makes the selection inspectable -- and a systematic trend
    # (curvature, roll-off, a drifting standard) is visible there long before
    # it is obvious in the calibration plot, where a straight line through
    # points spanning two decades hides a great deal.
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(7.5, 6.8), layout="constrained",
        gridspec_kw={"height_ratios": [3, 1]})

    used = np.asarray(result.get("used", np.ones(len(conc), bool)), bool)
    excl = (~used) & np.isfinite(conc) & np.isfinite(N)

    # Decide the axis scales up front: the shaded "excluded" regions below
    # need a positive lower bound on a log axis, where zero is not a
    # representable coordinate.
    pos_c = conc[np.isfinite(conc)]
    pos_y = N[np.isfinite(N)]
    log_x = bool(pos_c.size and np.all(pos_c > 0)
                 and pos_c.max() / max(pos_c.min(), 1e-300)
                 >= _LOG_SCALE_DECADES)
    log_y = bool(log_x and pos_y.size and np.all(pos_y > 0))
    if log_x:
        lo_x = float(pos_c.min()) / 2.0
        hi_x = float(pos_c.max()) * 2.0
    else:
        lo_x, hi_x = 0.0, float(conc.max() * 1.05)
    have_err = N_err is not None and np.all(np.isfinite(N_err))

    # Shade the trimmed low-/high-C regions.
    if excl.any() and used.any():
        lo, hi = float(conc[used].min()), float(conc[used].max())
        for _a in (ax, axr):
            if conc[excl].min() < lo:
                _a.axvspan(lo_x, lo, color="0.9", alpha=0.5, zorder=0,
                           label="_excluded range (low C)")
            if conc[excl].max() > hi:
                _a.axvspan(hi, hi_x, color="0.9", alpha=0.5,
                           zorder=0, label="_excluded range (high C)")

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

    # A calibration series normally spans decades of concentration, so log
    # axes are the sensible default: on a linear axis the low-concentration
    # points pile up against the origin and the very deviations the fit is
    # judged on become invisible.  Log needs strictly positive data, so fall
    # back to linear whenever anything is non-positive (a blank, a
    # background-subtracted <N> that went negative).
    if log_x:
        xline = np.logspace(np.log10(lo_x), np.log10(hi_x), 200)
    else:
        xline = np.linspace(lo_x, hi_x, 100)

    ax.plot(xline, result["slope"] * xline, color="tomato", linewidth=1.6,
            label=f"fit {ylab} = s·C", zorder=2)

    # Zero reference lines are meaningless on a log axis (and matplotlib
    # cannot draw them), so they are only added on linear axes.
    if not log_y:
        ax.axhline(0, color="grey", linewidth=0.6)
    if not log_x:
        ax.axvline(0, color="grey", linewidth=0.6)

    if log_x:
        for _a in (ax, axr):
            _a.set_xscale("log")
        ax.set_xlim(lo_x, hi_x)
    if log_y:
        ax.set_yscale("log")

    box = [
        f"s     = {result['slope']:.4g} ± {result['slope_err']:.2g}  {ylab}/{unit}",
        f"α     = {result['alpha']:.4g} ± {result['alpha_err']:.2g}  "
        f"{unit}/{'molecule' if is_occ else ylab}",
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

    axr.set_xlabel(f"Known concentration C ({unit})", fontsize=12)
    if is_occ:
        ax.set_ylabel("Mean occupancy ⟨N⟩"
                      + ("  (bg-corrected)" if result.get("corrected")
                         else " = 1/G0"),
                      fontsize=12)
        ax.set_title("Effective-volume calibration  ·  C = α·⟨N⟩", fontsize=11)
    else:
        # Not an occupancy, so neither "mean occupancy" nor "effective volume"
        # applies; say what is actually plotted.
        ax.set_ylabel(ylab, fontsize=12)
        ax.set_title(f"Linear calibration  ·  C = α·{ylab}", fontsize=11)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.85)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # ── Lower panel: relative deviation from the fitted line ─────────────────
    # This is exactly the quantity select_linear_subset thresholds on:
    #     rel = (N - s.C) / (s.C)
    # Signed rather than absolute, because the SHAPE carries the diagnosis.
    # Random scatter about zero means the linear model holds; a smooth trend
    # away from zero at one end is roll-off; a bow across the whole range
    # means the model is wrong everywhere, which a single alpha would hide.
    slope = result["slope"]
    with np.errstate(divide="ignore", invalid="ignore"):
        N_fit = slope * conc
        rel = np.where(np.abs(N_fit) > 0, (N - N_fit) / N_fit, np.nan) * 100.0
        rel_err = (np.where(np.abs(N_fit) > 0, N_err / N_fit, np.nan) * 100.0
                   if have_err else None)

    tol_int = result.get("max_rel_dev")
    tol_edge = result.get("max_rel_dev_edge")
    sel = result.get("selection") or {}
    if tol_int is None:
        tol_int = sel.get("max_rel_dev", _MAX_REL_DEV)
    if tol_edge is None:
        tol_edge = sel.get("max_rel_dev_edge", _MAX_REL_DEV_EDGE)

    for lvl, style, lab in ((tol_int, ":", "interior"),
                            (tol_edge, "-.", "ends")):
        if lvl is None or not np.isfinite(lvl):
            continue
        for sign in (+1.0, -1.0):
            axr.axhline(sign * lvl * 100.0, color="tomato", linewidth=0.9,
                        linestyle=style, alpha=0.75, zorder=1,
                        label=f"_±{lvl*100:g}% tolerance ({lab})")
    axr.axhline(0.0, color="grey", linewidth=0.8, zorder=1,
                label="_zero deviation")

    def _rpts(mask, **kw):
        if not mask.any():
            return
        if rel_err is not None:
            axr.errorbar(conc[mask], rel[mask], yerr=rel_err[mask], fmt="o",
                         capsize=3, markersize=5, zorder=3, **kw)
        else:
            axr.plot(conc[mask], rel[mask], "o", markersize=5, zorder=3, **kw)

    _rpts(used, color="steelblue", label="_deviation (fit)")
    _rpts(excl, color="0.6", markerfacecolor="none", label="_deviation (excluded)")

    axr.set_ylabel("deviation\nfrom fit (%)", fontsize=10)
    axr.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    # Fixed scale, set by the TOLERANCE rather than by the data.
    #
    # Scaling to the data max is useless here: a point at the bottom of the
    # concentration range can sit thousands of percent off the line, which
    # compresses the +/-15% and +/-20% criteria into a hairline at the centre
    # and hides exactly the structure this panel exists to show.  The axis is
    # therefore pinned to a fixed multiple of the edge tolerance and outliers
    # are allowed to run off it, with a count annotated so nothing is silently
    # invisible.
    _tol_span = float(tol_edge) * 100.0 if tol_edge else 20.0
    span = _tol_span * _RESID_SPAN_FACTOR
    axr.set_ylim(-span, span)

    n_off = int(np.count_nonzero(np.isfinite(rel) & (np.abs(rel) > span)))
    if n_off:
        axr.annotate(
            f"{n_off} point{'s' if n_off != 1 else ''} beyond ±{span:.0f}%",
            xy=(0.99, 0.06), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=8, color="tomato",
        )

    if show:
        #plt.show was static; now dynamic w/ fcs_plottools
        #plt.show()
        fcs_plottools.show_figure(fig, (ax, axr))
    return fig, (ax, axr)


# ── Rebuilding a plot from an exported CSV ────────────────────────────────────

def _meta_num(meta: dict, key: str, default=None, cast=float):
    v = meta.get(key)
    if v is None or str(v).strip() == "":
        return default
    try:
        return cast(float(v))
    except (TypeError, ValueError):
        return default


def rebuild_plot(meta: dict, columns: dict, show: bool = True, path=None):
    """
    Rebuild a calibration figure from an exported *_calibration_points.csv.

    Everything the plot needs is in that one file: the points and their
    errors, which of them the fit used, and the fitted statistics.  No .fcs
    file, no parameter table and no re-fitting are involved -- the numbers
    shown are the ones that were actually fitted, not a re-derivation that
    could differ if the selection rules changed in the meantime.

    Older exports lack the statistics keys, since the header used to carry
    only the unit and alpha.  In that case the slope is recovered from the
    N_fit column (N_fit = slope . C) and the box shows what can be derived,
    so an old calibration still reopens rather than refusing.

    Parameters
    ----------
    meta, columns : from fcs_export.read_export
    show : passed through to plot_calibration
    path : unused; part of the rebuilder contract

    Returns
    -------
    fig, ax
    """
    for need in ("concentration", "N"):
        if need not in columns:
            raise ValueError(
                f"No '{need}' column: this does not look like a calibration "
                f"points export."
            )

    conc = np.asarray(columns["concentration"], float)
    N    = np.asarray(columns["N"], float)
    N_err = (np.asarray(columns["N_err"], float)
             if "N_err" in columns else np.zeros_like(N))
    used = (np.asarray(columns["used"], float) > 0.5
            if "used" in columns else np.ones(N.size, bool))

    slope = _meta_num(meta, "slope")
    if slope is None and "N_fit" in columns:
        # Pre-dates the slope key: N_fit = slope . C, so any point with a
        # non-zero concentration recovers it exactly.
        nfit = np.asarray(columns["N_fit"], float)
        ok = np.isfinite(nfit) & np.isfinite(conc) & (np.abs(conc) > 0)
        if ok.any():
            slope = float(np.median(nfit[ok] / conc[ok]))
    if slope is None:
        raise ValueError(
            "This export records neither a slope nor an N_fit column, so the "
            "calibration line cannot be drawn."
        )

    alpha = _meta_num(meta, "alpha", 1.0 / slope if slope else float("nan"))

    result = {
        "conc": conc,
        "N": N,
        "N_err": N_err,
        "used": used,
        "unit": meta.get("unit", "nM"),
        "slope": slope,
        "slope_err": _meta_num(meta, "slope_err", float("nan")),
        "alpha": alpha,
        "alpha_err": _meta_num(meta, "alpha_err", float("nan")),
        "intercept": _meta_num(meta, "intercept", float("nan")),
        "r2": _meta_num(meta, "r2", float("nan")),
        "veff_um3": _meta_num(meta, "veff_um3"),
        "red_chi2": _meta_num(meta, "red_chi2", float("nan")),
        "gof_Q": _meta_num(meta, "gof_Q", float("nan")),
        "n_used": _meta_num(meta, "n_used", int(np.count_nonzero(used)),
                            cast=int),
        "n_total": _meta_num(meta, "n_total", int(N.size), cast=int),
        "corrected": str(meta.get("corrected", "")).strip().lower() == "yes",
        # Default to occupancy so calibrations exported before Y became
        # selectable reopen exactly as they always did.
        "y_label": meta.get("y_label") or "⟨N⟩",
        "y_is_occupancy": str(
            meta.get("y_is_occupancy", "yes")).strip().lower() != "no",
        "y_column": meta.get("y_column"),
        "y_err_column": meta.get("y_err_column"),
        # Tolerances the range was selected with, so the reopened deviation
        # panel draws the criterion that was actually applied rather than
        # whatever the current defaults happen to be.
        "max_rel_dev": _meta_num(meta, "select_max_rel_dev"),
        "max_rel_dev_edge": _meta_num(meta, "select_max_rel_dev_edge"),
    }
    return plot_calibration(result, show=show)


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
    _ylab = result.get("y_label") or "⟨N⟩"
    if result.get("y_is_occupancy", True):
        L.append(f"<N> source : "
                 f"{'background-corrected (N_corr)' if result.get('corrected') else 'raw (1/G0)'}")
    else:
        L.append(f"Y column   : {result.get('y_column') or _ylab}   "
                 f"(not an occupancy — no V_eff derived)")
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
        L.append(f"criterion  : |N - fit| / fit  <=  "
                 f"{100*sel.get('max_rel_dev', float('nan')):.3g}% "
                 f"(interior), "
                 f"{100*sel.get('max_rel_dev_edge', float('nan')):.3g}% "
                 f"(window ends)"
                 + ("\n             [FALLBACK: no window met the criteria; "
                    "the least-bad window is shown]"
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
                     f"{'max dev':>8}  {'ok':>3}  {'chi2_red':>9}")
            for r in sorted(tbl, key=lambda r: (r["i"], r["j"])):
                star = " *" if (best is not None and r["i"] == best["i"]
                                and r["j"] == best["j"]) else "  "
                dev = r.get("max_rel_dev", float("nan"))
                devs = f"{100*dev:.1f}%" if np.isfinite(dev) else "—"
                L.append(f"{star}  {r['n']:>2}  "
                         f"{r['c_lo']:>8.3g}–{r['c_hi']:<9.3g}  "
                         f"{r['slope']:>9.4g}  {devs:>8}  "
                         f"{'yes' if r.get('passes') else 'no':>3}  "
                         f"{r['red_chi2']:>9.3g}")
            L.append("  (* = selected window;  'max dev' is the largest "
                     "|N - fit|/fit in that window)")
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
        # Machine-readable header, so this file can be reopened as a live plot
        # (see fcs_plotopen).  The banner above carries no colon, so a
        # "key : value" parser cannot read it.  Everything the plot draws --
        # the fit line, the statistics box, the shaded excluded regions --
        # comes from these keys plus the columns below, so the figure is
        # reconstructible from this file alone.
        fh.write("# analysis : calibration points\n")
        fh.write(f"# unit : {unit}\n")
        fh.write(f"# slope : {result['slope']:.10g}\n")
        fh.write(f"# slope_err : {result['slope_err']:.10g}\n")
        fh.write(f"# alpha : {result['alpha']:.10g}\n")
        fh.write(f"# alpha_err : {result['alpha_err']:.10g}\n")
        fh.write(f"# intercept : {result['intercept']:.10g}\n")
        fh.write(f"# r2 : {result['r2']:.10g}\n")
        if result.get("veff_um3") is not None:
            fh.write(f"# veff_um3 : {result['veff_um3']:.10g}\n")
        if np.isfinite(result.get("red_chi2", float("nan"))):
            fh.write(f"# red_chi2 : {result['red_chi2']:.10g}\n")
        if np.isfinite(result.get("gof_Q", float("nan"))):
            fh.write(f"# gof_Q : {result['gof_Q']:.10g}\n")
        fh.write(f"# n_used : {result['n_used']}\n")
        fh.write(f"# n_total : {result['n_total']}\n")
        fh.write(f"# corrected : {'yes' if result.get('corrected') else 'no'}\n")
        # Which column was calibrated, so a reopened plot is labelled with the
        # quantity that was actually fitted rather than assuming occupancy.
        fh.write(f"# y_label : {result.get('y_label') or '⟨N⟩'}\n")
        fh.write(f"# y_is_occupancy : "
                 f"{'yes' if result.get('y_is_occupancy', True) else 'no'}\n")
        if result.get("y_column"):
            fh.write(f"# y_column : {result['y_column']}\n")
        if result.get("y_err_column"):
            fh.write(f"# y_err_column : {result['y_err_column']}\n")
        sel = result.get("selection")
        if sel is not None:
            fh.write(f"# select_max_rel_dev : {sel.get('max_rel_dev')}\n")
            fh.write(f"# select_max_rel_dev_edge : "
                     f"{sel.get('max_rel_dev_edge')}\n")
            fh.write(f"# select_fallback : "
                     f"{'yes' if sel.get('fallback') else 'no'}\n")
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

    # Every numeric column is a candidate Y.  <N> stays the default, but a
    # calibration is just a straight line through the origin, so any fitted
    # quantity can be calibrated against concentration the same way.
    def _numeric_cols(rr):
        out = []
        for k in rr[0]:
            if k == "dataset":
                continue
            vals = [r.get(k, np.nan) for r in rr]
            if any(np.isfinite(v) for v in vals if isinstance(v, float)):
                out.append(k)
        return out

    all_cols = _numeric_cols(rows)
    y_choices = [c for c in all_cols if not c.endswith("_err")] or all_cols
    err_choices = [_NO_ERR_COL] + all_cols

    default_y = "N_corr" if use_corr else "N"
    if default_y not in y_choices:
        default_y = y_choices[0]

    def _pair_err(ycol):
        """The natural error column for *ycol*, if the table has one."""
        cand = f"{ycol}_err"
        return cand if cand in all_cols else _NO_ERR_COL

    ycol_var  = tk.StringVar(value=default_y)
    yerr_var  = tk.StringVar(value=_pair_err(default_y))

    names = [r["dataset"] for r in rows]

    def _read_y():
        """Current Y and its error, straight from the loaded table."""
        yc = ycol_var.get()
        ec = yerr_var.get()
        y = np.array([r.get(yc, np.nan) for r in rows], float)
        if ec and ec != _NO_ERR_COL:
            ye = np.array([r.get(ec, np.nan) for r in rows], float)
        else:
            ye = np.full(len(rows), np.nan)
        return y, ye

    N, N_err = _read_y()
    has_err = bool(np.all(np.isfinite(N_err)) and np.all(N_err > 0))
    win = tk.Toplevel(parent)
    win.title("Calibration — Y vs concentration")
    win.resizable(True, True)
    win.grab_set()

    tk.Label(win, text="Enter the known concentration for each dataset",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text=f"Source: {csv_path.name}",
             font=("Helvetica", 9), fg="grey").pack()
    ysel = tk.Frame(win, padx=12, pady=2)
    ysel.pack(fill="x")
    tk.Label(ysel, text="Y column:", anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    tk.OptionMenu(ysel, ycol_var, *y_choices).pack(side="left", padx=(4, 12))
    tk.Label(ysel, text="its error:", anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    tk.OptionMenu(ysel, yerr_var, *err_choices).pack(side="left", padx=4)

    ynote_var = tk.StringVar(value="")
    tk.Label(win, textvariable=ynote_var, font=("Helvetica", 9),
             justify="left", anchor="w").pack(fill="x", padx=12)

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


    #ADDING BELOW:
    table_container = tk.Frame(win)
    table_container.pack(fill="both", expand=True, padx=12, pady=4)

    canvas = tk.Canvas(table_container, height=300)
    scrollbar = tk.Scrollbar(table_container,
                         orient="vertical",
                         command=canvas.yview)

    table = tk.Frame(canvas)

    table.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    canvas.create_window((0, 0), window=table, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    # END ADDITION -- next 2 lines also commented out


    #table = tk.Frame(win, padx=12, pady=4)
    #table.pack(fill="x")
    tk.Label(table, text="dataset", font=("Helvetica", 10, "bold")).grid(
        row=0, column=0, sticky="w", padx=4, pady=(0, 4))
    yhdr_var = tk.StringVar(value=default_y)
    tk.Label(table, textvariable=yhdr_var,
             font=("Helvetica", 10, "bold")).grid(row=0, column=1, padx=4)
    tk.Label(table, text="concentration", font=("Helvetica", 10, "bold")).grid(
        row=0, column=2, padx=4)

    conc_vars = []
    yval_vars = []
    for r, nm in enumerate(names, start=1):
        tk.Label(table, text=nm, anchor="w", width=22,
                 font=("Courier", 9)).grid(row=r, column=0, sticky="w", padx=4)
        yv = tk.StringVar(value="")
        yval_vars.append(yv)
        tk.Label(table, textvariable=yv, anchor="e", width=16,
                 fg="grey").grid(row=r, column=1, padx=4)
        cv = tk.StringVar(value="")
        tk.Entry(table, textvariable=cv, width=12).grid(row=r, column=2, padx=4)
        conc_vars.append(cv)

    weight_var = tk.BooleanVar(value=has_err)
    weight_cb = tk.Checkbutton(
        win, text="Weight by 1/err²", variable=weight_var, anchor="w",
    )
    weight_cb.pack(fill="x", padx=12, pady=(4, 0))

    # ── Keep the table and the weighting option in step with the Y choice ────
    def _refresh_y(*_):
        nonlocal N, N_err, has_err
        N, N_err = _read_y()
        has_err = bool(np.all(np.isfinite(N_err)) and np.all(N_err > 0))

        yc = ycol_var.get()
        yhdr_var.set(yc)
        for i, v in enumerate(yval_vars):
            val = N[i]
            if not np.isfinite(val):
                v.set("—")
            elif has_err:
                v.set(f"{val:.4g} ± {N_err[i]:.2g}")
            else:
                v.set(f"{val:.4g}")

        weight_var.set(has_err)
        weight_cb.config(
            state="normal" if has_err else "disabled",
            text=("Weight by 1/err²  (from "
                  f"{yerr_var.get()})" if has_err
                  else "Weight — unavailable (no usable error column)"),
        )

        # V_eff is only meaningful when Y is an occupancy.  Say so plainly
        # rather than quietly producing a volume from, say, a diffusion time.
        if yc in ("N", "N_corr"):
            ynote_var.set(
                "Using background-corrected ⟨N⟩ (N_corr)" if yc == "N_corr"
                else "Using raw ⟨N⟩ = 1/G0")
        else:
            ynote_var.set(
                f"Calibrating '{yc}' — not an occupancy, so no V_eff is "
                f"derived (slope and α are still reported)")

    ycol_var.trace_add("write",
                       lambda *_: (yerr_var.set(_pair_err(ycol_var.get())),
                                   _refresh_y()))
    yerr_var.trace_add("write", _refresh_y)
    _refresh_y()

    collapse_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text="Collapse repeated concentrations",
                   variable=collapse_var, anchor="w").pack(fill="x", padx=12, pady=(4, 0))

    sel_frame = tk.LabelFrame(win, text="Linear-range selection", padx=12, pady=6)
    sel_frame.pack(fill="x", padx=12, pady=(8, 0))
    auto_var = tk.BooleanVar(value=True)
    tk.Checkbutton(sel_frame, text="Auto-select the linear range",
                   variable=auto_var, anchor="w").grid(
        row=0, column=0, columnspan=4, sticky="w")
    tk.Label(sel_frame, text="max dev %:").grid(row=1, column=0, sticky="e",
                                               pady=(4, 0))
    dev_var = tk.StringVar(value=f"{100*_MAX_REL_DEV:g}")
    tk.Entry(sel_frame, textvariable=dev_var, width=5).grid(
        row=1, column=1, sticky="w", padx=(4, 12), pady=(4, 0))
    tk.Label(sel_frame, text="at ends %:").grid(row=1, column=2, sticky="e",
                                                pady=(4, 0))
    edge_var = tk.StringVar(value=f"{100*_MAX_REL_DEV_EDGE:g}")
    tk.Entry(sel_frame, textvariable=edge_var, width=5).grid(
        row=1, column=3, sticky="w", padx=4, pady=(4, 0))
    tk.Label(sel_frame, text="min points:").grid(row=2, column=0, sticky="e",
                                                 pady=(4, 0))
    nmin_var = tk.StringVar(value="3")
    tk.Entry(sel_frame, textvariable=nmin_var, width=5).grid(
        row=2, column=1, sticky="w", padx=4, pady=(4, 0))
    tk.Label(sel_frame,
             text="A point is kept when it sits within\n"
                  "'max dev' of the fitted line; the two\n"
                  "points at the ends of the range get\n"
                  "the looser 'at ends' tolerance.",
             font=("Helvetica", 8), fg="grey", justify="left",
             anchor="w").grid(row=3, column=0, columnspan=4, sticky="w",
                              pady=(6, 0))

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
            max_dev = float(dev_var.get()) / 100.0
            edge_dev = float(edge_var.get()) / 100.0
            if not (0 < max_dev and 0 < edge_dev):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "The deviation tolerances must be positive percentages.",
                parent=win)
            return
        try:
            _yc = ycol_var.get()
            _is_occ = _yc in ("N", "N_corr")
            _ylab = "⟨N⟩" if _is_occ else _yc
            result = calibrate(names, N, N_err if has_err else None, conc,
                               unit=unit, use_weights=weight_var.get(),
                               corrected=(_yc == "N_corr"),
                               collapse=collapse_var.get(),
                               select=("auto" if auto_var.get() else "none"),
                               n_min=n_min, max_rel_dev=max_dev,
                               max_rel_dev_edge=edge_dev,
                               y_label=_ylab, y_is_occupancy=_is_occ)
            result["y_column"] = _yc
            result["y_err_column"] = (yerr_var.get()
                                      if yerr_var.get() != _NO_ERR_COL
                                      else None)
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
                    f"  (max dev {100*result.get('max_rel_dev', 0):.3g}% / "
                    f"{100*result.get('max_rel_dev_edge', 0):.3g}% at ends)\n")
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
