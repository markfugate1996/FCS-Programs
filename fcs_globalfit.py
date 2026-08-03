"""
fcs_globalfit.py
================
The linked ("global") least-squares fit, shared by every analysis in the suite.

    fcs_fit.py            correlation curves      G(tau)
    fcs_pch_fit.py        counting histograms     n_k
    fcs_lifetime_fit.py   TCSPC decays            counts(t)

What a global fit is
--------------------
One model is fitted to SEVERAL datasets at once.  Each model parameter is one
of three things:

    fixed     held at its guess for every dataset; no free variable at all
    linked    ONE shared free variable used by every dataset
    unlinked  one independent free variable per dataset

Nothing linked and D datasets is a batch fit: D independent fits solved in one
pass, which is the same answer you would get one at a time, but with one report
and one parameter table at the end.  Linking a parameter is what makes it a
global fit, and it is the whole point: a quantity that is genuinely shared --
one diffusion time across a dilution series, one lifetime across a titration,
one molecular brightness across a concentration series -- is determined far
better by all the data together than by any dataset alone, and the parameters
it was trading against are freed to differ.

Why this module exists
----------------------
The layout above is arithmetic, not physics.  It knows about a free-parameter
vector, an index map from (parameter, dataset) to a slot in it, and a residual
built by concatenation.  It does NOT need to know whether the curve is a
correlation, a histogram or a decay.  That was already true of the original
implementation in fcs_fit.fit_global, which touched its data only through
``model.func(tau, **values)``; extracting it costs a callback and buys the same
capability for the other two analyses instead of two more copies of a subtle
piece of code drifting apart.

What an analysis supplies
-------------------------
One function::

    predict(ds, x, values) -> y_model

``ds`` is the caller's own dataset dict, so anything the model needs beyond the
parameters -- a sampled-bin count M, an IRF, a peak index, a window origin --
travels with the dataset rather than through this module.  That is the seam:

    correlation     model.func(tau, **values)
    PCH             M * model.func(k, **values)
    lifetime tail   model.func(t_rel, **values)

Weighting
---------
Weighted fitting divides each residual by that point's sigma, which makes the
sum of squares a chi-squared and the reduced chi-squared interpretable.  It is
used only when EVERY included dataset carries a usable sigma: mixing weighted
and unweighted datasets in one sum silently gives the unweighted ones a weight
of 1 in units the others do not share, so the total is not a chi-squared of
anything.  Better to fit them all unweighted and say so.

On linking parameters that should not be linked
-----------------------------------------------
This module does not police that, and that is deliberate.  Whether two datasets
share a lifetime is a question about the experiment, not about the arithmetic,
and the person running the fit is the one who knows.  But two hazards are worth
stating once, because both are silent:

* Scale-like parameters (amplitudes, occupations, brightness per bin,
  background offsets) depend on the acquisition settings.  PCH ``epsilon`` is
  counts per molecule PER BIN, so it scales with bin width; lifetime amplitudes
  scale with bin width and are measured from the fit-window origin.  Linking
  one across datasets acquired differently constrains two quantities that are
  not the same quantity.
* Rate-like parameters (diffusion times, lifetimes, brightness per second) do
  not have this problem and are the usual thing to link.

A fit that links a scale-like parameter across mismatched settings will
converge, and report an answer, and be wrong.  It is reported in full in the
result so the report can show what was linked.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares


__all__ = ["prepare_datasets", "fit_linked", "combined_guess_from"]


# ── Preparation ───────────────────────────────────────────────────────────────

def prepare_datasets(
    datasets: Sequence[dict],
    weighted: bool = False,
    extra_mask: Optional[Callable[[dict, np.ndarray, np.ndarray], np.ndarray]] = None,
    min_points: int = 2,
) -> List[dict]:
    """
    Mask each dataset to the points that can actually be fitted.

    Keeps points where x and y are both finite, and -- when *weighted* -- where
    sigma is finite and strictly positive, since a zero sigma is an infinite
    weight and a negative one is meaningless.

    *extra_mask* is the hook for censoring that only one analysis knows about:
    it receives ``(ds, x, y)`` and returns a boolean array of points to KEEP.
    Correlation uses it to drop pair-starved lag channels, whose G values are
    censored rather than measured.  Nothing here needs to know that; it just
    needs somewhere for the caller to say so.

    Returns a new list of dicts carrying the masked arrays plus every other key
    of the original, so per-dataset context (M, irf, meta) survives untouched.
    """
    prepped: List[dict] = []
    for ds in datasets:
        x = np.asarray(ds["x"], dtype=np.float64)
        y = np.asarray(ds["y"], dtype=np.float64)
        if x.shape != y.shape:
            raise ValueError(
                f"Dataset '{ds.get('name', '?')}': x has {x.shape} points but "
                f"y has {y.shape}.")
        m = np.isfinite(x) & np.isfinite(y)

        s = None
        if weighted and ds.get("sigma") is not None:
            s = np.asarray(ds["sigma"], dtype=np.float64)
            m &= np.isfinite(s) & (s > 0)

        if extra_mask is not None:
            m &= np.asarray(extra_mask(ds, x, y), dtype=bool)

        if int(np.count_nonzero(m)) < min_points:
            raise ValueError(
                f"Dataset '{ds.get('name', '?')}' has fewer than {min_points} "
                f"usable points after masking; there is nothing to fit.")

        out = dict(ds)
        out["x"] = x[m]
        out["y"] = y[m]
        out["sigma"] = s[m] if s is not None else None
        out["mask"] = m
        prepped.append(out)
    return prepped


def combined_guess_from(per_dataset_guesses: Sequence[Dict[str, float]],
                        fallback: Dict[str, float]) -> Dict[str, float]:
    """
    One starting guess per parameter from several per-dataset guesses.

    The median, not the mean: a starting guess only has to land in the right
    basin, and one pathological dataset -- an empty channel, a curve that never
    decayed -- should not drag the start for all the others.  Parameters no
    dataset produced a finite guess for fall back to the model default.
    """
    out = dict(fallback)
    for name in fallback:
        vals = [g[name] for g in per_dataset_guesses
                if name in g and np.isfinite(g[name])]
        if vals:
            out[name] = float(np.median(vals))
    return out


# ── The fit ───────────────────────────────────────────────────────────────────

def fit_linked(
    param_names: Sequence[str],
    datasets: Sequence[dict],
    predict: Callable[[dict, np.ndarray, Dict[str, float]], np.ndarray],
    linked: Dict[str, bool],
    guesses: Dict[str, float],
    lowers: Dict[str, float],
    uppers: Dict[str, float],
    fixed: Dict[str, bool],
    weighted: bool = False,
    maxfev: int = 20000,
) -> dict:
    """
    Fit one model to several datasets with shared, independent and held
    parameters.

    Parameters
    ----------
    param_names : sequence of str
        Model parameter names, in the order the report should list them.
    datasets : sequence of dict
        Already masked -- see :func:`prepare_datasets`.  Each needs ``name``,
        ``x``, ``y`` and optionally ``sigma``; any other key is carried through
        to *predict* and into the result.
    predict : callable
        ``predict(ds, x, values) -> y_model``.  Called once per dataset per
        residual evaluation, so it should be cheap; anything precomputable
        (an IRF's FFT, a normalisation) belongs on the dataset dict.
    linked, fixed : dict of str -> bool
        Per-parameter flags.  ``fixed`` wins over ``linked``: a parameter held
        fixed is the same value everywhere by definition.
    guesses, lowers, uppers : dict of str -> float
        Start and bounds, shared by every dataset.  A fixed parameter is held
        at its guess.
    weighted : bool
        Divide residuals by sigma.  Downgraded to False, without error, if any
        dataset lacks one -- see the module docstring.

    Returns
    -------
    dict
        Global goodness-of-fit plus a per-dataset breakdown: values, 1-sigma
        errors, fitted curve, residuals and R^2 for each.
    """
    names = list(param_names)
    prepped = list(datasets)
    D = len(prepped)
    if D == 0:
        raise ValueError("No datasets selected.")

    use_weights = weighted and all(p.get("sigma") is not None for p in prepped)

    # ── Lay out the free-parameter vector ────────────────────────────────────
    # One slot per linked parameter, D slots per unlinked one, none for fixed.
    free_spec: list = []                      # (param_name, dataset_index|None)
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
            r = pp["y"] - predict(pp, pp["x"], vals)
            if use_weights:
                r = r / pp["sigma"]
            chunks.append(r)
        return np.concatenate(chunks)

    theta0, lb, ub = [], [], []
    for (p, _di) in free_spec:
        theta0.append(guesses[p]); lb.append(lowers[p]); ub.append(uppers[p])
    # Nudge the start inside the bounds; least_squares rejects a start outside
    # them, and a guess sitting exactly on a bound is a common way to get there.
    theta0 = [min(max(v, lo), hi) for v, lo, hi in zip(theta0, lb, ub)]

    sol = least_squares(residuals, theta0, bounds=(lb, ub), max_nfev=maxfev)

    # ── Covariance / parameter errors ────────────────────────────────────────
    n_obs = int(sum(len(pp["x"]) for pp in prepped))
    n_par = len(free_spec)
    dof = n_obs - n_par
    cov = None
    try:
        JtJ = sol.jac.T @ sol.jac
        cov = np.linalg.pinv(JtJ)
        if not use_weights and dof > 0:
            # Unweighted: the residuals carry no absolute scale, so the
            # covariance is only known up to the residual variance.  Weighted:
            # sigma already sets the scale and rescaling would double-count it.
            cov = cov * (2.0 * sol.cost / dof)
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
        yfit = predict(pp, pp["x"], vals)
        resid = pp["y"] - yfit
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((pp["y"] - pp["y"].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        entry = dict(pp)
        entry.update({
            "yfit": yfit, "resid": resid,
            "values": vals, "errors": errs, "r2": r2,
            "ss_res": ss_res, "n_points": len(pp["x"]),
        })
        per_dataset.append(entry)

    full_res = residuals(theta)
    ss_res_tot = float(np.sum(full_res ** 2))
    if use_weights and dof > 0:
        chi2, red_chi2 = ss_res_tot, ss_res_tot / dof
    else:
        chi2 = red_chi2 = float("nan")

    free_labels = [p if di is None else f"{p}[{prepped[di]['name']}]"
                   for (p, di) in free_spec]

    return {
        "names": names, "datasets": per_dataset,
        "linked": dict(linked), "fixed": dict(fixed),
        "lowers": dict(lowers), "uppers": dict(uppers),
        "guesses": dict(guesses),
        "weighted": use_weights, "n_datasets": D,
        "dof": dof, "n_free": n_par, "n_obs": n_obs,
        "chi2": chi2, "red_chi2": red_chi2, "ss_res": ss_res_tot,
        "success": bool(sol.success), "message": str(sol.message),
        "cov": cov, "free_labels": free_labels,
    }
