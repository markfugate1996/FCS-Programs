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

Those three are the whole-parameter cases of something more general: a
parameter's datasets can be PARTITIONED, so that some share a value, some are
held, and the rest are independent.  "Link tau for datasets 1, 3 and 4, hold it
at 1e-6 for dataset 2, and let the rest float" is one parameter, four rules.
A titration in which two buffers were used, or a series with one control
measured under known conditions, needs exactly this and cannot be expressed by
a single checkbox.  :func:`parse_dataset_rule` reads that partition from a
short text rule; :func:`normalise_plan` turns any of the four ways of saying it
into the one internal form the fit uses.

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


__all__ = ["prepare_datasets", "fit_linked", "combined_guess_from",
           "parse_dataset_rule", "normalise_plan", "describe_plan",
           "RuleError", "RULE_HELP", "rule_help_for", "FREE", "LINK", "FIX"]


# ── Per-parameter dataset plans ───────────────────────────────────────────────

# A "plan" for one parameter is a tuple with one entry per dataset:
#
#   (FREE,)          this dataset gets its own free variable
#   (LINK, group_id) this dataset shares a free variable with its group
#   (FIX,  value)    this dataset holds the parameter at value; no free variable
#
# One representation for all four ways of expressing the same thing -- a Fix
# checkbox, a Link checkbox, neither, or a typed rule -- so the fit itself has
# exactly one case to handle and the checkboxes are not a separate code path
# that can disagree with the rules.
FREE = "free"
LINK = "link"
FIX  = "fix"


# The syntax reference, kept HERE rather than in the dialog that shows it:
# it describes what parse_dataset_rule accepts, so it belongs beside the parser
# and changes when the parser changes.  A help text living in the GUI is a help
# text that quietly goes stale.
RULE_HELP = """\
Per-dataset rules
=================
A rule says what one parameter does across the selected datasets.  Without a
rule, a parameter is either linked (one shared value), fixed (held), or free
(an independent value per dataset) for ALL of them.  A rule lets you mix those.

Dataset numbers are the ones shown in the dataset list, starting at 1.

Clauses, separated by semicolons
--------------------------------
  L(1,3,4)      LINK datasets 1, 3 and 4: they share one fitted value
  L(1-10)       link a range (inclusive, so 1-10 is ten datasets)
  L(*)          link every dataset (the same as ticking Link)
  F(2=1e-6)     FIX dataset 2 at 1e-6; it is held, not fitted
  F(2,5=0)      fix datasets 2 and 5 at 0
  F(2)          fix dataset 2 at the Guess value in this row
  U(7)          dataset 7 stays independent (the default; for emphasis)

Anything you do not mention stays independent.

Examples
--------
  L(1,3,4); F(2=1e-6)
        1, 3 and 4 share a value; 2 is held at 1e-6; the rest float freely.

  L(1-10); L(11-20)
        Two SEPARATE shared values -- one for the first ten datasets, one for
        the second ten.  Two L clauses are two groups, not one big one.

  F(1-3=0.5)
        Hold the first three at 0.5; everything else is fitted independently.

Rules
-----
  * Each dataset may appear in at most one clause per parameter.
  * L(...) needs two or more datasets; linking one is the same as leaving it
    free.
  * A rule replaces the Link and Fix boxes for that parameter.

What linking means
------------------
Linking says the datasets SHARE a quantity, so the fit determines one value
from all of them at once.  That makes the shared value better determined, and
frees the parameters it was trading against.  It is an assumption, not a
measurement: two datasets forced to share a value look, in the results, exactly
like two datasets that happened to agree, which is why the report records every
rule alongside the numbers.

Rate-like quantities (diffusion times, lifetimes, brightness per second) are
the usual things to link.  Scale-like ones (amplitudes, occupations, background
offsets, and PCH epsilon, which is counts per molecule per BIN) depend on the
acquisition settings, so linking them across datasets acquired with different
bin widths, windows or durations links two quantities that are not the same
quantity.
"""


class RuleError(ValueError):
    """
    A dataset rule that could not be parsed or does not make sense.

    Its own type so a dialog can catch exactly this and put the message beside
    the offending entry box, rather than catching ValueError and also swallowing
    a genuine numerical failure from the fit.
    """


def _parse_index_set(text: str, n_datasets: int, where: str) -> list:
    """
    Read ``1,3,4`` or ``1-10`` or ``*`` into a sorted list of 0-based indices.

    Indices are 1-based in the text because that is what the dataset list shows
    the user; they are 0-based everywhere inside.
    """
    text = text.strip()
    if not text:
        raise RuleError(f"{where}: no datasets listed inside the brackets.")
    if text == "*":
        return list(range(n_datasets))

    out: list = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            raise RuleError(f"{where}: empty item in the dataset list "
                            f"(a stray comma?).")
        if "-" in piece.lstrip("-"):
            lo_s, _, hi_s = piece.partition("-")
            try:
                lo, hi = int(lo_s.strip()), int(hi_s.strip())
            except ValueError:
                raise RuleError(
                    f"{where}: '{piece}' is not a dataset range like 1-10.")
            if hi < lo:
                raise RuleError(
                    f"{where}: range '{piece}' runs backwards; write "
                    f"{hi}-{lo} if that is what you meant.")
            span = list(range(lo, hi + 1))
        else:
            try:
                span = [int(piece)]
            except ValueError:
                raise RuleError(
                    f"{where}: '{piece}' is not a dataset number.")
        for one in span:
            if not (1 <= one <= n_datasets):
                raise RuleError(
                    f"{where}: dataset {one} does not exist; there "
                    f"{'is' if n_datasets == 1 else 'are'} {n_datasets}.")
            out.append(one - 1)
    return out


def parse_dataset_rule(text: str, n_datasets: int,
                       default_value: float = 0.0) -> tuple:
    """
    Parse a per-dataset rule into a plan.

    Grammar
    -------
    Clauses separated by ``;``.  Dataset numbers are 1-based and match the
    numbering shown in the dataset list::

        L(1,3,4)        link datasets 1, 3 and 4 -- one shared value
        L(1-10)         link a range
        L(*)            link every dataset (same as ticking Link)
        F(2=1e-6)       hold dataset 2 at 1e-6
        F(2,5=0)        hold datasets 2 and 5 at 0
        F(2)            hold dataset 2 at the row's guess
        U(7)            dataset 7 independent (the default; for emphasis)

    Two separate ``L`` clauses are two separate groups::

        L(1-10); L(11-20)     two shared values, not one

    Anything not mentioned is independent.  So ``F(2=1e-6)`` alone means
    "dataset 2 is held, everything else floats freely", which is usually what
    is meant by naming only the exception.

    Returns
    -------
    (plan, summary)
        *plan* is a tuple of per-dataset entries; *summary* is a one-line
        human-readable rendering for the report and for live feedback in the
        dialog.

    Raises
    ------
    RuleError
        On anything unparseable, out of range, or contradictory -- notably a
        dataset named twice, which is the easy mistake to make when editing a
        long rule and which would otherwise resolve silently to whichever
        clause happened to be applied last.
    """
    plan: list = [(FREE,)] * n_datasets
    assigned: dict = {}
    group_id = 0

    for raw in str(text).split(";"):
        clause = raw.strip()
        if not clause:
            continue
        head, sep, rest = clause.partition("(")
        kind = head.strip().upper()
        if not sep or not rest.rstrip().endswith(")"):
            raise RuleError(
                f"'{clause}' is not a rule.  Expected something like "
                f"L(1,3,4) or F(2=1e-6).")
        body = rest.rstrip()[:-1]
        if kind not in ("L", "F", "U"):
            raise RuleError(
                f"'{head.strip()}' is not a known rule.  Use L(...) to link, "
                f"F(...) to fix, U(...) to leave independent.")

        value = default_value
        if kind == "F" and "=" in body:
            body, _, val_s = body.partition("=")
            try:
                value = float(val_s.strip())
            except ValueError:
                raise RuleError(
                    f"F(...={val_s.strip()}): '{val_s.strip()}' is not a "
                    f"number.")

        members = _parse_index_set(body, n_datasets, clause)
        for di in members:
            if di in assigned:
                raise RuleError(
                    f"Dataset {di + 1} appears in more than one rule "
                    f"('{assigned[di]}' and '{clause}').  Each dataset can "
                    f"take only one rule per parameter.")
            assigned[di] = clause

        if kind == "L":
            if len(members) < 2:
                raise RuleError(
                    f"'{clause}' links a single dataset, which is the same as "
                    f"leaving it independent.  Link two or more, or use "
                    f"U({members[0] + 1}).")
            for di in members:
                plan[di] = (LINK, group_id)
            group_id += 1
        elif kind == "F":
            for di in members:
                plan[di] = (FIX, float(value))
        else:
            for di in members:
                plan[di] = (FREE,)

    return tuple(plan), describe_plan(tuple(plan))


def describe_plan(plan, names=None) -> str:
    """
    One-line rendering of a plan, for the report and for live feedback.

    Uses dataset numbers when *names* is None and dataset names when it is
    given, because a report read months later should not require the reader to
    reconstruct which file was number 7.
    """
    def label(di):
        return names[di] if names else str(di + 1)

    groups: dict = {}
    fixes: dict = {}
    free: list = []
    for di, entry in enumerate(plan):
        if entry[0] == LINK:
            groups.setdefault(entry[1], []).append(di)
        elif entry[0] == FIX:
            fixes.setdefault(entry[1], []).append(di)
        else:
            free.append(di)

    parts = []
    for gid in sorted(groups):
        parts.append("linked {" + ", ".join(label(d) for d in groups[gid]) + "}")
    for val in sorted(fixes):
        parts.append("fixed " + ", ".join(label(d) for d in fixes[val])
                     + f" = {val:g}")
    if free:
        parts.append("free " + ", ".join(label(d) for d in free))
    return "; ".join(parts) if parts else "free"


def normalise_plan(param: str, n_datasets: int,
                   linked: dict, fixed: dict, guesses: dict,
                   plans: Optional[dict] = None) -> tuple:
    """
    Resolve one parameter's setup into a plan, whichever way it was expressed.

    A plan supplied in *plans* wins; otherwise the whole-parameter ``fixed``
    and ``linked`` flags are expanded into the equivalent plan.  ``fixed`` wins
    over ``linked``, as it always has: a parameter held fixed is the same value
    everywhere by definition, so linking it says nothing extra.
    """
    if plans and param in plans and plans[param] is not None:
        plan = tuple(plans[param])
        if len(plan) != n_datasets:
            raise RuleError(
                f"The rule for '{param}' covers {len(plan)} datasets but "
                f"{n_datasets} were selected.")
        return plan
    if fixed.get(param, False):
        return tuple((FIX, float(guesses[param])) for _ in range(n_datasets))
    if linked.get(param, False):
        return tuple((LINK, 0) for _ in range(n_datasets))
    return tuple((FREE,) for _ in range(n_datasets))


# ── Preparation ───────────────────────────────────────────────────────────────

def prepare_datasets(
    datasets: Sequence[dict],
    weighted: bool = False,
    extra_mask: Optional[Callable[..., np.ndarray]] = None,
    min_points: int = 2,
) -> List[dict]:
    """
    Mask each dataset to the points that can actually be fitted.

    Keeps points where x and y are both finite, and -- when *weighted* -- where
    sigma is finite and strictly positive, since a zero sigma is an infinite
    weight and a negative one is meaningless.

    *extra_mask* is the hook for censoring that only one analysis knows about:
    it receives ``(ds, x, y, mask)`` and returns a boolean array of points to
    KEEP.  Correlation uses it to drop pair-starved lag channels, whose G
    values are censored rather than measured.  Nothing here needs to know that;
    it just needs somewhere for the caller to say so.

    *mask* is the keep-array built so far, and it is passed because a censoring
    rule usually needs to COUNT what it removes for the report -- and should
    count only points that survived the finite and sigma masks, or a lag that
    was already excluded for having no sigma gets reported a second time as
    pair-starved.

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
            m &= np.asarray(extra_mask(ds, x, y, m), dtype=bool)

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

def _spans_all(plan, group_id, D: int) -> bool:
    """True when *group_id* covers every dataset -- i.e. a plain whole-parameter
    link, which keeps its historical slot key and label."""
    return sum(1 for e in plan if e[0] == LINK and e[1] == group_id) == D


def _slot_key(param: str, plan, di: int, D: int):
    """
    Which free variable dataset *di* uses for *param*.

    Independent datasets must key on their own index -- ``(FREE,)`` is one and
    the same tuple for every dataset, so keying on the entry alone would quietly
    merge all of them into a single shared variable and fit one value where D
    were asked for.  A group spanning every dataset keeps the historical
    ``(param, None)`` key so the free-parameter ordering, and every label
    derived from it, is unchanged from before plans existed.
    """
    entry = plan[di]
    if entry[0] == LINK:
        if _spans_all(plan, entry[1], D):
            return (param, None)
        return (param, (LINK, entry[1]))
    return (param, (FREE, di))


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
    plans: Optional[Dict[str, tuple]] = None,
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
        fixed is the same value everywhere by definition.  Ignored for any
        parameter that appears in *plans*.
    plans : dict of str -> plan, optional
        Per-parameter dataset partitions, from :func:`parse_dataset_rule` or
        built directly.  A parameter absent from *plans* falls back to its
        ``linked`` / ``fixed`` flags, so passing nothing here behaves exactly
        as before.
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

    # ── Resolve every parameter to a plan ────────────────────────────────────
    resolved = {p: normalise_plan(p, D, linked, fixed, guesses, plans)
                for p in names}

    # ── Lay out the free-parameter vector ────────────────────────────────────
    # One slot per link group, one per independent dataset, none for held ones.
    # Slots are created in order of FIRST APPEARANCE scanning datasets 0..D-1
    # within each parameter, which is what makes the whole-parameter cases come
    # out in exactly the order they always did: a parameter linked across
    # everything yields one slot, an unlinked one yields D in dataset order.
    free_spec: list = []                      # (param_name, slot_key)
    slot_members: Dict[tuple, list] = {}
    fixed_value: Dict[tuple, float] = {}      # (param, dataset) -> value
    for p in names:
        for di, entry in enumerate(resolved[p]):
            if entry[0] == FIX:
                fixed_value[(p, di)] = float(entry[1])
                continue
            key = _slot_key(p, resolved[p], di, D)
            if key not in slot_members:
                slot_members[key] = []
                free_spec.append(key)
            slot_members[key].append(di)
    if not free_spec:
        raise ValueError("At least one parameter must be free (not fixed).")
    idx = {spec: i for i, spec in enumerate(free_spec)}

    def slot_of(p, di):
        if resolved[p][di][0] == FIX:
            return None
        return idx[_slot_key(p, resolved[p], di, D)]

    def value_of(theta, p, di):
        slot = slot_of(p, di)
        if slot is None:
            return fixed_value[(p, di)]
        return theta[slot]

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
    for (p, _slot) in free_spec:
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
        slot = slot_of(p, di)
        # A held parameter has no fitted error.  0.0, not NaN, because that is
        # what this has always returned and the exporters translate it into a
        # blank cell themselves.
        return 0.0 if slot is None else float(perr_free[slot])

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
            # Per-DATASET now, not per-parameter: with plans a parameter can be
            # held for one dataset and free for the next, and an exporter that
            # blanks an error cell has to know which.
            "fixed": {p: resolved[p][di][0] == FIX for p in names},
            "linked": {p: resolved[p][di][0] == LINK for p in names},
        })
        per_dataset.append(entry)

    full_res = residuals(theta)
    ss_res_tot = float(np.sum(full_res ** 2))
    if use_weights and dof > 0:
        chi2, red_chi2 = ss_res_tot, ss_res_tot / dof
    else:
        chi2 = red_chi2 = float("nan")

    # Labels: a group spanning every dataset is just the parameter name, as it
    # always was; anything narrower names its members, so a report shows which
    # datasets a shared value was shared BETWEEN.
    free_labels = []
    for key in free_spec:
        p, slot = key
        members = slot_members[key]
        if slot is None:
            free_labels.append(p)
        elif slot[0] == LINK:   # a group narrower than the whole selection
            free_labels.append(
                f"{p}[{'+'.join(prepped[d]['name'] for d in members)}]")
        else:
            free_labels.append(f"{p}[{prepped[members[0]]['name']}]")

    ds_names = [pp["name"] for pp in prepped]
    return {
        "names": names, "datasets": per_dataset,
        "plans": resolved,
        "plan_summaries": {p: describe_plan(resolved[p], ds_names)
                           for p in names},
        # Top-level flags stay whole-parameter: True only when the plan holds
        # or shares across EVERY dataset.  Callers that predate plans keep
        # reading the same thing, and a partial rule reads as neither -- which
        # is right, because it is neither.
        "linked": {p: all(e[0] == LINK for e in resolved[p]) and
                      len({e[1] for e in resolved[p]}) == 1 for p in names},
        "fixed": {p: all(e[0] == FIX for e in resolved[p]) for p in names},
        "lowers": dict(lowers), "uppers": dict(uppers),
        "guesses": dict(guesses),
        "weighted": use_weights, "n_datasets": D,
        "dof": dof, "n_free": n_par, "n_obs": n_obs,
        "chi2": chi2, "red_chi2": red_chi2, "ss_res": ss_res_tot,
        "success": bool(sol.success), "message": str(sol.message),
        "cov": cov, "free_labels": free_labels,
    }


def rule_help_for(names: Optional[Sequence[str]] = None) -> str:
    """
    :data:`RULE_HELP` with the current dataset list appended.

    Numbers in a rule mean nothing without knowing which file each one is, and
    a help window that shows the syntax but not the numbering leaves the reader
    to go and find it.  Called with no names it returns the plain reference.
    """
    if not names:
        return RULE_HELP
    lines = [RULE_HELP, "", f"Your {len(names)} dataset"
             f"{'s' if len(names) != 1 else ''}", "-" * 24]
    width = len(str(len(names)))
    lines += [f"  {i:>{width}}  {nm}" for i, nm in enumerate(names, start=1)]
    return "\n".join(lines)
