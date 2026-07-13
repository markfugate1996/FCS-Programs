"""
fcs_fisher.py
=============
Fisher information from a completed least-squares fit.

For a least-squares (Gaussian / Poisson-weighted) fit, the covariance matrix of
the fitted parameters is asymptotically the inverse of the Fisher information
matrix (FIM):

        Cov(theta_hat)  ~=  I(theta)^-1      <=>      I(theta)  ~=  Cov(theta_hat)^-1

So once a fit has produced a parameter covariance (scipy's ``curve_fit`` returns
it as ``pcov``; a ``least_squares`` fit builds it from the Jacobian as
``pinv(J^T J)``), the FIM is just its inverse over the free parameters -- no
re-fitting and no re-evaluation of the Jacobian required.

Interpretation depends on how the fit was weighted:

* Weighted (absolute sigma -- e.g. Poisson sigma = sqrt(counts) for TCSPC / PCH,
  or G_std for correlation): ``Cov^-1`` is the observed Fisher information in
  absolute units.  The 1-sigma Cramer-Rao lower bounds it implies are exactly
  the fit's reported standard errors.
* Unweighted: ``curve_fit`` / ``least_squares`` scale the covariance by the
  residual-estimated noise variance, so the information is expressed *relative*
  to that estimated noise level rather than an externally known sigma.

A note on the photon-counting fits (lifetime, PCH): the FIM taken from ``pcov``
is the Gauss-Newton / observed information, weighted by sqrt(data) because that
is how the fit itself was weighted.  The *expected* Poisson information would
weight by sqrt(model); the two agree asymptotically and differ only slightly in
finite samples.  Deriving it from the covariance is consistent with how each
fit actually ran.

This module has NO GUI or fcs_* dependency (numpy only), so it is safe to import
from every fitter without any circular-import risk.
"""

from __future__ import annotations

import textwrap
from typing import List, Optional, Sequence

import numpy as np


def fisher_from_covariance(
    cov: Optional[np.ndarray],
    free_names: Sequence[str],
    weighted: bool,
) -> dict:
    """
    Build the Fisher information matrix and scalar summaries from a fit's
    parameter covariance.

    Parameters
    ----------
    cov : (n, n) array or None
        Covariance of the FREE parameters, in the same order as ``free_names``.
        ``None`` (or a non-finite / mis-shaped matrix) yields ``ok=False`` with
        an explanatory ``reason`` rather than raising.
    free_names : sequence of str
        Names/labels of the free parameters, ordering the rows/cols of ``cov``.
    weighted : bool
        Whether the fit used absolute weights (affects interpretation only, not
        the arithmetic).

    Returns
    -------
    dict
        On success (``ok=True``): ``fim`` (n x n ndarray), ``crlb`` (1-sigma per
        parameter), ``marginal_precision`` (1/CRLB^2), ``conditional_information``
        (FIM diagonal), ``logdet``, ``det_sign``, ``cond``, ``used_pinv``,
        ``names``, ``weighted``.  On failure: ``ok=False`` and a ``reason``.
    """
    names = list(free_names)
    n = len(names)
    out: dict = {"ok": False, "names": names, "weighted": bool(weighted),
                 "reason": ""}

    if cov is None:
        out["reason"] = "the fit returned no covariance matrix"
        return out

    C = np.asarray(cov, dtype=np.float64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        out["reason"] = "covariance is not a square matrix"
        return out
    if C.shape[0] != n:
        out["reason"] = (f"covariance is {C.shape[0]}x{C.shape[0]} but there are "
                         f"{n} free parameters")
        return out
    if not np.all(np.isfinite(C)):
        out["reason"] = ("covariance has non-finite entries -- at least one "
                         "parameter is unconstrained by the data")
        return out

    diag = np.diag(C)
    if np.any(diag < 0):
        out["reason"] = "covariance has a negative variance on its diagonal"
        return out

    crlb = np.sqrt(diag)

    used_pinv = False
    try:
        F = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        F = np.linalg.pinv(C)
        used_pinv = True
    if not np.all(np.isfinite(F)):
        out["reason"] = "the Fisher matrix is non-finite (singular covariance)"
        return out

    sign, logabsdet = np.linalg.slogdet(F)
    try:
        cond = float(np.linalg.cond(F)) if n > 1 else 1.0
    except np.linalg.LinAlgError:
        cond = float("inf")

    with np.errstate(divide="ignore", invalid="ignore"):
        marg = np.where(diag > 0, 1.0 / diag, np.inf)

    out.update({
        "ok": True,
        "fim": F,
        "crlb": {names[i]: float(crlb[i]) for i in range(n)},
        "marginal_precision": {names[i]: float(marg[i]) for i in range(n)},
        "conditional_information": {names[i]: float(F[i, i]) for i in range(n)},
        "det_sign": float(sign),
        "logdet": float(logabsdet),
        "cond": cond,
        "used_pinv": used_pinv,
    })
    return out


def format_report_lines(
    fi: dict,
    matrix_max: int = 8,
    indent: str = "  ",
    width: int = 60,
) -> List[str]:
    """
    Render a Fisher-information section (a list of text lines) for a fit report.

    Always emits a header; when the FIM is available it adds a scalar summary
    (log-determinant / D-optimality and condition number) and a per-parameter
    table.  The full matrix is printed only when there are at most ``matrix_max``
    free parameters -- larger problems (e.g. big global fits) would otherwise
    swamp the report, so only the summary is shown.

    The returned lines carry no trailing blank; callers append their own spacing
    to match the surrounding report style.
    """
    L: List[str] = ["Fisher information", "-" * width]

    if not fi.get("ok"):
        L.append(f"{indent}not available -- {fi.get('reason', 'unknown reason')}")
        return L

    if fi["weighted"]:
        interp = ("absolute weighting: the matrix is the observed Fisher "
                  "information in absolute units; CRLB is the 1-sigma "
                  "Cramer-Rao lower bound and equals the fit's std errors.")
    else:
        interp = ("unweighted fit: the covariance was scaled by the "
                  "fit-estimated noise variance, so the information is relative "
                  "to that estimated noise level, not an externally known sigma.")
    for line in textwrap.wrap(interp, max(20, width - len(indent))):
        L.append(f"{indent}{line}")
    if fi.get("used_pinv"):
        L.append(f"{indent}(covariance was singular; used a pseudo-inverse.)")
    if fi.get("det_sign", 1.0) <= 0:
        L.append(f"{indent}(FIM is not positive-definite; near-degenerate fit.)")

    L.append("")
    L.append(f"{indent}{'log|FIM| (D-optimality)':<24}: {fi['logdet']:.6g}")
    L.append(f"{indent}{'cond(FIM)':<24}: {fi['cond']:.6g}")
    L.append("")

    L.append(f"{indent}{'parameter':<18}{'CRLB (1sigma)':>15}"
             f"{'I marginal':>15}{'I conditional':>15}")
    for nm in fi["names"]:
        L.append(f"{indent}{nm:<18}{fi['crlb'][nm]:>15.6g}"
                 f"{fi['marginal_precision'][nm]:>15.6g}"
                 f"{fi['conditional_information'][nm]:>15.6g}")
    L.append("")
    L.append(f"{indent}I marginal    = 1/CRLB^2   (precision incl. correlations)")
    L.append(f"{indent}I conditional = FIM diagonal (other parameters known)")

    names = fi["names"]
    F = fi.get("fim")
    if F is not None and len(names) <= matrix_max:
        L.append("")
        L.append(f"{indent}Fisher information matrix (rows/cols = free params):")
        colw = 13
        L.append(f"{indent}{'':<12}" + "".join(f"{nm[:12]:>{colw}}" for nm in names))
        for i, nm in enumerate(names):
            L.append(f"{indent}{nm[:12]:<12}"
                     + "".join(f"{F[i, j]:>{colw}.4g}" for j in range(len(names))))
        L.append(f"{indent}(entries carry mixed units 1/[param_i*param_j]; "
                 f"compare with care.)")
    elif F is not None:
        L.append("")
        L.append(f"{indent}(full {len(names)}x{len(names)} matrix omitted for "
                 f"size; per-parameter summary above.)")

    return L
