"""
fcs_basisfit.py
===============
Correlation decomposition onto a basis of MEASURED fit curves.

Where fcs_matrixfit generates hundreds of candidate curves from an analytic
generating function and asks how much of each is present, this module takes a
handful of curves the user has already fitted and exported -- one per known
species -- and asks how much of each is present in a mixture.

    G_mixture(tau)  ~  sum_j  d_j * C_j(tau)   +   baseline

where each C_j is an exported fit curve, normalised so that C_j = 1 at its
shortest lag.  With that normalisation d_j is the amplitude the component
contributes to G(0), so the amplitudes sum to the mixture's G(0) and <N> is
1/sum(d) exactly as it is elsewhere in the suite.

The model behind a basis curve is deliberately irrelevant here.  This module
never reads which model produced a curve, never reads its chi2, and never
judges whether the fit was good.  The user asserts that the exported curve is
what that species looks like; that assertion IS the basis vector.  A curve
exported by the global fitter, the matrix method, or anything else that writes
``tau_s`` and ``G_fit`` columns works identically.

Because the number of components is small and known, the answer is reported
the way an ordinary fit is -- an amplitude and an error per named component --
rather than as a distribution.

Saved bases
-----------
A basis is a named, reusable object stored as JSON under the per-user
application-data directory, alongside the preferences file:

    %APPDATA%/FCS-Programs/curve_bases/<name>.json     (Windows)
    ~/.fcs-programs/curve_bases/<name>.json            (fallback)

The curves themselves are stored, not paths to the files they came from: a
basis that breaks because someone tidied up a folder is not reusable, which was
the point of saving it.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np

import fcs_export
import fcs_linsolve
import fcs_plottools
from fcs_fitcommon import (fit_dialog_to_screen, new_fit_dir, scrollable_dialog,
                           slug_name, write_params_table)


# ── Module-level settings ─────────────────────────────────────────────────────

# Fewer usable lag points than this after trimming is not a decomposition.
_MIN_POINTS = 8

# A basis curve is normalised by its value at its shortest lag.  That is
# G(tau_min), not a fitted G(0): whatever the curve does inside its first lag
# is inside the reference.  Two curves measured over different lag ranges are
# therefore normalised at different places, which matters when their shapes
# differ at short lag.  Reported rather than corrected, because correcting it
# would mean extrapolating someone else's model, which this module does not do.
_NORMALISE_AT_SHORTEST_LAG = True

# Column names accepted when reading an exported curves file.  Every fitter in
# the suite writes tau_s and G_fit; the alternatives are here so a file written
# by an older version, or in milliseconds, still loads.
_LAG_COLUMNS_S = ("tau_s",)
_LAG_COLUMNS_MS = ("tau_ms",)
_FIT_COLUMNS = ("G_fit", "fit", "G_model")


# ── Basis curves ──────────────────────────────────────────────────────────────

class BasisCurve(object):
    """One exported fit curve, ready to serve as a basis vector.

    Attributes
    ----------
    name : str
        Display name, editable by the user; what the exported parameter table
        calls this component.
    tau : (k,) ndarray
        Lag times in seconds, ascending, strictly positive.
    g : (k,) ndarray
        The fit curve as exported, before normalisation.
    source : str
        Where it came from, kept for the report only.
    """

    def __init__(self, name, tau, g, source=""):
        tau = np.asarray(tau, dtype=float).ravel()
        g = np.asarray(g, dtype=float).ravel()
        if tau.size != g.size:
            raise ValueError("%s: %d lag values but %d curve values"
                             % (name, tau.size, g.size))
        keep = np.isfinite(tau) & np.isfinite(g) & (tau > 0.0)
        tau, g = tau[keep], g[keep]
        if tau.size < 2:
            raise ValueError("%s: fewer than two usable points" % name)
        order = np.argsort(tau, kind="stable")
        self.tau = tau[order]
        self.g = g[order]
        self.name = str(name)
        self.source = str(source)

    @property
    def g0(self):
        """The normalisation reference: the curve at its shortest lag."""
        return float(self.g[0])

    def normalised(self):
        """The curve divided by :attr:`g0`, so it starts at 1."""
        ref = self.g0
        if not np.isfinite(ref) or ref == 0.0:
            raise ValueError("%s: normalisation reference is %r; the curve "
                             "cannot be scaled to G0 = 1" % (self.name, ref))
        return self.g / ref

    def sample(self, tau_target):
        """The normalised curve interpolated onto *tau_target*.

        Interpolation is linear in log(lag), which is how these curves are
        plotted and read, and where a smooth correlation curve is closest to
        straight.  Points outside the curve's own range are NOT extrapolated;
        the caller is expected to have trimmed to :func:`usable_lag_range`
        first, and anything still outside comes back as NaN so it cannot pass
        silently into a fit.
        """
        t = np.asarray(tau_target, dtype=float).ravel()
        out = np.full(t.size, np.nan)
        inside = (t >= self.tau[0]) & (t <= self.tau[-1])
        if np.any(inside):
            out[inside] = np.interp(np.log(t[inside]), np.log(self.tau),
                                    self.normalised())
        return out

    def to_dict(self):
        return {"name": self.name, "source": self.source,
                "tau_s": [float(v) for v in self.tau],
                "g_fit": [float(v) for v in self.g]}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("name", "component"), d["tau_s"], d["g_fit"],
                   d.get("source", ""))

    def __repr__(self):
        return ("<BasisCurve %r %d pts, %.4g to %.4g ms>"
                % (self.name, self.tau.size, self.tau[0] * 1e3,
                   self.tau[-1] * 1e3))


def read_curve_file(path):
    """Load every fit curve in an exported ``*_curves.csv``.

    A curves file may hold several datasets, distinguished by a ``dataset``
    column, so this returns a LIST -- one BasisCurve per dataset found.

    Only the lag column and the fit column are read.  The data, the residuals,
    the model name and the fit statistics are all ignored on purpose: this
    module treats the exported curve as given, and reading the quality of the
    fit would only invite second-guessing it.
    """
    path = Path(path)
    _meta, columns = fcs_export.read_export(path)

    lag = None
    scale = 1.0
    for key in _LAG_COLUMNS_S:
        if key in columns:
            lag, scale = np.asarray(columns[key], dtype=float), 1.0
            break
    if lag is None:
        for key in _LAG_COLUMNS_MS:
            if key in columns:
                lag, scale = np.asarray(columns[key], dtype=float), 1e-3
                break
    if lag is None:
        raise ValueError("%s: no lag column (expected one of %s)"
                         % (path.name,
                            ", ".join(_LAG_COLUMNS_S + _LAG_COLUMNS_MS)))

    fit = None
    for key in _FIT_COLUMNS:
        if key in columns:
            fit = np.asarray(columns[key], dtype=float)
            break
    if fit is None:
        raise ValueError("%s: no fit-curve column (expected one of %s). This "
                         "does not look like a curves file exported by a "
                         "fitter." % (path.name, ", ".join(_FIT_COLUMNS)))

    lag = lag * scale
    names = columns.get("dataset")
    out = []
    if names is None:
        out.append(BasisCurve(path.stem, lag, fit, str(path)))
    else:
        names = [str(v) for v in names]
        for label in dict.fromkeys(names):           # preserve file order
            sel = np.array([n == label for n in names], dtype=bool)
            if int(np.sum(sel)) >= 2:
                out.append(BasisCurve(label, lag[sel], fit[sel], str(path)))
    if not out:
        raise ValueError("%s: no usable curves found" % path.name)
    return out


# ── A basis ───────────────────────────────────────────────────────────────────

class CurveBasis(object):
    """An ordered, named set of basis curves, plus a baseline column."""

    def __init__(self, name="", curves=None, include_baseline=True):
        self.name = str(name)
        self.curves = list(curves or [])
        self.include_baseline = bool(include_baseline)

    def __len__(self):
        return len(self.curves)

    @property
    def n_columns(self):
        return len(self.curves) + (1 if self.include_baseline else 0)

    def component_names(self):
        return [c.name for c in self.curves]

    def usable_lag_range(self):
        """The lag range every curve covers: (lo, hi) in seconds.

        The INTERSECTION, not the union.  A lag where even one basis curve has
        no value cannot be decomposed onto the full basis, so it is outside the
        usable range whatever the other curves do there.
        """
        if not self.curves:
            return None, None
        lo = max(float(c.tau[0]) for c in self.curves)
        hi = min(float(c.tau[-1]) for c in self.curves)
        return lo, hi

    def build(self, tau_target):
        """The (m, n_columns) design matrix on *tau_target*.

        Basis curves first in list order, then the baseline column of ones.
        Raises if any entry is not finite, which means the caller did not trim
        to :meth:`usable_lag_range` first.
        """
        if not self.curves:
            raise ValueError("the basis has no curves")
        t = np.asarray(tau_target, dtype=float).ravel()
        cols = [c.sample(t) for c in self.curves]
        A = np.column_stack(cols)
        if self.include_baseline:
            A = np.hstack([A, np.ones((t.size, 1))])
        if not np.all(np.isfinite(A)):
            lo, hi = self.usable_lag_range()
            raise ValueError(
                "the target lags run outside the basis (usable range %.6g to "
                "%.6g ms); trim with trim_to_basis() before building"
                % (lo * 1e3, hi * 1e3))
        return A

    def nonneg_mask(self):
        """Constrain the component amplitudes, leave the baseline free."""
        mask = np.ones(self.n_columns, dtype=bool)
        if self.include_baseline:
            mask[-1] = False
        return mask

    # ---- persistence ----

    def to_dict(self):
        return {
            "format": "fcs_basisfit.CurveBasis",
            "version": 1,
            "name": self.name,
            "saved": datetime.now().isoformat(timespec="seconds"),
            "include_baseline": self.include_baseline,
            "curves": [c.to_dict() for c in self.curves],
        }

    @classmethod
    def from_dict(cls, d):
        if d.get("format") != "fcs_basisfit.CurveBasis":
            raise ValueError("not a saved curve basis (format = %r)"
                             % d.get("format"))
        return cls(d.get("name", ""),
                   [BasisCurve.from_dict(c) for c in d.get("curves", [])],
                   bool(d.get("include_baseline", True)))

    def save(self, path=None):
        """Write to *path*, or to the shared basis directory under its name."""
        path = Path(path) if path is not None else basis_path(self.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")
        print("[basisfit] wrote %s" % path)
        return path

    @classmethod
    def load(cls, path):
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def __repr__(self):
        return "<CurveBasis %r %d curves>" % (self.name, len(self.curves))


# ── Where saved bases live ────────────────────────────────────────────────────

def bases_dir():
    """Directory holding saved bases, beside the preferences file.

    APPDATA is read per call rather than at import, matching fcs_prefs, so a
    test can redirect storage by setting the environment variable.
    """
    appdata = os.environ.get("APPDATA")
    root = Path(appdata) / "FCS-Programs" if appdata \
        else Path.home() / ".fcs-programs"
    return root / "curve_bases"


def basis_path(name):
    """File a basis called *name* is stored in."""
    slug = slug_name(name) or "basis"
    return bases_dir() / ("%s.json" % slug)


def list_saved_bases():
    """Every saved basis as (display name, path), sorted by name."""
    out = []
    d = bases_dir()
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as fh:
                label = json.load(fh).get("name") or p.stem
        except Exception as exc:                     # noqa: BLE001
            print("[basisfit] skipping %s (%s)" % (p, exc))
            continue
        out.append((str(label), p))
    return sorted(out, key=lambda t: t[0].lower())


# ── Trimming ──────────────────────────────────────────────────────────────────

def trim_to_basis(tau, G, sigma, basis):
    """Restrict a dataset to the lag range the basis actually covers.

    Returns (tau, G, sigma, info) where info records what was dropped, so the
    caller can warn rather than silently shortening someone's curve.
    """
    tau = np.asarray(tau, dtype=float).ravel()
    G = np.asarray(G, dtype=float).ravel()
    sig = None if sigma is None else np.asarray(sigma, dtype=float).ravel()

    lo, hi = basis.usable_lag_range()
    if lo is None:
        raise ValueError("the basis has no curves")

    keep = np.isfinite(tau) & np.isfinite(G) & (tau > 0.0)
    if sig is not None:
        keep &= np.isfinite(sig) & (sig > 0.0)
    n_bad = int(np.sum(~keep))

    inside = keep & (tau >= lo) & (tau <= hi)
    n_fast = int(np.sum(keep & (tau < lo)))
    n_slow = int(np.sum(keep & (tau > hi)))

    info = {
        "n_bad": n_bad, "n_fast": n_fast, "n_slow": n_slow,
        "n_kept": int(np.sum(inside)), "n_total": tau.size,
        "basis_lo": lo, "basis_hi": hi,
        "message": "",
    }
    parts = []
    if n_fast:
        parts.append("%d point%s below the basis's shortest lag (%.4g ms)"
                     % (n_fast, "" if n_fast == 1 else "s", lo * 1e3))
    if n_slow:
        parts.append("%d point%s above its longest (%.4g ms)"
                     % (n_slow, "" if n_slow == 1 else "s", hi * 1e3))
    if parts:
        info["message"] = (
            "The data extend beyond the basis curves: %s. Those points cannot "
            "be decomposed and have been trimmed; the fit uses the remaining "
            "%d of %d." % (" and ".join(parts), info["n_kept"], tau.size))

    return (tau[inside], G[inside],
            None if sig is None else sig[inside], info)


# ── The fit ───────────────────────────────────────────────────────────────────

def fit_dataset(ds, basis, weighted=True):
    """Decompose one correlation curve onto *basis*.

    Uses the joint bounded solve, not the iterative search: the basis has a few
    named columns rather than hundreds of candidates, so it is well conditioned
    and there is nothing to select.
    """
    from fcs_matrixfit import resolve_sigma

    notes = []
    tau = np.asarray(ds["tau"], dtype=float).ravel()
    G = np.asarray(ds["G"], dtype=float).ravel()
    if tau.size != G.size:
        raise ValueError("%s: tau has %d points but G has %d"
                         % (ds.get("name", "?"), tau.size, G.size))

    sigma, sigma_note = resolve_sigma(ds.get("sigma"), ds.get("meta"), weighted)
    notes.append(sigma_note)

    tau_k, G_k, sig_k, trim = trim_to_basis(tau, G, sigma, basis)
    if trim["n_bad"]:
        notes.append("dropped %d point%s with a missing or non-positive tau, "
                     "G or sigma" % (trim["n_bad"],
                                     "" if trim["n_bad"] == 1 else "s"))
    if trim["message"]:
        notes.append(trim["message"])
    if tau_k.size < _MIN_POINTS:
        raise ValueError(
            "%s: only %d lag points fall inside the basis's range (%.4g to "
            "%.4g ms); need at least %d"
            % (ds.get("name", "?"), tau_k.size, trim["basis_lo"] * 1e3,
               trim["basis_hi"] * 1e3, _MIN_POINTS))

    A = basis.build(tau_k)
    result = fcs_linsolve.solve_linear(G_k, A, sigma=sig_k,
                                       nonneg=basis.nonneg_mask(),
                                       rescale_errors="auto")
    notes.extend(result.warnings)

    n_comp = len(basis.curves)
    amp = result.d[:n_comp]
    amp_err = result.d_err[:n_comp]
    baseline = float(result.d[n_comp]) if basis.include_baseline else 0.0
    baseline_err = (float(result.d_err[n_comp]) if basis.include_baseline
                    else float("nan"))

    # Amplitudes are the components' contributions to G(0) because every basis
    # curve was normalised to 1 at its shortest lag, so they sum to the
    # mixture's G(0) and <N> follows exactly as it does elsewhere.  The
    # baseline is an offset, not a population, and stays out of both.
    G0 = float(np.sum(amp))
    N = 1.0 / G0 if G0 > 0.0 else float("nan")
    frac = amp / G0 if G0 > 0.0 else np.full(n_comp, np.nan)

    # Per-component curves, for the plot: what each one contributes.
    contributions = [amp[j] * A[:, j] for j in range(n_comp)]

    coh = fcs_linsolve.column_coherence(
        A[:, :n_comp] if sig_k is None else (A / sig_k[:, None])[:, :n_comp])

    return {
        "name": ds.get("name", "?"), "path": ds.get("path"),
        "meta": ds.get("meta", {}),
        "basis_name": basis.name, "components": basis.component_names(),
        "tau": tau_k, "G": G_k, "sigma": sig_k,
        "G_fit": result.G_fit, "residual": result.residuals,
        "residual_weighted": result.residuals_weighted,
        "contributions": contributions,
        "amp": amp, "amp_err": amp_err, "frac": frac,
        "pinned": result.pinned[:n_comp],
        "baseline": baseline, "baseline_err": baseline_err,
        "G0": G0, "N": N,
        "chi2": result.chi2, "chi2_red": result.chi2_red, "dof": result.dof,
        "durbin_watson": float(getattr(result, "durbin_watson", float("nan"))),
        "cond": result.cond, "rank": result.rank,
        "coherence_max": coh["max"], "coherence_pair": coh["pair"],
        "weighted": sig_k is not None,
        "trim": trim, "notes": notes, "result": result,
    }


def fit_datasets(datasets, basis, weighted=True):
    return [fit_dataset(ds, basis, weighted=weighted) for ds in datasets]


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_basis_fit(fit, show=True):
    """Curve, total fit, per-component contributions, and residuals."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(8.0, 6.2))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08, figure=fig)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    tau_ms = fit["tau"] * 1e3
    if fit["sigma"] is not None:
        ax.errorbar(tau_ms, fit["G"], yerr=fit["sigma"], fmt="o", ms=3,
                    color="0.35", ecolor="0.75", elinewidth=0.8, capsize=0,
                    label="G (data)", zorder=2)
    else:
        ax.plot(tau_ms, fit["G"], "o", ms=3, color="0.35", label="G (data)",
                zorder=2)
    ax.plot(tau_ms, fit["G_fit"], "-", lw=1.8, color="#c0392b",
            label="sum of components", zorder=4)

    palette = getattr(fcs_plottools, "palette", None)
    colours = (palette(len(fit["components"])) if callable(palette)
               else ["#2c6fbb", "#2e8b57", "#8e44ad", "#d68910", "#16a085"])
    for j, nm in enumerate(fit["components"]):
        ax.plot(tau_ms, fit["contributions"][j], "--", lw=1.1,
                color=colours[j % len(colours)], zorder=3,
                label="%s  (%.1f%%)" % (nm, 100.0 * fit["frac"][j]))
    if fit["baseline"] != 0.0:
        ax.axhline(fit["baseline"], color="0.55", lw=0.8, ls=":",
                   label="_baseline")

    ax.set_xscale("log")
    ax.set_ylabel("G(τ)", fontsize=12)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    fcs_plottools.adaptive_legend(ax, base_fontsize=9, loc="upper right")
    ax.set_title("Curve-basis decomposition — %s\n%d components from %r  ·  "
                 "χ²/dof = %.4g  ·  DW = %.3f"
                 % (fit["name"], len(fit["components"]),
                    fit["basis_name"] or "unnamed basis",
                    fit["chi2_red"], fit["durbin_watson"]), fontsize=10)
    plt.setp(ax.get_xticklabels(), visible=False)

    res = fit["residual_weighted"] if fit["weighted"] else fit["residual"]
    axr.axhline(0.0, color="grey", linewidth=0.8, label="_zero")
    axr.plot(tau_ms, res, color="#2c6fbb", label="_residuals",
             **fcs_plottools.residual_style(marker="o", markersize=2.5))
    axr.set_xscale("log")
    axr.set_xlabel("Lag time τ (ms)", fontsize=12)
    axr.set_ylabel("resid/σ" if fit["weighted"] else "resid", fontsize=10)
    axr.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)

    axes = np.array([ax, axr])
    if show:
        fcs_plottools.show_figure(fig, axes)
    return fig, axes


# ── Export ────────────────────────────────────────────────────────────────────

def _stem(label=None, when=None):
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    slug = slug_name(label)
    return "%s_%s" % (stamp, slug) if slug else "basisfit_%s" % stamp


def _write_curves(path, fits):
    cv = fcs_export.csv_value
    comps = fits[0]["components"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write("# FCS curve-basis decomposition — curves\n")
        fh.write("# written : %s\n"
                 % datetime.now().isoformat(timespec="seconds"))
        fh.write("# basis : %s\n" % (fits[0]["basis_name"] or "unnamed"))
        fh.write("# components : %s\n" % ", ".join(comps))
        fh.write("# note : long format; filter on the dataset column\n")
        head = ["dataset", "tau_s", "G", "sigma", "G_fit", "residual",
                "residual_weighted"]
        head += ["comp_%s" % slug_name(c) for c in comps]
        fh.write(",".join(head) + "\n")
        for f in fits:
            nm = f["name"].replace(",", "_")
            sig = f["sigma"]
            for i in range(f["tau"].size):
                row = [nm, cv(f["tau"][i]), cv(f["G"][i]),
                       cv(sig[i] if sig is not None else np.nan),
                       cv(f["G_fit"][i]), cv(f["residual"][i]),
                       cv(f["residual_weighted"][i])]
                row += [cv(c[i]) for c in f["contributions"]]
                fh.write(",".join(row) + "\n")
    print("[basisfit] wrote %s" % path)
    return Path(path)


def _params_header(components):
    head = ["dataset", "basis"]
    for name in components:
        tag = slug_name(name) or "component"
        head += [tag, "%s_err" % tag, "%s_frac" % tag]
    head += ["G0_sum", "N", "baseline", "baseline_err",
             "chi2_red", "durbin_watson", "dof", "cond", "coherence_max",
             "n_lags", "n_trimmed", "weighted"]
    return head


def _params_rows(fits):
    rows = []
    for f in fits:
        row = [f["name"], f["basis_name"] or "unnamed"]
        for j in range(len(f["components"])):
            row += [f["amp"][j], f["amp_err"][j], f["frac"][j]]
        trim = f.get("trim", {})
        row += [f["G0"], f["N"], f["baseline"], f["baseline_err"],
                f["chi2_red"], f["durbin_watson"], f["dof"], f["cond"],
                f.get("coherence_max", float("nan")), f["tau"].size,
                int(trim.get("n_fast", 0)) + int(trim.get("n_slow", 0)),
                "yes" if f["weighted"] else "no"]
        rows.append(row)
    return rows


def _report_lines(fits):
    L = []
    L.append("FCS curve-basis decomposition")
    L.append("=" * 64)
    L.append("method   : least squares onto exported fit curves, amplitudes >= 0")
    L.append("fitted   : %s" % datetime.now().isoformat(timespec="seconds"))
    L.append("basis    : %s" % (fits[0]["basis_name"] or "unnamed"))
    L.append("")
    L.append("Each basis curve is normalised to 1 at its own shortest lag, so")
    L.append("an amplitude is that component's contribution to G(0) and the")
    L.append("amplitudes sum to the mixture's G(0).  The baseline is an offset,")
    L.append("not a population, and is excluded from G0 and <N>.")
    L.append("")
    L.append("No property of the fits that produced the basis curves is used:")
    L.append("not the model, not chi2, not the parameter values.  The curves")
    L.append("are taken as given.")
    L.append("")
    for f in fits:
        L.append("-" * 64)
        L.append("dataset   : %s" % f["name"])
        if f.get("path") is not None:
            L.append("source    : %s" % f["path"])
        trim = f.get("trim", {})
        L.append("lags used : %d of %d  (basis covers %.4g to %.4g ms)"
                 % (f["tau"].size, trim.get("n_total", f["tau"].size),
                    trim.get("basis_lo", float("nan")) * 1e3,
                    trim.get("basis_hi", float("nan")) * 1e3))
        L.append("weighting : %s" % ("per-point sigma" if f["weighted"]
                                     else "UNWEIGHTED"))
        L.append("chi2/dof  : %.6g   (dof %d)" % (f["chi2_red"], f["dof"]))
        L.append("Durbin-Watson: %.4f" % f["durbin_watson"])
        L.append("cond      : %.4g   worst pair coherence %.6f"
                 % (f["cond"], f.get("coherence_max", float("nan"))))
        L.append("components:")
        L.append("      name                       amplitude          fraction")
        for j, nm in enumerate(f["components"]):
            L.append("   %-24s %11.6g +/- %-11.6g %7.2f%%"
                     % (nm[:24], f["amp"][j], f["amp_err"][j],
                        100.0 * f["frac"][j]))
        L.append("   %-24s %11.6g +/- %-11.6g"
                 % ("baseline", f["baseline"], f["baseline_err"]))
        L.append("G0 = sum  : %.6g" % f["G0"])
        L.append("<N>       : %.6g" % f["N"])
        if f["notes"]:
            L.append("notes:")
            for n in f["notes"]:
                L.append("  - %s" % n)
    L.append("-" * 64)
    return L


def export_basis_fit(fits, out_source, label=None):
    """Report, curves and a parameter table, in the usual fits/ folder."""
    out_dir = new_fit_dir(Path(out_source), label)
    stem = _stem(label)

    report = out_dir / ("%s_report.txt" % stem)
    report.write_text("\n".join(_report_lines(fits)) + "\n", encoding="utf-8")
    print("[basisfit] wrote %s" % report)

    curves = _write_curves(out_dir / ("%s_curves.csv" % stem), fits)

    params_csv, _x = write_params_table(
        out_dir / ("%s_params.csv" % stem),
        ["FCS curve-basis decomposition — parameters",
         "basis : %s" % (fits[0]["basis_name"] or "unnamed"),
         "components : %s" % ", ".join(fits[0]["components"]),
         "note : amplitudes are contributions to G(0); the baseline is excluded "
         "from G0 and N"],
        _params_header(fits[0]["components"]), _params_rows(fits),
        log_tag="basisfit", sheet_title="curve-basis fit")

    return out_dir, report, curves, params_csv


# ── GUI ───────────────────────────────────────────────────────────────────────

def run_basis_dialog(fcs_data=None, parent=None, workspace_order=None):
    """Entry point: pick the mixtures, build or load a basis, decompose.

    Datasets first, then the basis, matching the order the other Model Data
    entries use.
    """
    import fcs_fit

    init_dir = fcs_fit.default_dataset_dir(fcs_data)

    def _after(loaded):
        _basis_dialog(parent, loaded, init_dir)

    fcs_fit.select_datasets_dialog(parent, init_dir, _after,
                                   order=workspace_order)


def _basis_dialog(parent, datasets, init_dir=None):
    """Build or load a basis of exported fit curves, then decompose."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog

    win, body, btnbar = scrollable_dialog(parent, "Curve basis — setup")

    basis = CurveBasis("")

    tk.Label(body, text="Correlation — curve-basis decomposition",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(body, text="%d dataset%s to decompose."
                        % (len(datasets), "" if len(datasets) == 1 else "s"),
             font=("Helvetica", 9), fg="grey").pack()

    # ---- saved bases ----
    saved_f = tk.LabelFrame(body, text="Saved basis", padx=10, pady=6)
    saved_f.pack(fill="x", padx=12, pady=(8, 4))
    saved_var = tk.StringVar(master=win, value="")
    saved_menu = {"widget": None}

    def _refresh_saved():
        entries = list_saved_bases()
        names = [n for n, _p in entries] or ["(none saved)"]
        if saved_menu["widget"] is not None:
            saved_menu["widget"].destroy()
        om = tk.OptionMenu(saved_f, saved_var, *names)
        om.config(font=("Helvetica", 9), anchor="w")
        om["menu"].config(font=("Helvetica", 9))
        om.pack(fill="x")
        saved_menu["widget"] = om
        if saved_var.get() not in names:
            saved_var.set(names[0])
        return dict(entries)

    known = {"map": _refresh_saved()}

    # ---- the component list ----
    comp_f = tk.LabelFrame(body, text="Basis curves", padx=10, pady=6)
    comp_f.pack(fill="both", expand=True, padx=12, pady=4)
    listbox = tk.Listbox(comp_f, height=6, font=("Helvetica", 9),
                         exportselection=False)
    listbox.pack(fill="both", expand=True)

    range_lbl = tk.Label(comp_f, text="", font=("Helvetica", 8), fg="grey",
                         anchor="w", justify="left", wraplength=430)
    range_lbl.pack(fill="x")
    warn_lbl = tk.Label(comp_f, text="", font=("Helvetica", 8, "bold"),
                        fg="#c0392b", anchor="w", justify="left",
                        wraplength=430)
    warn_lbl.pack(fill="x")

    def _refresh_list():
        listbox.delete(0, "end")
        for c in basis.curves:
            listbox.insert("end", "%s   [%.4g – %.4g ms, %d pts]"
                           % (c.name, c.tau[0] * 1e3, c.tau[-1] * 1e3,
                              c.tau.size))
        if not basis.curves:
            range_lbl.config(text="      no curves yet")
            warn_lbl.config(text="")
            return
        lo, hi = basis.usable_lag_range()
        range_lbl.config(text="      every curve covers %.4g to %.4g ms; that "
                              "is the range the data can be decomposed over."
                              % (lo * 1e3, hi * 1e3))
        if lo >= hi:
            warn_lbl.config(text="      These curves have no lag range in "
                                 "common. Remove one, or the fit has nothing "
                                 "to work with.")
            return
        # How much of the actual data this basis will cost.
        worst, total = 0, 0
        for ds in datasets:
            t = np.asarray(ds.get("tau"), dtype=float).ravel()
            t = t[np.isfinite(t) & (t > 0.0)]
            if not t.size:
                continue
            outside = int(np.sum((t < lo) | (t > hi)))
            worst = max(worst, outside)
            total = max(total, t.size)
        if worst:
            warn_lbl.config(text="      Up to %d of %d lag points fall outside "
                                 "this range and will be trimmed before "
                                 "fitting." % (worst, total))
        else:
            warn_lbl.config(text="")

    def _add_files():
        paths = filedialog.askopenfilenames(
            parent=win, title="Select exported fit curves",
            initialdir=str(init_dir) if init_dir else None,
            filetypes=[("Exported curves", "*_curves.csv"),
                       ("CSV files", "*.csv"), ("All files", "*.*")])
        added, failed = 0, []
        for p in paths:
            try:
                for c in read_curve_file(p):
                    basis.curves.append(c)
                    added += 1
            except Exception as exc:                 # noqa: BLE001
                failed.append("%s: %s" % (Path(p).name, exc))
        _refresh_list()
        if failed:
            messagebox.showwarning(
                "Some files could not be read",
                "\n\n".join(failed) +
                "\n\nA basis curve comes from a fitter's *_curves.csv, which "
                "carries tau_s and G_fit columns.", parent=win)
        elif added:
            print("[basisfit] added %d curve(s)" % added)

    def _remove():
        sel = listbox.curselection()
        if not sel:
            return
        for i in sorted(sel, reverse=True):
            del basis.curves[i]
        _refresh_list()

    def _rename():
        sel = listbox.curselection()
        if not sel:
            return
        cur = basis.curves[sel[0]]
        new = simpledialog.askstring("Rename component",
                                     "Name for this basis curve:",
                                     initialvalue=cur.name, parent=win)
        if new and new.strip():
            cur.name = new.strip()
            _refresh_list()

    def _load_saved():
        path = known["map"].get(saved_var.get())
        if path is None:
            messagebox.showinfo("No basis", "Nothing saved yet.", parent=win)
            return
        try:
            loaded = CurveBasis.load(path)
        except Exception as exc:                     # noqa: BLE001
            messagebox.showerror("Could not load", str(exc), parent=win)
            return
        basis.curves = list(loaded.curves)
        basis.name = loaded.name
        basis.include_baseline = loaded.include_baseline
        base_var.set(basis.include_baseline)
        name_var.set(basis.name)
        _refresh_list()

    def _save_as():
        if not basis.curves:
            messagebox.showinfo("Nothing to save",
                                "Add at least one curve first.", parent=win)
            return
        nm = simpledialog.askstring("Save basis", "Name for this basis:",
                                    initialvalue=basis.name or "", parent=win)
        if not nm or not nm.strip():
            return
        basis.name = nm.strip()
        basis.include_baseline = bool(base_var.get())
        try:
            path = basis.save()
        except Exception as exc:                     # noqa: BLE001
            messagebox.showerror("Could not save", str(exc), parent=win)
            return
        name_var.set(basis.name)
        known["map"] = _refresh_saved()
        saved_var.set(basis.name)
        messagebox.showinfo("Saved", "Saved as\n%s" % path, parent=win)

    row = tk.Frame(saved_f)
    row.pack(fill="x", pady=(4, 0))
    tk.Button(row, text="Load", width=9, command=_load_saved).pack(side="left")
    tk.Button(row, text="Save as…", width=9,
              command=_save_as).pack(side="left", padx=4)

    crow = tk.Frame(comp_f)
    crow.pack(fill="x", pady=(4, 0))
    tk.Button(crow, text="Add curves file…", command=_add_files).pack(side="left")
    tk.Button(crow, text="Rename", width=8,
              command=_rename).pack(side="left", padx=4)
    tk.Button(crow, text="Remove", width=8, command=_remove).pack(side="left")

    # ---- options ----
    opt_f = tk.LabelFrame(body, text="Options", padx=10, pady=6)
    opt_f.pack(fill="x", padx=12, pady=4)
    have_sigma = all(ds.get("sigma") is not None for ds in datasets)
    weight_var = tk.BooleanVar(master=win, value=have_sigma)
    tk.Checkbutton(opt_f, variable=weight_var, anchor="w",
                   state="normal" if have_sigma else "disabled",
                   text=("Weight by σ (G_std)" if have_sigma
                         else "Weight by σ — no G_std in every dataset")
                   ).pack(fill="x")
    base_var = tk.BooleanVar(master=win, value=True)
    tk.Checkbutton(opt_f, variable=base_var, anchor="w",
                   text="Fit a free baseline column").pack(fill="x")
    tk.Label(opt_f, text="      Component amplitudes are held at or above "
                         "zero; the baseline is not.\n"
                         "      Each curve is normalised to 1 at its own "
                         "shortest lag, so an amplitude\n"
                         "      is that component's share of G(0).",
             font=("Helvetica", 8), fg="grey", anchor="w",
             justify="left").pack(fill="x")

    nrow = tk.Frame(body)
    nrow.pack(fill="x", padx=12, pady=(4, 2))
    tk.Label(nrow, text="Label for the output (optional):", anchor="w",
             font=("Helvetica", 9)).pack(side="left")
    name_var = tk.StringVar(master=win, value="")
    tk.Entry(nrow, textvariable=name_var, width=20,
             font=("Helvetica", 9)).pack(side="right")

    export_var = tk.BooleanVar(master=win, value=True)
    tk.Checkbutton(body, variable=export_var, anchor="w",
                   text="Write report, curves and parameter table"
                   ).pack(fill="x", padx=16)

    def _run():
        if not basis.curves:
            messagebox.showinfo("No basis",
                                "Add at least one fit curve to decompose onto.",
                                parent=win)
            return
        lo, hi = basis.usable_lag_range()
        if lo >= hi:
            messagebox.showerror(
                "Basis curves do not overlap",
                "These curves share no lag range, so there is nothing to fit "
                "over. Remove whichever one covers a different window.",
                parent=win)
            return
        basis.include_baseline = bool(base_var.get())
        try:
            fits = fit_datasets(datasets, basis, weighted=weight_var.get())
        except ValueError as exc:
            messagebox.showerror("Decomposition failed", str(exc), parent=win)
            return

        trimmed = [f for f in fits if f["trim"].get("message")]
        win.destroy()
        if trimmed:
            messagebox.showwarning(
                "Data trimmed to the basis",
                "\n\n".join("%s:\n%s" % (f["name"], f["trim"]["message"])
                             for f in trimmed),
                parent=parent)
        for f in fits:
            plot_basis_fit(f, show=True)
        if export_var.get():
            export_basis_fit(fits, datasets[0]["path"],
                             name_var.get().strip() or basis.name or None)

    inner = tk.Frame(btnbar)
    inner.pack(pady=10)
    tk.Button(inner, text="Decompose", width=12, command=_run,
              pady=4).pack(side="left", padx=6)
    tk.Button(inner, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(side="left", padx=6)

    _refresh_list()
    fit_dialog_to_screen(win)
    win.wait_window()


# ── Self-test ─────────────────────────────────────────────────────────────────

def _selftest():
    import tempfile
    import warnings

    rng = np.random.RandomState(1234)

    def curve(tau, tau_d):
        return fcs_linsolve.build_basis(
            tau, [tau_d / fcs_linsolve.get_basis_model().anchor_s],
            t_units="s", include_baseline=False)[:, 0]

    print("=" * 68)
    print("TEST 1: BasisCurve normalisation and sampling")
    t_a = np.logspace(-5.5, -0.5, 90)
    c_a = BasisCurve("species A", t_a, 0.037 * curve(t_a, 1e-4), "A.csv")
    assert np.isclose(c_a.normalised()[0], 1.0)
    assert np.isclose(c_a.g0, 0.037 * curve(t_a, 1e-4)[0])
    # order and bad points are cleaned on construction
    messy = BasisCurve("m", [1e-3, np.nan, 1e-5, -1.0], [0.1, 0.2, 0.5, 0.9])
    assert messy.tau.size == 2 and messy.tau[0] < messy.tau[1]
    # sampling interpolates inside and refuses to extrapolate outside
    probe = np.array([t_a[0] / 10, t_a[0], 3e-4, t_a[-1], t_a[-1] * 10])
    got = c_a.sample(probe)
    assert np.isnan(got[0]) and np.isnan(got[-1]), "must not extrapolate"
    assert np.isclose(got[1], 1.0) and np.all(np.isfinite(got[1:4]))
    print("  normalised to 1 at the shortest lag; outside its range -> NaN")

    print("=" * 68)
    print("TEST 2: reading an exported curves file")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "g_curves.csv"
        tt = np.logspace(-5, -1, 40)
        with p.open("w", encoding="utf-8-sig", newline="") as fh:
            fh.write("# FCS global fit curves\n")
            fh.write("# analysis   : global fit curves\n")
            fh.write("dataset,tau_s,tau_ms,G_data,G_fit,residual\n")
            for label, td_ in (("A", 8e-5), ("B", 9e-4)):
                gg = 0.04 * curve(tt, td_)
                for i in range(tt.size):
                    fh.write("%s,%.10g,%.10g,%.10g,%.10g,%.10g\n"
                             % (label, tt[i], tt[i] * 1e3, gg[i] + 1e-4,
                                gg[i], 1e-4))
        loaded = read_curve_file(p)
        assert len(loaded) == 2, "one BasisCurve per dataset"
        assert [c.name for c in loaded] == ["A", "B"]
        assert loaded[0].tau.size == tt.size
        print("  two datasets in one file -> %s" % [c.name for c in loaded])

        bad = Path(td) / "not_curves.csv"
        bad.write_text("dataset,tau_s,G\nA,1,2\n", encoding="utf-8")
        try:
            read_curve_file(bad)
        except ValueError as exc:
            assert "fit-curve column" in str(exc)
            print("  a file without a fit column is refused, by name")
        else:
            raise AssertionError("expected ValueError")

    print("=" * 68)
    print("TEST 3: a two-species mixture is recovered")
    t_b = np.logspace(-5.5, -0.5, 90)
    c_b = BasisCurve("species B", t_b, 0.052 * curve(t_b, 1.2e-3), "B.csv")
    basis = CurveBasis("A + B", [c_a, c_b])
    assert basis.n_columns == 3 and basis.component_names() == ["species A",
                                                                "species B"]

    tau = np.logspace(-5, -1, 70)
    true_amp = np.array([0.030, 0.018])
    true_base = -0.0008
    mix = (true_amp[0] * c_a.sample(tau) + true_amp[1] * c_b.sample(tau)
           + true_base)
    sig = 0.0004 * np.ones(tau.size)
    obs = mix + sig * rng.randn(tau.size)
    ds = {"name": "mixture", "path": None, "tau": tau, "G": obs, "sigma": sig,
          "meta": {"G_std_definition": "sem"}}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = fit_dataset(ds, basis)
    print("  amplitudes  : %s   (truth %s)"
          % (np.array2string(f["amp"], precision=5),
             np.array2string(true_amp, precision=5)))
    print("  fractions   : %s" % np.array2string(100 * f["frac"], precision=2))
    print("  baseline    : %+.6f (truth %+.6f)" % (f["baseline"], true_base))
    print("  chi2_red %.4f, cond %.4g, worst coherence %.6f"
          % (f["chi2_red"], f["cond"], f["coherence_max"]))
    dev = np.abs(f["amp"] - true_amp) / f["amp_err"]
    assert np.all(dev < 4.0), dev
    assert np.all(f["amp"] >= 0.0)
    assert abs(f["baseline"] - true_base) < 5e-4
    assert 0.4 < f["chi2_red"] < 2.0
    assert np.isclose(f["G0"], f["amp"].sum())
    assert np.isclose(f["N"], 1.0 / f["G0"])
    assert abs(f["frac"].sum() - 1.0) < 1e-12

    print("=" * 68)
    print("TEST 4: data beyond the basis is trimmed, and says so")
    wide = np.logspace(-6.5, 0.0, 90)            # well outside the basis
    gw = np.interp(np.log(np.clip(wide, t_a[0], t_a[-1])), np.log(tau),
                   mix)
    dsw = {"name": "wide", "path": None, "tau": wide, "G": gw,
           "sigma": 0.0004 * np.ones(wide.size),
           "meta": {"G_std_definition": "sem"}}
    _t, _g, _s, info = trim_to_basis(dsw["tau"], dsw["G"], dsw["sigma"], basis)
    print("  basis covers %.4g to %.4g ms" % (info["basis_lo"] * 1e3,
                                              info["basis_hi"] * 1e3))
    print("  %s" % info["message"])
    assert info["n_fast"] > 0 and info["n_slow"] > 0
    assert info["n_kept"] == _t.size
    assert _t[0] >= info["basis_lo"] and _t[-1] <= info["basis_hi"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fw = fit_dataset(dsw, basis)
    assert any("trimmed" in n for n in fw["notes"])
    # building without trimming first must fail loudly, not silently NaN
    try:
        basis.build(wide)
    except ValueError as exc:
        assert "usable range" in str(exc)
        print("  building on untrimmed lags raises instead of producing NaNs")
    else:
        raise AssertionError("expected ValueError")

    print("=" * 68)
    print("TEST 5: save and reload a basis")
    with tempfile.TemporaryDirectory() as td:
        os.environ["APPDATA"] = td
        assert bases_dir() == Path(td) / "FCS-Programs" / "curve_bases"
        saved = basis.save()
        assert saved.exists()
        again = CurveBasis.load(saved)
        assert again.name == basis.name
        assert again.component_names() == basis.component_names()
        for c1, c2 in zip(basis.curves, again.curves):
            assert np.allclose(c1.tau, c2.tau) and np.allclose(c1.g, c2.g)
        A1, A2 = basis.build(tau), again.build(tau)
        assert np.allclose(A1, A2), "a reloaded basis must build the same matrix"
        listed = list_saved_bases()
        assert [n for n, _p in listed] == ["A + B"], listed
        junk = bases_dir() / "junk.json"
        junk.write_text("{ not json", encoding="utf-8")
        assert [n for n, _p in list_saved_bases()] == ["A + B"], \
            "a corrupt file must be skipped, not fatal"
        try:
            CurveBasis.from_dict({"format": "something else"})
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
        print("  saved to %s, reloaded identically, corrupt files skipped"
              % saved.name)

    print("=" * 68)
    print("TEST 6: exports")
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "data" / "analysis" / "mix_correlation_cross.csv"
        src.parent.mkdir(parents=True)
        src.touch()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out_dir, rep, cur, par = export_basis_fit([f], src, "two species")
        assert out_dir.parent.name == "fits"
        for p in (rep, cur, par):
            assert p.exists() and p.stat().st_size > 0
        head = [l for l in cur.read_text(encoding="utf-8-sig").splitlines()
                if not l.startswith("#")][0]
        assert head.endswith("comp_species_A,comp_species_B"), head
        prow = [l for l in par.read_text(encoding="utf-8-sig").splitlines()
                if not l.startswith("#")]
        assert prow[0].startswith("dataset,basis,species_A,species_A_err,"
                                  "species_A_frac,")
        print("  curves carry a column per component; params one row per dataset")

    print("=" * 68)
    print("TEST 7: plotting, headless")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plot_basis_fit(f, show=False)
    assert len(axes) == 2
    labels = [ln.get_label() for ln in axes[0].get_lines()]
    assert any("species A" in l for l in labels)
    plt.close(fig)
    print("  two panels, one dashed contribution per component")

    print("=" * 68)
    print("TEST 8: the basis-builder dialog")
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:                        # noqa: BLE001
        print("  no display (%s); skipped the dialog check" % type(exc).__name__)
    else:
        root.withdraw()
        captured = {}
        real_wait = tk.Toplevel.wait_window
        tk.Toplevel.wait_window = lambda self, *a: captured.setdefault("w", self)
        old_appdata = os.environ.get("APPDATA")
        tmp = tempfile.mkdtemp()
        os.environ["APPDATA"] = tmp
        try:
            _basis_dialog(root, [ds])
            win = captured["w"]
            win.update_idletasks()
            assert win.resizable() == (True, True)
            assert win.winfo_height() <= int(win.winfo_screenheight() * 0.85) + 1
            found = {}

            def _walk(w):
                for c in w.winfo_children():
                    cls = c.winfo_class()
                    if cls == "Listbox":
                        found["list"] = c
                    elif cls == "Button":
                        found.setdefault("buttons", []).append(c.cget("text"))
                    _walk(c)

            _walk(win)
            assert "list" in found, "the component list must exist"
            for label in ("Add curves file…", "Rename", "Remove", "Load",
                          "Save as…", "Decompose", "Cancel"):
                assert label in found["buttons"], "missing button %r" % label
            assert found["list"].size() == 0, "a new basis starts empty"
            print("  dialog built: %dx%d, buttons %s"
                  % (win.winfo_width(), win.winfo_height(),
                     ", ".join(found["buttons"])))
            win.destroy()
        finally:
            tk.Toplevel.wait_window = real_wait
            root.destroy()
            if old_appdata is None:
                os.environ.pop("APPDATA", None)
            else:
                os.environ["APPDATA"] = old_appdata

    print("=" * 68)
    print("fcs_basisfit: all self-tests passed.")


if __name__ == "__main__":
    _selftest()
