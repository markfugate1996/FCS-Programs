#!/usr/bin/env python3
"""
Report an uncertainty on the mean lifetime of a multi-exponential fit.

Both fitters already print <tau>_amp and <tau>_int with no error beside
them.  The covariance needed to propagate one is already kept (both store
pcov for the Fisher section of their reports), so this adds the
propagation and prints the result.

Touches fcs_fitcommon.py (the shared helper), fcs_lifetime_fit.py (tail
fits) and fcs_lifetime_recon.py (reconvolution fits).  Every anchor is
validated before anything is written; line endings are preserved per file.

Usage:  python3 patch_tau_mean_error.py [path/to/FCS-Programs]
"""
import pathlib
import sys

COMMON_EDITS = [
    ("mean-lifetime error propagation",
'''# ── Dialog windows ────────────────────────────────────────────────────────────
''',
'''# ── Derived-quantity uncertainties ────────────────────────────────────────────

def mean_lifetime_errors(values: dict, pairs, free, pcov) -> Optional[dict]:
    """
    1σ uncertainties on the amplitude- and intensity-weighted mean lifetimes.

        <tau>_amp = SUM A_i tau_i / SUM A_i
        <tau>_int = SUM A_i tau_i^2 / SUM A_i tau_i

    Propagated through the FULL covariance matrix rather than from the
    per-parameter errors alone.  In a multi-exponential fit the amplitudes
    and lifetimes are strongly anticorrelated -- a longer tau is bought by a
    smaller A -- and treating them as independent overstates the uncertainty
    on a mean that is typically far better determined than the components it
    is made of.  That is the whole reason the mean is worth quoting on a fit
    whose individual taus are poorly separated, so the covariance is not
    optional here.

    Gradients are analytic:

        d<tau>amp/dA_i   = (tau_i - <tau>amp) / SUM A
        d<tau>amp/dtau_i = A_i / SUM A
        d<tau>int/dA_i   = tau_i (tau_i - <tau>int) / SUM A tau
        d<tau>int/dtau_i = A_i (2 tau_i - <tau>int) / SUM A tau

    and sigma^2 = g^T C g over the FREE parameters; a fixed parameter
    contributes nothing, having no uncertainty to contribute.

    *pairs* is [(amplitude_name, lifetime_name), ...] -- the two fitters name
    their parameters differently, so neither naming is assumed here.

    The result inherits whatever convention the fit's own covariance follows
    (scaled by reduced chi-squared for an unweighted fit, absolute for a
    weighted one), exactly as the per-parameter errors do.  Returns None when
    the decomposition itself is not meaningful (fewer than two components,
    non-positive amplitudes or lifetimes), and NaN errors when the fit
    returned no usable covariance -- a parameter resting on a bound being the
    common cause, where the quoted error would be meaningless anyway.
    """
    import numpy as np

    comps = [(float(values[a]), float(values[t]), a, t)
             for a, t in pairs if a in values and t in values]
    if len(comps) < 2:
        return None
    A = np.array([c[0] for c in comps], dtype=float)
    T = np.array([c[1] for c in comps], dtype=float)
    if np.any(A <= 0) or np.any(T <= 0):
        return None
    sum_a = float(A.sum())
    sum_at = float((A * T).sum())
    if sum_a <= 0 or sum_at <= 0:
        return None

    nan = {"tau_mean_amp_err": float("nan"), "tau_mean_int_err": float("nan")}
    if pcov is None:
        return nan
    C = np.asarray(pcov, dtype=float)
    free = list(free)
    if (C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != len(free)
            or not np.all(np.isfinite(C))):
        return nan

    tau_amp = sum_at / sum_a
    tau_int = float((A * T * T).sum()) / sum_at

    index = {name: i for i, name in enumerate(free)}
    g_amp = np.zeros(len(free), dtype=float)
    g_int = np.zeros(len(free), dtype=float)
    for a_val, t_val, a_name, t_name in comps:
        if a_name in index:
            g_amp[index[a_name]] += (t_val - tau_amp) / sum_a
            g_int[index[a_name]] += t_val * (t_val - tau_int) / sum_at
        if t_name in index:
            g_amp[index[t_name]] += a_val / sum_a
            g_int[index[t_name]] += a_val * (2.0 * t_val - tau_int) / sum_at

    def _sigma(g) -> float:
        var = float(g @ C @ g)
        # A tiny negative variance is round-off on a near-singular covariance,
        # not a result; anything else non-finite is the fit's problem, not
        # something to paper over with a number.
        if not np.isfinite(var):
            return float("nan")
        return float(np.sqrt(var)) if var > 0 else 0.0

    return {"tau_mean_amp_err": _sigma(g_amp),
            "tau_mean_int_err": _sigma(g_int)}


# ── Dialog windows ────────────────────────────────────────────────────────────
'''),
]

FIT_EDITS = [
    ("tail fit computes the error",
'''    fractions = component_fractions(values, names)
    tau_mean  = fractions["tau_mean_amp"] if fractions else float("nan")
''',
'''    fractions = component_fractions(values, names)
    tau_mean  = fractions["tau_mean_amp"] if fractions else float("nan")
    if fractions:
        # Folded into `fractions` so every consumer that already reads the
        # means from there gets the errors without being handed a second
        # dict to look in.
        _tau_errs = mean_lifetime_errors(
            values, [(f"A{i}", f"tau{i}") for i in fractions["indices"]],
            free, pcov)
        if _tau_errs:
            fractions.update(_tau_errs)
    tau_mean_err = (fractions.get("tau_mean_amp_err", float("nan"))
                    if fractions else float("nan"))
'''),

    ("tail fit result carries it",
'''        "drop_edges": drop_edges, "tau_mean": tau_mean,
''',
'''        "drop_edges": drop_edges, "tau_mean": tau_mean,
        "tau_mean_err": tau_mean_err,
'''),

    ("tail fit figure shows it",
'''        if np.isfinite(result.get("tau_mean", float("nan"))):
            lines.append(f"⟨τ⟩ = {result['tau_mean']:.4g} ns  (ampl.)")
        if fr:
            lines.append(f"⟨τ⟩ = {fr['tau_mean_int']:.4g} ns  (int.)")
''',
'''        if np.isfinite(result.get("tau_mean", float("nan"))):
            _e = result.get("tau_mean_err", float("nan"))
            _pm = f" ± {_e:.2g}" if np.isfinite(_e) else ""
            lines.append(f"⟨τ⟩ = {result['tau_mean']:.4g}{_pm} ns  (ampl.)")
        if fr:
            _e = fr.get("tau_mean_int_err", float("nan"))
            _pm = f" ± {_e:.2g}" if np.isfinite(_e) else ""
            lines.append(f"⟨τ⟩ = {fr['tau_mean_int']:.4g}{_pm} ns  (int.)")
'''),

    ("tail fit report shows it",
'''        L.append(f"derived    : amplitude-weighted mean lifetime "
                 f"⟨τ⟩ = {result['tau_mean']:.6g} ns")
        if fr:
            L.append(f"             intensity-weighted mean lifetime "
                     f"⟨τ⟩ = {fr['tau_mean_int']:.6g} ns")
''',
'''        _ea = result.get("tau_mean_err", float("nan"))
        _pma = f" ± {_ea:.6g}" if np.isfinite(_ea) else ""
        L.append(f"derived    : amplitude-weighted mean lifetime "
                 f"⟨τ⟩ = {result['tau_mean']:.6g}{_pma} ns")
        if fr:
            _ei = fr.get("tau_mean_int_err", float("nan"))
            _pmi = f" ± {_ei:.6g}" if np.isfinite(_ei) else ""
            L.append(f"             intensity-weighted mean lifetime "
                     f"⟨τ⟩ = {fr['tau_mean_int']:.6g}{_pmi} ns")
        if fr and not np.isfinite(_ea):
            # Silence here would read as "no uncertainty", which is the one
            # thing it does not mean.
            L.append("             (no uncertainty: the fit returned no "
                     "usable covariance)")
'''),

    ("tail fit params table gains the columns",
'''        p_header += ["tau_mean_amp_ns", "tau_mean_int_ns"]
''',
'''        p_header += ["tau_mean_amp_ns", "tau_mean_amp_err_ns",
                     "tau_mean_int_ns", "tau_mean_int_err_ns"]
'''),

    ("tail fit params row gains the values",
'''        row.append(float(fr["tau_mean_amp"]))
        row.append(float(fr["tau_mean_int"]))
''',
'''        row.append(float(fr["tau_mean_amp"]))
        row.append(float(fr.get("tau_mean_amp_err", float("nan"))))
        row.append(float(fr["tau_mean_int"]))
        row.append(float(fr.get("tau_mean_int_err", float("nan"))))
'''),

    ("tail fit note explains the propagation",
'''        comments.append("note_tau_mean : tau_mean_amp = sum(A*tau)/sum(A) ; "
                        "tau_mean_int = sum(A*tau^2)/sum(A*tau)")
''',
'''        comments.append("note_tau_mean : tau_mean_amp = sum(A*tau)/sum(A) ; "
                        "tau_mean_int = sum(A*tau^2)/sum(A*tau)")
        comments.append("note_tau_mean_err : 1-sigma, propagated through the "
                        "full parameter covariance (amplitudes and lifetimes "
                        "are correlated, so this is not a sum in quadrature "
                        "of the individual errors)")
'''),

    ("tail fit summary line",
'''        if np.isfinite(result.get("tau_mean", float("nan"))):
            summary += f"\\n⟨τ⟩ = {result['tau_mean']:.4g} ns (amplitude-weighted)"
''',
'''        if np.isfinite(result.get("tau_mean", float("nan"))):
            _e = result.get("tau_mean_err", float("nan"))
            _pm = f" ± {_e:.2g}" if np.isfinite(_e) else ""
            summary += (f"\\n⟨τ⟩ = {result['tau_mean']:.4g}{_pm} ns "
                        f"(amplitude-weighted)")
'''),
]

RECON_EDITS = [
    ("reconvolution computes the error",
'''    derived = _derived_lifetimes(model, values)
''',
'''    derived = _derived_lifetimes(model, values)
    if model.n_exp > 1:
        _tau_errs = mean_lifetime_errors(
            values, list(zip(model.amp_names, model.tau_names)), free, pcov)
        if _tau_errs:
            derived.update(_tau_errs)
'''),

    ("reconvolution figure shows it",
'''    if model.n_exp > 1:
        lines.append(f"⟨τ⟩amp = {result['derived']['tau_mean_amp']:.3g} ns")
        lines.append(f"⟨τ⟩int = {result['derived']['tau_mean_int']:.3g} ns")
''',
'''    if model.n_exp > 1:
        _der = result["derived"]
        _ea = _der.get("tau_mean_amp_err", float("nan"))
        _ei = _der.get("tau_mean_int_err", float("nan"))
        _pma = f" ± {_ea:.2g}" if np.isfinite(_ea) else ""
        _pmi = f" ± {_ei:.2g}" if np.isfinite(_ei) else ""
        lines.append(f"⟨τ⟩amp = {_der['tau_mean_amp']:.3g}{_pma} ns")
        lines.append(f"⟨τ⟩int = {_der['tau_mean_int']:.3g}{_pmi} ns")
'''),

    ("reconvolution report shows it",
'''        L.append(f"  amplitude-weighted <tau>   : {der['tau_mean_amp']:.4f} ns")
        L.append(f"  intensity-weighted <tau>   : {der['tau_mean_int']:.4f} ns")
''',
'''        _ea = der.get("tau_mean_amp_err", float("nan"))
        _ei = der.get("tau_mean_int_err", float("nan"))
        _pma = f" +/- {_ea:.4f}" if np.isfinite(_ea) else ""
        _pmi = f" +/- {_ei:.4f}" if np.isfinite(_ei) else ""
        L.append(f"  amplitude-weighted <tau>   : "
                 f"{der['tau_mean_amp']:.4f}{_pma} ns")
        L.append(f"  intensity-weighted <tau>   : "
                 f"{der['tau_mean_int']:.4f}{_pmi} ns")
        if not np.isfinite(_ea):
            L.append("  (no uncertainty on <tau>: the Jacobian step returned "
                     "no usable covariance)")
'''),

    ("reconvolution params table gains the columns",
'''        p_header += ["tau_mean_amp_ns", "tau_mean_int_ns"]
''',
'''        p_header += ["tau_mean_amp_ns", "tau_mean_amp_err_ns",
                     "tau_mean_int_ns", "tau_mean_int_err_ns"]
'''),

    ("reconvolution params row gains the values",
'''        row.append(float(der["tau_mean_amp"]))
        row.append(float(der["tau_mean_int"]))
''',
'''        row.append(float(der["tau_mean_amp"]))
        row.append(float(der.get("tau_mean_amp_err", float("nan"))))
        row.append(float(der["tau_mean_int"]))
        row.append(float(der.get("tau_mean_int_err", float("nan"))))
'''),
]

IMPORT_EDITS = {
    "fcs_lifetime_fit.py": ("import the helper",
'''from fcs_fitcommon import (
''',
'''from fcs_fitcommon import (
    mean_lifetime_errors,
'''),
    "fcs_lifetime_recon.py": ("import the helper",
'''from fcs_fitcommon import (
''',
'''from fcs_fitcommon import (
    mean_lifetime_errors,
'''),
}


def _target(argv):
    p = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    if p.is_file():
        p = p.parent
    for name in ("fcs_fitcommon.py", "fcs_lifetime_fit.py",
                 "fcs_lifetime_recon.py"):
        if not (p / name).is_file():
            print(f"ERROR: {name} not found in {p}. Run this from the "
                  f"FCS-Programs directory, or pass its path.")
            raise SystemExit(2)
    return p


def _plan(path, edits):
    data = path.read_bytes()
    crlf = data.count(b"\r\n")
    lf_only = data.count(b"\n") - crlf
    if crlf and lf_only:
        raise ValueError(f"{path.name}: mixed line endings "
                         f"({crlf} CRLF, {lf_only} LF)")
    newline = b"\r\n" if crlf else b"\n"

    def enc(text):
        return text.encode("utf-8").replace(b"\n", newline)

    applied = []
    for name, old_lf, new_lf in edits:
        old, new = enc(old_lf), enc(new_lf)
        if data.count(new) and not data.count(old):
            raise ValueError(f"{path.name}: '{name}' looks already applied")
        n = data.count(old)
        if n != 1:
            raise ValueError(f"{path.name}: anchor for '{name}' matched "
                             f"{n} times, expected 1")
        data = data.replace(old, new, 1)
        applied.append(name)
    return data, applied, ("CRLF" if newline == b"\r\n" else "LF")


def main(argv) -> int:
    root = _target(argv)
    jobs = {
        "fcs_fitcommon.py": COMMON_EDITS,
        "fcs_lifetime_fit.py": [IMPORT_EDITS["fcs_lifetime_fit.py"]] + FIT_EDITS,
        "fcs_lifetime_recon.py": ([IMPORT_EDITS["fcs_lifetime_recon.py"]]
                                  + RECON_EDITS),
    }
    planned = {}
    for fname, edits in jobs.items():
        try:
            planned[fname] = _plan(root / fname, edits)
        except ValueError as exc:
            print(f"ERROR: {exc}\nNothing was written.")
            return 2
    for fname, (data, applied, eol) in planned.items():
        (root / fname).write_bytes(data)
        print(f"  {fname} ({eol}, {len(applied)} edits)")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
