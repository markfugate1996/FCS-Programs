#!/usr/bin/env python3
"""
Give the GLOBAL lifetime fit uncertainties on its derived quantities.

The global/batch path (fit_lifetime_global) builds tau_mean_amp, f1 and f2
by hand and reported them bare -- the earlier covariance work reached only
the single-decay tail fitter and the reconvolution fitter.  This propagates
errors there too, and adds the intensity-weighted mean the other two paths
already report.

Everything needed was already returned by fcs_globalfit.fit_linked: the
global covariance, plus each dataset's free_slots mapping its parameters to
covariance rows.

Touches fcs_fitcommon.py and fcs_lifetime_fit.py.  Anchors are validated
before anything is written; line endings preserved per file.

Usage:  python3 patch_global_derived_errors.py [path/to/FCS-Programs]
"""
import pathlib
import sys

COMMON_EDITS = [
    ("intensity-fraction error propagation",
'''    return {"tau_mean_amp_err": _sigma(g_amp),
            "tau_mean_int_err": _sigma(g_int)}


# ── Dialog windows ────────────────────────────────────────────────────────────
''',
'''    return {"tau_mean_amp_err": _sigma(g_amp),
            "tau_mean_int_err": _sigma(g_int)}


def intensity_fraction_errors(values: dict, pairs, free, pcov):
    """
    1σ uncertainties on the intensity fractions

        f_i = A_i tau_i / SUM_j A_j tau_j

    returned as a list in the order of *pairs*.

    Through the full covariance, for the same reason as the mean lifetime: a
    fraction is a ratio of fitted products, so an amplitude and its own
    lifetime trade against each other directly, and an error built from the
    diagonal alone is wrong.  Gradients, with w = SUM A_j tau_j:

        df_i/dA_i   =  tau_i (w - A_i tau_i) / w^2
        df_i/dtau_i =  A_i   (w - A_i tau_i) / w^2
        df_i/dA_j   = -A_i tau_i tau_j / w^2        (j != i)
        df_i/dtau_j = -A_i tau_i A_j   / w^2        (j != i)

    For two components f2 = 1 - f1, so the two uncertainties come out equal.
    That identity is left to fall out of the arithmetic rather than being
    imposed, so an error in the gradients shows up instead of being hidden
    by a shortcut.

    *free* is the covariance's parameter order, with None in any slot that no
    parameter of this dataset owns; *pcov* is indexed by it.  Returns None
    when the fractions are not meaningful (fewer than two components,
    non-positive values), and a list of NaNs when the fit returned no usable
    covariance -- the same contract as mean_lifetime_errors.
    """
    import numpy as np

    comps = [(float(values[a]), float(values[t]), a, t)
             for a, t in pairs if a in values and t in values]
    if len(comps) < 2:
        return None
    if any(a <= 0 or t <= 0 for a, t, _an, _tn in comps):
        return None
    w = float(sum(a * t for a, t, _an, _tn in comps))
    if w <= 0:
        return None

    if pcov is None:
        return [float("nan")] * len(comps)
    C = np.asarray(pcov, dtype=float)
    free = list(free)
    if (C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] != len(free)
            or not np.all(np.isfinite(C))):
        return [float("nan")] * len(comps)

    index = {name: i for i, name in enumerate(free) if name is not None}
    out = []
    for a_i, t_i, an_i, tn_i in comps:
        num_i = a_i * t_i
        g = np.zeros(len(free), dtype=float)
        for a_j, t_j, an_j, tn_j in comps:
            if an_j is an_i or an_j == an_i:
                d_a = t_i * (w - num_i) / w ** 2
                d_t = a_i * (w - num_i) / w ** 2
            else:
                d_a = -num_i * t_j / w ** 2
                d_t = -num_i * a_j / w ** 2
            if an_j in index:
                g[index[an_j]] += d_a
            if tn_j in index:
                g[index[tn_j]] += d_t
        var = float(g @ C @ g)
        if not np.isfinite(var):
            out.append(float("nan"))
        else:
            out.append(float(np.sqrt(var)) if var > 0 else 0.0)
    return out


# ── Dialog windows ────────────────────────────────────────────────────────────
'''),
]

FIT_EDITS = [
    ("import the fraction helper",
'''from fcs_fitcommon import (
    mean_lifetime_errors,
''',
'''from fcs_fitcommon import (
    mean_lifetime_errors,
    intensity_fraction_errors,
'''),

    ("global extras carry their uncertainties",
'''    for entry in out["datasets"]:
        vals = entry["values"]
        extras: Dict[str, float] = {}
        if "A1" in vals and "A2" in vals and "tau1" in vals and "tau2" in vals:
            a1, a2, t1, t2 = vals["A1"], vals["A2"], vals["tau1"], vals["tau2"]
            denom = a1 + a2
            if denom > 0:
                extras["tau_mean_amp"] = (a1 * t1 + a2 * t2) / denom
            w = a1 * t1 + a2 * t2
            if w > 0:
                extras["f1"] = a1 * t1 / w
                extras["f2"] = a2 * t2 / w
        entry["extras"] = extras
''',
'''    # Derived quantities, with uncertainties propagated through the global
    # covariance.  fit_linked returns one covariance for the whole fit and,
    # per dataset, the free slot each parameter used -- which is the only
    # thing that knows which row and column a (parameter, dataset) pair owns.
    # Reversing that mapping per dataset gives the parameter order the
    # covariance is indexed by, which is what the propagation helpers want.
    _cov = out.get("cov")
    _n_free = (len(out.get("free_labels") or ())
               if _cov is None else len(_cov))
    for entry in out["datasets"]:
        vals = entry["values"]
        extras: Dict[str, float] = {}
        if "A1" in vals and "A2" in vals and "tau1" in vals and "tau2" in vals:
            a1, a2, t1, t2 = vals["A1"], vals["A2"], vals["tau1"], vals["tau2"]
            denom = a1 + a2
            w = a1 * t1 + a2 * t2
            if denom > 0:
                extras["tau_mean_amp"] = (a1 * t1 + a2 * t2) / denom
            if w > 0:
                extras["tau_mean_int"] = (a1 * t1 * t1 + a2 * t2 * t2) / w
                extras["f1"] = a1 * t1 / w
                extras["f2"] = a2 * t2 / w

            # A held parameter owns no slot, hence None: it contributes no
            # uncertainty, which is exactly how the helpers treat a gap.
            _slots = entry.get("free_slots", {}) or {}
            _order = [None] * _n_free
            for _p, _s in _slots.items():
                if _s is not None and 0 <= _s < _n_free:
                    _order[_s] = _p
            _pairs = [("A1", "tau1"), ("A2", "tau2")]
            _tau_errs = mean_lifetime_errors(vals, _pairs, _order, _cov)
            if _tau_errs:
                if "tau_mean_amp" in extras:
                    extras["tau_mean_amp_err"] = _tau_errs["tau_mean_amp_err"]
                if "tau_mean_int" in extras:
                    extras["tau_mean_int_err"] = _tau_errs["tau_mean_int_err"]
            _f_errs = intensity_fraction_errors(vals, _pairs, _order, _cov)
            if _f_errs and "f1" in extras:
                extras["f1_err"] = _f_errs[0]
                extras["f2_err"] = _f_errs[1]
        entry["extras"] = extras
'''),

    ("global report pairs each value with its error",
'''        for key, val in d["extras"].items():
            L.append(f"    {key:<20} = {val:.6g}")
''',
'''        for key, val in d["extras"].items():
            if key.endswith("_err"):
                continue            # printed beside the value it belongs to
            _e = d["extras"].get(f"{key}_err", float("nan"))
            _pm = f"  ± {_e:.6g}" if np.isfinite(_e) else ""
            L.append(f"    {key:<20} = {val:.6g}{_pm}")
'''),

    ("global params table gains the error columns",
'''    extra_keys = ["tau_mean_amp", "f1", "f2"]
''',
'''    extra_keys = ["tau_mean_amp", "tau_mean_amp_err",
                  "tau_mean_int", "tau_mean_int_err",
                  "f1", "f1_err", "f2", "f2_err"]
'''),
]


def _target(argv):
    p = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    if p.is_file():
        p = p.parent
    for name in ("fcs_fitcommon.py", "fcs_lifetime_fit.py"):
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
    jobs = {"fcs_fitcommon.py": COMMON_EDITS,
            "fcs_lifetime_fit.py": FIT_EDITS}
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
