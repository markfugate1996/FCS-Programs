"""
Test the uncertainty on the mean lifetime of a multi-exponential fit.

The propagation is checked against Monte Carlo sampling of the same
covariance -- an independent route to the same number -- and then end to
end through a real two-exponential tail fit and its written report.

Usage: python3 test_tau_mean_error.py [path/to/FCS-Programs]
"""
import os
import pathlib
import shutil
import sys
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ok = 0


def check(label, cond, detail=""):
    global ok
    assert cond, f"{label}{(': ' + str(detail)) if detail else ''}"
    ok += 1
    print(f"PASS  {label}")


def main(root):
    sys.path.insert(0, root)
    import fcs_fitcommon as fc
    import fcs_lifetime_fit as lf
    import fcs_models

    PAIRS = [("A1", "tau1"), ("A2", "tau2")]
    FREE = ["A1", "tau1", "A2", "tau2"]
    VALS = {"A1": 600.0, "tau1": 0.80, "A2": 400.0, "tau2": 3.50}

    def tau_amp(v):
        return ((v["A1"] * v["tau1"] + v["A2"] * v["tau2"])
                / (v["A1"] + v["A2"]))

    def tau_int(v):
        num = v["A1"] * v["tau1"] ** 2 + v["A2"] * v["tau2"] ** 2
        return num / (v["A1"] * v["tau1"] + v["A2"] * v["tau2"])

    # ── against Monte Carlo, for a correlated covariance ──────────────────
    rng = np.random.default_rng(7)
    sd = np.array([35.0, 0.05, 30.0, 0.22])
    corr = np.array([[1.0, -0.82, -0.55, 0.30],
                     [-0.82, 1.0, 0.40, -0.25],
                     [-0.55, 0.40, 1.0, -0.78],
                     [0.30, -0.25, -0.78, 1.0]])
    C = corr * np.outer(sd, sd)

    errs = fc.mean_lifetime_errors(VALS, PAIRS, FREE, C)
    draws = rng.multivariate_normal([VALS[n] for n in FREE], C, size=400_000)
    keep = (draws[:, 0] > 0) & (draws[:, 2] > 0)
    mc = [dict(zip(FREE, row)) for row in draws[keep][:200_000]]
    mc_amp = np.std([tau_amp(v) for v in mc])
    mc_int = np.std([tau_int(v) for v in mc])

    check("amplitude-weighted error matches Monte Carlo within 3%",
          abs(errs["tau_mean_amp_err"] - mc_amp) / mc_amp < 0.03,
          (errs["tau_mean_amp_err"], mc_amp))
    check("intensity-weighted error matches Monte Carlo within 3%",
          abs(errs["tau_mean_int_err"] - mc_int) / mc_int < 0.03,
          (errs["tau_mean_int_err"], mc_int))

    # ── correlations matter: the diagonal-only answer is different ────────
    diag_only = fc.mean_lifetime_errors(VALS, PAIRS, FREE, np.diag(np.diag(C)))
    check("ignoring the correlations would give a different (here larger) "
          "error", diag_only["tau_mean_amp_err"]
          > errs["tau_mean_amp_err"] * 1.1,
          (diag_only["tau_mean_amp_err"], errs["tau_mean_amp_err"]))

    # ── fixed parameters ──────────────────────────────────────────────────
    free3 = ["A1", "tau1", "tau2"]          # A2 held fixed
    C3 = C[np.ix_([0, 1, 3], [0, 1, 3])]
    errs3 = fc.mean_lifetime_errors(VALS, PAIRS, free3, C3)
    check("a fixed amplitude contributes no uncertainty",
          np.isfinite(errs3["tau_mean_amp_err"])
          and errs3["tau_mean_amp_err"] != errs["tau_mean_amp_err"],
          errs3)

    all_fixed = fc.mean_lifetime_errors(VALS, PAIRS, [], np.zeros((0, 0)))
    check("with everything fixed the mean has no uncertainty",
          all_fixed["tau_mean_amp_err"] == 0.0, all_fixed)

    # ── refusals ──────────────────────────────────────────────────────────
    check("a single component has no mean to quote",
          fc.mean_lifetime_errors({"A1": 1.0, "tau1": 2.0}, [("A1", "tau1")],
                                  FREE, C) is None)
    check("a non-positive amplitude is refused, not propagated",
          fc.mean_lifetime_errors({**VALS, "A2": -5.0}, PAIRS, FREE, C)
          is None)
    check("no covariance gives NaN, not a fabricated number",
          np.isnan(fc.mean_lifetime_errors(VALS, PAIRS, FREE,
                                           None)["tau_mean_amp_err"]))
    check("a covariance of the wrong shape is rejected",
          np.isnan(fc.mean_lifetime_errors(VALS, PAIRS, FREE,
                                           C[:3, :3])["tau_mean_amp_err"]))
    check("a non-finite covariance is rejected",
          np.isnan(fc.mean_lifetime_errors(
              VALS, PAIRS, FREE,
              np.where(np.eye(4) > 0, np.inf, C))["tau_mean_amp_err"]))

    # ── end to end through a real two-exponential fit ─────────────────────
    model = next(m for m in fcs_models.list_lifetime_models()
                 if "A2" in [p.name for p in m.params]
                 and "tau2" in [p.name for p in m.params])
    t = np.arange(1024) * 0.024
    truth = {"A1": 800.0, "tau1": 0.7, "A2": 350.0, "tau2": 3.4}
    clean = (truth["A1"] * np.exp(-t / truth["tau1"])
             + truth["A2"] * np.exp(-t / truth["tau2"]))
    names = [p.name for p in model.params]
    extra = {n: 0.0 for n in names if n not in truth}
    counts = np.random.default_rng(11).poisson(clean + 3.0).astype(float)

    guesses = {**{n: v for n, v in truth.items()}, **extra}
    for n in names:
        guesses.setdefault(n, 1.0)
    result = lf.fit_lifetime(
        model, t, counts, float(t[0]), float(t[-1]), guesses,
        {p.name: p.lower for p in model.params},
        {p.name: p.upper for p in model.params},
        {p.name: p.fixed for p in model.params},
        weighted=True, channel=1, n_bins=len(t))

    check("the fit result carries an uncertainty on <tau>",
          np.isfinite(result["tau_mean_err"]) and result["tau_mean_err"] > 0,
          result.get("tau_mean_err"))
    check("the fractions dict carries both errors",
          np.isfinite(result["fractions"]["tau_mean_amp_err"])
          and np.isfinite(result["fractions"]["tau_mean_int_err"]),
          {k: v for k, v in result["fractions"].items() if "err" in k})
    check("the error is small next to the mean it qualifies",
          result["tau_mean_err"] < 0.5 * result["tau_mean"],
          (result["tau_mean"], result["tau_mean_err"]))
    check("<tau> itself still agrees with the truth",
          abs(result["tau_mean"] - tau_amp(truth)) < 0.25,
          (result["tau_mean"], tau_amp(truth)))

    # the propagated error must be consistent with resampling THIS fit
    pcov, free = result["pcov"], result["free"]
    idx = {n: i for i, n in enumerate(free)}
    sample = np.random.default_rng(3).multivariate_normal(
        [result["values"][n] for n in free], pcov, size=60_000)
    vals_mc = []
    for row in sample:
        v = dict(result["values"])
        v.update({n: row[idx[n]] for n in free})
        if v["A1"] > 0 and v["A2"] > 0:
            vals_mc.append(tau_amp(v))
    check("and it agrees with resampling the fit's own covariance",
          abs(np.std(vals_mc) - result["tau_mean_err"])
          / result["tau_mean_err"] < 0.05,
          (np.std(vals_mc), result["tau_mean_err"]))

    # ── what gets written ─────────────────────────────────────────────────
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fcs_taumean_"))
    try:
        src = tmp / "synthetic_lifetime.csv"
        src.write_text("time_ns,ch1_counts\n", encoding="utf-8")
        result["origin"] = "synthetic"
        rep, _curve, params = lf.export_lifetime_fit(result, src)[:3]
        text = pathlib.Path(rep).read_text(encoding="utf-8")
        check("the report prints the mean with its uncertainty",
              "±" in [ln for ln in text.splitlines()
                      if "amplitude-weighted mean" in ln][0],
              [ln for ln in text.splitlines() if "mean lifetime" in ln])
        head = pathlib.Path(params).read_text(encoding="utf-8").splitlines()
        hdr = next(ln for ln in head if "tau_mean_amp_ns" in ln)
        check("the params table gains error columns",
              "tau_mean_amp_err_ns" in hdr and "tau_mean_int_err_ns" in hdr,
              hdr)
        check("and the note says how the error was obtained",
              any("full parameter covariance" in ln for ln in head),
              [ln for ln in head if ln.startswith("#")][:6])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── the reconvolution fitter, for real ────────────────────────────────
    import fcs_lifetime_recon as rc
    import fcs_lifetime_models as lm

    rmodel = lm.get_model("lifetime_exp2")
    tr = np.arange(1024) * 0.024
    irf = np.exp(-0.5 * ((tr - 1.2) / 0.12) ** 2)
    irf /= irf.sum()
    pure = 900.0 * np.exp(-tr / 0.65) + 300.0 * np.exp(-tr / 3.1)
    conv = np.convolve(pure, irf)[:len(tr)]
    rcounts = np.random.default_rng(5).poisson(conv + 4.0).astype(float)

    rres = rc.fit_lifetime(
        rmodel, tr, rcounts, irf,
        {"a1": 900.0, "tau1": 0.65, "a2": 300.0, "tau2": 3.1,
         "bg": 4.0, "shift": 0.0},
        {p.name: p.lower for p in rmodel.params},
        {p.name: p.upper for p in rmodel.params},
        {"a1": False, "tau1": False, "a2": False, "tau2": False,
         "bg": False, "shift": True},
        weighted=True)
    rder = rres["derived"]
    check("a reconvolution fit reports an uncertainty on <tau> too",
          np.isfinite(rder.get("tau_mean_amp_err", float("nan")))
          and rder["tau_mean_amp_err"] > 0,
          rder.get("tau_mean_amp_err"))
    check("and on the intensity-weighted mean",
          np.isfinite(rder.get("tau_mean_int_err", float("nan"))),
          rder.get("tau_mean_int_err"))

    rfree, rpcov = rres["free"], rres["pcov"]
    ridx = {n: i for i, n in enumerate(rfree)}
    rs = np.random.default_rng(9).multivariate_normal(
        [rres["values"][n] for n in rfree], rpcov, size=40_000)
    mc_r = []
    for row in rs:
        v = dict(rres["values"])
        v.update({n: row[ridx[n]] for n in rfree})
        if v["a1"] > 0 and v["a2"] > 0:
            mc_r.append((v["a1"] * v["tau1"] + v["a2"] * v["tau2"])
                        / (v["a1"] + v["a2"]))
    check("the reconvolution error also matches resampling its covariance",
          abs(np.std(mc_r) - rder["tau_mean_amp_err"])
          / rder["tau_mean_amp_err"] < 0.08,
          (np.std(mc_r), rder["tau_mean_amp_err"]))

    print(f"\nAll {ok} checks passed.")
    return 0


if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    if not (p / "fcs_fitcommon.py").is_file():
        print(f"ERROR: fcs_fitcommon.py not found in {p}.")
        raise SystemExit(2)
    sys.exit(main(str(p.resolve())))
