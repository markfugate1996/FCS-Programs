"""
Test uncertainties on the global lifetime fit's derived quantities.

Runs a real two-dataset global fit of a two-exponential decay, then checks
the propagated errors against Monte Carlo sampling of the fit's own global
covariance, and checks what gets written.

Usage: python3 test_global_derived_errors.py [path/to/FCS-Programs]
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

    # ── the fraction helper, against Monte Carlo ──────────────────────────
    PAIRS = [("A1", "tau1"), ("A2", "tau2")]
    FREE = ["A1", "tau1", "A2", "tau2"]
    VALS = {"A1": 700.0, "tau1": 0.75, "A2": 300.0, "tau2": 3.2}
    sd = np.array([40.0, 0.06, 28.0, 0.25])
    corr = np.array([[1.0, -0.8, -0.5, 0.25],
                     [-0.8, 1.0, 0.35, -0.2],
                     [-0.5, 0.35, 1.0, -0.75],
                     [0.25, -0.2, -0.75, 1.0]])
    C = corr * np.outer(sd, sd)

    f_errs = fc.intensity_fraction_errors(VALS, PAIRS, FREE, C)
    draws = np.random.default_rng(4).multivariate_normal(
        [VALS[n] for n in FREE], C, size=300_000)
    keep = (draws[:, 0] > 0) & (draws[:, 2] > 0)
    f1s = []
    for a1, t1, a2, t2 in draws[keep]:
        w = a1 * t1 + a2 * t2
        if w > 0:
            f1s.append(a1 * t1 / w)
    mc = float(np.std(f1s))
    check("f1 error matches Monte Carlo within 3%",
          abs(f_errs[0] - mc) / mc < 0.03, (f_errs[0], mc))
    check("f2 error equals f1's, as f2 = 1 - f1 demands",
          abs(f_errs[0] - f_errs[1]) < 1e-12, f_errs)
    check("ignoring correlations would give a different answer",
          abs(fc.intensity_fraction_errors(
              VALS, PAIRS, FREE, np.diag(np.diag(C)))[0] - f_errs[0])
          > 0.05 * f_errs[0])
    check("a held parameter simply contributes nothing",
          np.isfinite(fc.intensity_fraction_errors(
              VALS, PAIRS, ["A1", "tau1", None, "tau2"], C)[0]))
    check("no covariance gives NaN, not a number",
          np.isnan(fc.intensity_fraction_errors(VALS, PAIRS, FREE, None)[0]))
    check("one component has no fractions to quote",
          fc.intensity_fraction_errors({"A1": 1.0, "tau1": 2.0},
                                        [("A1", "tau1")], FREE, C) is None)

    # ── a real two-dataset global fit ─────────────────────────────────────
    model = next(m for m in fcs_models.list_lifetime_models()
                 if {"A1", "tau1", "A2", "tau2"} <= {p.name
                                                     for p in m.params})
    names = [p.name for p in model.params]
    t = np.arange(900) * 0.024
    truth = {"A1": 800.0, "tau1": 0.70, "A2": 320.0, "tau2": 3.30}

    def synth(scale, seed):
        clean = (scale * truth["A1"] * np.exp(-t / truth["tau1"])
                 + scale * truth["A2"] * np.exp(-t / truth["tau2"]))
        return np.random.default_rng(seed).poisson(clean + 3.0).astype(float)

    datasets = [{"name": "A", "path": "A.csv", "series": "Ch1",
                 "t_ns": t, "counts": synth(1.0, 21), "n_bins": len(t)},
                {"name": "B", "path": "B.csv", "series": "Ch1",
                 "t_ns": t, "counts": synth(0.6, 22), "n_bins": len(t)}]
    windows = [(float(t[0]), float(t[-1]))] * 2
    guesses = {n: 1.0 for n in names}
    guesses.update(truth)
    res = lf.fit_lifetime_global(
        model, datasets, windows,
        {"A1": False, "tau1": True, "A2": False, "tau2": True,
         **{n: False for n in names if n not in truth}},
        guesses,
        {p.name: p.lower for p in model.params},
        {p.name: p.upper for p in model.params},
        {n: False for n in names},
        weighted=True)

    ex = res["datasets"][0]["extras"]
    check("the global fit now reports an error on the mean lifetime",
          np.isfinite(ex.get("tau_mean_amp_err", float("nan")))
          and ex["tau_mean_amp_err"] > 0, ex.get("tau_mean_amp_err"))
    check("and the intensity-weighted mean, which it did not report at all",
          np.isfinite(ex.get("tau_mean_int", float("nan")))
          and np.isfinite(ex.get("tau_mean_int_err", float("nan"))),
          {k: v for k, v in ex.items() if "int" in k})
    check("and errors on f1 and f2",
          np.isfinite(ex.get("f1_err", float("nan")))
          and abs(ex["f1_err"] - ex["f2_err"]) < 1e-9,
          (ex.get("f1_err"), ex.get("f2_err")))
    check("the fractions still sum to one",
          abs(ex["f1"] + ex["f2"] - 1.0) < 1e-12)
    check("every dataset gets its own errors",
          all(np.isfinite(d["extras"]["f1_err"]) for d in res["datasets"]),
          [d["extras"].get("f1_err") for d in res["datasets"]])

    # against resampling the fit's own global covariance
    cov, labels = res["cov"], res["free_labels"]
    d0 = res["datasets"][0]
    slots = d0["free_slots"]
    mean = np.zeros(len(cov))
    for p, s in slots.items():
        if s is not None:
            mean[s] = d0["values"][p]
    # fill any slot this dataset does not own from another dataset
    for d in res["datasets"][1:]:
        for p, s in d["free_slots"].items():
            if s is not None and mean[s] == 0.0:
                mean[s] = d["values"][p]
    sample = np.random.default_rng(13).multivariate_normal(cov.shape[0] * [0.0],
                                                           cov, size=40_000)
    f1_mc, tm_mc = [], []
    for row in sample:
        v = dict(d0["values"])
        for p, s in slots.items():
            if s is not None:
                v[p] = d0["values"][p] + row[s]
        if v["A1"] <= 0 or v["A2"] <= 0:
            continue
        w = v["A1"] * v["tau1"] + v["A2"] * v["tau2"]
        f1_mc.append(v["A1"] * v["tau1"] / w)
        tm_mc.append(w / (v["A1"] + v["A2"]))
    check("the f1 error agrees with resampling the global covariance",
          abs(np.std(f1_mc) - ex["f1_err"]) / ex["f1_err"] < 0.08,
          (np.std(f1_mc), ex["f1_err"]))
    check("so does the mean-lifetime error",
          abs(np.std(tm_mc) - ex["tau_mean_amp_err"])
          / ex["tau_mean_amp_err"] < 0.08,
          (np.std(tm_mc), ex["tau_mean_amp_err"]))

    # ── what gets written ─────────────────────────────────────────────────
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fcs_globalderiv_"))
    try:
        src = tmp / "A_lifetime.csv"
        src.write_text("time_ns,ch1_counts\n", encoding="utf-8")
        paths = lf.export_lifetime_global(res, src)
        report = next(pathlib.Path(p) for p in paths
                      if str(p).endswith("report.txt"))
        text = report.read_text(encoding="utf-8")
        line = next(ln for ln in text.splitlines() if "tau_mean_amp" in ln)
        check("the report prints the mean with its error on one line",
              "±" in line, line)
        check("and no bare *_err line of its own",
              not any(ln.strip().startswith("tau_mean_amp_err")
                      for ln in text.splitlines()),
              [ln for ln in text.splitlines() if "_err" in ln][:3])
        fline = next(ln for ln in text.splitlines()
                     if ln.strip().startswith("f1"))
        check("f1 is printed with its error too", "±" in fline, fline)

        params = next(pathlib.Path(p) for p in paths
                      if str(p).endswith("params.csv"))
        header = next(ln for ln in params.read_text(
            encoding="utf-8").splitlines() if "tau_mean_amp" in ln)
        for col in ("tau_mean_amp_err", "tau_mean_int", "tau_mean_int_err",
                    "f1_err", "f2_err"):
            check(f"the params table has a {col} column", col in header)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\nAll {ok} checks passed.")
    return 0


if __name__ == "__main__":
    p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    if not (p / "fcs_lifetime_fit.py").is_file():
        print(f"ERROR: fcs_lifetime_fit.py not found in {p}.")
        raise SystemExit(2)
    sys.exit(main(str(p.resolve())))
