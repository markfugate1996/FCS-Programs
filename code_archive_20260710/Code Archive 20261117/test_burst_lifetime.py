"""
Test that a lifetime decay exported by the burst-excision tool loads, and
that all of its decays are offered for fitting.

Builds fixtures in the burst export's own column schema, then exercises
the real loader, the LoadedDecay adapter, the batch picker's discover/load
pair, and an actual fit + export.  The fit screen needs a display, so its
new branch is checked statically.

Usage: python3 test_burst_lifetime.py [path/to/FCS-Programs]
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


def decay(n, tau_ns, amp, bin_ns, seed):
    t = np.arange(n) * bin_ns
    rng = np.random.default_rng(seed)
    return t, rng.poisson(amp * np.exp(-t / tau_ns) + 2.0).astype(float)


def write_csv(path, header, columns, n):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(",".join(columns) + "\n")
        for i in range(n):
            fh.write(",".join(
                "" if np.isnan(columns[c][i]) else f"{columns[c][i]:.10g}"
                for c in columns) + "\n")


def write_table(path, header, cols, n):
    names = list(cols)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        for line in header:
            fh.write(line + "\n")
        fh.write(",".join(names) + "\n")
        for i in range(n):
            row = []
            for name in names:
                v = cols[name][i]
                row.append("" if (isinstance(v, float) and np.isnan(v))
                           else f"{v:.10g}")
            fh.write(",".join(row) + "\n")


def main(root):
    sys.path.insert(0, root)
    import fcs_lifetime_fit as lf
    import fcs_models

    N, BIN = 512, 0.048
    t, ch1_all = decay(N, 3.2, 900, BIN, 1)
    _, ch1_burst = decay(N, 1.1, 120, BIN, 2)
    ch1_clean = ch1_all - ch1_burst
    _, ch2_all = decay(N, 3.0, 600, BIN, 3)
    _, ch2_burst = decay(N, 1.0, 80, BIN, 4)
    ch2_clean = ch2_all - ch2_burst
    blank = np.full(N, np.nan)

    HEAD = ["# FCS analysis export — burst lifetime",
            "# analysis    : burst lifetime",
            "# source file : sample_A.fcs",
            "# n_bins : 512",
            "# laser_period_ns : 24.576",
            "# channels_recorded : Ch1+Ch2",
            "# M : 20", "# F : 3"]

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="fcs_burst_lt_"))
    try:
        burst = tmp / "sample_A_burst_lifetime_512bins.csv"
        write_table(burst, HEAD, {
            "time_ns": t, "ch1_all": ch1_all, "ch1_burst": ch1_burst,
            "ch1_clean": ch1_clean, "ch2_all": ch2_all,
            "ch2_burst": ch2_burst, "ch2_clean": ch2_clean}, N)

        plain = tmp / "sample_B_lifetime_512bins.csv"
        write_table(plain, ["# FCS analysis export — lifetime decay",
                            "# analysis    : lifetime decay",
                            "# source file : sample_B.fcs",
                            "# laser_period_ns : 24.576"],
                    {"time_ns": t, "ch1_counts": ch1_all,
                     "ch2_counts": ch2_all}, N)

        one_ch = tmp / "sample_C_burst_lifetime_512bins.csv"
        write_table(one_ch, HEAD, {
            "time_ns": t, "ch1_all": ch1_all, "ch1_burst": ch1_burst,
            "ch1_clean": ch1_clean, "ch2_all": blank, "ch2_burst": blank,
            "ch2_clean": blank}, N)

        # ── the loader ────────────────────────────────────────────────────
        t_ns, series, irf, cols, meta = lf.load_lifetime_csv(burst)
        check("a burst decay export loads instead of being rejected",
              len(series) == 6, sorted(series))
        check("its six decays are named by channel and component",
              set(series) == {"Ch1 all photons", "Ch1 bursts",
                              "Ch1 burst-free", "Ch2 all photons",
                              "Ch2 bursts", "Ch2 burst-free"}, sorted(series))

        # ── the adapter ───────────────────────────────────────────────────
        d = lf.load_decay_object(burst)
        check("it reports the three components",
              d.components == ("all photons", "bursts", "burst-free"),
              d.components)
        check("it opens on the burst-free decay",
              d.component == "burst-free", d.component)
        check("both channels are available", d.channels == (1, 2), d.channels)
        check("the fit screen is offered six decays",
              [lbl for _c, _p, lbl in d.decay_options()]
              == ["Ch1 all photons", "Ch1 bursts", "Ch1 burst-free",
                  "Ch2 all photons", "Ch2 bursts", "Ch2 burst-free"],
              d.decay_options())

        check("the selected component is what counts() returns",
              np.allclose(d.counts(1), ch1_clean))
        d.select_component("bursts")
        check("selecting another component swaps the curves",
              np.allclose(d.counts(1), ch1_burst)
              and np.allclose(d.counts(2), ch2_burst))
        d.select_component("all photons")
        check("and 'all photons' is the undivided decay",
              np.allclose(d.counts(1), ch1_all))
        try:
            d.select_component("nonsense")
            check("an unknown component raises", False, "no error")
        except ValueError as exc:
            check("an unknown component raises rather than falling back",
                  "nonsense" in str(exc), str(exc))

        check("the summary says which decay is showing",
              "all photons" in d.channel_summary, d.channel_summary)

        # the FCSData-shaped calls, on the selected component
        d.select_component("burst-free")
        tt, cc, _irf = d.decay_curve(channel=1)
        check("decay_curve returns the selected component",
              np.allclose(cc, ch1_clean) and np.allclose(tt, t))
        tb, cb = d.lifetime_histogram(channel=2, n_bins=256)
        check("rebinning still sums adjacent bins exactly",
              len(cb) == 256
              and abs(cb.sum() - ch2_clean.sum()) < 1e-6, cb.sum())

        # ── a channel that was never recorded ─────────────────────────────
        d1 = lf.load_decay_object(one_ch)
        check("a single-channel burst export offers three decays",
              [lbl for _c, _p, lbl in d1.decay_options()]
              == ["Ch1 all photons", "Ch1 bursts", "Ch1 burst-free"],
              d1.decay_options())
        check("and reports only the channel it recorded",
              d1.channels == (1,), d1.channels)

        # ── ordinary and .ifx exports are unchanged ───────────────────────
        p = lf.load_decay_object(plain)
        check("an ordinary two-column decay still loads as two channels",
              p.channels == (1, 2) and p.components == (), p.components)
        check("its options are the plain channel pair",
              [lbl for _c, _p2, lbl in p.decay_options()] == ["Ch1", "Ch2"])
        check("its summary is unchanged",
              p.channel_summary == "Ch1 + Ch2", p.channel_summary)
        check("counts still come from ch1_counts",
              np.allclose(p.counts(1), ch1_all))

        # an .ifx-style single-curve export: the path that has no channel
        ifx = tmp / "sample_D_lifetime_decay.csv"
        write_table(ifx, ["# FCS analysis export \u2014 lifetime decay",
                          "# analysis    : lifetime decay",
                          "# source file : sample_D.ifx"],
                    {"time_ns": t, "intensity": ch1_all}, N)
        single = lf.load_decay_object(ifx)
        check("a single-curve .ifx export still loads as one decay",
              single._single and single.channels == (1,)
              and single.components == (),
              (single._single, single.channels))
        check("and still reports itself as a decay, not a detector",
              single.channel_summary == "decay", single.channel_summary)

        # ── the batch picker ──────────────────────────────────────────────
        src = lf.lifetime_dataset_source()
        entries = src.discover(tmp)
        labels = sorted(e.series for e in entries if "sample_A" in
                        pathlib.Path(e.path).name)
        check("a global fit can pick any of the six decays",
              labels == ["Ch1 all photons", "Ch1 burst-free", "Ch1 bursts",
                         "Ch2 all photons", "Ch2 burst-free", "Ch2 bursts"],
              labels)
        entry = next(e for e in entries
                     if e.series == "Ch2 bursts" and "sample_A" in
                     pathlib.Path(e.path).name)
        ds = src.load(entry)
        check("loading one picks the right channel AND component",
              ds["channel"] == 2 and np.allclose(ds["counts"], ch2_burst),
              (ds["channel"], ds["counts"][:3], ch2_burst[:3]))
        plain_entry = next(e for e in entries if "sample_B" in
                           pathlib.Path(e.path).name)
        check("an ordinary decay still loads through the same path",
              src.load(plain_entry)["channel"] in (1, 2))

        # ── an actual fit, and what it records ────────────────────────────
        model = fcs_models.get_lifetime_model(
            fcs_models.list_lifetime_models()[0].key)
        d.select_component("burst-free")
        t_fit, c_fit = d.lifetime_histogram(channel=1)
        lo, hi = float(t_fit[c_fit.argmax()]), float(t_fit[-1])
        guesses = lf.auto_guess_lifetime(model, *lf.prepare_decay(
            t_fit, c_fit, lo, hi)[2:0:-1])
        result = lf.fit_lifetime(
            model, t_fit, c_fit, lo, hi, guesses,
            {p.name: p.lower for p in model.params},
            {p.name: p.upper for p in model.params},
            {p.name: p.fixed for p in model.params},
            channel=1, n_bins=d.n_bins)
        check("a burst-free decay actually fits",
              np.isfinite(result["r2"]) and result["r2"] > 0.5,
              result["r2"])
        result["decay_component"] = d.component
        result["origin"] = d.source_name
        rep, curve = lf.export_lifetime_fit(result, burst)[:2]
        text = pathlib.Path(rep).read_text(encoding="utf-8")
        check("the report says which decay was fitted",
              "decay      : burst-free" in text,
              [ln for ln in text.splitlines() if "decay" in ln][:3])
        head = pathlib.Path(curve).read_text(encoding="utf-8")[:900]
        check("so does the curve header",
              "# decay_component : burst-free" in head,
              [ln for ln in head.splitlines() if ln.startswith("#")])

        # ── the fit screen's new branch ───────────────────────────────────
        src_text = (pathlib.Path(root) / "fcs_lifetime_fit.py").read_text(
            encoding="utf-8")
        check("the setup screen has a burst branch driven by decay_options",
              'elif getattr(fcs_data, "components", ())' in src_text
              and "fcs_data.decay_options()" in src_text)
        check("picking one selects the component and the channel",
              "fcs_data.select_component(comp)" in src_text
              and "ch_var.set(int(ch_text))" in src_text)
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
