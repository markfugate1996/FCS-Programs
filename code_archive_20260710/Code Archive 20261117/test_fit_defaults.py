"""
Test the Fit defaults preferences tab: the override layer in fcs_models,
the storage round-trip in fcs_prefs, and the tab's own collect/restore.

Imports the real modules.  The tab is built against a stub tkinter, so no
display is needed.

Usage: python3 test_fit_defaults.py [path/to/FCS-Programs]
"""
import json
import os
import shutil
import sys
import tempfile
import types

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ok = 0


def check(label, cond, detail=""):
    global ok
    assert cond, f"{label}{(': ' + str(detail)) if detail else ''}"
    ok += 1
    print(f"PASS  {label}")


# ── stub tkinter, enough for the tab ──────────────────────────────────────
class _W:
    def __init__(self, master=None, *a, **kw):
        self.master, self.kw, self.args = master, kw, a
        self.children = []
        if isinstance(master, _W):
            master.children.append(self)
    def pack(self, **kw): pass
    def pack_forget(self): pass
    def config(self, **kw): self.kw.update(kw)
    configure = config
    def bind(self, *a): pass
    def __getitem__(self, k): return _W(self)


class _Var:
    def __init__(self, master=None, value=""):
        self._v, self._cbs = value, []
    def get(self): return self._v
    def set(self, v):
        self._v = v
        for cb in self._cbs:
            cb()
    def trace_add(self, _m, cb): self._cbs.append(lambda *a: cb())


class _Bool(_Var):
    def get(self): return bool(self._v)


class _Notebook:
    def __init__(self): self.tabs = []
    def add(self, frame, text=""): self.tabs.append(text)


def tk_stub():
    m = types.ModuleType("tkinter")
    for n in ("Frame", "Label", "Entry", "Button", "Checkbutton",
              "OptionMenu", "LabelFrame", "Spinbox", "Scrollbar"):
        setattr(m, n, type(n, (_W,), {}))
    m.StringVar, m.BooleanVar = _Var, _Bool
    return m


def main(root):
    sys.path.insert(0, root)
    sys.modules.setdefault("tkinter", tk_stub())
    import fcs_models
    import fcs_prefs
    import fcs_fit

    MODELS = fcs_models.MODELS
    builtin = fcs_models.BUILTIN_PARAM_DEFAULTS

    # ── the snapshot ──────────────────────────────────────────────────────
    check("every correlation model is in the built-in snapshot",
          set(builtin) == set(MODELS), set(builtin) ^ set(MODELS))
    n_pairs = sum(len(v) for v in builtin.values())
    check(f"the snapshot covers all {n_pairs} (model, parameter) pairs",
          n_pairs == sum(len(m.params) for m in MODELS.values()))
    m1 = MODELS["diffusion_3d_1comp"]
    k = next(p for p in m1.params if p.name == "kappa")
    b = builtin["diffusion_3d_1comp"]["kappa"]
    check("the snapshot matches the shipped Param",
          (b["guess"], b["lower"], b["upper"], b["fixed"], b["link"])
          == (k.default, k.lower, k.upper, k.fixed, k.link_default))
    check("PCH and lifetime registries are untouched by this feature",
          not (set(builtin) & set(getattr(fcs_models, "PCH_MODELS", {}))))

    # ── applying, re-applying, removing ───────────────────────────────────
    n = fcs_models.apply_param_overrides({
        "diffusion_3d_1comp": {
            "kappa": {"fixed": True, "link": True, "lower": 3.0,
                      "upper": 8.0, "guess": 5.5},
            "offset": {"lower": "-inf", "upper": "inf"},
        }})
    check("applying reports how many fields it set", n == 7, n)
    check("bounds, flags and guess all reach the Param",
          (k.lower, k.upper, k.fixed, k.link_default, k.default)
          == (3.0, 8.0, True, True, 5.5),
          (k.lower, k.upper, k.fixed, k.link_default, k.default))

    off = next(p for p in m1.params if p.name == "offset")
    check("'inf' and '-inf' survive the JSON round trip as bounds",
          off.lower == -np.inf and off.upper == np.inf,
          (off.lower, off.upper))

    fcs_models.apply_param_overrides({
        "diffusion_3d_1comp": {"kappa": {"fixed": True}}})
    check("re-applying a smaller table restores what it no longer names",
          (k.lower, k.upper, k.default, k.fixed)
          == (b["lower"], b["upper"], b["guess"], True),
          (k.lower, k.upper, k.default, k.fixed))

    fcs_models.apply_param_overrides({})
    check("an empty table puts every parameter back to built-in",
          (k.lower, k.upper, k.fixed, k.link_default, k.default)
          == (b["lower"], b["upper"], b["fixed"], b["link"], b["guess"]))

    # ── junk in the stored file is survivable ─────────────────────────────
    n_bad = fcs_models.apply_param_overrides({
        "no_such_model": {"x": {"lower": 1}},
        "diffusion_3d_1comp": {
            "no_such_param": {"lower": 1},
            "kappa": {"no_such_field": 1, "lower": "not a number",
                      "upper": 9.0},
        }})
    check("unknown models, parameters and fields are skipped, not raised",
          n_bad == 1 and k.upper == 9.0, (n_bad, k.upper))
    fcs_models.apply_param_overrides({})

    # ── what the fit dialog actually reads ────────────────────────────────
    src = open(os.path.join(root, "fcs_fitdialogs.py"), encoding="utf-8").read()
    for attr in ("p.link_default", "p.lower", "p.upper", "p.fixed",
                 "p.default"):
        check(f"the fit dialog's ParamTable reads {attr}", attr in src)

    # ── auto_guess keeps its three, and only its three ────────────────────
    tau = np.logspace(-6, -1, 120)
    G = 0.4 / ((1 + tau / 3e-4) * np.sqrt(1 + tau / (25 * 3e-4)))
    fcs_models.apply_param_overrides({
        "diffusion_3d_1comp": {
            "G0": {"guess": 9.0}, "tau_D": {"guess": 0.05},
            "offset": {"guess": 0.7}, "kappa": {"guess": 7.25},
        }})
    guess = fcs_fit.auto_guess(m1, tau, G)
    check("a stored guess for kappa is what the fit starts from",
          guess["kappa"] == 7.25, guess["kappa"])
    check("G0, tau_D and offset stay data-derived, as you asked",
          guess["G0"] != 9.0 and guess["tau_D"] != 0.05
          and guess["offset"] != 0.7,
          {kk: guess[kk] for kk in ("G0", "tau_D", "offset")})
    check("and the tab's auto list names exactly those three",
          set(fcs_models.AUTO_GUESS_PARAMS) == {"G0", "tau_D", "offset"},
          fcs_models.AUTO_GUESS_PARAMS)
    fcs_models.apply_param_overrides({})

    # ── storage round trip through the real prefs file ────────────────────
    tmp = tempfile.mkdtemp(prefix="fcs_prefs_test_")
    old_appdata = os.environ.get("APPDATA")
    os.environ["APPDATA"] = tmp
    try:
        table = {"diffusion_3d_1comp":
                 {"kappa": {"fixed": True, "upper": "inf"}}}
        saved = fcs_prefs.save({"figure_dpi": 200,
                                fcs_prefs.FIT_DEFAULTS_KEY: table})
        check("the table is written to the prefs file", saved is not None)
        raw = open(saved, encoding="utf-8").read()
        json.loads(raw)
        check("the file is still strict JSON (no Infinity literal)",
              "Infinity" not in raw and '"inf"' in raw)

        back = fcs_prefs.load()
        check("it reads back unchanged",
              back[fcs_prefs.FIT_DEFAULTS_KEY] == table,
              back.get(fcs_prefs.FIT_DEFAULTS_KEY))

        applied = fcs_prefs.apply()
        check("apply() pushes both the registry and the table",
              applied > len(fcs_prefs.PREFS), applied)
        check("the stored override is live on the model",
              k.fixed is True and k.upper == np.inf, (k.fixed, k.upper))

        # ── the tab: collect is sparse, restore is total ──────────────────
        stored = fcs_prefs.load()
        tab = fcs_prefs._build_fit_defaults_tab(_Notebook(), _W(), stored)
        check("the tab builds when fcs_models is imported", tab is not None)

        first = tab.collect()
        check("it opens showing what was stored",
              first == table, first)

        tab2 = fcs_prefs._build_fit_defaults_tab(_Notebook(), _W(), {})
        empty = tab2.collect()
        check("with nothing overridden, the table collects empty",
              empty == {}, empty)

        # Validation, driven through stored values the widgets open with.
        bad_bounds = {"diffusion_3d_1comp":
                      {"kappa": {"lower": 10.0, "upper": 1.0}}}
        tab3 = fcs_prefs._build_fit_defaults_tab(_Notebook(), _W(),
                                                 {fcs_prefs.FIT_DEFAULTS_KEY:
                                                  bad_bounds})
        try:
            tab3.collect()
            check("an inverted range is refused", False, "no error")
        except ValueError as exc:
            check("an inverted range is refused, naming the parameter",
                  "kappa" in str(exc) and "minimum" in str(exc), str(exc))

        bad_guess = {"diffusion_3d_1comp":
                     {"kappa": {"guess": 100.0}}}      # built-in max is 50
        tab4 = fcs_prefs._build_fit_defaults_tab(_Notebook(), _W(),
                                                 {fcs_prefs.FIT_DEFAULTS_KEY:
                                                  bad_guess})
        try:
            tab4.collect()
            check("a guess outside its own bounds is refused", False,
                  "no error")
        except ValueError as exc:
            check("a guess outside its own bounds is refused",
                  "kappa" in str(exc) and "outside" in str(exc), str(exc))

        # A stored value equal to the built-in must not be re-stored: that
        # is what keeps the file sparse and lets shipped defaults move.
        redundant = {"diffusion_3d_1comp":
                     {"kappa": {"upper": builtin["diffusion_3d_1comp"]
                                ["kappa"]["upper"]}}}
        tab5 = fcs_prefs._build_fit_defaults_tab(_Notebook(), _W(),
                                                 {fcs_prefs.FIT_DEFAULTS_KEY:
                                                  redundant})
        check("a value equal to the built-in is dropped, not stored",
              tab5.collect() == {}, tab5.collect())
    finally:
        if old_appdata is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = old_appdata
        shutil.rmtree(tmp, ignore_errors=True)
        fcs_models.apply_param_overrides({})

    print(f"\nAll {ok} checks passed.")
    return 0


if __name__ == "__main__":
    import pathlib
    p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    if not (p / "fcs_prefs.py").is_file():
        print(f"ERROR: fcs_prefs.py not found in {p}. Run this from the "
              f"FCS-Programs directory, or pass its path.")
        raise SystemExit(2)
    sys.exit(main(str(p.resolve())))
