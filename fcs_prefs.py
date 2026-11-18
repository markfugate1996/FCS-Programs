"""
fcs_prefs.py
============
Session-persistent user preferences for the FCS analysis suite.

Design
------
A preference is a registry entry naming a TARGET: a (module, attribute)
pair such as ``("fcs_plottools", "_NORMALIZE_K")``.  Applying a preference
sets that attribute on the target module -- nothing more.  This is general
by construction: any module-level constant anywhere in the suite becomes
user-adjustable by adding one entry to ``PREFS`` below, whether it belongs
to plotting, correlation, fitting, or anything else.  Entries carry a
``section`` string purely so the dialog can group them as the list grows.

The coupling is strictly one-way.  No module in the suite imports
fcs_prefs; each keeps its constant as the built-in default, so every
module works unchanged if this file is absent -- the same graceful
degradation the rest of the suite follows.  fcs_prefs reaches INTO the
others through ``sys.modules``, and only touches modules that are already
imported.  For that reason :func:`apply` must run AFTER the suite's
imports (fcs_main imports everything at the top, then calls apply()); it
is also safe to call again at any time, and the dialog's OK re-applies.

Persistence is a human-readable JSON file under the per-user
application-data directory:

    %APPDATA%/FCS-Programs/prefs.json          (Windows)
    ~/.fcs-programs/prefs.json                 (fallback, non-Windows)

Unknown keys in the file are ignored (a preference removed from the
registry does not break old files).  Missing keys fall back to defaults.
An invalid stored value is reported by name and replaced by the default
for the session -- reported, never silently accepted.

The Fit defaults tab is the one setting that is NOT a registry entry.  The
correlation models carry 48 (model, parameter) pairs, each with bounds, a
guess and two flags, which is a table rather than a list of scalars.  It is
stored sparsely under the single JSON key "fit_defaults" and pushed into
fcs_models.apply_param_overrides(), the same one-way coupling every other
preference uses.

Public interface
----------------
    PREFS                          -- the registry (append to extend)
    apply()                        -- load file, push values into modules
    run_preferences_dialog(parent) -- Settings > Preferences... dialog
    load(), save(values)           -- JSON round-trip, used by the above
    prefs_path()                   -- where the JSON lives
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass
class Pref:
    """
    One user-adjustable setting.

    key      : identifier used in the JSON file; never renamed once released,
               or existing files silently lose the setting.
    label    : text shown in the dialog.
    section  : dialog grouping ("Figures", "Correlation", ...).  Purely
               presentational; the mechanism is identical for every section.
    kind     : "choice" | "int" | "float" | "bool" | "str".
    default  : built-in value; must equal the target module's own constant,
               so an absent or empty prefs file changes nothing.
    target   : (module name, attribute name) the value is written to.
    choices  : allowed values for kind == "choice".
    lo, hi   : inclusive range for int/float; None leaves that side open.
    help     : one grey hint line in the dialog.  Empty for none.
    """
    key: str
    label: str
    section: str
    kind: str
    default: object
    target: tuple
    choices: tuple = field(default_factory=tuple)
    lo: Optional[float] = None
    hi: Optional[float] = None
    help: str = ""


# The registry.  Add entries here; the dialog and the JSON schema follow
# automatically.
PREFS: list = [
    Pref(
        key="figure_format",
        label="Figure export format",
        section="Figures",
        kind="choice",
        default="svg",
        target=("fcs_plottools", "_FIGURE_FORMAT"),
        choices=("svg", "pdf", "png"),
        help="SVG keeps text editable in Illustrator/Inkscape.",
    ),
    Pref(
        key="figure_dpi",
        label="Raster figure DPI",
        section="Figures",
        kind="int",
        default=150,
        target=("fcs_plottools", "_FIGURE_DPI"),
        lo=50, hi=600,
        help="Applies to PNG and other raster formats only.",
    ),
    Pref(
        key="normalize_k",
        label="Normalization: leading points averaged",
        section="Figures",
        kind="int",
        default=3,
        target=("fcs_plottools", "_NORMALIZE_K"),
        lo=1, hi=20,
        help="Reference for the Normalize checkbox: mean of the first "
             "k plotted points.",
    ),
    Pref(
        key="linsolve_taufactor_min",
        label="Shortest taufactor",
        section="Linear decomposition",
        kind="float",
        default=1.0e0,
        target=("fcs_linsolve", "_TAUFACTOR_MIN"),
        lo=1e-3, hi=1e9,
        help="taufactor 1 is a characteristic lag time of 1 microsecond.",
    ),
    Pref(
        key="linsolve_taufactor_max",
        label="Longest taufactor",
        section="Linear decomposition",
        kind="float",
        default=1.0e3,
        target=("fcs_linsolve", "_TAUFACTOR_MAX"),
        lo=1e-3, hi=1e9,
        help="Must exceed the shortest taufactor; 1e3 is 1 millisecond.",
    ),
    Pref(
        key="linsolve_per_decade",
        label="Basis vectors per decade",
        section="Linear decomposition",
        kind="int",
        default=20,
        target=("fcs_linsolve", "_TAUFACTOR_PER_DECADE"),
        lo=1, hi=200,
        help="Spacing is set per decade, so widening the span above does "
             "not coarsen the grid.",
    ),
    Pref(
        key="matrixfit_auto_taufactor_span",
        label="Start from the span the data support",
        section="Linear decomposition",
        kind="bool",
        default=False,
        target=("fcs_matrixfit", "_AUTO_TAUFACTOR_SPAN"),
        help="Prefill the matrix-method span from each dataset's own lag "
             "range instead of the values above. Still editable per run.",
    ),
    Pref(
        key="normalize_mode",
        label="Normalization reference",
        section="Figures",
        kind="choice",
        default="leading",
        target=("fcs_plottools", "_NORMALIZE_MODE"),
        choices=("leading", "peak"),
        help="Leading points suit correlation curves; peak suits lifetime "
             "decays, whose first points sit on the rise.",
    ),
    Pref(
        key="normalize_ymax",
        label="Normalized y-axis top",
        section="Figures",
        kind="float",
        default=1.1,
        target=("fcs_plottools", "_NORMALIZE_YMAX"),
        lo=1.0, hi=5.0,
        help="Slightly above 1 so a trace at full scale is not flush with "
             "the top of the frame.",
    ),
    Pref(
        key="matrixfit_show_stats_box",
        label="Show the statistics box on the distribution plot",
        section="Linear decomposition",
        kind="bool",
        default=False,
        target=("fcs_matrixfit", "_SHOW_STATS_BOX"),
        help="Off by default because it sits over the plot. The plot-controls "
             "Annotations section can turn it on for a single figure.",
    ),
    Pref(
        key="matrixfit_summary_components",
        label="Ranked components in the summary",
        section="Linear decomposition",
        kind="int",
        default=10,
        target=("fcs_matrixfit", "_SUMMARY_MAX_COMPONENTS"),
        lo=0, hi=100,
        help="How many non-zero amplitudes the summary table lists per "
             "dataset, largest first.",
    ),
    Pref(
        key="resid_style",
        label="Residual trace style",
        section="Plot defaults",
        kind="choice",
        default="line",
        target=("fcs_plottools", "_RESID_STYLE"),
        choices=("points", "line", "line+points"),
        help="A thin line shows a trend in the residuals that a cloud of "
             "dots hides. Applies to FCS, lifetime, PCH and matrix/basis "
             "fit panels; the Traces section still overrides one figure.",
    ),
    Pref(
        key="resid_linewidth",
        label="Residual line width",
        section="Plot defaults",
        kind="float",
        default=0.8,
        target=("fcs_plottools", "_RESID_LINEWIDTH"),
        lo=0.1, hi=5.0,
        help="Points, not pixels. Ignored when the style above is points.",
    ),
    Pref(
        key="save_tight",
        label="Crop saved figures to content",
        section="Plot defaults",
        kind="bool",
        default=True,
        target=("fcs_plottools", "_SAVE_TIGHT"),
        help="On, the file is cropped to the drawn artists and is NOT the "
             "figure size. Off, it is exactly the size set in the controls "
             "panel -- what a journal column width needs.",
    ),
    Pref(
        key="svg_editable_text",
        label="Keep SVG text editable",
        section="Plot defaults",
        kind="bool",
        default=True,
        target=("fcs_plottools", "_SVG_EDITABLE_TEXT"),
        help="On, labels stay as text for Illustrator/Inkscape. Off, every "
             "glyph becomes a path: renders anywhere, editable nowhere.",
    ),
    Pref(
        key="grid_default",
        label="Gridlines on new figures",
        section="Plot defaults",
        kind="choice",
        default="as drawn",
        target=("fcs_plottools", "_GRID_DEFAULT"),
        choices=("as drawn", "on", "off"),
        help="'as drawn' keeps each plot's own choice. The others override "
             "every axis when the figure is shown; the Grid section still "
             "toggles one figure.",
    ),
    Pref(
        key="grid_show_minor",
        label="Include minor gridlines",
        section="Plot defaults",
        kind="bool",
        default=True,
        target=("fcs_plottools", "_GRID_SHOW_MINOR"),
        help="Only consulted when the setting above is 'on'.",
    ),
    Pref(
        key="grid_major_linestyle",
        label="Major gridline style",
        section="Plot defaults",
        kind="choice",
        default="--",
        target=("fcs_plottools", "_GRID_MAJOR_LINESTYLE"),
        choices=("--", ":", "-.", "-"),
        help="Also used by the Grid section when it switches gridlines on.",
    ),
    Pref(
        key="linsolve_cd_target_chi2_red",
        label="Stop at chi2/dof",
        section="Matrix method stopping",
        kind="float",
        default=1.0,
        target=("fcs_linsolve", "_CD_TARGET_CHI2_RED"),
        lo=0.0, hi=1e6,
        help="0 switches this off. It trusts G_std to be a fair estimate of "
             "the scatter; a generous G_std stops the search while the "
             "residuals still show structure.",
    ),
    Pref(
        key="linsolve_cd_min_dchi2",
        label="Minimum chi2 improvement",
        section="Matrix method stopping",
        kind="float",
        default=1.0,
        target=("fcs_linsolve", "_CD_MIN_DCHI2"),
        lo=0.0, hi=1e6,
        help="A round buying less than this counts as stalled. Lower it to "
             "keep adding components until the residuals are structureless.",
    ),
    Pref(
        key="linsolve_cd_patience",
        label="Stalled rounds allowed",
        section="Matrix method stopping",
        kind="int",
        default=3,
        target=("fcs_linsolve", "_CD_PATIENCE"),
        lo=1, hi=100,
        help="The improvement sequence is not monotonic, so one small round "
             "is not evidence of having arrived.",
    ),
    Pref(
        key="linsolve_cd_target_dw",
        label="Stop at Durbin-Watson",
        section="Matrix method stopping",
        kind="float",
        default=0.0,
        target=("fcs_linsolve", "_CD_TARGET_DW"),
        lo=0.0, hi=4.0,
        help="0 switches this off. Around 1.8 keeps fitting until the "
             "residuals have no structure left, which does not rely on G_std "
             "being a fair error estimate the way chi2/dof does.",
    ),
    Pref(
        key="linsolve_cd_max_iter",
        label="Maximum rounds",
        section="Matrix method stopping",
        kind="int",
        default=500,
        target=("fcs_linsolve", "_CD_MAX_ITER"),
        lo=1, hi=100000,
        help="Hard cap. A fit that hits it is reported as not converged.",
    ),
]


# ── Storage ───────────────────────────────────────────────────────────────────

def prefs_path() -> Path:
    """
    Location of the JSON file.

    %APPDATA% is read per call rather than at import, so a test (or an
    unusual launcher) can redirect storage by setting the environment
    variable before use.
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "FCS-Programs" / "prefs.json"
    return Path.home() / ".fcs-programs" / "prefs.json"


def load() -> dict:
    """
    Read the stored values.  Missing file -> {}.

    A corrupt file is reported on the console and treated as empty for the
    session; it is NOT overwritten here, so the broken content survives for
    inspection until the next explicit save.
    """
    p = prefs_path()
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("top level is not an object")
        return data
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"[prefs] could not read {p} ({exc}); using defaults")
        return {}


def save(values: dict) -> Optional[Path]:
    """
    Write *values* (already validated) to the JSON file, creating the
    directory if needed.  Returns the path written, or None on failure --
    a preferences write that fails must never take the application down.
    """
    p = prefs_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(values, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return p
    except Exception as exc:
        print(f"[prefs] could not write {p} ({exc})")
        return None


# ── Validation and application ────────────────────────────────────────────────

def _coerce(pref: Pref, raw):
    """
    Validate *raw* against *pref*; return the typed value.

    Raises ValueError with a human-readable reason.  bool accepts real
    booleans only -- JSON has them, so "1"/"yes" strings would only paper
    over a malformed file.
    """
    if pref.kind == "choice":
        if raw in pref.choices:
            return raw
        raise ValueError(f"must be one of {', '.join(map(str, pref.choices))}")
    if pref.kind == "bool":
        if isinstance(raw, bool):
            return raw
        raise ValueError("must be true or false")
    if pref.kind == "int":
        try:
            v = int(raw)
            if float(raw) != v:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("must be a whole number")
    elif pref.kind == "float":
        try:
            v = float(raw)
        except (TypeError, ValueError):
            raise ValueError("must be a number")
    elif pref.kind == "str":
        if not isinstance(raw, str):
            raise ValueError("must be text")
        return raw
    else:
        raise ValueError(f"unknown preference kind '{pref.kind}'")
    if pref.lo is not None and v < pref.lo:
        raise ValueError(f"must be at least {pref.lo:g}")
    if pref.hi is not None and v > pref.hi:
        raise ValueError(f"must be at most {pref.hi:g}")
    return v


def effective_value(pref: Pref, stored: dict):
    """
    The value *pref* should have right now: the stored value when present
    and valid, else the default.  Invalid stored values are reported by
    name on the console and never applied.
    """
    if pref.key in stored:
        try:
            return _coerce(pref, stored[pref.key])
        except ValueError as exc:
            print(f"[prefs] stored value for '{pref.key}' ignored: {exc}; "
                  f"using default {pref.default!r}")
    return pref.default


FIT_DEFAULTS_KEY = "fit_defaults"


def _apply_fit_defaults(stored: dict) -> int:
    """
    Push the Fit defaults table into fcs_models.

    Same rules as the registry: fcs_models is touched only when it is
    already imported, failures are reported rather than raised, and the
    module's own built-in values are the fallback -- so an absent or empty
    table changes nothing.
    """
    mod = sys.modules.get("fcs_models")
    if mod is None or not hasattr(mod, "apply_param_overrides"):
        return 0
    table = stored.get(FIT_DEFAULTS_KEY) or {}
    if not isinstance(table, dict):
        print(f"[prefs] '{FIT_DEFAULTS_KEY}' is not an object; ignored")
        table = {}
    try:
        return int(mod.apply_param_overrides(table))
    except Exception as exc:                            # noqa: BLE001
        print(f"[prefs] could not apply fit defaults ({exc})")
        return 0


def _fmt_param_value(value) -> str:
    """Format a bound or guess for an entry box, infinities included."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.10g}"


def _parse_param_value(text: str, label: str) -> float:
    """Read a bound or guess back; 'inf' / '-inf' allowed."""
    s = str(text).strip().lower()
    if s in ("inf", "+inf", "infinity"):
        return float("inf")
    if s in ("-inf", "-infinity"):
        return float("-inf")
    if not s:
        raise ValueError(f"{label} is empty")
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"{label} is not a number: {text!r}")


def _storable(value):
    """JSON-safe form of a parameter value (JSON has no infinity literal)."""
    if isinstance(value, bool):
        return value
    v = float(value)
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return v


def _same_value(a, b) -> bool:
    """Whether an edited value still equals the built-in it came from."""
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    return float(a) == float(b)


def _build_fit_defaults_tab(notebook, dialog, stored):
    """
    Build the "Fit defaults" tab: one model at a time, one row per
    parameter, columns guess / min / max / fix / link.

    Returns an object with ``collect()`` -- the sparse override table, or
    ValueError with a message naming the offending cell -- and
    ``restore()``.  Returns None when fcs_models is not imported, in which
    case the dialog simply has one tab fewer, the same graceful degradation
    the rest of this module follows.

    Every model's widgets are built once and the selected one is packed, so
    edits to a model survive switching away and back before OK.
    """
    import tkinter as tk

    models_mod = sys.modules.get("fcs_models")
    builtin = getattr(models_mod, "BUILTIN_PARAM_DEFAULTS", None)
    if not builtin:
        return None
    auto = set(getattr(models_mod, "AUTO_GUESS_PARAMS", ()))
    table = stored.get(FIT_DEFAULTS_KEY) or {}
    if not isinstance(table, dict):
        table = {}

    tab = tk.Frame(notebook, padx=10, pady=8)
    notebook.add(tab, text="Fit defaults")

    label_of = {}
    for key in builtin:
        model = models_mod.MODELS.get(key)
        label_of[key] = f"{getattr(model, 'name', key)}" if model else key
    keys = list(builtin)

    top = tk.Frame(tab)
    top.pack(fill="x")
    tk.Label(top, text="Model:", font=("Helvetica", 9)).pack(side="left")
    model_var = tk.StringVar(master=dialog, value=label_of[keys[0]])
    om = tk.OptionMenu(top, model_var, *[label_of[k] for k in keys])
    om.config(font=("Helvetica", 9), width=34, anchor="w")
    om["menu"].config(font=("Helvetica", 9))
    om.pack(side="left", padx=4)

    cells: dict = {}
    frames: dict = {}

    def _initial(mkey, pname, field):
        base = builtin[mkey][pname][field]
        over = ((table.get(mkey) or {}).get(pname) or {})
        if field not in over:
            return base
        try:
            if field in ("fixed", "link"):
                return bool(over[field])
            return models_mod._as_bound(over[field])
        except (TypeError, ValueError):
            return base

    for mkey in keys:
        frame = tk.Frame(tab)
        frames[mkey] = frame

        head = tk.Frame(frame)
        head.pack(fill="x", pady=(6, 2))
        for text, width in (("parameter", 11), ("guess", 11), ("min", 11),
                            ("max", 11), ("fix", 4), ("link", 5)):
            tk.Label(head, text=text, width=width, anchor="w",
                     font=("Helvetica", 8, "bold")).pack(side="left")

        for pname in builtin[mkey]:
            row = tk.Frame(frame)
            row.pack(fill="x")
            tk.Label(row, text=pname, width=11, anchor="w",
                     font=("Helvetica", 9)).pack(side="left")

            if pname in auto:
                # Shown, not editable: auto_guess rewrites these from the
                # curve on every fit, so an entry box here would be a
                # setting that silently does nothing.
                tk.Label(row, text="auto", width=11, anchor="w", fg="grey",
                         font=("Helvetica", 9, "italic")).pack(side="left")
                cells[(mkey, pname, "guess")] = None
            else:
                gv = tk.StringVar(master=dialog,
                                  value=_fmt_param_value(
                                      _initial(mkey, pname, "guess")))
                tk.Entry(row, textvariable=gv, width=10,
                         font=("Helvetica", 9),
                         justify="right").pack(side="left", padx=(0, 2))
                cells[(mkey, pname, "guess")] = gv

            for field in ("lower", "upper"):
                v = tk.StringVar(master=dialog,
                                 value=_fmt_param_value(
                                     _initial(mkey, pname, field)))
                tk.Entry(row, textvariable=v, width=10,
                         font=("Helvetica", 9),
                         justify="right").pack(side="left", padx=(0, 2))
                cells[(mkey, pname, field)] = v

            for field, width in (("fixed", 4), ("link", 5)):
                b = tk.BooleanVar(master=dialog,
                                  value=bool(_initial(mkey, pname, field)))
                holder = tk.Frame(row, width=width * 8)
                holder.pack(side="left")
                tk.Checkbutton(holder, variable=b).pack(side="left")
                cells[(mkey, pname, field)] = b

        tk.Label(
            frame,
            text=("      guess, min and max accept inf and -inf · fix holds "
                  "a parameter at its guess ·\n      link shares one value "
                  "across datasets · these set what the fit dialog OPENS "
                  "with,\n      and every fit can still override them"),
            font=("Helvetica", 8), fg="grey", anchor="w",
            justify="left").pack(fill="x", pady=(6, 0))

    def _show(*_a):
        wanted = model_var.get()
        for mkey, frame in frames.items():
            if label_of[mkey] == wanted:
                frame.pack(fill="x")
            else:
                frame.pack_forget()

    model_var.trace_add("write", _show)
    _show()

    def _restore_current():
        wanted = model_var.get()
        for mkey in keys:
            if label_of[mkey] != wanted:
                continue
            for pname, base in builtin[mkey].items():
                for field in ("guess", "lower", "upper"):
                    var = cells[(mkey, pname, field)]
                    if var is not None:
                        var.set(_fmt_param_value(base[field]))
                for field in ("fixed", "link"):
                    cells[(mkey, pname, field)].set(bool(base[field]))

    tk.Button(tab, text="Restore this model's defaults", pady=3,
              command=_restore_current).pack(pady=(8, 0))

    def restore():
        for mkey in keys:
            for pname, base in builtin[mkey].items():
                for field in ("guess", "lower", "upper"):
                    var = cells[(mkey, pname, field)]
                    if var is not None:
                        var.set(_fmt_param_value(base[field]))
                for field in ("fixed", "link"):
                    cells[(mkey, pname, field)].set(bool(base[field]))

    def collect() -> dict:
        out: dict = {}
        for mkey in keys:
            for pname, base in builtin[mkey].items():
                where = f"{label_of[mkey]} · {pname}"
                lo = _parse_param_value(
                    cells[(mkey, pname, "lower")].get(), f"{where} min")
                hi = _parse_param_value(
                    cells[(mkey, pname, "upper")].get(), f"{where} max")
                if not lo < hi:
                    raise ValueError(
                        f"{where}: minimum ({_fmt_param_value(lo)}) must be "
                        f"below maximum ({_fmt_param_value(hi)}).")
                gvar = cells[(mkey, pname, "guess")]
                if gvar is None:
                    guess = base["guess"]
                else:
                    guess = _parse_param_value(gvar.get(), f"{where} guess")
                    if not lo <= guess <= hi:
                        raise ValueError(
                            f"{where}: guess {_fmt_param_value(guess)} lies "
                            f"outside its own bounds "
                            f"{_fmt_param_value(lo)} to "
                            f"{_fmt_param_value(hi)}.")
                values = {"guess": guess, "lower": lo, "upper": hi,
                          "fixed": bool(cells[(mkey, pname, "fixed")].get()),
                          "link": bool(cells[(mkey, pname, "link")].get())}
                entry = {f: _storable(v) for f, v in values.items()
                         if not _same_value(v, base[f])}
                if entry:
                    out.setdefault(mkey, {})[pname] = entry
        return out

    class _Tab:
        pass

    handle = _Tab()
    handle.collect = collect
    handle.restore = restore
    return handle


def apply(values: Optional[dict] = None) -> int:
    """
    Push every registered preference into its target module.

    *values* defaults to the stored file.  Only modules already present in
    ``sys.modules`` are touched: fcs_prefs never imports suite modules
    itself, so module load order (and hot-swapping) stays untouched.  Run
    this AFTER the suite's imports; calling it again later is safe and is
    what the dialog's OK does.

    Returns the number of attributes actually set, which a caller can log.
    """
    stored = load() if values is None else values
    n = _apply_fit_defaults(stored)
    for pref in PREFS:
        mod = sys.modules.get(pref.target[0])
        if mod is None:
            continue
        try:
            setattr(mod, pref.target[1], effective_value(pref, stored))
            n += 1
        except Exception as exc:
            print(f"[prefs] could not apply '{pref.key}' to "
                  f"{pref.target[0]}.{pref.target[1]} ({exc})")
    return n


# ── Dialog ────────────────────────────────────────────────────────────────────

def run_preferences_dialog(parent=None) -> None:
    """
    Show the Preferences dialog, generated from the registry.

    Fields show the value currently IN EFFECT: the live module attribute
    when the target module is imported (the ground truth for what the
    session is doing), else the stored value, else the default.  OK
    validates every field -- reporting the first offender by label and
    leaving the dialog open -- then saves the JSON and applies live.
    "Restore defaults" resets the FIELDS only; nothing is saved or applied
    until OK.
    """
    import tkinter as tk
    from tkinter import messagebox, ttk

    stored = load()

    dialog = tk.Toplevel(parent) if parent is not None else tk.Toplevel()
    dialog.title("Preferences")
    # Resizable, and sized to fit the screen further down.  The registry only
    # ever grows, and a fixed-size settings window eventually puts its OK
    # button somewhere unreachable.
    dialog.resizable(True, True)
    try:
        dialog.grab_set()
    except Exception:
        pass

    def _current(pref: Pref):
        mod = sys.modules.get(pref.target[0])
        if mod is not None and hasattr(mod, pref.target[1]):
            live = getattr(mod, pref.target[1])
            try:
                return _coerce(pref, live)
            except ValueError:
                pass       # a live value outside today's range: fall back
        return effective_value(pref, stored)

    # ── One TAB per section, one row per pref ─────────────────────────────────
    #
    # Sections were stacked LabelFrames, which worked while there were three of
    # them and stopped working as the registry grew: the window simply got
    # taller until it ran off the screen.  A notebook bounds the height at the
    # tallest single section instead of the sum of all of them, and the tab
    # strip doubles as the list of categories.
    #
    # Tab order follows first appearance in PREFS, so the registry stays the
    # single place that decides both what exists and how it is grouped.
    notebook = ttk.Notebook(dialog)
    notebook.pack(side="top", fill="both", expand=True, padx=8, pady=(8, 2))

    vars_by_key: dict = {}
    sections: dict = {}
    for pref in PREFS:
        if pref.section not in sections:
            tab = tk.Frame(notebook, padx=10, pady=8)
            notebook.add(tab, text=pref.section)
            sections[pref.section] = tab
        lf = sections[pref.section]

        row = tk.Frame(lf)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=pref.label + ":", anchor="w",
                 font=("Helvetica", 9)).pack(side="left")

        cur = _current(pref)
        if pref.kind == "choice":
            var = tk.StringVar(master=dialog, value=str(cur))
            om = tk.OptionMenu(row, var, *pref.choices)
            om.config(font=("Helvetica", 9), width=8)
            om["menu"].config(font=("Helvetica", 9))
            om.pack(side="right")
        elif pref.kind == "bool":
            var = tk.BooleanVar(master=dialog, value=bool(cur))
            tk.Checkbutton(row, variable=var).pack(side="right")
        else:
            var = tk.StringVar(master=dialog, value=str(cur))
            tk.Entry(row, textvariable=var, width=10, font=("Helvetica", 9),
                     justify="right").pack(side="right")
        vars_by_key[pref.key] = var

        if pref.help:
            tk.Label(lf, text="      " + pref.help, font=("Helvetica", 8),
                     fg="grey", anchor="w", justify="left").pack(fill="x")

    fit_defaults = _build_fit_defaults_tab(notebook, dialog, stored)

    tk.Label(dialog, text=f"Stored in: {prefs_path()}",
             font=("Helvetica", 8), fg="grey", anchor="w",
             justify="left").pack(side="top", fill="x", padx=12, pady=(6, 0))

    # ── Buttons ───────────────────────────────────────────────────────────────
    def _restore_defaults():
        for pref in PREFS:
            v = vars_by_key[pref.key]
            if pref.kind == "bool":
                v.set(bool(pref.default))
            else:
                v.set(str(pref.default))
        if fit_defaults is not None:
            fit_defaults.restore()

    def _ok():
        new_values = dict(stored)      # keep unknown keys from older files
        for pref in PREFS:
            raw = vars_by_key[pref.key].get()
            try:
                new_values[pref.key] = _coerce(pref, raw)
            except ValueError as exc:
                messagebox.showerror(
                    "Invalid value", f"{pref.label}: {exc}", parent=dialog)
                return
        if fit_defaults is not None:
            try:
                table = fit_defaults.collect()
            except ValueError as exc:
                messagebox.showerror("Invalid fit default", str(exc),
                                     parent=dialog)
                return
            if table:
                new_values[FIT_DEFAULTS_KEY] = table
            else:
                # Everything back at its built-in: drop the key rather than
                # write an empty object, so the file says "nothing
                # overridden" the same way it did before the tab existed.
                new_values.pop(FIT_DEFAULTS_KEY, None)

        if save(new_values) is None:
            messagebox.showerror(
                "Save failed",
                f"Could not write {prefs_path()}.\n\n"
                "The values were NOT applied.  See the console for details.",
                parent=dialog)
            return
        apply(new_values)
        dialog.destroy()

    # [button-strip] Claim the bottom slot before anything that can grow.
    # side="bottom" decides where the strip is DRAWN, not what survives when
    # space runs short: pack() hands out space in packing order, so this
    # frame -- packed after the notebook, which expands -- is the first thing
    # squeezed when fit_dialog_to_screen caps the window below the content's
    # height, and OK/Cancel end up below the bottom edge.  Re-ordering here
    # rather than moving the code keeps the build order intact, and matches
    # the idiom the fit and PCH dialogs already use.
    btns = tk.Frame(dialog)
    btns.pack(side="bottom", pady=10)
    _above = [_w for _w in dialog.pack_slaves() if _w is not btns]
    if _above:
        btns.pack_configure(before=_above[0])
    tk.Button(btns, text="OK", width=10, command=_ok,
              pady=3).pack(side="left", padx=5)
    tk.Button(btns, text="Cancel", width=10, command=dialog.destroy,
              pady=3).pack(side="left", padx=5)
    tk.Button(btns, text="Restore defaults", width=14,
              command=_restore_defaults, pady=3).pack(side="left", padx=5)

    # Size to content, capped at a fraction of the screen.  fcs_fitcommon owns
    # the sizing logic; if it is absent this module still works, which is the
    # same graceful degradation the rest of the file follows.
    try:
        from fcs_fitcommon import fit_dialog_to_screen
    except Exception:              # noqa: BLE001
        dialog.update_idletasks()
    else:
        fit_dialog_to_screen(dialog, min_width=420, min_height=260)

    dialog.wait_window()
