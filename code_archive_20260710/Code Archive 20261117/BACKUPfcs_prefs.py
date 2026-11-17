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
    n = 0
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
    from tkinter import messagebox

    stored = load()

    dialog = tk.Toplevel(parent) if parent is not None else tk.Toplevel()
    dialog.title("Preferences")
    dialog.resizable(False, False)
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

    # ── One row per pref, grouped by section ──────────────────────────────────
    vars_by_key: dict = {}
    sections: dict = {}
    for pref in PREFS:
        if pref.section not in sections:
            lf = tk.LabelFrame(dialog, text=pref.section, padx=10, pady=6)
            lf.pack(fill="x", padx=12, pady=(8, 2))
            sections[pref.section] = lf
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

    tk.Label(dialog, text=f"Stored in: {prefs_path()}",
             font=("Helvetica", 8), fg="grey").pack(padx=12, pady=(6, 0))

    # ── Buttons ───────────────────────────────────────────────────────────────
    def _restore_defaults():
        for pref in PREFS:
            v = vars_by_key[pref.key]
            if pref.kind == "bool":
                v.set(bool(pref.default))
            else:
                v.set(str(pref.default))

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
        if save(new_values) is None:
            messagebox.showerror(
                "Save failed",
                f"Could not write {prefs_path()}.\n\n"
                "The values were NOT applied.  See the console for details.",
                parent=dialog)
            return
        apply(new_values)
        dialog.destroy()

    btns = tk.Frame(dialog)
    btns.pack(pady=10)
    tk.Button(btns, text="OK", width=10, command=_ok,
              pady=3).pack(side="left", padx=5)
    tk.Button(btns, text="Cancel", width=10, command=dialog.destroy,
              pady=3).pack(side="left", padx=5)
    tk.Button(btns, text="Restore defaults", width=14,
              command=_restore_defaults, pady=3).pack(side="left", padx=5)

    dialog.wait_window()
