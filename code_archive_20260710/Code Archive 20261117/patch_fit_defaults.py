#!/usr/bin/env python3
"""
Add a "Fit defaults" tab to Preferences: per correlation model, per
parameter, the bounds and the link / fix flags that the global-fit dialog
starts from (plus the initial guess, where it is not derived from the data).

Why a table and not registry rows: the correlation models carry 48
(model, parameter) pairs, and the existing Pref registry is one scalar per
row.  Overrides are stored sparsely under one JSON key, so a parameter you
have not touched keeps following the built-in value.

Touches fcs_models.py and fcs_prefs.py.  Every anchor in both files is
validated before anything is written; line endings are preserved per file.

Usage:  python3 patch_fit_defaults.py [path/to/FCS-Programs]
"""
import pathlib
import sys

# ── fcs_models.py ─────────────────────────────────────────────────────────
MODELS_EDITS = [
    ("built-in snapshot and override application",
'''# ── Accessors ─────────────────────────────────────────────────────────────────

def list_models() -> List[FCSModel]:
''',
'''# ── User-adjustable parameter defaults ────────────────────────────────────────
#
# A Param carries the state the fit dialog opens with: initial guess, bounds,
# and the suggested "hold fixed" / "link across datasets" flags.  Those are
# judgement calls, not physics, and they are the kind of judgement that
# changes with experience -- so they are preferences rather than constants
# edited in this file.  fcs_prefs pushes a stored table in through
# apply_param_overrides(); nothing here imports fcs_prefs, so the suite runs
# unchanged when that file is absent.
#
# Correlation models only (MODELS).  The PCH and lifetime registries are not
# touched: their dialogs are separate and were not part of this.

# The shipped values, captured before anything can overwrite them.  Both the
# "restore defaults" button and the sparse-override comparison need to know
# what the built-in value WAS, which a mutated Param can no longer say.
BUILTIN_PARAM_DEFAULTS: Dict[str, Dict[str, dict]] = {
    _key: {
        _p.name: {"guess": _p.default, "lower": _p.lower, "upper": _p.upper,
                  "fixed": _p.fixed, "link": _p.link_default}
        for _p in _model.params
    }
    for _key, _model in MODELS.items()
}

PARAM_FIELDS = ("guess", "lower", "upper", "fixed", "link")

# Parameters whose starting guess fcs_fit.auto_guess derives from the curve
# itself (short-lag plateau, half-amplitude crossing) rather than reading
# from the Param.  A stored guess for these would be overwritten on every
# fit, so the preferences tab shows them as automatic instead of pretending
# to set them.  Mirrors auto_guess; test_fit_defaults.py checks they agree.
AUTO_GUESS_PARAMS = ("G0", "tau_D", "offset")


def _as_bound(value) -> float:
    """
    A bound from stored JSON: a number, or "inf" / "-inf".

    JSON has no infinity literal, and the suite's bounds routinely are
    infinite, so they round-trip as those two strings rather than as
    something non-standard that only Python would read back.
    """
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("inf", "+inf", "infinity", "+infinity"):
            return float(np.inf)
        if text in ("-inf", "-infinity"):
            return float(-np.inf)
        return float(text)
    return float(value)


def apply_param_overrides(overrides: dict) -> int:
    """
    Reset every correlation parameter to its built-in default, then apply
    *overrides*.

    Resetting first is what makes this idempotent, and what makes REMOVING
    an override take effect without restarting: the dialog's OK re-applies
    the whole table, and a value no longer in it has to go back to the
    built-in rather than linger from the last apply.

    *overrides* is ``{model_key: {param: {field: value}}}`` and sparse --
    only fields that differ from the built-in are stored, so a later change
    to a shipped default still reaches a parameter the user never touched.

    Unknown model keys, parameter names and fields are reported on the
    console and skipped, never raised: a table written before a model was
    renamed must not stop the suite from starting.

    Returns the number of fields actually applied.
    """
    for _key, _model in MODELS.items():
        base = BUILTIN_PARAM_DEFAULTS.get(_key, {})
        for _p in _model.params:
            b = base.get(_p.name)
            if b is None:
                continue
            _p.default = b["guess"]
            _p.lower = b["lower"]
            _p.upper = b["upper"]
            _p.fixed = b["fixed"]
            _p.link_default = b["link"]

    if not isinstance(overrides, dict) or not overrides:
        return 0

    n = 0
    for mkey, params in overrides.items():
        model = MODELS.get(mkey)
        if model is None:
            print(f"[models] fit default for unknown model '{mkey}' ignored")
            continue
        by_name = {p.name: p for p in model.params}
        if not isinstance(params, dict):
            print(f"[models] fit defaults for '{mkey}' are not an object; "
                  f"ignored")
            continue
        for pname, fields in params.items():
            p = by_name.get(pname)
            if p is None:
                print(f"[models] fit default for unknown parameter "
                      f"'{mkey}.{pname}' ignored")
                continue
            if not isinstance(fields, dict):
                print(f"[models] fit defaults for '{mkey}.{pname}' are not "
                      f"an object; ignored")
                continue
            for field, value in fields.items():
                try:
                    if field == "guess":
                        p.default = float(value)
                    elif field == "lower":
                        p.lower = _as_bound(value)
                    elif field == "upper":
                        p.upper = _as_bound(value)
                    elif field == "fixed":
                        p.fixed = bool(value)
                    elif field == "link":
                        p.link_default = bool(value)
                    else:
                        raise ValueError(f"unknown field '{field}'")
                except (TypeError, ValueError) as exc:
                    print(f"[models] fit default {mkey}.{pname}.{field} "
                          f"ignored ({exc})")
                    continue
                n += 1
    return n


# ── Accessors ─────────────────────────────────────────────────────────────────

def list_models() -> List[FCSModel]:
'''),
]

# ── fcs_prefs.py ──────────────────────────────────────────────────────────
PREFS_EDITS = [
    ("docstring: the fit-defaults table",
'''Public interface
----------------
    PREFS                          -- the registry (append to extend)
''',
'''The Fit defaults tab is the one setting that is NOT a registry entry.  The
correlation models carry 48 (model, parameter) pairs, each with bounds, a
guess and two flags, which is a table rather than a list of scalars.  It is
stored sparsely under the single JSON key "fit_defaults" and pushed into
fcs_models.apply_param_overrides(), the same one-way coupling every other
preference uses.

Public interface
----------------
    PREFS                          -- the registry (append to extend)
'''),

    ("apply() pushes the fit defaults too",
'''    stored = load() if values is None else values
    n = 0
    for pref in PREFS:
        mod = sys.modules.get(pref.target[0])
''',
'''    stored = load() if values is None else values
    n = _apply_fit_defaults(stored)
    for pref in PREFS:
        mod = sys.modules.get(pref.target[0])
'''),

    ("the fit-defaults applier",
'''def apply(values: Optional[dict] = None) -> int:
''',
'''FIT_DEFAULTS_KEY = "fit_defaults"


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
                  "a parameter at its guess ·\\n      link shares one value "
                  "across datasets · these set what the fit dialog OPENS "
                  "with,\\n      and every fit can still override them"),
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
'''),

    ("dialog builds the tab",
'''    tk.Label(dialog, text=f"Stored in: {prefs_path()}",
''',
'''    fit_defaults = _build_fit_defaults_tab(notebook, dialog, stored)

    tk.Label(dialog, text=f"Stored in: {prefs_path()}",
'''),

    ("restore defaults covers the table",
'''    def _restore_defaults():
        for pref in PREFS:
            v = vars_by_key[pref.key]
            if pref.kind == "bool":
                v.set(bool(pref.default))
            else:
                v.set(str(pref.default))
''',
'''    def _restore_defaults():
        for pref in PREFS:
            v = vars_by_key[pref.key]
            if pref.kind == "bool":
                v.set(bool(pref.default))
            else:
                v.set(str(pref.default))
        if fit_defaults is not None:
            fit_defaults.restore()
'''),

    ("OK saves the table",
'''        if save(new_values) is None:
''',
'''        if fit_defaults is not None:
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
'''),
]


def _target(argv):
    p = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    if p.is_file():
        p = p.parent
    for name in ("fcs_models.py", "fcs_prefs.py"):
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
    jobs = {"fcs_models.py": MODELS_EDITS, "fcs_prefs.py": PREFS_EDITS}

    planned = {}
    for fname, edits in jobs.items():
        try:
            planned[fname] = _plan(root / fname, edits)
        except ValueError as exc:
            print(f"ERROR: {exc}\nNothing was written.")
            return 2

    for fname, (data, applied, eol) in planned.items():
        (root / fname).write_bytes(data)
        print(f"  {fname} ({eol})")
        for name in applied:
            print(f"      - {name}")
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
