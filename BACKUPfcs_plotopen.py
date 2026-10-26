"""
fcs_plotopen.py
===============
Reopen a previously exported analysis as a LIVE figure.

A PNG of a fit from three months ago is a picture.  This module reopens the
same result as a real matplotlib figure with the full plot-controls panel:
rescale it, restyle the traces, relabel the axes, re-export it.

How it works
------------
Every analysis in this suite already writes a self-describing CSV: a
commented ``# key : value`` header followed by the exact columns that were
plotted.  Reopening is therefore not a matter of restoring a saved *picture*
but of re-reading that data and calling the SAME plotting function that drew
it the first time.  There is no second rendering path to drift out of step
with the first, and the reopened figure is identical to a fresh one.

This is why figures are not pickled.  ``pickle.dump(fig)`` is five lines and
tempting, but matplotlib gives no compatibility guarantee across versions:
figures pickled today can fail to load after an upgrade, with nothing to fall
back on.  It also stores rendered state rather than the science, so nothing
else -- Excel, Origin, a collaborator -- can read it.  A CSV stays readable
by everything, forever.

Adding an analysis
------------------
Implement ``rebuild_plot(meta, columns, show=True, path=None) -> (fig, ax)``
in the module that owns the analysis, then register it::

    register_rebuilder("intensity", "fcs_plot", "rebuild_plot")

The rebuilder receives exactly what :func:`fcs_export.read_export` returns,
plus the source path, so it can find sibling files -- a global fit reads its
parameter table from alongside the curves.
Modules are imported lazily, on dispatch, so this module can depend on all of
them without any of them depending on it.

Overlaying several exports
--------------------------
Several exports can be reopened onto ONE figure for comparison.  Each file
is still rebuilt by its own registered rebuilder -- the single rendering
path is preserved -- but silently (``show=False``), and the resulting
curves are then composited onto a fresh combined figure.  Only the data
traces travel: shaded error bands, annotation boxes, residual panels and
similar auxiliary artists cannot be composited faithfully and are dropped,
which is reported rather than passed off as a complete picture.  See
:func:`open_exports`.

Public API
----------
    supported_analyses()                  -> list[str]
    register_rebuilder(name, mod, func)   -> None
    open_export(path, show=True)          -> (fig, ax)
    open_exports(paths, show=True)        -> (fig, ax, notes)
    run_open_dialog(parent=None)          -> (fig, ax) | None
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Optional, Tuple

import fcs_export


# ── Registry ──────────────────────────────────────────────────────────────────

# analysis name, exactly as written into the CSV header -> (module, function)
_REBUILDERS: dict = {
    "correlation":        ("fcs_corr",     "rebuild_plot"),
    "global fit curves":  ("fcs_fit",      "rebuild_plot"),
    "calibration points": ("fcs_calib",    "rebuild_plot"),
    "lifetime":           ("fcs_lifetime", "rebuild_plot"),
    "lifetime fit curve": ("fcs_lifetime_fit", "rebuild_plot"),
    "pch":                ("fcs_pch",      "rebuild_plot"),
    "intensity":          ("fcs_plot",     "rebuild_plot"),
}


def register_rebuilder(analysis: str, module_name: str,
                       func_name: str = "rebuild_plot") -> None:
    """
    Register a rebuilder for an analysis.

    *module_name* is imported only when a file of that analysis is actually
    opened, so registering costs nothing and cannot create an import cycle.
    """
    _REBUILDERS[str(analysis).strip().lower()] = (module_name, func_name)


def supported_analyses() -> list:
    """Analysis names that can currently be reopened, sorted."""
    return sorted(_REBUILDERS)


# ── Opening ───────────────────────────────────────────────────────────────────

def _resolve(analysis: str):
    """Look up and import the rebuilder for *analysis*."""
    key = str(analysis or "").strip().lower()
    if key not in _REBUILDERS:
        known = ", ".join(supported_analyses()) or "(none registered)"
        raise ValueError(
            f"No rebuilder is registered for analysis '{analysis}'.\n\n"
            f"Currently supported: {known}."
        )
    module_name, func_name = _REBUILDERS[key]
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Cannot reopen '{analysis}': module {module_name} failed to "
            f"import ({exc})."
        ) from exc
    fn = getattr(mod, func_name, None)
    if fn is None:
        raise AttributeError(
            f"{module_name} has no '{func_name}', so '{analysis}' exports "
            f"cannot be rebuilt."
        )
    return fn


def _call_rebuilder(fn, meta, columns, show, path, options):
    """
    Call a rebuilder, forwarding *options* only to those that accept it.

    ``options`` is an optional extension to the rebuilder contract: display
    choices the reopen path collected from the user, which only some
    analyses have any use for (the lifetime fit curve's time axis is the
    first).  Rebuilders written before it existed take four arguments and
    must keep working untouched, so the parameter is passed only when the
    target's signature actually has it.
    """
    kw = {"show": show, "path": path}
    if options:
        try:
            accepts = "options" in inspect.signature(fn).parameters
        except (TypeError, ValueError):   # builtins, C functions, exotic wrappers
            accepts = False
        if accepts:
            kw["options"] = options
    return fn(meta, columns, **kw)


def open_export(path, show: bool = True, options=None) -> Tuple:
    """
    Reopen an exported analysis CSV as a live figure.

    Parameters
    ----------
    path : str | Path
        A CSV written by this suite.
    show : bool
        Display the figure and its plot-controls panel.
    options : dict, optional
        Display choices forwarded to the rebuilder if it accepts them; see
        :func:`_call_rebuilder`.

    Returns
    -------
    (fig, ax) from the owning module's rebuilder.

    Raises
    ------
    FileNotFoundError, ValueError, ImportError, AttributeError
        With a message naming the actual problem, so a mis-picked file says
        so rather than failing deep inside a plotting routine.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")

    meta, columns = fcs_export.read_export(p)

    analysis = meta.get("analysis")
    if not analysis:
        raise ValueError(
            f"{p.name} does not identify which analysis produced it.\n\n"
            f"Exports written by this suite carry an 'analysis' header line "
            f"(or the older 'FCS analysis export' banner).  This file has "
            f"neither, so it may not be an FCS analysis export."
        )

    if not columns:
        raise ValueError(f"{p.name} contains a header but no data rows.")

    return _call_rebuilder(_resolve(analysis), meta, columns, show, p,
                           options)


# ── Overlaying several exports ────────────────────────────────────────────────

# Line properties carried across when a trace is composited onto the combined
# figure.  Copying the drawn appearance (rather than re-parenting the artist)
# keeps this independent of how any given matplotlib version stores an
# artist's internal state.
_LINE_PROPS = (
    "color", "linestyle", "linewidth", "marker", "markersize",
    "markerfacecolor", "markeredgecolor", "markeredgewidth", "alpha",
    "zorder", "drawstyle", "fillstyle",
)


def _is_reference_line(line, ax) -> bool:
    """
    True for an axhline/axvline rather than a data trace.

    Those draw with a blended transform -- one coordinate in axes fraction
    -- so comparing against ax.transData identifies them exactly.  The same
    test is used by the Normalize section of the plot-controls panel.
    """
    try:
        return line.get_transform() != ax.transData
    except Exception:
        return True


def _main_axes(fig, axes):
    """
    The primary axes of a rebuilt figure.

    Rebuilders return either a single Axes or a sequence whose first entry
    is the main panel and whose remainder are residual panels; only the
    main panel is composited.
    """
    if axes is None:
        return fig.axes[0] if fig.axes else None
    if hasattr(axes, "get_xlim"):
        return axes
    try:
        seq = list(axes)
    except TypeError:
        return None
    return seq[0] if seq else None


def _axis_labels(fig, main_ax):
    """
    The x and y labels that describe *main_ax*'s data.

    In a stacked fit figure the x axis is SHARED and only the bottom panel
    carries its label, so reading the x label off the main panel alone
    returns "" -- the overlay then has an unlabelled x axis, and the
    disagreement check in open_exports can never fire because every
    two-panel figure reports the same empty string.  Falling back to the
    lowest panel that has one fixes both, and changes nothing for a
    single-panel figure, which labels its own axis.
    """
    xlabel = main_ax.get_xlabel()
    if not xlabel:
        for other in reversed(fig.axes):
            lbl = other.get_xlabel()
            if lbl:
                xlabel = lbl
                break
    return xlabel, main_ax.get_ylabel()


def _qualified_label(label, stem: str) -> str:
    """
    Prefix a trace's label with the file it came from.

    Overlaid files usually carry identical labels -- every correlation
    export names its traces "Ch1 ACF" -- so the source file is what
    distinguishes them: "sampleA — Ch1 ACF".

    The leading underscore that hides a fit's data points from the legend is
    preserved through the rename, so hidden traces stay hidden.  A trace with
    no label at all, or matplotlib's "_nolegend_" placeholder, is left alone:
    naming it would not make it visible in the legend, only odd-looking if it
    were ever unhidden.
    """
    lbl = "" if label is None else str(label)
    if not lbl or lbl == "_nolegend_" or not stem:
        return lbl
    if lbl.startswith("_"):
        return f"_{stem} — {lbl[1:]}"
    return f"{stem} — {lbl}"


def open_exports(paths, show: bool = True, options=None) -> Tuple:
    """
    Reopen several exported analyses overlaid on a single figure.

    Each file is rebuilt by its own registered rebuilder with ``show=False``
    -- so the curves are drawn by the same code that drew them originally --
    and the resulting data traces are then copied onto one combined figure.

    What travels, and what does not
    -------------------------------
    Data traces (Line2D) are carried with their drawn appearance, and their
    labels are qualified with the source file's stem ("sampleA — Ch1 ACF")
    because overlaid exports routinely carry identical trace names.  Reference lines (axhline/axvline) are dropped, since N
    overlaid copies of the same y=0 line are clutter; one is re-drawn if any
    source had one.  Shaded error bands, annotation boxes such as the linked
    parameter table, and residual panels are NOT carried: they cannot be
    composited faithfully, and a band silently detached from its curve would
    be worse than an honest omission.  Everything dropped is counted and
    reported in the returned notes.

    Data/fit pairing (``_norm_ref``, set by the fit plotters so the plot
    tools' Normalize scales a dataset by its fit rather than by its own
    noisy leading points) is re-established between the copies, so
    normalizing an overlay behaves as it does on an individual figure.

    Axis scales, labels and limits are taken from the FIRST file.  When a
    later file's axis labels disagree -- overlaying a correlation with a
    lifetime decay, say -- the overlay still happens and the disagreement is
    reported, because occasionally that is exactly what is wanted.

    Parameters
    ----------
    paths : sequence of str | Path
        CSVs written by this suite.  A single path is allowed and simply
        produces a one-trace overlay.
    show : bool
        Display the figure and its plot-controls panel.

    Returns
    -------
    (fig, ax, notes) where *notes* is a list of human-readable strings
    describing anything that was dropped or that disagreed.  An empty list
    means everything composited cleanly.

    Raises
    ------
    ValueError
        If *paths* is empty, or if no file yielded a usable trace.  Errors
        from individual files are collected into *notes* rather than raised,
        so one unreadable file cannot lose the whole overlay.
    """
    import matplotlib.pyplot as plt
    import fcs_plottools

    items = [Path(p) for p in paths]
    if not items:
        raise ValueError("No files given to overlay.")

    notes: list = []
    n_lines = 0
    ref_y: list = []
    first_axes_style = None
    # id() of a source line -> its copy, so _norm_ref pairing survives.
    copy_of: dict = {}
    pending_pairs: list = []

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    for p in items:
        try:
            src_fig, src_axes = open_export(p, show=False, options=options)
        except Exception as exc:            # noqa: BLE001 — collected below
            notes.append(f"{p.name}: skipped — {exc}")
            continue

        try:
            src_ax = _main_axes(src_fig, src_axes)
            if src_ax is None:
                notes.append(f"{p.name}: skipped — rebuilt figure has no axes.")
                continue

            # Caveats the rebuilder itself raised about this file.  Part of
            # the rebuilder contract rather than a special case here: this
            # module knows nothing about any analysis, and a lifetime fit
            # that did not start at its peak is exactly the sort of thing
            # only the owning module can notice.
            for _note in getattr(src_fig, "_fcs_rebuild_notes", None) or []:
                notes.append(str(_note))

            src_xlabel, src_ylabel = _axis_labels(src_fig, src_ax)
            if first_axes_style is None:
                first_axes_style = (
                    src_ax.get_xscale(), src_ax.get_yscale(),
                    src_xlabel, src_ylabel,
                )
            else:
                if (src_xlabel, src_ylabel) != first_axes_style[2:]:
                    notes.append(
                        f"{p.name}: axis labels differ from the first file "
                        f"(\"{src_xlabel}\" / "
                        f"\"{src_ylabel}\" vs "
                        f"\"{first_axes_style[2]}\" / "
                        f"\"{first_axes_style[3]}\") — overlaid anyway, so "
                        f"read the axes with that in mind."
                    )

            # Auxiliary artists that cannot be composited faithfully.
            n_coll = len(getattr(src_ax, "collections", []))
            if n_coll:
                notes.append(
                    f"{p.name}: {n_coll} shaded band(s)/collection(s) not "
                    f"carried over.")
            n_extra = max(0, len(src_fig.axes) - 1)
            if n_extra:
                notes.append(
                    f"{p.name}: {n_extra} extra panel(s) (residuals) not "
                    f"carried over.")
            n_text = len([t for t in src_ax.texts if (t.get_text() or "").strip()])
            if n_text:
                notes.append(
                    f"{p.name}: {n_text} annotation(s) not carried over.")

            for line in src_ax.get_lines():
                if _is_reference_line(line, src_ax):
                    ydata = line.get_ydata()
                    if len(ydata):
                        ref_y.append(float(ydata[0]))
                    continue
                props = {}
                for name in _LINE_PROPS:
                    getter = getattr(line, f"get_{name}", None)
                    if getter is not None:
                        try:
                            props[name] = getter()
                        except Exception:
                            pass
                # Labels are qualified with the source file's stem, since
                # overlaid exports routinely carry identical trace names.
                # The producer's own label is kept after the stem, including
                # the leading underscore that hides a fit's data points.
                new_line, = ax.plot(line.get_xdata(), line.get_ydata(),
                                    label=_qualified_label(line.get_label(),
                                                           p.stem),
                                    **props)
                copy_of[id(line)] = new_line
                ref = getattr(line, "_norm_ref", None)
                if ref is not None:
                    pending_pairs.append((new_line, id(ref)))
                n_lines += 1
        finally:
            # The rebuilt figure has served its purpose; leaving it open
            # would leak one window per file opened.
            try:
                plt.close(src_fig)
            except Exception:
                pass

    if not n_lines:
        plt.close(fig)
        detail = "\n  ".join(notes) if notes else "no traces found."
        raise ValueError(f"Nothing could be overlaid:\n  {detail}")

    # Re-establish data/fit pairing between the copies.
    for new_line, old_ref_id in pending_pairs:
        target = copy_of.get(old_ref_id)
        if target is not None:
            new_line._norm_ref = target

    xscale, yscale, xlabel, ylabel = first_axes_style
    try:
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
    except Exception:
        pass
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Overlay — {n_lines} trace"
                 f"{'s' if n_lines != 1 else ''} from {len(items)} file"
                 f"{'s' if len(items) != 1 else ''}")

    # One reference line for the whole overlay rather than N identical ones.
    if ref_y:
        ax.axhline(ref_y[0], color="grey", linewidth=0.6, linestyle="--",
                   zorder=0)

    ax.relim()
    ax.autoscale_view()
    try:
        fcs_plottools.adaptive_legend(ax)
    except Exception:
        pass

    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax, notes


def _peek_lifetime_x_axis(path) -> Optional[str]:
    """
    If *path* is a lifetime fit curve, the time axis recorded in it.

    Returns the recorded mode, or "" for a lifetime fit curve that records
    none, or None for anything else.

    Deliberately cheap and best-effort: it reads comment lines only and
    stops at the column header, because read_export parses the whole file
    and an intensity export can be a hundred thousand rows.  Its only
    consequence is whether the time-axis dialog appears, so a false
    negative costs a dialog and never a wrong plot -- the rebuilder reads
    the real header through read_export either way.
    """
    try:
        with Path(path).open("r", encoding="utf-8-sig") as fh:
            is_lifetime_fit = False
            recorded = ""
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("#"):
                    break
                body = line[1:].strip()
                if ":" in body:
                    key, val = body.split(":", 1)
                    key = key.strip().lower()
                    if key == "analysis" and val.strip().lower() == \
                            "lifetime fit curve":
                        is_lifetime_fit = True
                    elif key == "x_axis":
                        recorded = val.strip().lower()
                elif "—" in body:
                    if body.split("—", 1)[0].strip().lower() == \
                            "lifetime fit curve":
                        is_lifetime_fit = True
            return recorded if is_lifetime_fit else None
    except OSError:
        return None


def _ask_lifetime_x_axis(parent, n_files: int, n_total: int,
                         preselect: str) -> Optional[str]:
    """
    Ask which time axis to draw reopened lifetime fit curves on.

    The reopen path has no setup dialog, so without this a curve exported on
    an absolute axis could only ever be reopened on one -- and overlaying
    decays whose peaks sit at different arrival times needs the relative
    axis to line them up at all.

    Returns "absolute", "relative", or None if cancelled.
    """
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Lifetime fit curves — time axis")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="Time axis", font=("Helvetica", 12, "bold"),
             pady=6).pack()
    scope = (f"{n_files} of the {n_total} selected files are lifetime fit "
             f"curves." if n_files != n_total else
             ("This file is a lifetime fit curve." if n_total == 1 else
              f"All {n_total} selected files are lifetime fit curves."))
    tk.Label(win, text=scope, font=("Helvetica", 9), fg="grey").pack(padx=14)

    var = tk.StringVar(master=win, value=preselect)
    frame = tk.Frame(win, padx=16, pady=8)
    frame.pack(fill="x")
    tk.Radiobutton(frame, variable=var, value="absolute", anchor="w",
                   text="Arrival time within the laser cycle").pack(fill="x")
    tk.Radiobutton(frame, variable=var, value="relative", anchor="w",
                   text="Time since the start of the fit window").pack(fill="x")
    tk.Label(win,
             text=("      Relative lines several decays up when their peaks "
                   "sit at different\n      arrival times; on the absolute "
                   "axis they read as offset.\n"
                   "      Preselected from what the first file recorded."),
             font=("Helvetica", 8), fg="grey", anchor="w",
             justify="left").pack(fill="x", padx=4)

    choice = {"val": None}

    def _ok():
        choice["val"] = var.get()
        win.destroy()

    btns = tk.Frame(win)
    btns.pack(pady=10)
    tk.Button(btns, text="Open", width=10, command=_ok, pady=4
              ).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4
              ).pack(side="left", padx=6)

    win.wait_window()
    return choice["val"]


def run_open_dialog(parent=None, initialdir=None) -> Optional[Tuple]:
    """
    Ask for one or more exported CSVs and reopen them.

    Selecting a single file reopens it exactly as before, on its own
    rebuilder's own figure with every annotation and residual panel intact.
    Selecting several overlays them on one combined figure via
    :func:`open_exports`, which is a lossier composite -- so the single-file
    path is deliberately NOT routed through it.

    Returns the rebuilt (fig, ax), or None if the user cancelled.  Errors
    are reported in a message box rather than raised, so a mis-picked file
    cannot take the application down.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    paths = filedialog.askopenfilenames(
        parent=parent,
        title="Open exported plot(s) — select several to overlay",
        initialdir=str(initialdir) if initialdir else None,
        filetypes=[("FCS analysis export", "*.csv"), ("All files", "*.*")],
    )
    if not paths:
        return None
    paths = list(paths)

    # Lifetime fit curves are the one analysis with a display choice the
    # reopen path has to make, because the figure's time axis is fixed at
    # fit time and overlaying decays with different peak positions needs
    # the other one.
    options = None
    peeked = [_peek_lifetime_x_axis(p) for p in paths]
    n_lifetime = sum(1 for v in peeked if v is not None)
    if n_lifetime:
        preselect = next((v for v in peeked if v), "") or "absolute"
        choice = _ask_lifetime_x_axis(parent, n_lifetime, len(paths),
                                      preselect)
        if choice is None:
            return None
        options = {"x_axis": choice}

    try:
        if len(paths) == 1:
            return open_export(paths[0], show=True, options=options)

        fig, ax, notes = open_exports(paths, show=True, options=options)
        if notes:
            # Reported after the figure is up, so the overlay is on screen
            # to look at while reading what did not make it across.
            messagebox.showwarning(
                "Overlay — what was not carried over",
                "The overlay is built from the data traces of each file.\n\n"
                + "\n\n".join(notes),
                parent=parent)
        return fig, ax
    except Exception as exc:  # noqa: BLE001 — surfaced to the user below
        messagebox.showerror("Cannot reopen plot", str(exc), parent=parent)
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"Supported analyses: {', '.join(supported_analyses())}")
        print("\nUsage:  python fcs_plotopen.py <exported.csv> [more.csv ...]")
        sys.exit(0)
    if len(sys.argv) == 2:
        open_export(sys.argv[1], show=True)
    else:
        _fig, _ax, _notes = open_exports(sys.argv[1:], show=True)
        for _n in _notes:
            print(f"[overlay] {_n}")
