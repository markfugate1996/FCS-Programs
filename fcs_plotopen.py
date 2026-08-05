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

Public API
----------
    supported_analyses()                  -> list[str]
    register_rebuilder(name, mod, func)   -> None
    open_export(path, show=True)          -> (fig, ax)
    run_open_dialog(parent=None)          -> (fig, ax) | None
"""

from __future__ import annotations

import importlib
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


def open_export(path, show: bool = True) -> Tuple:
    """
    Reopen an exported analysis CSV as a live figure.

    Parameters
    ----------
    path : str | Path
        A CSV written by this suite.
    show : bool
        Display the figure and its plot-controls panel.

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

    return _resolve(analysis)(meta, columns, show=show, path=p)


def run_open_dialog(parent=None, initialdir=None) -> Optional[Tuple]:
    """
    Ask for an exported CSV and reopen it.

    Returns the rebuilt (fig, ax), or None if the user cancelled.  Errors are
    reported in a message box rather than raised, so a mis-picked file cannot
    take the application down.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    path = filedialog.askopenfilename(
        parent=parent,
        title="Open exported plot",
        initialdir=str(initialdir) if initialdir else None,
        filetypes=[("FCS analysis export", "*.csv"), ("All files", "*.*")],
    )
    if not path:
        return None

    try:
        return open_export(path, show=True)
    except Exception as exc:  # noqa: BLE001 — surfaced to the user below
        messagebox.showerror("Cannot reopen plot", str(exc), parent=parent)
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"Supported analyses: {', '.join(supported_analyses())}")
        print("\nUsage:  python fcs_plotopen.py <exported.csv>")
        sys.exit(0)
    open_export(sys.argv[1], show=True)
