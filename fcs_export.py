"""
fcs_export.py
=============
Shared CSV export for the FCS analysis suite.

When the "Export plotted data to CSV" option is ticked in the main window,
each analysis writes the exact arrays it plots to a CSV file inside an
``analysis`` folder created alongside the source ``.fcs`` file.

File layout
-----------
Each export is a plain CSV with a commented metadata block on top::

    # FCS analysis export — intensity
    # source file : experiment.fcs
    # exported    : 2026-06-10T17:30:00
    # bin_width_s : 0.01
    time_s,ch1_counts,ch2_counts,ch1_cps,ch2_cps
    0,12,9,1200,900
    ...

The ``#`` lines are comments (skip them with ``pandas.read_csv(path,
comment='#')`` or ``numpy.loadtxt`` defaults).  The first non-comment line
is the real column header, followed by the data.

Filenames
---------
``<source-stem>_<analysis>[_<suffix>].csv``

The optional suffix captures the primary plot variant (e.g. lifetime bin
count, correlation type, PCH channel + bin width) so that two genuinely
different plots of the same file do not overwrite one another, while
re-running the *identical* plot overwrites its previous export.

Set ``TIMESTAMP_FILENAMES = True`` to instead keep every export by
appending a timestamp to the filename.

Public API
----------
    analysis_dir(d)                                  -> Path
    export_columns(d, analysis, columns, meta, ...)  -> Path   (raises)
    safe_export(d, analysis, columns, meta, ...)     -> Path | None
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from fcs_reader import FCSData


# If True, a timestamp is inserted into each filename so repeated exports are
# all preserved.  If False (default), re-running an analysis with the same
# settings overwrites that analysis's previous export for the same file.
TIMESTAMP_FILENAMES = False

_OUTPUT_DIRNAME = "analysis"


# ── Folder ────────────────────────────────────────────────────────────────────

def analysis_dir(d: FCSData) -> Path:
    """
    Return the ``analysis`` folder beside the source file, creating it
    (and any missing parents) if it does not already exist.
    """
    out = d.filepath.parent / _OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Filename helper ───────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    """Reduce arbitrary text to a filesystem-safe lower-case token."""
    out = []
    for ch in str(text).strip().lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "data"


# ── Core writer ───────────────────────────────────────────────────────────────

def export_columns(
    d: FCSData,
    analysis: str,
    columns: Mapping[str, np.ndarray],
    meta: Optional[Mapping[str, object]] = None,
    *,
    suffix: str = "",
) -> Path:
    """
    Write a set of named, equal-length columns to a CSV in the analysis folder.

    Parameters
    ----------
    d        : FCSData
        Source dataset — provides the output folder, filename stem and the
        file-name fields written into the header.
    analysis : str
        Short analysis name, e.g. ``"intensity"`` or ``"correlation"``.
    columns  : mapping of column-label -> 1-D array
        All arrays must share the same length; they become the CSV columns
        in insertion order.  The label is used verbatim as the column header.
    meta     : mapping, optional
        Extra ``key : value`` lines written into the commented header block
        (parameters, statistics, photon counts, …).
    suffix   : str, keyword-only
        Extra tag folded into the filename to distinguish plot variants.

    Returns
    -------
    Path to the written file.

    Raises
    ------
    ValueError
        If no columns are supplied or their lengths differ.
    """
    names = list(columns.keys())
    if not names:
        raise ValueError("No columns supplied to export.")

    arrays = [np.asarray(columns[k], dtype=np.float64).ravel() for k in names]
    n = len(arrays[0])
    for name, arr in zip(names, arrays):
        if len(arr) != n:
            raise ValueError(
                f"Column '{name}' has length {len(arr)}, expected {n}; "
                f"all export columns must be the same length."
            )

    # ── Build the output path ────────────────────────────────────────────────
    parts = [_slug(d.filepath.stem), _slug(analysis)]
    if suffix:
        parts.append(_slug(suffix))
    if TIMESTAMP_FILENAMES:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_path = analysis_dir(d) / ("_".join(parts) + ".csv")

    # ── Write commented header + real CSV header + data ──────────────────────
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(f"# FCS analysis export \u2014 {analysis}\n")
        fh.write(f"# source file : {d.filepath.name}\n")
        fh.write(f"# exported    : {datetime.now().isoformat(timespec='seconds')}\n")
        if meta:
            for key, val in meta.items():
                fh.write(f"# {key} : {val}\n")
        fh.write(",".join(names) + "\n")
        for row in zip(*arrays):
            fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

    return out_path


def safe_export(
    d: FCSData,
    analysis: str,
    columns: Mapping[str, np.ndarray],
    meta: Optional[Mapping[str, object]] = None,
    *,
    suffix: str = "",
) -> Optional[Path]:
    """
    Like :func:`export_columns` but never raises.

    A failed export (e.g. a read-only directory) prints a warning and
    returns ``None`` so the plot still appears.  On success the path is
    printed and returned.
    """
    try:
        path = export_columns(d, analysis, columns, meta=meta, suffix=suffix)
        print(f"[export] wrote {path}")
        return path
    except Exception as exc:  # noqa: BLE001 — export must never break a plot
        print(f"[export] FAILED for {analysis}: {exc}", file=sys.stderr)
        return None
