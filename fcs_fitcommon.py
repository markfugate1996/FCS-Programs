"""
fcs_fitcommon.py
================
Helpers shared by every fitting module in the suite:

    fcs_fit.py            (correlation / global fit)
    fcs_lifetime_fit.py   (lifetime tail fit)
    fcs_lifetime_recon.py (lifetime IRF-reconvolution fit)
    fcs_pch_fit.py        (photon-counting histogram fit)

Before this module existed each of the four carried its own copy of
``_fits_dir``, ``_new_fit_dir``, ``_fmt`` and ``_parse_bound``.  The copies
drifted: only ``fcs_lifetime_recon`` guarded against re-fitting a file that
already lives inside a ``fits/`` folder, so the other three would nest
``fits/fits/``.  A single definition removes that whole class of bug.

Scope and dependencies
----------------------
Deliberately depends on nothing but the standard library (``pathlib``,
``datetime``, ``math``).  No numpy, no matplotlib, no tkinter, and no ``fcs_*``
imports — so every fit module can import it with no risk of an import cycle and
no cost at start-up.

Note on ``math.inf`` vs ``np.inf``: they are the same float value
(``math.inf == np.inf`` and ``np.isinf(math.inf)`` is True), so callers that
compare against or store ``np.inf`` are unaffected.

Import convention
-----------------
The functions are public (no leading underscore) because they cross module
boundaries.  To avoid touching call sites, each consumer aliases them back to
its existing private names:

    from fcs_fitcommon import (
        fits_dir as _fits_dir,
        new_fit_dir as _new_fit_dir,
        fmt_bound as _fmt,
        parse_bound as _parse_bound,
    )
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path


# ── Output directories ────────────────────────────────────────────────────────

def fits_dir(source_path: Path) -> Path:
    """
    Return (creating if needed) the shared 'fits' folder for a source file.

    Correlation CSVs normally live in ``<datadir>/analysis/``, so the fits
    folder is placed as a sibling: ``<datadir>/fits/``.  If the source is not
    inside an 'analysis' folder, 'fits' is created next to it instead.

    A source that already lives inside a ``fits/`` folder (e.g. re-fitting an
    exported curve) also resolves to that same shared ``fits/`` directory
    rather than creating ``fits/fits/``.  This guard previously existed only in
    fcs_lifetime_recon; it is now the behaviour everywhere.
    """
    source_path = Path(source_path)
    base = source_path.parent
    if base.name.lower() in ("analysis", "fits"):
        base = base.parent
    out = base / "fits"
    out.mkdir(parents=True, exist_ok=True)
    return out


def new_fit_dir(source_path: Path) -> Path:
    """
    Create and return a fresh, uniquely-named subfolder inside the shared
    'fits' directory for ONE export.

    The folder name is just the export timestamp; sample, model and parameter
    detail lives inside the files.  A numeric suffix is appended if two exports
    land in the same second.
    """
    fits = fits_dir(source_path)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = fits / stamp
    n = 2
    while out.exists():                    # avoid same-second clashes
        out = fits / f"{stamp}_{n}"
        n += 1
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── Parameter-bound display and entry ─────────────────────────────────────────

def fmt_bound(x: float) -> str:
    """
    Format a parameter bound for display in a dialog entry box.

    Infinities render as 'inf' / '-inf' so they round-trip through
    :func:`parse_bound`; everything else uses 4 significant figures.
    """
    if x == math.inf:
        return "inf"
    if x == -math.inf:
        return "-inf"
    return f"{x:.4g}"


def parse_bound(text: str, default: float) -> float:
    """
    Parse a bound typed into a dialog entry, accepting blank, 'inf', '-inf'.

    A blank entry returns ``default`` (the caller's sentinel for "unset").
    Raises ``ValueError`` on anything else that is not a number, which the
    dialogs catch to show an "Invalid input" message.
    """
    t = text.strip().lower()
    if t in ("", "inf", "+inf", "infinity"):
        return math.inf if t != "" else default
    if t in ("-inf", "-infinity"):
        return -math.inf
    return float(t)


# ── Self-test ─────────────────────────────────────────────────────────────────

def _selftest() -> None:
    """Round-trip and directory checks.  Run:  python fcs_fitcommon.py"""
    import tempfile

    # fmt_bound / parse_bound round-trip
    assert fmt_bound(math.inf) == "inf"
    assert fmt_bound(-math.inf) == "-inf"
    assert fmt_bound(0.000123456) == "0.0001235"
    assert parse_bound("inf", 0.0) == math.inf
    assert parse_bound("-inf", 0.0) == -math.inf
    assert parse_bound("", 7.5) == 7.5
    assert parse_bound(" 1E-9 ", 0.0) == 1e-9
    assert parse_bound(fmt_bound(math.inf), 0.0) == math.inf
    try:
        parse_bound("abc", 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("parse_bound should raise on non-numeric text")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # source beside data  ->  <datadir>/fits/
        (root / "data").mkdir()
        src = root / "data" / "G1.fcs"
        src.touch()
        assert fits_dir(src) == root / "data" / "fits"

        # source in analysis/ ->  sibling <datadir>/fits/
        (root / "data" / "analysis").mkdir()
        src2 = root / "data" / "analysis" / "g1_correlation_cross.csv"
        src2.touch()
        assert fits_dir(src2) == root / "data" / "fits"

        # source already inside fits/ -> same fits/, NOT fits/fits/
        src3 = root / "data" / "fits" / "curve.csv"
        src3.touch()
        assert fits_dir(src3) == root / "data" / "fits", "fits/fits/ regression"

        # new_fit_dir is unique even within the same second
        a = new_fit_dir(src)
        b = new_fit_dir(src)
        assert a != b and a.is_dir() and b.is_dir()
        assert a.parent == b.parent == root / "data" / "fits"

    print("fcs_fitcommon: all self-tests passed.")


if __name__ == "__main__":
    _selftest()
