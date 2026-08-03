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
``datetime``, ``math``) AT IMPORT TIME.  No numpy, no matplotlib, no tkinter,
and no ``fcs_*`` imports — so every fit module can import it with no risk of an
import cycle and no cost at start-up.

:func:`write_params_table` makes one narrow exception: it imports
``fcs_export.write_table_xlsx`` INSIDE the function body, not at module scope.
That keeps both properties the rule exists to protect — nothing is imported
until a fit is actually exported, and the import graph at module load is
unchanged — while letting all four fitters share one spreadsheet writer instead
of each wrapping their own.  If ``fcs_export`` is unavailable for any reason the
CSV is still written and only the ``.xlsx`` is skipped.

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
from typing import Optional


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


def slug_name(text: Optional[str]) -> str:
    """
    Reduce a user-supplied label to a filesystem-safe token.

    Letters, digits, '-' and '_' survive; everything else becomes '_', runs of
    underscores collapse, and the result is trimmed to 60 characters so a long
    label cannot push a path past the filesystem limit.  Returns "" for None or
    for text that contains nothing usable, which every caller treats as
    "no label given".
    """
    if not text:
        return ""
    out = []
    for ch in str(text).strip():
        out.append(ch if (ch.isalnum() or ch in "-_") else "_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:60]


def new_fit_dir(source_path: Path, name: Optional[str] = None) -> Path:
    """
    Create and return a fresh, uniquely-named subfolder inside the shared
    'fits' directory for ONE export.

    The folder name is the export timestamp, followed by the user's label if
    one was given::

        fits/2026-07-31_14-25-30/                 (no label)
        fits/2026-07-31_14-25-30_titration_5uM/   (label "titration 5uM")

    Timestamp FIRST so that a plain alphabetical listing is also chronological,
    which is what makes a fits/ folder with dozens of runs navigable.  The
    label is what makes a particular run findable within it.

    A numeric suffix is appended if two exports land in the same second.
    """
    fits = fits_dir(source_path)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = slug_name(name)
    base = f"{stamp}_{slug}" if slug else stamp
    out = fits / base
    n = 2
    while out.exists():                    # avoid same-second clashes
        out = fits / f"{base}_{n}"
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


# ── Parameter tables ──────────────────────────────────────────────────────────

def params_cell(v) -> str:
    """
    Render one value as a CSV cell for a parameter table.

    * Non-finite numbers (NaN, +/-inf) become an EMPTY cell.  NaN in these
      tables means "not estimated" -- a parameter held fixed, a background
      correction the source CSV did not carry, a count rate that was never
      recorded.  A blank reads as absence to a human, to pandas, and to
      :func:`fcs_export.read_export`, which maps it straight back to NaN.  The
      literal text "nan" reads as a value, and Excel treats it as text and then
      refuses to plot the column.
    * Finite numbers use 10 significant figures, matching fcs_export.
    * Anything else is passed through as text.
    * Cells containing a comma, quote or newline are quoted per RFC 4180, so a
      dataset name with a comma in it no longer silently shifts every later
      column in the row by one.

    Note that ``bool`` is a subclass of ``int`` and would render as 1/0; the
    fitters pass "yes"/"no" strings instead, which is what the readers expect.
    """
    if isinstance(v, str):
        cell = v
    else:
        try:
            f = float(v)
        except (TypeError, ValueError):
            cell = str(v)
        else:
            cell = "" if not math.isfinite(f) else f"{f:.10g}"
    if any(c in cell for c in (',', '"', '\n', '\r')):
        return '"' + cell.replace('"', '""') + '"'
    return cell


def write_params_table(
    csv_path: Path,
    comments: list,
    header: list,
    rows: list,
    *,
    log_tag: str = "fit",
    sheet_title: str = "fit parameters",
) -> tuple:
    """
    Write a fit's parameter table as a CSV plus a matching ``.xlsx`` mirror.

    Every fitter in the suite ends by writing the same artefact: a commented
    header block, one column-title row, then one row per fitted dataset with
    each parameter as a value + ``_err`` pair.  Four modules each had their own
    copy of that loop, and they had already drifted -- three wrote a blank cell
    for a non-finite value while fcs_fit wrote the text "nan", so the same
    missing quantity looked different depending on which fitter produced it.
    One definition removes that whole class of bug, exactly as this module did
    for ``fits_dir`` and ``new_fit_dir``.

    The ``.xlsx`` path is derived from *csv_path* rather than passed in: the
    two files are the same table in two formats and must not be allowed to
    drift apart in name any more than in content.

    Parameters
    ----------
    csv_path : Path
        Destination ``.csv``.  The spreadsheet is written alongside it with the
        same stem and a ``.xlsx`` suffix.
    comments : list of str
        Header block.  Written as ``# line`` in the CSV and as grey italics
        above the table in the spreadsheet.  Use DISTINCT keys for
        ``key : value`` lines -- fcs_export.read_export parses them into a dict,
        so four lines all called "note" collapse to one.
    header : list of str
        Column titles.
    rows : list of sequences
        Table body, one sequence per dataset.  Values are rendered by
        :func:`params_cell`.
    log_tag : str
        Prefix for the console messages, e.g. "globalfit" or "reconv fit".
    sheet_title : str
        Worksheet name.

    Returns
    -------
    (csv_path, xlsx_path or None)
        The spreadsheet is None when openpyxl is missing or the write failed;
        that never costs you the CSV, which is written first.
    """
    csv_path = Path(csv_path)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        for line in comments:
            fh.write(f"# {line}\n")
        fh.write(",".join(str(h) for h in header) + "\n")
        for row in rows:
            fh.write(",".join(params_cell(v) for v in row) + "\n")
    print(f"[{log_tag}] wrote {csv_path}")

    xlsx_path = csv_path.with_suffix(".xlsx")
    try:
        from fcs_export import write_table_xlsx
    except Exception as exc:   # noqa: BLE001 - the CSV must survive anything
        print(f"[{log_tag}] could not load the spreadsheet writer ({exc}); "
              f".csv is unaffected.")
        return csv_path, None

    written = write_table_xlsx(xlsx_path, comments, header, rows,
                               sheet_title=sheet_title, log_tag=log_tag)
    return csv_path, written


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

        # slug_name
        assert slug_name(None) == ""
        assert slug_name("   ") == ""
        assert slug_name("!!!") == ""
        assert slug_name("titration 5uM") == "titration_5uM"
        assert slug_name("a//b  c") == "a_b_c"
        assert slug_name("keep-me_1") == "keep-me_1"
        assert len(slug_name("x" * 200)) == 60

        # new_fit_dir folds the label in after the timestamp
        c = new_fit_dir(src, "titration 5uM")
        assert c.name.endswith("_titration_5uM"), c.name
        assert c.is_dir()
        stamp_part = c.name[: -len("_titration_5uM")]
        assert len(stamp_part) == len("2026-07-31_14-25-30"), stamp_part

        # a label that slugs to nothing behaves exactly like no label:
        # bare timestamp, plus only the numeric clash suffix.  (a is the bare
        # timestamp folder made above, so e collides with it and gets a suffix.)
        e = new_fit_dir(src, "***")
        assert e.name.startswith(a.name), e.name
        assert e.name[len(a.name):].lstrip("_").isdigit(), e.name

        # write_params_table: blanks, quoting, and the .xlsx sibling
        pt = root / "data" / "fits" / "t_params.csv"
        csvp, _xl = write_params_table(
            pt,
            ["Test table", "note_a : first", "note_b : second"],
            ["dataset", "tau", "tau_err", "n"],
            [["plain", 1.5, float("nan"), 12],
             ["has,comma", float("inf"), 2.0, 3]],
            log_tag="selftest")
        body = [l for l in csvp.read_text(encoding="utf-8").splitlines()
                if not l.startswith("#")]
        assert body[0] == "dataset,tau,tau_err,n", body[0]
        assert body[1] == "plain,1.5,,12", body[1]          # NaN -> blank
        assert body[2] == '"has,comma",,2,3', body[2]       # quoted; inf -> blank

    # params_cell
    assert params_cell(float("nan")) == ""
    assert params_cell(float("inf")) == ""
    assert params_cell(1.23456789012345) == "1.23456789"
    assert params_cell(12) == "12"
    assert params_cell("Ch1") == "Ch1"
    assert params_cell('say "hi"') == '"say ""hi"""'

    print("fcs_fitcommon: all self-tests passed.")


if __name__ == "__main__":
    _selftest()
