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

def fits_dir(source_path: Path, create: bool = True) -> Path:
    """
    Return (creating if needed) the shared 'fits' folder for a source file.

    Correlation CSVs normally live in ``<datadir>/analysis/``, so the fits
    folder is placed as a sibling: ``<datadir>/fits/``.  If the source is not
    inside an 'analysis' folder, 'fits' is created next to it instead.

    A source that already lives inside a ``fits/`` folder (e.g. re-fitting an
    exported curve) also resolves to that same shared ``fits/`` directory
    rather than creating ``fits/fits/``.  This guard previously existed only in
    fcs_lifetime_recon; it is now the behaviour everywhere.

    ``create=False`` works the path out WITHOUT touching the disk, for callers
    that only want to show a destination.  The dataset picker recomputes it on
    every click, and creating as it went would leave an empty 'fits' folder in
    every directory browsed past.  The path logic stays here either way: a
    second copy of these four lines somewhere else is how the displayed
    destination and the real one come to disagree.
    """
    source_path = Path(source_path)
    base = source_path.parent
    if base.name.lower() in ("analysis", "fits"):
        base = base.parent
    out = base / "fits"
    if create:
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


# ── Dialog windows ────────────────────────────────────────────────────────────
#
# Every dialog in the suite should be resizable and should scroll, because
# neither is optional in practice: a settings window grows an entry at a time
# until it is taller than a laptop screen, and a fixed-size window then puts its
# OK button somewhere the user cannot reach.  The matrix-method setup window hit
# exactly that at 1031 pixels tall.
#
# tkinter has no scrollable frame, so it is built the usual way: a Canvas with a
# Frame inside it, the Canvas scrolled by a Scrollbar.  The buttons are packed
# OUTSIDE the scrolling area so they stay put no matter how far the body
# scrolls -- a scrollbar alone would still let someone lose the OK button off
# the bottom.
#
# tkinter is imported inside these functions, not at module scope, for the same
# reason write_params_table imports fcs_export inside its body: the module's
# import graph stays free of GUI dependencies, and a headless caller that never
# opens a dialog never pays for one.


def scrollable_dialog(parent=None, title="", modal=True):
    """Create a resizable Toplevel whose body scrolls vertically.

    Returns
    -------
    (win, body, buttons)
        ``win``     the Toplevel.
        ``body``    pack content in here; it scrolls.
        ``buttons`` pinned to the bottom, always visible.

    Call :func:`fit_dialog_to_screen` once the body is filled, then
    ``win.wait_window()`` as usual.
    """
    import tkinter as tk

    win = tk.Toplevel(parent) if parent is not None else tk.Toplevel()
    win.title(title)
    win.resizable(True, True)
    if modal:
        try:
            win.grab_set()
        except Exception:      # noqa: BLE001 - a dialog that cannot grab is
            pass               # still a usable dialog

    # Buttons first: packing them to the bottom BEFORE the expanding body is
    # what reserves their space, so they are never squeezed out.
    buttons = tk.Frame(win)
    buttons.pack(side="bottom", fill="x")

    outer = tk.Frame(win)
    outer.pack(side="top", fill="both", expand=True)
    canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
    vbar = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vbar.set)
    vbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    body = tk.Frame(canvas)
    body_id = canvas.create_window((0, 0), window=body, anchor="nw")

    def _body_resized(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _canvas_resized(event):
        # Keep the body as wide as the canvas so content stretches instead of
        # leaving a dead strip beside the scrollbar.
        canvas.itemconfigure(body_id, width=event.width)

    body.bind("<Configure>", _body_resized)
    canvas.bind("<Configure>", _canvas_resized)

    def _wheel(event):
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        else:
            step = -1 if getattr(event, "delta", 0) > 0 else 1
        canvas.yview_scroll(step, "units")

    _SEQS = ("<MouseWheel>", "<Button-4>", "<Button-5>")

    def _grab_wheel(_event=None):
        for seq in _SEQS:
            canvas.bind_all(seq, _wheel)

    def _release_wheel(_event=None):
        for seq in _SEQS:
            try:
                canvas.unbind_all(seq)
            except Exception:          # noqa: BLE001 - teardown must not raise
                pass

    # Bound only while the pointer is over this window, and released again on
    # destroy, so a dialog cannot leave a global binding behind that scrolls it
    # after it has closed.
    win.bind("<Enter>", _grab_wheel)
    win.bind("<Leave>", _release_wheel)
    win.bind("<Destroy>",
             lambda e: _release_wheel() if e.widget is win else None)

    win._scroll_canvas = canvas          # read by fit_dialog_to_screen
    win._scroll_body = body
    return win, body, buttons


def fit_dialog_to_screen(win, max_fraction=0.85, min_width=360,
                         min_height=200):
    """Size *win* to its content, capped at a fraction of the screen.

    A window taller than the screen is the failure this exists to prevent, so
    the cap is the point: past it the body scrolls instead of growing.  The
    window is then centred horizontally and placed a third of the way down,
    which keeps it clear of a taskbar at the bottom.
    """
    win.update_idletasks()

    body = getattr(win, "_scroll_body", None)
    canvas = getattr(win, "_scroll_canvas", None)
    if body is not None and canvas is not None:
        # A Canvas has no natural size, so ask it for the body's.
        canvas.configure(width=body.winfo_reqwidth(),
                         height=body.winfo_reqheight())
        win.update_idletasks()

    want_w = max(win.winfo_reqwidth(), min_width)
    want_h = max(win.winfo_reqheight(), min_height)
    screen_w, screen_h = win.winfo_screenwidth(), win.winfo_screenheight()
    w = min(want_w, int(screen_w * max_fraction))
    h = min(want_h, int(screen_h * max_fraction))

    win.geometry("%dx%d+%d+%d" % (w, h, max((screen_w - w) // 2, 0),
                                  max((screen_h - h) // 3, 0)))
    win.minsize(min(min_width, w), min(min_height, h))
    return w, h


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

    # scrollable_dialog / fit_dialog_to_screen, when a display is available
    try:
        import tkinter as tk
        root = tk.Tk()
    except Exception as exc:               # noqa: BLE001
        print("  no display (%s); skipped the dialog checks"
              % type(exc).__name__)
    else:
        root.withdraw()
        try:
            win, body, buttons = scrollable_dialog(root, "test", modal=False)
            for i in range(60):            # taller than any screen
                tk.Label(body, text="row %d" % i).pack(fill="x")
            tk.Button(buttons, text="OK").pack()
            w, h = fit_dialog_to_screen(win)
            win.update_idletasks()

            assert win.resizable() == (True, True), "dialog must be resizable"
            sh = win.winfo_screenheight()
            assert h <= int(sh * 0.85) + 1, "height must be capped: %d" % h

            canvas = win._scroll_canvas
            region = [float(v) for v in canvas.cget("scrollregion").split()]
            assert region[3] > canvas.winfo_height(), \
                "the body must overflow, or nothing is being scrolled"
            bars = [c for c in canvas.master.winfo_children()
                    if c.winfo_class() == "Scrollbar"]
            assert len(bars) == 1, "expected one scrollbar"

            # Buttons live outside the scrolling area, so scrolling to the
            # bottom cannot take them with it.
            assert buttons.master is win
            assert buttons not in canvas.winfo_children()
            canvas.yview_moveto(1.0)
            win.update_idletasks()
            assert buttons.winfo_ismapped(), "buttons must stay visible"
            print("  scrollable dialog: %dx%d, capped from %d, scrollbar "
                  "present, buttons pinned" % (w, h, win.winfo_reqheight()))
            win.destroy()
        finally:
            root.destroy()

    print("fcs_fitcommon: all self-tests passed.")


if __name__ == "__main__":
    _selftest()
