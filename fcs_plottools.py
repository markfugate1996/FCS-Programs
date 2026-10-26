"""
fcs_plottools.py
================
Post-plot interactive controls panel for the FCS analysis suite.

Provides a single public function::

    show_figure(fig, axes=None)

Drop-in replacement for ``plt.show()`` at the end of every ``plot_*``
function.  After the figure is rendered, a small Tkinter panel appears
beside it offering:

  * **Axis scale toggles** — flip any axis between Linear and Log.
  * **Axis limits** — manually set X/Y bounds, or reset to auto.
  * **Legend position** — dropdown to move the legend.
  * **Label editor** — edit title, X label, and Y label.
  * **Export** — save figure as PNG, PDF, or SVG.

Multi-axes figures (e.g. the two-panel fit plot with G(τ) + residuals)
are handled automatically:

  * X-scale and X-limits are applied to all axes (sharex panels stay in sync).
  * Y-scale, Y-limits, and legend are applied to the main (first) axis only.
  * Title and Y-label come from the first axis; X-label from the last (bottom).

Graceful fallback
-----------------
If the Tk figure manager is not available (headless CI, non-Tk backend)
the function falls back to a plain ``plt.show()`` with no panel.

Usage
-----
In each plot function, replace::

    if show:
        plt.show()

with::

    if show:
        fcs_plottools.show_figure(fig, ax)

For multi-axes figures pass the axes array::

    if show:
        fcs_plottools.show_figure(fig, np.array([ax_main, ax_resid]))
"""

from __future__ import annotations

import re

import numpy as np
import matplotlib.pyplot as plt


# ── Public API ────────────────────────────────────────────────────────────────

def palette(n: int) -> list:
    """
    Return *n* visually distinct colours for overlaying multiple datasets.

    Up to 10 datasets use matplotlib's categorical ``tab10`` map (highly
    distinguishable); beyond that a continuous map is sampled so arbitrarily
    many files still get unique colours.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = max(1, int(n))
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return [cmap(i) for i in range(n)]
    cmap = plt.get_cmap("turbo")
    return [cmap(x) for x in np.linspace(0.05, 0.95, n)]


# Longest figure name carried into a window title.  Beyond this the title
# bar truncates anyway, and on Windows an over-long title squeezes out the
# figure number -- the one part that has to stay readable, since it is what
# ties a panel to its figure when two titles are near-identical.
_TITLE_MAX_CHARS = 44


def _figure_name(fig, axes_list) -> str:
    """
    A short human name for *fig*, taken from its main axes title.

    Producers already title every figure ("Lifetime global fit — 2-exp
    · 5 datasets", "PCH global fit — ...").  Reading that title here keeps
    the naming automatic: no producer passes a name, and a new plotting
    routine is named correctly the day it is written.

    Titles are frequently two-line (``f"{title}\\n{subtitle}"``), so only
    the first line is used, collapsed and truncated.  Returns "" when the
    figure has no title, in which case callers fall back to the bare
    figure number.
    """
    try:
        raw = (axes_list[0].get_title() or "").strip()
    except Exception:
        return ""
    if not raw:
        return ""
    first = raw.splitlines()[0]
    first = " ".join(first.split())          # collapse runs of whitespace
    if len(first) > _TITLE_MAX_CHARS:
        first = first[:_TITLE_MAX_CHARS - 1].rstrip() + "…"
    return first


def show_figure(
    fig: plt.Figure,
    axes=None,
) -> None:
    """
    Display *fig* and attach a live-edit controls panel beside it.

    The figure window and its panel are named after the figure's own title
    and share its number ("Figure 3 — Lifetime global fit", "Plot controls
    — Figure 3 — Lifetime global fit"), and the panel is marked transient
    for the figure window, so with several figures open it is always
    apparent which panel drives which figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to display.
    axes : Axes, array of Axes, or None
        The axes to control.  If *None*, ``fig.axes`` is used.
        For subplot figures pass the full list so scale changes propagate
        to all panels.
    """
    import matplotlib.pyplot as plt

    # Normalise axes to a plain Python list
    if axes is None:
        axes_list = list(fig.axes)
    elif hasattr(axes, '__iter__') and not hasattr(axes, 'get_xlim'):
        axes_list = list(axes)
    else:
        axes_list = [axes]

    if not axes_list:
        plt.show()
        return

    # Try to get the Tk window that backs this figure
    try:
        tk_win = fig.canvas.manager.window
    except AttributeError:
        # Non-Tk backend or headless environment — degrade gracefully
        plt.show()
        return

    import tkinter as tk

    # ── Name both windows after the figure ────────────────────────────────
    # Default titles are matplotlib's "Figure 1/2/3..." against a generic
    # "Plot controls", which leaves nothing on screen connecting a panel to
    # its figure once more than one is open.  Both now carry the figure
    # number and its title, so the pairing is readable at a glance and in
    # the taskbar.
    _fig_no = getattr(fig, "number", None)
    _stem = f"Figure {_fig_no}" if _fig_no is not None else "Figure"
    _name = _figure_name(fig, axes_list)
    _label = f"{_stem} — {_name}" if _name else _stem
    try:
        fig.canvas.manager.set_window_title(_label)
    except Exception:
        pass          # non-Tk manager, or one without a settable title

    # ── Build the controls panel ──────────────────────────────────────────
    panel = tk.Toplevel(tk_win)
    panel.title(f"Plot controls — {_label}")
    panel.resizable(False, True)

    # Tie the panel to its figure at the window-manager level.  A transient
    # window stays above its master, minimises and restores with it, and
    # drops its own taskbar button -- so a panel can never be buried under
    # an unrelated figure, and minimising a figure no longer leaves an
    # orphaned panel behind.  Guarded: on some non-Windows managers
    # transient windows lose their minimise control, and a failure here
    # must not cost the panel itself.
    try:
        panel.transient(tk_win)
    except Exception:
        pass

    # The sections stack to well over a laptop screen's height once Traces is
    # added, so they live in a scrollable body rather than directly on the
    # panel.  Without this the Export button falls off the bottom and cannot
    # be reached at all.
    _canvas = tk.Canvas(panel, borderwidth=0, highlightthickness=0)
    _vsb    = tk.Scrollbar(panel, orient="vertical", command=_canvas.yview)
    _canvas.configure(yscrollcommand=_vsb.set)
    _vsb.pack(side="right", fill="y")
    _canvas.pack(side="left", fill="both", expand=True)

    body     = tk.Frame(_canvas)
    _body_id = _canvas.create_window((0, 0), window=body, anchor="nw")

    def _on_body_configure(_e=None):
        _canvas.configure(scrollregion=_canvas.bbox("all"))

    def _on_canvas_configure(e):
        # Keep the body as wide as the canvas so nothing scrolls sideways.
        _canvas.itemconfigure(_body_id, width=e.width)

    body.bind("<Configure>", _on_body_configure)
    _canvas.bind("<Configure>", _on_canvas_configure)

    # Mouse wheel, bound only while the pointer is over the panel so it does
    # not hijack scrolling in the figure window or anywhere else.
    def _on_wheel(event):
        if getattr(event, "num", None) == 4:
            _canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            _canvas.yview_scroll(1, "units")
        else:
            _canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _bind_wheel(_e=None):
        _canvas.bind_all("<MouseWheel>", _on_wheel)
        _canvas.bind_all("<Button-4>", _on_wheel)
        _canvas.bind_all("<Button-5>", _on_wheel)

    def _unbind_wheel(_e=None):
        _canvas.unbind_all("<MouseWheel>")
        _canvas.unbind_all("<Button-4>")
        _canvas.unbind_all("<Button-5>")

    panel.bind("<Enter>", _bind_wheel)
    panel.bind("<Leave>", _unbind_wheel)

    # Traces is built first (it is what people reach for) but the legend
    # rebuild it needs does not exist until the Legend section is built, so
    # the callback is resolved lazily through this holder.
    _legend_cb = {"fn": None}

    def _refresh_legend():
        fn = _legend_cb["fn"]
        if fn is not None:
            fn()

    _build_traces_section(body, fig, axes_list, on_change=_refresh_legend)
    _build_scale_section(body, fig, axes_list)
    _build_normalize_section(body, fig, axes_list)
    _build_limits_section(body, fig, axes_list)
    _build_grid_section(body, fig, axes_list)
    _legend_cb["fn"] = _build_legend_section(body, fig, axes_list)
    _build_annotation_section(body, fig, axes_list)
    _build_label_section(body, fig, axes_list)
    _build_save_section(body, fig)

    # ── Size to content, clamped to the screen ───────────────────────────────
    panel.update_idletasks()
    _panel_w = max(300, body.winfo_reqwidth() + _vsb.winfo_reqwidth() + 6)
    _content_h = body.winfo_reqheight() + 8
    _panel_h = min(_content_h,
                   panel.winfo_screenheight() - _SCREEN_MARGIN
                   - _SCREEN_MARGIN_BOTTOM)
    panel.geometry(f"{_panel_w}x{_panel_h}")
    panel.minsize(_panel_w, min(_content_h, 320))

    # ── Position both windows fully on screen (deferred) ──────────────────
    def _position_panel():
        try:
            # The figure window first: matplotlib leaves its placement to the
            # window manager, which can open it partly below the desktop and
            # take the navigation toolbar with it.  The panel is then placed
            # relative to wherever the figure actually ended up.
            place_figure_window(fig)
            tk_win.update_idletasks()
            fx = tk_win.winfo_x()
            fy = tk_win.winfo_y()
            fw = tk_win.winfo_width()
            sw = tk_win.winfo_screenwidth()

            px = fx + fw + 6
            if px + _panel_w > sw - _SCREEN_MARGIN:
                # No room to the right; try the left, else overlap the figure
                # rather than run off the edge.
                px = fx - _panel_w - 6
                if px < _SCREEN_MARGIN:
                    px = max(_SCREEN_MARGIN, sw - _panel_w - _SCREEN_MARGIN)
            # place_window clamps y as well, so a panel taller than the space
            # below the figure is pulled up instead of hanging off the bottom.
            place_window(panel, x=px, y=fy, w=_panel_w, h=_panel_h)
            panel.lift()
        except Exception:
            pass

    panel.after(150, _position_panel)

    # ── Mutual close-on-close ─────────────────────────────────────────────
    def _on_fig_close(_event=None):
        try:
            panel.destroy()
        except Exception:
            pass

    fig.canvas.mpl_connect("close_event", _on_fig_close)
    panel.protocol("WM_DELETE_WINDOW", panel.destroy)

    plt.show(block=False)

    # plt.show() on a never-before-shown figure only deiconifies the window;
    # it does NOT schedule a draw — it relies on the <Configure> event from
    # deiconification to trigger resize() -> draw_idle().  When show_figure is
    # called from inside a nested Tk event loop (e.g. a fit dialog's
    # wait_window), that <Configure> event stays queued until the nested loop
    # unwinds, so the canvas stays blank.  draw() here is synchronous and
    # guarantees the figure is rendered regardless of event-loop depth.
    try:
        fig.canvas.draw()
    except Exception:
        pass


# ── Figure export ─────────────────────────────────────────────────────────────

# Default format for saved figures.  SVG is vector: it stays sharp at any
# magnification, survives a journal's resizing, and can be opened in
# Illustrator or Inkscape to adjust a label.  A PNG can do none of that, and
# re-running an analysis to regenerate a figure at a different size is a poor
# substitute.  PNG and PDF remain available in the save dialog.
_FIGURE_FORMAT = "svg"

# Raster resolution.  Ignored for svg/pdf, which have no pixel grid.
_FIGURE_DPI = 150

# Keep SVG text as real <text> elements rather than converting it to outlines.
#
# Matplotlib defaults to svg.fonttype='path', which traces every glyph as a
# vector path: the file renders identically anywhere, but the text can no
# longer be selected or edited, and the file is roughly 40% larger.  With
# 'none' the text stays editable downstream -- normally the whole reason for
# exporting SVG rather than PNG -- at the cost of needing the font present on
# the machine that opens it (matplotlib's DejaVu Sans, or a substitute).
_SVG_EDITABLE_TEXT = True

_RASTER_FORMATS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def save_figure(fig, path, *, fmt: str = None, dpi: int = None,
                tight: bool = True):
    """
    Save *fig*, defaulting to vector SVG.

    *path* may be given with or without a suffix; when it has none, or when
    *fmt* is supplied, the format decides the extension.  DPI is applied only
    to raster formats, where it means something.

    Returns the Path written, or None on failure -- a figure that cannot be
    saved must never take down the analysis that produced it.
    """
    from pathlib import Path as _P
    import matplotlib as _mpl

    p = _P(path)
    ext = (f".{fmt.lstrip('.')}" if fmt
           else (p.suffix if p.suffix else f".{_FIGURE_FORMAT}"))
    p = p.with_suffix(ext)

    kw = {}
    if tight:
        kw["bbox_inches"] = "tight"
    if ext.lower() in _RASTER_FORMATS:
        kw["dpi"] = dpi if dpi is not None else _FIGURE_DPI

    # Scope the rcParam change to this call so it cannot leak into any other
    # figure the session goes on to draw.
    old_ft = _mpl.rcParams.get("svg.fonttype")
    try:
        if ext.lower() == ".svg" and _SVG_EDITABLE_TEXT:
            _mpl.rcParams["svg.fonttype"] = "none"
        fig.savefig(p, **kw)
        return p
    except Exception as exc:
        print(f"[plot] could not save figure ({exc})")
        return None
    finally:
        if old_ft is not None:
            _mpl.rcParams["svg.fonttype"] = old_ft


# ── Window placement ──────────────────────────────────────────────────────────

# Keep this much of the screen clear at the bottom for a taskbar / dock, so a
# window placed flush against the screen edge does not hide its own buttons
# behind it.
_SCREEN_MARGIN_BOTTOM = 64
_SCREEN_MARGIN = 8


def place_window(win, x=None, y=None, w=None, h=None) -> tuple:
    """
    Position *win* so that all of it is on screen.

    Windows that open partly off-screen are not merely untidy: a dialog whose
    lower edge falls below the desktop takes its OK button with it, and there
    is no way to reach it without moving the window first.  This clamps the
    requested geometry to the visible desktop and shrinks the window if it is
    larger than the screen.

    Any of x, y, w, h may be None, in which are taken from the window itself.

    Returns the (x, y, w, h) actually applied.
    """
    try:
        win.update_idletasks()
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        w = int(w if w is not None else (win.winfo_width() or
                                         win.winfo_reqwidth()))
        h = int(h if h is not None else (win.winfo_height() or
                                         win.winfo_reqheight()))
        x = int(x if x is not None else win.winfo_x())
        y = int(y if y is not None else win.winfo_y())

        # A window taller or wider than the desktop can never be fully
        # reachable, so shrink it rather than leave part of it inaccessible.
        avail_h = sh - _SCREEN_MARGIN - _SCREEN_MARGIN_BOTTOM
        w = max(200, min(w, sw - 2 * _SCREEN_MARGIN))
        h = max(150, min(h, avail_h))

        x = max(_SCREEN_MARGIN, min(x, sw - w - _SCREEN_MARGIN))
        y = max(_SCREEN_MARGIN, min(y, sh - h - _SCREEN_MARGIN_BOTTOM))

        win.geometry(f"{w}x{h}+{x}+{y}")
        return x, y, w, h
    except Exception:
        return (0, 0, 0, 0)


def place_figure_window(fig, x=None, y=None) -> None:
    """
    Bring a matplotlib figure window fully on screen.

    Matplotlib leaves placement to the window manager, which on Windows
    happily opens a figure with its lower portion below the desktop -- taking
    the navigation toolbar with it on some styles.
    """
    try:
        mgr = fig.canvas.manager
        win = getattr(mgr, "window", None)
        if win is None or not hasattr(win, "winfo_screenwidth"):
            return
        place_window(win, x=x, y=y)
    except Exception:
        pass


# ── Adaptive legends ──────────────────────────────────────────────────────────

# Above this many entries the legend is shrunk and split into columns.
_LEGEND_SHRINK_ABOVE = 10
# Above this many it is built but hidden, since at that point it covers more
# of the plot than it explains.  The Legend section of the controls panel can
# switch it back on.
_LEGEND_HIDE_ABOVE = 24


def adaptive_legend(ax, handles=None, labels=None, *,
                    base_fontsize: int = 10, **kw):
    """
    Add a legend whose size is scaled to the number of entries.

    A twenty-file overlay with a fixed 9 pt single-column legend produces a
    legend box TALLER than the axes it sits in -- measured at 409 px against
    316 px of plot -- so it hides the very data it is labelling.  Entry count
    is known only at draw time, so the sizing has to be decided here rather
    than fixed at the call site.

    Above ``_LEGEND_HIDE_ABOVE`` entries the legend is created but hidden.
    Creating it (rather than skipping it) matters: the Legend section of the
    plot-controls panel keys off ``ax.get_legend()``, so a legend that was
    never created could not be switched back on.

    Returns the Legend, or None when there is nothing to label.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    n = len(handles)
    if n <= _LEGEND_SHRINK_ABOVE:
        fontsize, ncol = base_fontsize, 1
    elif n <= _LEGEND_HIDE_ABOVE:
        fontsize = max(6, base_fontsize - 3)
        ncol = 2 if n <= 16 else 3
    else:
        fontsize = max(6, base_fontsize - 4)
        ncol = 3

    kw.setdefault("framealpha", 0.85)
    leg = ax.legend(handles, labels, fontsize=fontsize, ncol=ncol, **kw)
    if leg is not None and n > _LEGEND_HIDE_ABOVE:
        leg.set_visible(False)
    return leg


# ── Section builders ──────────────────────────────────────────────────────────

# ── Trace styling ─────────────────────────────────────────────────────────────

# (display name, matplotlib value)
_MARKER_CHOICES = [
    ("none",       "None"), ("point  .",   "."), ("circle  o",  "o"),
    ("square  s",  "s"),    ("triangle ^", "^"), ("diamond  D", "D"),
    ("plus  +",    "+"),    ("cross  x",   "x"), ("star  *",    "*"),
    ("down  v",    "v"),
]
_LINESTYLE_CHOICES = [
    ("none",     "None"), ("solid",  "-"),
    ("dashed",   "--"),   ("dashdot", "-."), ("dotted", ":"),
]

# Labels matplotlib generates for itself; these carry no meaning for a user.
_AUTO_LABEL = re.compile(r"^_+(child|line|collection|nolegend)[_\d]*$")

# Visibility glyphs for the trace selector.
_DOT_SHOWN  = "\u25cf"
_DOT_HIDDEN = "\u25cb"


def _artist_description(art, index: int) -> str:
    """
    Describe an artist that carries no usable label.

    Reference lines (a horizontal axhline, a vertical axvline) are by far the
    most common unlabelled artists in this suite, so they are named for what
    they are rather than by position in a list.
    """
    try:
        xd = np.asarray(art.get_xdata(), dtype=float)
        yd = np.asarray(art.get_ydata(), dtype=float)
        if yd.size and np.allclose(yd, yd[0]):
            return f"reference line (y = {yd[0]:.4g})"
        if xd.size and np.allclose(xd, xd[0]):
            return f"reference line (x = {xd[0]:.4g})"
    except Exception:
        pass
    return f"unnamed trace {index}"


def collect_artists(axes_list: list) -> list:
    """
    Enumerate the styleable artists across *axes_list*, with display names.

    The display name is the artist's own matplotlib label wherever there is
    one -- which for a correlation overlay is the source FILENAME, and for a
    fit is the dataset name.  Naming traces after the data they hold is the
    whole point: a selector that reads "Trace 1, Trace 2, Trace 3" forces you
    to click each one to find out what it is.

    A label starting with "_" is hidden from matplotlib legends by convention.
    Such a trace is still a real trace, so it is listed here with the
    underscore stripped -- unless the label is one matplotlib auto-generated
    (``_child0``), which means nothing and is replaced by a description.

    Returns a list of dicts: ``{artist, name, axis, is_line}``.
    """
    from matplotlib.lines import Line2D  # noqa: F401 — kept for clarity

    entries: list = []
    n_axes = len(axes_list)
    for ai, ax in enumerate(axes_list):
        # An errorbar puts its label on the CONTAINER, not on the artists it
        # is made of, so the data line, the caps and the bars all arrive here
        # unlabelled.  Map each member back to its container's label first,
        # otherwise a plot built from errorbars shows nothing but "unnamed
        # trace N" -- exactly the failure this selector exists to avoid.
        owned: dict = {}
        for cont in list(getattr(ax, "containers", [])):
            try:
                clbl = cont.get_label() or ""
            except Exception:
                clbl = ""
            # A leading underscore means "keep out of the legend", not
            # "unnamed" -- the same convention individual artists follow
            # below, where it is stripped rather than treated as absent.
            # Skipping such containers here would leave every member of an
            # underscore-labelled errorbar anonymous, which is exactly what
            # happened to the calibration deviation panel.
            if not clbl or _AUTO_LABEL.match(clbl):
                continue
            try:
                kids = list(cont.get_children())
            except Exception:
                kids = [k for k in cont if k is not None]
            for pos, kid in enumerate(kids):
                if kid is None:
                    continue
                # First member is the data series itself; the rest are the
                # caps and bars, which are worth naming but not confusing
                # with it.
                owned[id(kid)] = clbl if pos == 0 else f"{clbl}  (error bars)"

        pairs = [(a, True) for a in ax.get_lines()]
        pairs += [(a, False) for a in list(getattr(ax, "collections", []))]
        pairs += [(a, False) for a in list(getattr(ax, "patches", []))]
        for i, (art, is_line) in enumerate(pairs, start=1):
            try:
                lbl = art.get_label() or ""
            except Exception:
                lbl = ""
            if (not lbl or lbl.startswith("_")) and id(art) in owned:
                lbl = owned[id(art)]
            if lbl and not lbl.startswith("_"):
                name = lbl
            elif lbl and not _AUTO_LABEL.match(lbl):
                name = lbl.lstrip("_")
            elif is_line:
                name = _artist_description(art, i)
            else:
                name = f"unnamed shading {i}"
            # Only tag the panel index when there is more than one panel,
            # e.g. the fit plot's G(tau) + residuals pair.
            if n_axes > 1:
                name = f"[{ai + 1}] {name}"
            entries.append({"artist": art, "name": name,
                            "axis": ax, "is_line": is_line})
    return entries


def _build_traces_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
    on_change=None,
) -> None:
    """
    Add the "Traces" LabelFrame to *panel*.

    Layout is a selector plus ONE control block that retargets to whichever
    trace is selected, rather than a row of controls per trace.  An overlay of
    twenty files would otherwise produce a panel taller than the screen.

    Visibility is shown in the list itself (● shown, ○ hidden) so the state of
    every trace is readable at a glance without clicking through them.

    *on_change* is called after any edit that the legend needs to know about
    (colour, marker, line style, width, visibility, label).
    """
    import tkinter as tk
    from tkinter import colorchooser
    from matplotlib.colors import to_hex, to_rgba

    entries = collect_artists(axes_list)
    if not entries:
        return

    tf = tk.LabelFrame(panel, text="Traces", padx=10, pady=6)
    tf.pack(fill="x", padx=10, pady=(10, 4))

    # ── Selector ─────────────────────────────────────────────────────────────
    list_row = tk.Frame(tf)
    list_row.pack(fill="x")
    lb_sb = tk.Scrollbar(list_row, orient="vertical")
    lb = tk.Listbox(
        list_row, height=min(8, max(3, len(entries))), activestyle="none",
        font=("Courier", 9), exportselection=False, yscrollcommand=lb_sb.set,
    )
    lb_sb.config(command=lb.yview)
    lb_sb.pack(side="right", fill="y")
    lb.pack(side="left", fill="both", expand=True)

    def _row_text(e) -> str:
        try:
            vis = bool(e["artist"].get_visible())
        except Exception:
            vis = True
        return f"{_DOT_SHOWN if vis else _DOT_HIDDEN} {e['name']}"

    for e in entries:
        lb.insert("end", _row_text(e))

    # ── Control block ────────────────────────────────────────────────────────
    # True while the controls are being repopulated from a newly selected
    # artist, so the trace-apply callbacks do not fire and write the previous
    # trace's settings onto the new one.
    loading = {"busy": False}

    vis_var    = tk.BooleanVar(master=panel, value=True)
    label_var  = tk.StringVar(master=panel, value="")
    marker_var = tk.StringVar(master=panel, value="none")
    ls_var     = tk.StringVar(master=panel, value="solid")
    width_var  = tk.StringVar(master=panel, value="1.0")
    msize_var  = tk.StringVar(master=panel, value="4")
    alpha_var  = tk.DoubleVar(master=panel, value=1.0)
    colour_box = {"hex": "#000000"}

    def _redraw():
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass

    def _notify():
        if on_change is not None:
            try:
                on_change()
            except Exception:
                pass

    def _selected():
        sel = lb.curselection()
        return entries[sel[0]] if sel else None

    def _refresh_row(idx):
        """Rewrite one listbox row without disturbing the selection."""
        e = entries[idx]
        lb.delete(idx)
        lb.insert(idx, _row_text(e))
        lb.selection_set(idx)

    # ── Visibility ───────────────────────────────────────────────────────────
    vis_cb = tk.Checkbutton(
        tf, text="Visible", variable=vis_var, anchor="w",
        font=("Helvetica", 9),
    )
    vis_cb.pack(fill="x", pady=(6, 0))

    def _apply_visible(*_):
        if loading["busy"]:
            return
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        try:
            entries[idx]["artist"].set_visible(vis_var.get())
        except Exception:
            return
        _refresh_row(idx)
        _redraw()
        _notify()

    vis_cb.config(command=_apply_visible)

    # ── Label ────────────────────────────────────────────────────────────────
    lab_row = tk.Frame(tf)
    lab_row.pack(fill="x", pady=(4, 0))
    tk.Label(lab_row, text="Label:", width=7, anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    lab_entry = tk.Entry(lab_row, textvariable=label_var, width=22,
                         font=("Helvetica", 9))
    lab_entry.pack(side="left", padx=4, fill="x", expand=True)

    def _apply_label(_event=None):
        if loading["busy"]:
            return
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        new = label_var.get().strip()
        if not new:
            return
        try:
            entries[idx]["artist"].set_label(new)
        except Exception:
            return
        entries[idx]["name"] = new
        _refresh_row(idx)
        _redraw()
        _notify()

    lab_entry.bind("<Return>", _apply_label)
    lab_entry.bind("<FocusOut>", _apply_label)

    # ── Colour + alpha ───────────────────────────────────────────────────────
    col_row = tk.Frame(tf)
    col_row.pack(fill="x", pady=(4, 0))
    tk.Label(col_row, text="Colour:", width=7, anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    swatch = tk.Button(col_row, text="    ", relief="solid", borderwidth=1,
                       width=4)
    swatch.pack(side="left", padx=4)

    def _pick_colour():
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        rgb, hx = colorchooser.askcolor(
            color=colour_box["hex"], parent=panel, title="Trace colour")
        if not hx:
            return
        art = entries[idx]["artist"]
        try:
            art.set_color(hx)
        except Exception:
            try:
                art.set_facecolor(hx)
            except Exception:
                return

        # set_color reaches a Line2D's MARKERS only while their face and
        # edge colours are 'auto'.  Set explicitly, they ignore it and
        # this control silently does nothing -- and on a points-only
        # trace (linestyle "none") the line colour is never drawn, so
        # nothing at all changes on screen.  A composited overlay trace
        # is exactly that case: fcs_plotopen copies markerfacecolor and
        # markeredgecolor across, and the getters resolve 'auto' to a
        # literal on the way, so the copy arrives with them pinned.
        # 'none' is left alone: an unfilled marker is a deliberate
        # style, not a colour waiting to be set.
        for _getter, _setter in (("get_markerfacecolor",
                                  "set_markerfacecolor"),
                                 ("get_markeredgecolor",
                                  "set_markeredgecolor")):
            try:
                _cur = getattr(art, _getter)()
            except AttributeError:
                break          # not a Line2D -- no markers to recolour
            except Exception:
                continue
            if isinstance(_cur, str) and _cur.lower() == "none":
                continue
            try:
                getattr(art, _setter)(hx)
            except Exception:
                pass

        colour_box["hex"] = hx
        swatch.config(bg=hx)
        _redraw()
        _notify()

    swatch.config(command=_pick_colour)

    tk.Label(col_row, text="Alpha:", anchor="e",
             font=("Helvetica", 9)).pack(side="left", padx=(10, 0))

    def _apply_alpha(*_):
        if loading["busy"]:
            return
        sel = lb.curselection()
        if not sel:
            return
        try:
            entries[sel[0]]["artist"].set_alpha(float(alpha_var.get()))
        except Exception:
            return
        _redraw()

    tk.Scale(col_row, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
             variable=alpha_var, command=_apply_alpha, showvalue=True,
             length=90, font=("Helvetica", 8)).pack(side="left", padx=4)

    # ── Marker + marker size ─────────────────────────────────────────────────
    mk_row = tk.Frame(tf)
    mk_row.pack(fill="x", pady=(4, 0))
    tk.Label(mk_row, text="Marker:", width=7, anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    mk_menu = tk.OptionMenu(mk_row, marker_var,
                            *[n for n, _v in _MARKER_CHOICES])
    mk_menu.config(font=("Helvetica", 9), width=9)
    mk_menu["menu"].config(font=("Helvetica", 9))
    mk_menu.pack(side="left", padx=4)
    tk.Label(mk_row, text="Size:", anchor="e",
             font=("Helvetica", 9)).pack(side="left", padx=(8, 0))
    msize_sp = tk.Spinbox(mk_row, from_=0, to=30, increment=0.5, width=5,
                          textvariable=msize_var, font=("Helvetica", 9))
    msize_sp.pack(side="left", padx=4)

    # ── Line style + width ───────────────────────────────────────────────────
    ln_row = tk.Frame(tf)
    ln_row.pack(fill="x", pady=(4, 0))
    tk.Label(ln_row, text="Line:", width=7, anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    ls_menu = tk.OptionMenu(ln_row, ls_var,
                            *[n for n, _v in _LINESTYLE_CHOICES])
    ls_menu.config(font=("Helvetica", 9), width=9)
    ls_menu["menu"].config(font=("Helvetica", 9))
    ls_menu.pack(side="left", padx=4)
    tk.Label(ln_row, text="Width:", anchor="e",
             font=("Helvetica", 9)).pack(side="left", padx=(8, 0))
    width_sp = tk.Spinbox(ln_row, from_=0, to=10, increment=0.5, width=5,
                          textvariable=width_var, font=("Helvetica", 9))
    width_sp.pack(side="left", padx=4)

    _LINE_ONLY = (mk_menu, msize_sp, ls_menu, width_sp)

    def _apply_marker(*_):
        if loading["busy"]:
            return
        sel = lb.curselection()
        if not sel:
            return
        val = dict(_MARKER_CHOICES).get(marker_var.get())
        try:
            entries[sel[0]]["artist"].set_marker(val)
        except Exception:
            return
        _redraw()
        _notify()

    def _apply_linestyle(*_):
        if loading["busy"]:
            return
        sel = lb.curselection()
        if not sel:
            return
        val = dict(_LINESTYLE_CHOICES).get(ls_var.get())
        try:
            entries[sel[0]]["artist"].set_linestyle(val)
        except Exception:
            return
        _redraw()
        _notify()

    def _apply_width(*_):
        if loading["busy"]:
            return
        sel = lb.curselection()
        if not sel:
            return
        try:
            entries[sel[0]]["artist"].set_linewidth(float(width_var.get()))
        except (ValueError, tk.TclError, AttributeError):
            return
        _redraw()
        _notify()

    def _apply_msize(*_):
        if loading["busy"]:
            return
        sel = lb.curselection()
        if not sel:
            return
        try:
            entries[sel[0]]["artist"].set_markersize(float(msize_var.get()))
        except (ValueError, tk.TclError, AttributeError):
            return
        _redraw()
        _notify()

    marker_var.trace_add("write", _apply_marker)
    ls_var.trace_add("write", _apply_linestyle)
    width_var.trace_add("write", _apply_width)
    msize_var.trace_add("write", _apply_msize)
    msize_sp.config(command=_apply_msize)
    width_sp.config(command=_apply_width)

    # ── Selection -> repopulate controls ─────────────────────────────────────
    def _on_select(_event=None):
        e = _selected()
        if e is None:
            return
        art = e["artist"]
        loading["busy"] = True
        try:
            try:
                vis_var.set(bool(art.get_visible()))
            except Exception:
                vis_var.set(True)

            label_var.set(e["name"])

            # Colour: lines expose get_color; collections may only have
            # facecolors, and an empty array means "no explicit colour".
            hx = "#000000"
            try:
                hx = to_hex(to_rgba(art.get_color()))
            except Exception:
                try:
                    fc = art.get_facecolor()
                    if len(fc):
                        hx = to_hex(fc[0])
                except Exception:
                    pass
            colour_box["hex"] = hx
            try:
                swatch.config(bg=hx)
            except Exception:
                pass

            try:
                a = art.get_alpha()
                alpha_var.set(1.0 if a is None else float(a))
            except Exception:
                alpha_var.set(1.0)

            if e["is_line"]:
                inv_mk = {v: n for n, v in _MARKER_CHOICES}
                inv_ls = {v: n for n, v in _LINESTYLE_CHOICES}
                try:
                    marker_var.set(inv_mk.get(str(art.get_marker()), "none"))
                except Exception:
                    marker_var.set("none")
                try:
                    ls_var.set(inv_ls.get(str(art.get_linestyle()), "solid"))
                except Exception:
                    ls_var.set("solid")
                try:
                    width_var.set(f"{float(art.get_linewidth()):g}")
                except Exception:
                    width_var.set("1")
                try:
                    msize_var.set(f"{float(art.get_markersize()):g}")
                except Exception:
                    msize_var.set("4")

            # Marker / line-style / width / size are Line2D concepts.  A
            # collection (fill_between shading, scatter) has no equivalent, so
            # those controls are disabled rather than silently doing nothing.
            state = "normal" if e["is_line"] else "disabled"
            for w in _LINE_ONLY:
                try:
                    w.config(state=state)
                except Exception:
                    pass
        finally:
            loading["busy"] = False

    lb.bind("<<ListboxSelect>>", _on_select)
    lb.selection_set(0)
    _on_select()


def _panel_names(axes_list: list) -> list:
    """
    Short, recognisable names for each panel of a multi-panel figure.

    Named from the y-label where there is one -- "deviation from fit (%)"
    beats "Panel 2" when deciding which axis to rescale.
    """
    names = []
    for i, ax in enumerate(axes_list, start=1):
        try:
            lbl = (ax.get_ylabel() or "").replace("\n", " ").strip()
        except Exception:
            lbl = ""
        names.append(f"{i}: {lbl[:28]}" if lbl else f"Panel {i}")
    # Duplicate labels would make the menu ambiguous.
    if len(set(names)) != len(names):
        names = [f"Panel {i}" for i in range(1, len(axes_list) + 1)]
    return names


def _build_scale_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
    """
    Add the "Axis scale" LabelFrame to *panel*.

    X-scale changes are applied to every axis in axes_list, so a shared-x
    subplot pair stays in sync.  Y is per-panel: a residuals panel has its own
    scale and its own sensible range, and forcing it to follow the main axis
    would be wrong.  When the figure has more than one panel a selector
    chooses which panel the Y controls act on -- previously they always acted
    on axes[0], leaving a residuals panel with no Y control at all.
    """
    import tkinter as tk
    from tkinter import messagebox

    ax0 = axes_list[0]

    sf = tk.LabelFrame(panel, text="Axis scale", padx=10, pady=6)
    sf.pack(fill="x", padx=10, pady=(10, 4))

    y_target = {"i": 0}

    x_var = tk.StringVar(master=panel, value=ax0.get_xscale())
    y_var = tk.StringVar(master=panel, value=ax0.get_yscale())

    def _yax():
        return axes_list[min(y_target["i"], len(axes_list) - 1)]

    def _apply(*_args):
        xs = x_var.get()
        ys = y_var.get()

        x_failed = False
        for ax in axes_list:
            try:
                ax.set_xscale(xs)
            except Exception as e:
                x_var.set(ax.get_xscale())
                messagebox.showwarning(
                    "X scale not applied",
                    f"Could not set X axis to {xs!r}:\n\n{e}",
                    parent=panel,
                )
                x_failed = True
                break

        y_failed = False
        try:
            _yax().set_yscale(ys)
        except Exception as e:
            y_var.set(_yax().get_yscale())
            messagebox.showwarning(
                "Y scale not applied",
                f"Could not set Y axis to {ys!r}:\n\n{e}\n\n"
                "Log scale requires all plotted values to be positive.\n"
                "G(\u03c4) in correlation plots often has negative values,\n"
                "which prevents log-Y display.",
                parent=panel,
            )
            y_failed = True

        if x_failed and y_failed:
            return

        try:
            fig.canvas.draw()
        except Exception:
            pass

    for axis_label, var in [("X:", x_var), ("Y:", y_var)]:
        row = tk.Frame(sf)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=axis_label, width=3, anchor="e").pack(side="left")
        for val, txt in [("linear", "Linear"), ("log", "Log")]:
            tk.Radiobutton(
                row, text=txt, variable=var, value=val,
                command=_apply,
            ).pack(side="left", padx=5)

    # Panel selector, only where there is more than one panel to choose from.
    if len(axes_list) > 1:
        prow = tk.Frame(sf)
        prow.pack(fill="x", pady=(4, 0))
        tk.Label(prow, text="Y of:", width=5, anchor="e",
                 font=("Helvetica", 9)).pack(side="left")
        pv = tk.StringVar(master=panel, value=_panel_names(axes_list)[0])

        def _on_panel(*_):
            names = _panel_names(axes_list)
            try:
                y_target["i"] = names.index(pv.get())
            except ValueError:
                y_target["i"] = 0
            # Show the newly selected panel's current scale, without
            # re-applying the previous panel's setting to it.
            y_var.set(_yax().get_yscale())

        om = tk.OptionMenu(prow, pv, *_panel_names(axes_list))
        om.config(font=("Helvetica", 9))
        om["menu"].config(font=("Helvetica", 9))
        om.pack(side="left", padx=4)
        pv.trace_add("write", _on_panel)


# ── Trace normalization ───────────────────────────────────────────────────────

# Number of leading points averaged as the normalization reference.  The
# first plotted point alone is a single noisy estimate of G(tau_min);
# averaging a few leading points beats it at no cost for a monotone decay.
# Module-level so the preferences system can retarget it at runtime, the
# same pattern _FIGURE_FORMAT follows; it is re-read on every toggle.
def _autoscale_axes(a) -> None:
    """Rescale one axes to its currently visible data.

    Two things make the obvious ``a.relim(); a.autoscale_view()`` a no-op, and
    both bit the Axis-limits "Auto" button, which never worked:

    1. ``set_xlim``/``set_ylim`` turn autoscaling OFF for that axis, and every
       plotter in the suite sets limits when it draws.  ``autoscale_view()``
       then respects the disabled flag and returns without doing anything, so
       autoscaling has to be switched back on first.
    2. ``relim()`` walks lines and patches but IGNORES collections, so a
       fill_between, a vlines stem plot or the error bars of an errorbar call
       are invisible to it.  Their extents have to be folded into the data
       limits by hand.

    ``visible_only=True`` keeps traces hidden in the Traces section out of the
    range, which is what someone who just hid a trace expects.
    """
    try:
        a.set_autoscalex_on(True)
        a.set_autoscaley_on(True)
        a.relim(visible_only=True)
        for col in a.collections:
            try:
                a.update_datalim(col.get_datalim(a.transData))
            except Exception:      # noqa: BLE001 - one odd collection must
                pass               # not stop the rest rescaling
        a.autoscale_view()
    except Exception:              # noqa: BLE001 - rescaling is cosmetic
        pass


_NORMALIZE_K = 3

# How the normalization reference is chosen: "leading" divides by the mean of
# the first _NORMALIZE_K plotted points, "peak" by the largest value.
#
# Leading points suit correlation curves, which start at their maximum.  They
# are wrong for a lifetime decay, whose first plotted points sit on the RISE
# before the peak, so dividing by them scales every curve by an essentially
# arbitrary number partway up the leading edge.
_NORMALIZE_MODE = "leading"

# Upper y limit applied while normalized.  Slightly above 1 on purpose: a trace
# touching the very top of the frame looks clipped even when it is not.
_NORMALIZE_YMAX = 1.1


def _build_normalize_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
    """
    Add the "Normalize" LabelFrame to *panel*.

    One checkbox that divides every data trace on the PRIMARY axis by the
    mean of its own first ``_NORMALIZE_K`` plotted points, so an overlay of
    correlation curves with different amplitudes can be compared by shape
    alone.  Display-only and fully reversible: original ydata are cached and
    restored on uncheck, and CSV exports (written at compute time, upstream
    of this panel) are untouched — though a figure saved while normalized
    captures the normalized view, which is the point.

    Scoped to axes_list[0] deliberately: in stacked layouts the lower panel
    is residuals, and dividing residuals by their own leading points would be
    meaningless noise amplification, not normalization.

    The reference is the mean of the first PLOTTED points — an estimate of
    G(tau_min), not a fitted G(0).  Triplet dynamics or afterpulsing inside
    tau_min therefore stay in the reference; curves differing only in
    triplet fraction will not converge to exactly 1.  That is a property of
    post-hoc normalization, stated in the hint rather than hidden.

    Reference lines (drawn with a blended transform — axhline/axvline) are
    skipped silently — they are not data.  A DATA trace whose reference is
    zero, NaN, or infinite is left unscaled and reported in one warning box,
    never silently mis-scaled.

    Data/fit pairing: a fit plotter that draws one dataset as two artists
    tags them ``data_line._norm_ref = fit_line``.  Such a pair normalizes by
    the FIT's leading-point mean — the smooth model amplitude at tau_min —
    with the data scaling by the same factor, never independently.  The
    contract is an explicit attribute set at plot-creation time; no label or
    colour heuristics, which a rename in the Traces section would break.
    """
    import tkinter as tk
    from tkinter import messagebox

    ax = axes_list[0]

    lf = tk.LabelFrame(panel, text="Normalize", padx=10, pady=6)
    lf.pack(fill="x", padx=10, pady=4)

    norm_var = tk.BooleanVar(master=panel, value=False)

    # id(line) -> (line, original ydata copy).  Holding the copy (rather than
    # a scale factor) makes restore exact to the bit, whatever floating-point
    # the division did.
    saved: dict = {}
    ylabel_box = {"orig": None}
    ylim_box: dict = {"orig": None}
    mode_var = None                # bound below; read by _ref_value

    def _apply_norm_ylim() -> None:
        """Pin the y range to 0 .. _NORMALIZE_YMAX while normalized.

        Normalized traces all top out near 1, so letting matplotlib autoscale
        wastes the frame or crops the peak depending on the noise.  A log y
        axis cannot include 0, so there only the TOP is pinned and the bottom
        is left wherever autoscaling put it.
        """
        try:
            if ax.get_yscale() == "log":
                lo, _hi = ax.get_ylim()
                ax.set_ylim(lo, _NORMALIZE_YMAX)
            else:
                ax.set_ylim(0.0, _NORMALIZE_YMAX)
        except Exception:          # noqa: BLE001 - cosmetic
            pass

    def _is_reference(line) -> bool:
        """
        True for artists that are not data in data space.

        axhline/axvline draw with a BLENDED transform (one coordinate in
        axes fraction), so comparing against ax.transData separates them
        from plotted data exactly — unlike the constant-y/constant-x test
        _artist_description uses for naming, which cannot tell an axhline
        from a genuinely constant data trace.  Constancy is the right
        heuristic for choosing a display NAME, but the wrong one here: a
        constant-zero data trace should reach the explicit zero-reference
        warning below, not vanish silently into the reference-line skip.
        """
        try:
            return line.get_transform() != ax.transData
        except Exception:
            return True        # cannot read it -> do not touch it

    def _rescale_and_draw():
        _autoscale_axes(ax)
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass

    def _apply(*_):
        if norm_var.get():
            k = max(1, int(_NORMALIZE_K))
            skipped, did_any = [], False

            # Reference values are computed from PRE-normalization ydata and
            # cached per reference line, so a data trace and the fit it is
            # paired to divide by the IDENTICAL number whatever order the
            # lines are visited in: a line's reference is always cached
            # before that line's ydata is mutated.
            ref_cache: dict = {}

            def _ref_value(ln) -> float:
                key = id(ln)
                if key not in ref_cache:
                    try:
                        yv = np.asarray(ln.get_ydata(), dtype=float)
                    except Exception:
                        yv = np.empty(0)
                    if mode_var.get() == "peak":
                        good = yv[np.isfinite(yv)]
                        ref_cache[key] = (float(good.max()) if good.size
                                          else float("nan"))
                    else:
                        lead = yv[:k]
                        lead = lead[np.isfinite(lead)]
                        ref_cache[key] = (float(lead.mean()) if lead.size
                                          else float("nan"))
                return ref_cache[key]

            for line in ax.get_lines():
                if id(line) in saved or _is_reference(line):
                    continue
                # The pairing contract: a fit plotter that draws a dataset as
                # separate data and fit artists sets data._norm_ref = fit, so
                # the pair is normalized by the FIT's leading points (smooth
                # model amplitude at tau_min) and never separately.  A line
                # without the attribute is its own reference — plain
                # overlays and figures predating the contract behave exactly
                # as before.  Explicit attribute only: no label or colour
                # matching, which a rename in the Traces section would break.
                ref_line = getattr(line, "_norm_ref", None)
                if ref_line is None:
                    ref_line = line
                ref = _ref_value(ref_line)
                y = np.asarray(line.get_ydata(), dtype=float)
                if not np.isfinite(ref) or ref == 0.0:
                    lbl = line.get_label() or ""
                    skipped.append(lbl if lbl and not lbl.startswith("_")
                                   else "(unnamed trace)")
                    continue
                saved[id(line)] = (line, y.copy())
                line.set_ydata(y / ref)
                did_any = True
            if did_any and ylabel_box["orig"] is None:
                ylabel_box["orig"] = ax.get_ylabel()
                if ylabel_box["orig"]:
                    ax.set_ylabel(ylabel_box["orig"] + "  (normalized)")
            if did_any and ylim_box["orig"] is None:
                ylim_box["orig"] = ax.get_ylim()
            _rescale_and_draw()
            if did_any:
                _apply_norm_ylim()
            if skipped:
                messagebox.showwarning(
                    "Some traces not normalized",
                    "The normalization reference (the trace's own leading "
                    "points, or its paired fit's) is zero or not finite "
                    "for:\n\n  " + "\n  ".join(skipped) +
                    "\n\nThese traces were left unscaled.",
                    parent=panel)
        else:
            for _id, (line, y) in list(saved.items()):
                try:
                    line.set_ydata(y)
                except Exception:
                    pass
            saved.clear()
            if ylabel_box["orig"] is not None:
                ax.set_ylabel(ylabel_box["orig"])
                ylabel_box["orig"] = None
            _rescale_and_draw()
            # AFTER the rescale, not before: _rescale_and_draw autoscales, so
            # restoring first would just be overwritten and unchecking would
            # leave a different range than the user had.
            if ylim_box["orig"] is not None:
                ax.set_ylim(*ylim_box["orig"])
                ylim_box["orig"] = None
                try:
                    fig.canvas.draw_idle()
                except Exception:      # noqa: BLE001 - cosmetic
                    pass

    tk.Checkbutton(
        lf, text="Normalize traces", variable=norm_var, command=_apply,
        anchor="w", font=("Helvetica", 9),
    ).pack(fill="x")

    mode_var = tk.StringVar(master=panel, value=_NORMALIZE_MODE)

    def _mode_changed(*_):
        """Re-normalize in place when the reference changes."""
        if norm_var.get():
            norm_var.set(False)
            _apply()
            norm_var.set(True)
            _apply()

    mrow = tk.Frame(lf)
    mrow.pack(fill="x")
    tk.Label(mrow, text="      divide by:", font=("Helvetica", 9),
             anchor="w").pack(side="left")
    tk.Radiobutton(mrow, text=f"first {max(1, int(_NORMALIZE_K))} pts",
                   variable=mode_var, value="leading", command=_mode_changed,
                   font=("Helvetica", 9)).pack(side="left")
    tk.Radiobutton(mrow, text="peak", variable=mode_var, value="peak",
                   command=_mode_changed,
                   font=("Helvetica", 9)).pack(side="left")

    tk.Label(
        lf,
        text=("      display only — exported CSVs keep physical units.\n"
              "      Leading points suit correlation curves; use the peak for\n"
              "      lifetime decays, whose first points are on the rise.\n"
              f"      While normalized the y axis is pinned to 0 – "
              f"{_NORMALIZE_YMAX:g}."),
        font=("Helvetica", 8), fg="grey", anchor="w", justify="left",
    ).pack(fill="x")


def _build_limits_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
    """
    Add the "Axis limits" LabelFrame to *panel*.

    X-limits are applied to all axes, since a multi-panel figure shares its
    x-axis.  Y-limits act on ONE panel, chosen by the selector when the figure
    has several -- a residuals panel needs its own range, and it previously
    had no way to be rescaled at all because these controls were wired to
    axes[0] unconditionally.
    "Auto" calls autoscale_view() and refreshes the displayed values.
    """
    import tkinter as tk
    from tkinter import messagebox

    ax = axes_list[0]
    y_target = {"i": 0}

    def _yax():
        return axes_list[min(y_target["i"], len(axes_list) - 1)]

    lf = tk.LabelFrame(panel, text="Axis limits", padx=10, pady=6)
    lf.pack(fill="x", padx=10, pady=4)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()

    xlo_var = tk.StringVar(master=panel, value=f"{xlo:.6g}")
    xhi_var = tk.StringVar(master=panel, value=f"{xhi:.6g}")
    ylo_var = tk.StringVar(master=panel, value=f"{ylo:.6g}")
    yhi_var = tk.StringVar(master=panel, value=f"{yhi:.6g}")

    for axis_lbl, lo_var, hi_var in [("X:", xlo_var, xhi_var),
                                      ("Y:", ylo_var, yhi_var)]:
        row = tk.Frame(lf)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=axis_lbl, width=3, anchor="e",
                 font=("Helvetica", 9)).pack(side="left")
        tk.Entry(row, textvariable=lo_var, width=9,
                 font=("Helvetica", 9)).pack(side="left", padx=(2, 0))
        tk.Label(row, text="\u2013", font=("Helvetica", 9)).pack(side="left", padx=3)
        tk.Entry(row, textvariable=hi_var, width=9,
                 font=("Helvetica", 9)).pack(side="left")

    def _apply_limits():
        x_ok = True
        try:
            xlo, xhi = float(xlo_var.get()), float(xhi_var.get())
            if xlo >= xhi:
                raise ValueError("min must be less than max")
            for a in axes_list:
                a.set_xlim(xlo, xhi)
        except Exception as e:
            messagebox.showwarning("X limits", f"Invalid X limits:\n{e}",
                                   parent=panel)
            x_ok = False

        y_ok = True
        try:
            ylo, yhi = float(ylo_var.get()), float(yhi_var.get())
            if ylo >= yhi:
                raise ValueError("min must be less than max")
            _yax().set_ylim(ylo, yhi)
        except Exception as e:
            messagebox.showwarning("Y limits", f"Invalid Y limits:\n{e}",
                                   parent=panel)
            y_ok = False

        if x_ok or y_ok:
            try:
                fig.canvas.draw()
            except Exception:
                pass

    def _auto_limits():
        for a in axes_list:
            _autoscale_axes(a)
        xlo, xhi = axes_list[0].get_xlim()
        ylo, yhi = _yax().get_ylim()
        xlo_var.set(f"{xlo:.6g}")
        xhi_var.set(f"{xhi:.6g}")
        ylo_var.set(f"{ylo:.6g}")
        yhi_var.set(f"{yhi:.6g}")
        try:
            fig.canvas.draw()
        except Exception:
            pass

    if len(axes_list) > 1:
        prow = tk.Frame(lf)
        prow.pack(fill="x", pady=(4, 0))
        tk.Label(prow, text="Y of:", width=5, anchor="e",
                 font=("Helvetica", 9)).pack(side="left")
        pv = tk.StringVar(master=panel, value=_panel_names(axes_list)[0])

        def _on_panel(*_):
            names = _panel_names(axes_list)
            try:
                y_target["i"] = names.index(pv.get())
            except ValueError:
                y_target["i"] = 0
            # Show the selected panel's own limits rather than carrying the
            # previous panel's numbers across.
            lo, hi = _yax().get_ylim()
            ylo_var.set(f"{lo:.6g}")
            yhi_var.set(f"{hi:.6g}")

        om = tk.OptionMenu(prow, pv, *_panel_names(axes_list))
        om.config(font=("Helvetica", 9))
        om["menu"].config(font=("Helvetica", 9))
        om.pack(side="left", padx=4)
        pv.trace_add("write", _on_panel)

    btn_row = tk.Frame(lf)
    btn_row.pack(fill="x", pady=(6, 0))
    tk.Button(btn_row, text="Apply", command=_apply_limits,
              width=8, pady=2).pack(side="left", padx=(0, 4))
    tk.Button(btn_row, text="Auto", command=_auto_limits,
              width=8, pady=2).pack(side="left")


# ── Gridlines ─────────────────────────────────────────────────────────────────

# Style applied when gridlines are switched ON from the controls panel.  The
# major values are the ones the suite's own plotters already pass to
# ax.grid(), so toggling a grid off and back on returns the figure to the way
# it was drawn rather than to the matplotlib default.  Module-level so the
# preferences system can retarget them at runtime, the same pattern
# _FIGURE_FORMAT and _NORMALIZE_K follow; both are re-read on every toggle.
_GRID_MAJOR_KW = {"linestyle": "--", "linewidth": 0.4, "alpha": 0.5}
# Minor lines are deliberately lighter.  A semilog correlation axis carries
# eight minor lines per decade, and at major weight they read as a hatch over
# the data rather than as a reference to measure against.
_GRID_MINOR_KW = {"linestyle": ":", "linewidth": 0.3, "alpha": 0.3}


def _grid_is_on(ax, which: str) -> bool:
    """
    Whether *ax* currently draws *which* ("major" or "minor") gridlines.

    Read from the tick artists, not from the private tick kwargs: a
    gridline's visibility is what actually renders, and ``get_xgridlines``
    and ``Tick.gridline`` are public on every matplotlib this suite runs on.
    Either axis counts, so a figure gridded in x only still opens with the
    box ticked.
    """
    try:
        if which == "major":
            lines = list(ax.get_xgridlines()) + list(ax.get_ygridlines())
        else:
            lines = [t.gridline for t in ax.xaxis.get_minor_ticks()]
            lines += [t.gridline for t in ax.yaxis.get_minor_ticks()]
    except Exception:
        return False
    for ln in lines:
        try:
            if ln.get_visible():
                return True
        except Exception:
            continue
    return False


def _build_grid_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
    """
    Add the "Gridlines" LabelFrame to *panel*.

    Major and minor are separate toggles, and both act on EVERY panel of the
    figure.  Grid is unlike Y scale and Y limits in this respect: it is a
    reading aid rather than a statement about the data, and a residual strip
    with no grid under a main panel that has one reads as an oversight.

    Initial state is read from the first panel, so a figure drawn with a grid
    opens with its box already ticked and the first click turns the grid OFF
    rather than appearing to do nothing.

    Minor gridlines need minor TICKS to hang on, and a linear axis has none
    by default -- ``ax.grid(True, which="minor")`` on one draws nothing at
    all.  Enabling minor therefore switches minor ticks on for any axis that
    lacks them, so the control cannot silently do nothing.  Disabling leaves
    the ticks in place: removing them again would undo a tick setting the
    figure may have arrived with.
    """
    import tkinter as tk

    ax0 = axes_list[0]

    maj_var = tk.BooleanVar(master=panel, value=_grid_is_on(ax0, "major"))
    min_var = tk.BooleanVar(master=panel, value=_grid_is_on(ax0, "minor"))

    gf = tk.LabelFrame(panel, text="Gridlines", padx=10, pady=6)
    gf.pack(fill="x", padx=10, pady=4)

    def _has_minor_ticks(ax) -> bool:
        try:
            # get_minorticklocs returns an ndarray, and bool() on an empty
            # array raises rather than returning False -- len() is the only
            # safe emptiness test here.
            return (len(ax.xaxis.get_minorticklocs()) > 0 or
                    len(ax.yaxis.get_minorticklocs()) > 0)
        except Exception:
            return False

    def _apply(*_):
        want_major = bool(maj_var.get())
        want_minor = bool(min_var.get())
        for ax in axes_list:
            if want_minor and not _has_minor_ticks(ax):
                try:
                    ax.minorticks_on()
                except Exception:
                    pass
            try:
                if want_major:
                    ax.grid(True, which="major", **dict(_GRID_MAJOR_KW))
                else:
                    ax.grid(False, which="major")
            except Exception:
                pass
            try:
                if want_minor:
                    ax.grid(True, which="minor", **dict(_GRID_MINOR_KW))
                else:
                    ax.grid(False, which="minor")
            except Exception:
                pass
        try:
            fig.canvas.draw()
        except Exception:
            pass

    row = tk.Frame(gf)
    row.pack(fill="x")
    tk.Checkbutton(
        row, text="Major", variable=maj_var, command=_apply, anchor="w",
        font=("Helvetica", 9),
    ).pack(side="left")
    tk.Checkbutton(
        row, text="Minor", variable=min_var, command=_apply, anchor="w",
        font=("Helvetica", 9),
    ).pack(side="left", padx=(12, 0))
    tk.Label(
        gf,
        text=("      applied to every panel — display only, exports\n"
              "      and fitted values are untouched"),
        font=("Helvetica", 8), fg="grey", anchor="w", justify="left",
    ).pack(fill="x")


def _build_legend_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
):
    """
    Add the "Legend" LabelFrame to *panel*, if the main axis has a legend.

    Offers these controls:
      * Show legend — a checkbox to hide/show it (useful when an overlay of
        many datasets produces an enormous legend).
      * Position — dropdown to move it.
      * Text size — slider to shrink or grow it.
      * Entries — a per-entry list with a legend-only display text and an
        include/exclude toggle.  Both are keyed by id(handle) and live only
        in this closure: nothing here calls set_label(), so the Traces
        section, exports, and any later analysis see the original names
        unchanged, and an excluded entry's trace stays visible on the plot.
        Exclusion is reversible (re-tick "Show in legend"); clearing the
        text reverts to the entry's own label.  Neither survives the
        session — reopening the figure via fcs_plotopen starts clean.

    Position and size changes rebuild the legend from a snapshot of its
    HANDLES captured when the panel is built; the snapshot is taken from the
    legend object itself (not the axes), so custom entries such as the
    intensity plot's "Mean Ch1/Ch2 CPS" rows are preserved.

    The label TEXT, by contrast, is re-read from each handle every rebuild
    rather than snapshotted, so a trace renamed in the Traces section is
    renamed in the legend too.  Hidden traces drop out, and a trace that had
    no label originally joins the legend once it is given one.

    Returns the rebuild function, so other sections (Traces) can refresh the
    legend after an edit that changes it.
    """
    import tkinter as tk

    ax  = axes_list[0]
    leg = ax.get_legend()
    if leg is None:
        return

    _LOC_NAMES = [
        "best",
        "upper right", "upper left",
        "lower right", "lower left",
        "center left", "center right",
        "lower center", "upper center",
        "center",
    ]
    _LOC_INT = {
        "best": 0, "upper right": 1, "upper left": 2,
        "lower left": 3, "lower right": 4,
        "center left": 6, "center right": 7,
        "lower center": 8, "upper center": 9, "center": 10,
    }
    _INT_LOC = {v: k for k, v in _LOC_INT.items()}

    # ── Snapshot the legend so rebuilds preserve every entry ──────────────────
    # legend_handles keeps custom proxy artists (e.g. the invisible Patches the
    # intensity plot uses for its mean-CPS rows); ax.get_legend_handles_labels()
    # would drop them.  Fall back to the axes handles if unavailable.
    handles = getattr(leg, "legend_handles", None)
    if handles is None:
        handles = getattr(leg, "legendHandles", None)
    labels = [t.get_text() for t in leg.get_texts()]
    if not handles:
        handles, labels = ax.get_legend_handles_labels()

    # ── Resolve display copies back to the axes artists ───────────────────────
    # On matplotlib >= 3.7 (observed on 3.10) legend_handles holds the
    # handler-created DISPLAY artists, not the originals, so matching them
    # against the axes by identity fails and every labelled trace appears
    # twice on rebuild: once from the snapshot and once from the axes scan.
    # The copies inherit the original artist's label (even when the legend
    # was created with different display text), so an id miss is resolved by
    # label instead.
    #
    # The label match must be ONE-TO-ONE.  Duplicate labels are ordinary --
    # an overlay of two files each with a "Ch1 ACF" trace, or a global fit
    # over same-named datasets -- and mapping every copy of a label to the
    # same first artist collapses those entries into one identity: they then
    # exclude and rename as a group, and the artists that lost the match are
    # re-appended by the axes scan as phantom duplicates.  Each artist is
    # therefore consumed once, in axes order, so equal-labelled entries keep
    # distinct identities.  A handle matching nothing (a true proxy such as
    # the intensity plot's mean-CPS Patch, whose copy carries "_nolegend_")
    # is kept as-is.
    _axes_arts = []
    for _ax in axes_list:
        _axes_arts.extend(_ax.get_lines())
        _axes_arts.extend(getattr(_ax, "collections", []))
    _art_ids  = {id(_a) for _a in _axes_arts}
    _by_label = {}
    for _a in _axes_arts:
        try:
            _l = _a.get_label() or ""
        except Exception:
            continue
        if _l and not _l.startswith("_"):
            _by_label.setdefault(_l, []).append(_a)

    def _canonical(h):
        if id(h) in _art_ids:
            # A real axes artist: claim it so no copy can take it as well.
            try:
                _l = h.get_label() or ""
            except Exception:
                _l = ""
            queue = _by_label.get(_l)
            if queue is not None and h in queue:
                queue.remove(h)
            return h
        try:
            _l = h.get_label() or ""
        except Exception:
            return h
        queue = _by_label.get(_l)
        if queue:
            return queue.pop(0)
        return h

    handles = [_canonical(h) for h in handles]

    title_artist = leg.get_title()
    title_text   = title_artist.get_text() if title_artist is not None else ""
    title_text   = title_text or None

    try:
        framealpha = leg.get_frame().get_alpha()
        if framealpha is None:
            framealpha = 0.85
    except Exception:
        framealpha = 0.85

    try:
        current_loc = _INT_LOC.get(leg._loc, "best")
    except Exception:
        current_loc = "best"

    try:
        current_size = int(round(leg.prop.get_size()))
    except Exception:
        current_size = 10
    current_size = min(20, max(4, current_size))

    # Column count, so a rebuild does not collapse a 3-column legend back to
    # one enormous column.  The private attribute moved in matplotlib 3.6.
    _ncol = 1
    for _attr in ("_ncols", "_ncol"):
        _v = getattr(leg, _attr, None)
        if isinstance(_v, int) and _v > 0:
            _ncol = _v
            break

    # Read the real visibility rather than assuming True: adaptive_legend
    # hides legends with very many entries, and a checkbox that claimed
    # "shown" over a hidden legend would be simply wrong.
    try:
        _vis = bool(leg.get_visible())
    except Exception:
        _vis = True
    show_var = tk.BooleanVar(master=panel, value=_vis)
    loc_var  = tk.StringVar(master=panel, value=current_loc)
    size_var = tk.IntVar(master=panel, value=current_size)

    lf = tk.LabelFrame(panel, text="Legend", padx=10, pady=6)
    lf.pack(fill="x", padx=10, pady=4)

    # Per-entry legend edits, keyed by id(handle).  Legend-only by design:
    # an override never touches the artist's label, and exclusion never
    # touches the artist's visibility.  Session-only: neither is exported.
    excluded  = set()   # ids of handles dropped from the legend
    overrides = {}      # id(handle) -> display text for the legend

    def _candidate_entries():
        """
        Every (handle, label) that COULD appear in the legend right now,
        with display-text overrides applied but exclusions NOT yet filtered
        out — the Entries list must show excluded entries too, or there
        would be no way to bring one back.

        Built from the snapshot first, so proxy artists keep their original
        order and survive, then extended with any axes artist that has since
        been given a real label.  Labels come from the artists themselves;
        the snapshot text is only a fallback for proxies whose label was set
        on the legend rather than on the handle.
        """
        out, seen = [], set()
        snap = list(zip(handles, labels)) if len(handles) == len(labels) \
            else [(h, "") for h in handles]
        for h, snap_lbl in snap:
            seen.add(id(h))
            try:
                if not h.get_visible():
                    continue
            except Exception:
                pass
            try:
                lbl = h.get_label() or ""
            except Exception:
                lbl = ""
            if not lbl or lbl.startswith("_"):
                lbl = snap_lbl
            if not lbl or lbl.startswith("_"):
                continue
            out.append((h, overrides.get(id(h), lbl)))

        for _ax in axes_list:
            arts = list(_ax.get_lines()) + list(getattr(_ax, "collections", []))
            for art in arts:
                if id(art) in seen:
                    continue
                try:
                    if not art.get_visible():
                        continue
                    lbl = art.get_label() or ""
                except Exception:
                    continue
                if lbl and not lbl.startswith("_"):
                    out.append((art, overrides.get(id(art), lbl)))
        return out

    def _live_entries():
        """Current (handles, labels) for the legend: candidates minus
        excluded entries."""
        pairs = [(h, l) for h, l in _candidate_entries()
                 if id(h) not in excluded]
        return [h for h, _ in pairs], [l for _, l in pairs]

    # The Entries list is built further down (widgets pack in code order, and
    # it sits below the existing controls), but _apply_legend must refresh it
    # so edits made in the Traces section show up there too; resolved lazily
    # through this holder, same pattern as _legend_cb in build_plot_controls.
    _entries_ui = {"refresh": None}

    def _refresh_entries_list():
        fn = _entries_ui["refresh"]
        if fn is not None:
            fn()

    def _apply_legend(*_):
        _refresh_entries_list()
        live_h, live_l = _live_entries()
        if not live_h:
            # Everything hidden or excluded: drop the legend rather than
            # draw an empty box.
            existing = ax.get_legend()
            if existing is not None:
                existing.set_visible(False)
            try:
                fig.canvas.draw()
            except Exception:
                pass
            return
        new = ax.legend(
            live_h, live_l,
            loc=loc_var.get(),
            fontsize=size_var.get(),
            title=title_text,
            title_fontsize=size_var.get(),
            framealpha=framealpha,
            ncol=_ncol,
        )
        new.set_visible(show_var.get())
        try:
            fig.canvas.draw()
        except Exception:
            pass

    # ── Show / hide ───────────────────────────────────────────────────────────
    _n_entries = len(handles)
    show_cb = tk.Checkbutton(
        lf, text=f"Show legend  ({_n_entries} entries)", variable=show_var,
        command=_apply_legend, anchor="w", font=("Helvetica", 9),
    )
    show_cb.pack(fill="x")
    if _n_entries > _LEGEND_HIDE_ABOVE:
        tk.Label(
            lf,
            text=("      hidden by default — at this many entries the legend\n"
                  "      covers more of the plot than it explains"),
            font=("Helvetica", 8), fg="grey", anchor="w", justify="left",
        ).pack(fill="x")

    # ── Position ──────────────────────────────────────────────────────────────
    row = tk.Frame(lf)
    row.pack(fill="x", pady=(4, 0))
    tk.Label(row, text="Position:", width=9, anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    om = tk.OptionMenu(row, loc_var, *_LOC_NAMES)
    om.config(font=("Helvetica", 9), width=12)
    om["menu"].config(font=("Helvetica", 9))
    om.pack(side="left", padx=4)

    # ── Text size ─────────────────────────────────────────────────────────────
    size_row = tk.Frame(lf)
    size_row.pack(fill="x", pady=(4, 0))
    tk.Label(size_row, text="Text size:", width=9, anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    tk.Scale(
        size_row, from_=4, to=20, orient="horizontal",
        variable=size_var, command=_apply_legend,
        showvalue=True, length=130, font=("Helvetica", 8),
    ).pack(side="left", padx=4, fill="x", expand=True)

    # ── Entries ───────────────────────────────────────────────────────────────
    # Layout mirrors the Traces section: a selector plus ONE control block
    # that retargets to whichever entry is selected.  Inclusion is shown in
    # the list itself (● in legend, ○ excluded) so the state of every entry
    # is readable at a glance.
    tk.Label(lf, text="Entries:", anchor="w",
             font=("Helvetica", 9)).pack(fill="x", pady=(8, 0))

    ent_row = tk.Frame(lf)
    ent_row.pack(fill="x")
    ent_sb = tk.Scrollbar(ent_row, orient="vertical")
    ent_lb = tk.Listbox(
        ent_row, height=min(6, max(3, _n_entries)), activestyle="none",
        font=("Courier", 9), exportselection=False,
        yscrollcommand=ent_sb.set,
    )
    ent_sb.config(command=ent_lb.yview)
    ent_sb.pack(side="right", fill="y")
    ent_lb.pack(side="left", fill="both", expand=True)

    # Handle of each listbox row, parallel to the rows themselves.  Rebuilt
    # on every refresh; handles are compared by id(), never held past it.
    ent_rows = []

    # True while the control block is being repopulated from a newly selected
    # entry, so the apply callbacks do not fire and write the previous
    # entry's settings onto the new one (same guard as the Traces section).
    ent_loading = {"busy": False}

    inc_var  = tk.BooleanVar(master=panel, value=True)
    text_var = tk.StringVar(master=panel, value="")

    def _ent_row_text(h, lbl) -> str:
        dot = _DOT_HIDDEN if id(h) in excluded else _DOT_SHOWN
        return f"{dot} {lbl}"

    def _refresh_entries():
        """
        Repopulate the listbox from the current candidates, preserving the
        selection by handle identity — the selected entry may have moved
        rows if a trace was hidden or renamed in the Traces section.
        """
        sel_id = None
        sel = ent_lb.curselection()
        if sel and sel[0] < len(ent_rows):
            sel_id = id(ent_rows[sel[0]])

        ent_loading["busy"] = True
        try:
            ent_lb.delete(0, "end")
            ent_rows.clear()
            for h, lbl in _candidate_entries():
                ent_rows.append(h)
                ent_lb.insert("end", _ent_row_text(h, lbl))
                if id(h) == sel_id:
                    ent_lb.selection_set(len(ent_rows) - 1)
        finally:
            ent_loading["busy"] = False

    _entries_ui["refresh"] = _refresh_entries

    def _ent_selected():
        sel = ent_lb.curselection()
        return ent_rows[sel[0]] if sel and sel[0] < len(ent_rows) else None

    def _on_entry_select(_event=None):
        h = _ent_selected()
        if h is None:
            return
        ent_loading["busy"] = True
        try:
            inc_var.set(id(h) not in excluded)
            cur = dict((id(ch), l) for ch, l in _candidate_entries())
            text_var.set(cur.get(id(h), ""))
        finally:
            ent_loading["busy"] = False

    ent_lb.bind("<<ListboxSelect>>", _on_entry_select)

    inc_cb = tk.Checkbutton(
        lf, text="Show in legend", variable=inc_var, anchor="w",
        font=("Helvetica", 9),
    )
    inc_cb.pack(fill="x", pady=(4, 0))

    def _apply_include(*_):
        if ent_loading["busy"]:
            return
        h = _ent_selected()
        if h is None:
            return
        if inc_var.get():
            excluded.discard(id(h))
        else:
            excluded.add(id(h))
        _apply_legend()

    inc_cb.config(command=_apply_include)

    txt_row = tk.Frame(lf)
    txt_row.pack(fill="x", pady=(2, 0))
    tk.Label(txt_row, text="Text:", width=7, anchor="e",
             font=("Helvetica", 9)).pack(side="left")
    txt_entry = tk.Entry(txt_row, textvariable=text_var,
                         font=("Helvetica", 9))
    txt_entry.pack(side="left", padx=4, fill="x", expand=True)
    tk.Label(
        lf,
        text="      legend only — the trace keeps its own name;\n"
             "      clear the text to revert",
        font=("Helvetica", 8), fg="grey", anchor="w", justify="left",
    ).pack(fill="x")

    def _apply_text(_event=None):
        if ent_loading["busy"]:
            return
        h = _ent_selected()
        if h is None:
            return
        new_text = text_var.get().strip()
        if new_text:
            overrides[id(h)] = new_text
        else:
            # Empty text reverts to the entry's own label rather than
            # producing a blank legend row.
            overrides.pop(id(h), None)
        _apply_legend()
        # Rewriting the listbox row moves the reverted label back into the
        # text field so the control shows what the legend now shows.
        _on_entry_select()

    txt_entry.bind("<Return>", _apply_text)
    txt_entry.bind("<FocusOut>", _apply_text)

    _refresh_entries()

    loc_var.trace_add("write", _apply_legend)

    return _apply_legend


# ── Annotation boxes (fit-result text) ────────────────────────────────────────

# Longest annotation summary shown in the selector row.
_ANNOT_LABEL_CHARS = 34


def _annotation_artists(axes_list: list) -> list:
    """
    The editable text boxes on *axes_list*, in drawing order.

    ``ax.texts`` holds exactly the artists added by ``ax.text(...)`` -- the
    fit-result and linked-parameter boxes the fitters draw.  The title and
    the axis labels are separate artists and are NOT in this list, so they
    cannot be reached from here (the Labels section owns them) and there is
    no risk of this section quietly editing them.

    Blank artists are skipped: several plotting routines pre-create an empty
    text they may never fill, and offering an empty row to edit would be
    noise.
    """
    out = []
    for ax in axes_list:
        for t in getattr(ax, "texts", []):
            try:
                if (t.get_text() or "").strip():
                    out.append(t)
            except Exception:
                continue
    return out


def _annotation_summary(text_artist, index: int) -> str:
    """A one-line description of an annotation for the selector."""
    try:
        raw = (text_artist.get_text() or "").strip()
    except Exception:
        raw = ""
    first = raw.splitlines()[0] if raw else ""
    first = " ".join(first.split())
    if len(first) > _ANNOT_LABEL_CHARS:
        first = first[:_ANNOT_LABEL_CHARS - 1].rstrip() + "…"
    return first or f"annotation {index + 1}"


# Corner presets for annotation placement, in axes-fraction coordinates:
# name -> (x, y, horizontal alignment, vertical alignment).  The alignment
# is what keeps a box inside the axes as its text changes length -- a
# right-aligned box grows leftwards from x=0.98 rather than off the edge.
# "as drawn" is handled separately: it restores the placement the producer
# used, so a box anchored to data coordinates can always be put back.
_ANNOT_AS_DRAWN = "as drawn"
_ANNOT_POSITIONS = {
    "upper left":   (0.02, 0.95, "left",   "top"),
    "upper center": (0.50, 0.95, "center", "top"),
    "upper right":  (0.98, 0.95, "right",  "top"),
    "center left":  (0.02, 0.50, "left",   "center"),
    "center":       (0.50, 0.50, "center", "center"),
    "center right": (0.98, 0.50, "right",  "center"),
    "lower left":   (0.02, 0.05, "left",   "bottom"),
    "lower center": (0.50, 0.05, "center", "bottom"),
    "lower right":  (0.98, 0.05, "right",  "bottom"),
}


def _build_annotation_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
    """
    Add the "Annotations" LabelFrame to *panel*, if the figure has any.

    Fit plots carry a text box of results -- linked parameters,
    chi-squared, fitted lifetimes.  This section shows or hides each box,
    moves it between the nine standard positions, and edits its text.  The
    fitters draw these boxes in the upper right, which is also where a
    legend most often wants to sit, so moving one out of the other's way is
    the common case.

    Display-only and session-only, like the Legend section: the parameter
    tables written at fit time (CSV and xlsx) are untouched by anything
    here, so an edited box never changes the recorded numbers -- only what
    this figure shows.  "Revert" restores the text AND the placement the
    fitter generated.

    Positions are set in axes fractions.  A box the producer anchored to
    data coordinates is re-anchored to the axes when a preset is chosen --
    that is what "upper left" has to mean -- and choosing "as drawn" puts
    it back on its original anchor, so the conversion is never one-way.

    The section is absent when the figure has no annotations, rather than
    present and empty; overlays built by fcs_plotopen drop annotations, so
    those figures simply will not show it.
    """
    import tkinter as tk

    boxes = _annotation_artists(axes_list)
    if not boxes:
        return

    lf = tk.LabelFrame(panel, text="Annotations", padx=10, pady=6)
    lf.pack(fill="x", padx=10, pady=4)

    # The text and placement each box was created with, so Revert has
    # something to go back to for the whole session, not just the last edit.
    original = {}
    original_pos = {}
    for t in boxes:
        try:
            original[id(t)] = t.get_text()
        except Exception:
            original[id(t)] = ""
        try:
            original_pos[id(t)] = (t.get_position(),
                                   t.get_horizontalalignment(),
                                   t.get_verticalalignment(),
                                   t.get_transform())
        except Exception:
            original_pos[id(t)] = None

    # True while the widgets are being repopulated from a newly selected
    # box, so the visibility callback does not write the previous box's
    # state onto the new one.
    loading = {"busy": False}
    current = {"artist": boxes[0]}

    # ── Selector, only when there is a choice to make ─────────────────────
    lb = None
    if len(boxes) > 1:
        tk.Label(lf, text="Box:", anchor="w",
                 font=("Helvetica", 9)).pack(fill="x")
        lb = tk.Listbox(lf, height=min(4, len(boxes)), activestyle="none",
                        font=("Courier", 9), exportselection=False)
        for i, t in enumerate(boxes):
            lb.insert("end", _annotation_summary(t, i))
        lb.selection_set(0)
        lb.pack(fill="x", pady=(0, 4))

    vis_var = tk.BooleanVar(master=panel, value=True)
    # Declared here rather than beside its OptionMenu below, because _load
    # (defined before the widgets) sets it when a box is selected.
    pos_var = tk.StringVar(master=panel, value=_ANNOT_AS_DRAWN)

    def _current_preset(artist) -> str:
        """
        The preset name matching *artist*'s placement, or "as drawn".

        A match requires axes-fraction anchoring as well as the right
        coordinates: the same numbers in data coordinates mean somewhere
        else entirely.
        """
        try:
            if artist.axes is None or \
                    artist.get_transform() != artist.axes.transAxes:
                return _ANNOT_AS_DRAWN
            x, y = artist.get_position()
            ha = artist.get_horizontalalignment()
            va = artist.get_verticalalignment()
        except Exception:
            return _ANNOT_AS_DRAWN
        for name, (px, py, pha, pva) in _ANNOT_POSITIONS.items():
            if (abs(float(x) - px) < 5e-3 and abs(float(y) - py) < 5e-3
                    and ha == pha and va == pva):
                return name
        return _ANNOT_AS_DRAWN

    def _restore_placement(artist) -> None:
        """Put *artist* back on the anchor the producer drew it with."""
        saved = original_pos.get(id(artist))
        if not saved:
            return
        pos, ha, va, transform = saved
        try:
            artist.set_transform(transform)
            artist.set_position(pos)
            artist.set_horizontalalignment(ha)
            artist.set_verticalalignment(va)
        except Exception:
            pass

    def _draw():
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass

    def _refresh_row(artist):
        """Keep the selector row in step with an edited box."""
        if lb is None or artist not in boxes:
            return
        i = boxes.index(artist)
        sel = lb.curselection()
        lb.delete(i)
        lb.insert(i, _annotation_summary(artist, i))
        if sel and sel[0] == i:
            lb.selection_set(i)

    def _load(artist):
        loading["busy"] = True
        try:
            current["artist"] = artist
            try:
                vis_var.set(bool(artist.get_visible()))
            except Exception:
                vis_var.set(True)
            pos_var.set(_current_preset(artist))
            box.delete("1.0", "end")
            try:
                box.insert("1.0", artist.get_text() or "")
            except Exception:
                pass
        finally:
            loading["busy"] = False

    def _on_select(_event=None):
        if lb is None:
            return
        sel = lb.curselection()
        if sel and sel[0] < len(boxes):
            _load(boxes[sel[0]])

    vis_cb = tk.Checkbutton(lf, text="Show annotation", variable=vis_var,
                            anchor="w", font=("Helvetica", 9))
    vis_cb.pack(fill="x")

    def _apply_visible(*_):
        if loading["busy"]:
            return
        try:
            current["artist"].set_visible(vis_var.get())
        except Exception:
            return
        _draw()

    vis_cb.config(command=_apply_visible)

    pos_row = tk.Frame(lf)
    pos_row.pack(fill="x", pady=(2, 0))
    tk.Label(pos_row, text="Position:", width=9, anchor="w",
             font=("Helvetica", 9)).pack(side="left")
    pos_menu = tk.OptionMenu(pos_row, pos_var, _ANNOT_AS_DRAWN,
                             *_ANNOT_POSITIONS.keys())
    pos_menu.config(font=("Helvetica", 9), width=12)
    pos_menu["menu"].config(font=("Helvetica", 9))
    pos_menu.pack(side="left", padx=4)

    def _apply_position(*_):
        if loading["busy"]:
            return
        artist = current["artist"]
        name = pos_var.get()
        if name == _ANNOT_AS_DRAWN:
            _restore_placement(artist)
            _draw()
            return
        spec = _ANNOT_POSITIONS.get(name)
        if spec is None:
            return
        x, y, ha, va = spec
        try:
            # Anchor to the axes before positioning: a box the producer
            # placed in DATA coordinates would otherwise land at data
            # (0.98, 0.95) -- somewhere arbitrary, usually outside the
            # visible area.  "as drawn" restores the original anchor, so
            # the conversion is never one-way.
            if artist.axes is not None:
                artist.set_transform(artist.axes.transAxes)
            artist.set_position((x, y))
            artist.set_horizontalalignment(ha)
            artist.set_verticalalignment(va)
        except Exception:
            return
        _draw()

    pos_var.trace_add("write", _apply_position)

    box = tk.Text(lf, height=6, width=30, wrap="none",
                  font=("Courier", 9), relief="solid", borderwidth=1)
    box.pack(fill="x", pady=(4, 2))

    def _apply_text(_event=None):
        artist = current["artist"]
        try:
            artist.set_text(box.get("1.0", "end-1c"))
        except Exception:
            return
        _refresh_row(artist)
        _draw()
        return "break"        # keep Ctrl+Return from also inserting a newline

    def _revert(_event=None):
        artist = current["artist"]
        text = original.get(id(artist), "")
        try:
            artist.set_text(text)
        except Exception:
            return
        _restore_placement(artist)
        loading["busy"] = True
        try:
            box.delete("1.0", "end")
            box.insert("1.0", text)
            pos_var.set(_current_preset(artist))
        finally:
            loading["busy"] = False
        _refresh_row(artist)
        _draw()

    btns = tk.Frame(lf)
    btns.pack(pady=(2, 0))
    tk.Button(btns, text="Apply text", command=_apply_text, width=11,
              pady=3).pack(side="left", padx=3)
    tk.Button(btns, text="Revert", command=_revert, width=9,
              pady=3).pack(side="left", padx=3)
    box.bind("<Control-Return>", _apply_text)
    tk.Label(lf, text="      Ctrl+Return also applies · Revert restores the "
                      "fitted\n      text and placement · display only, the "
                      "exported\n      parameter tables keep the fitted "
                      "values",
             font=("Helvetica", 8), fg="grey", anchor="w",
             justify="left").pack(fill="x")

    if lb is not None:
        lb.bind("<<ListboxSelect>>", _on_select)
    _load(boxes[0])


def _build_label_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
    """
    Add the "Labels" LabelFrame to *panel*.

    Title and Y-label come from axes_list[0] (the primary axis).
    X-label comes from axes_list[-1] (the bottom axis in stacked layouts).
    Changes are applied on the "Apply labels" button or Ctrl+Return.
    """
    import tkinter as tk

    ax_main   = axes_list[0]
    ax_bottom = axes_list[-1]

    lf = tk.LabelFrame(panel, text="Labels", padx=10, pady=6)
    lf.pack(fill="x", padx=10, pady=4)

    tk.Label(lf, text="Title:", anchor="w", font=("Helvetica", 9)).pack(
        fill="x", pady=(2, 0))
    title_box = tk.Text(
        lf, height=4, width=30, wrap="word",
        font=("Helvetica", 9), relief="solid", borderwidth=1,
    )
    title_box.insert("1.0", ax_main.get_title())
    title_box.pack(fill="x", pady=(0, 6))

    xlabel_var = tk.StringVar(master=panel, value=ax_bottom.get_xlabel())
    ylabel_var = tk.StringVar(master=panel, value=ax_main.get_ylabel())

    for label_text, var in [("X label:", xlabel_var), ("Y label:", ylabel_var)]:
        row = tk.Frame(lf)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label_text, width=9, anchor="e",
                 font=("Helvetica", 9)).pack(side="left")
        tk.Entry(row, textvariable=var, width=20,
                 font=("Helvetica", 9)).pack(side="left", padx=4)

    def _apply_labels(_event=None):
        new_title = title_box.get("1.0", "end-1c")
        fs = ax_main.title.get_fontsize()
        ax_main.set_title(new_title, fontsize=fs)
        ax_bottom.set_xlabel(xlabel_var.get(), fontsize=12)
        ax_main.set_ylabel(ylabel_var.get(), fontsize=12)
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass

    tk.Button(lf, text="Apply labels", command=_apply_labels,
              width=14, pady=3).pack(pady=(6, 2))
    title_box.bind("<Control-Return>", _apply_labels)
    tk.Label(lf, text="Ctrl+Return in title box also applies",
             font=("Helvetica", 8), fg="grey").pack()


def _build_save_section(panel, fig: plt.Figure) -> None:
    """
    Add a "Save figure…" button that opens a file-save dialog.

    Defaults to SVG (see _FIGURE_FORMAT); PNG and PDF remain available from
    the file-type list.
    """
    import tkinter as tk
    from tkinter import filedialog

    sf = tk.LabelFrame(panel, text="Export", padx=10, pady=4)
    sf.pack(fill="x", padx=10, pady=(4, 10))

    def _save():
        path = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=f".{_FIGURE_FORMAT}",
            filetypes=[
                ("SVG vector",   "*.svg"),
                ("PDF document", "*.pdf"),
                ("PNG image",    "*.png"),
                ("All files",    "*.*"),
            ],
        )
        if not path:
            return
        if save_figure(fig, path) is None:
            from tkinter import messagebox
            messagebox.showerror(
                "Save failed",
                f"Could not write {path}.\n\nSee the console for details.",
                parent=panel)

    tk.Button(sf, text="Save figure\u2026", command=_save,
              width=14, pady=3).pack()
