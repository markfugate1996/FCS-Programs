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


def show_figure(
    fig: plt.Figure,
    axes=None,
) -> None:
    """
    Display *fig* and attach a live-edit controls panel beside it.

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

    # ── Build the controls panel ──────────────────────────────────────────
    panel = tk.Toplevel(tk_win)
    panel.title("Plot controls")
    panel.resizable(False, True)

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
    _build_limits_section(body, fig, axes_list)
    _legend_cb["fn"] = _build_legend_section(body, fig, axes_list)
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
            a.relim()
            a.autoscale_view()
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


def _build_legend_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
):
    """
    Add the "Legend" LabelFrame to *panel*, if the main axis has a legend.

    Offers three controls:
      * Show legend — a checkbox to hide/show it (useful when an overlay of
        many datasets produces an enormous legend).
      * Position — dropdown to move it.
      * Text size — slider to shrink or grow it.

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

    def _live_entries():
        """
        Current (handles, labels) for the legend.

        Built from the snapshot first, so proxy artists keep their original
        order and survive, then extended with any axes artist that has since
        been given a real label.  Labels come from the artists themselves;
        the snapshot text is only a fallback for proxies whose label was set
        on the legend rather than on the handle.
        """
        out_h, out_l, seen = [], [], set()
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
            out_h.append(h)
            out_l.append(lbl)

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
                    out_h.append(art)
                    out_l.append(lbl)
        return out_h, out_l

    def _apply_legend(*_):
        live_h, live_l = _live_entries()
        if not live_h:
            # Everything hidden: drop the legend rather than draw an empty box.
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

    loc_var.trace_add("write", _apply_legend)

    return _apply_legend


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
    Supports PNG, PDF, and SVG.
    """
    import tkinter as tk
    from tkinter import filedialog

    sf = tk.LabelFrame(panel, text="Export", padx=10, pady=4)
    sf.pack(fill="x", padx=10, pady=(4, 10))

    def _save():
        path = filedialog.asksaveasfilename(
            title="Save figure",
            defaultextension=".png",
            filetypes=[
                ("PNG image",    "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG vector",   "*.svg"),
                ("All files",    "*.*"),
            ],
        )
        if not path:
            return
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Save failed", str(exc), parent=panel)

    tk.Button(sf, text="Save figure\u2026", command=_save,
              width=14, pady=3).pack()
