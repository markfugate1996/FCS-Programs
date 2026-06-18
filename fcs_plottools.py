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

from typing import Union
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

    _build_scale_section(panel, fig, axes_list)
    _build_limits_section(panel, fig, axes_list)
    _build_legend_section(panel, fig, axes_list)
    _build_label_section(panel, fig, axes_list)
    _build_save_section(panel, fig)

    # ── Position panel to the right of the figure (deferred) ─────────────
    def _position_panel():
        try:
            tk_win.update_idletasks()
            fx = tk_win.winfo_x()
            fy = tk_win.winfo_y()
            fw = tk_win.winfo_width()
            sw = tk_win.winfo_screenwidth()
            panel_w = 280
            px = fx + fw + 6
            if px + panel_w > sw:
                px = max(0, fx - panel_w - 6)
            panel.geometry(f"+{px}+{fy}")
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


# ── Section builders ──────────────────────────────────────────────────────────

def _build_scale_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
    """
    Add the "Axis scale" LabelFrame to *panel*.

    X-scale changes are applied to every axis in axes_list (so shared-x
    subplot pairs stay in sync).  Y-scale changes are applied to axes[0]
    only (keeps residuals / overlay panels at their intended scale).
    """
    import tkinter as tk
    from tkinter import messagebox

    ax0 = axes_list[0]

    sf = tk.LabelFrame(panel, text="Axis scale", padx=10, pady=6)
    sf.pack(fill="x", padx=10, pady=(10, 4))

    x_var = tk.StringVar(master=panel, value=ax0.get_xscale())
    y_var = tk.StringVar(master=panel, value=ax0.get_yscale())

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
            axes_list[0].set_yscale(ys)
        except Exception as e:
            y_var.set(axes_list[0].get_yscale())
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


def _build_limits_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
    """
    Add the "Axis limits" LabelFrame to *panel*.

    X-limits are applied to all axes; Y-limits to axes[0] only.
    "Auto" calls autoscale_view() and refreshes the displayed values.
    """
    import tkinter as tk
    from tkinter import messagebox

    ax = axes_list[0]

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
            axes_list[0].set_ylim(ylo, yhi)
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
        ylo, yhi = axes_list[0].get_ylim()
        xlo_var.set(f"{xlo:.6g}")
        xhi_var.set(f"{xhi:.6g}")
        ylo_var.set(f"{ylo:.6g}")
        yhi_var.set(f"{yhi:.6g}")
        try:
            fig.canvas.draw()
        except Exception:
            pass

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
) -> None:
    """
    Add the "Legend" LabelFrame to *panel*, if the main axis has a legend.

    Offers three controls:
      * Show legend — a checkbox to hide/show it (useful when an overlay of
        many datasets produces an enormous legend).
      * Position — dropdown to move it.
      * Text size — slider to shrink or grow it.

    Position and size changes rebuild the legend from a snapshot of its
    handles/labels captured when the panel is built; the snapshot is taken
    from the legend object itself (not the axes), so custom entries such as
    the intensity plot's "Mean Ch1/Ch2 CPS" rows are preserved.
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

    show_var = tk.BooleanVar(master=panel, value=True)
    loc_var  = tk.StringVar(master=panel, value=current_loc)
    size_var = tk.IntVar(master=panel, value=current_size)

    lf = tk.LabelFrame(panel, text="Legend", padx=10, pady=6)
    lf.pack(fill="x", padx=10, pady=4)

    def _apply_legend(*_):
        if not handles:
            return
        new = ax.legend(
            handles, labels,
            loc=loc_var.get(),
            fontsize=size_var.get(),
            title=title_text,
            title_fontsize=size_var.get(),
            framealpha=framealpha,
        )
        new.set_visible(show_var.get())
        try:
            fig.canvas.draw()
        except Exception:
            pass

    # ── Show / hide ───────────────────────────────────────────────────────────
    show_cb = tk.Checkbutton(
        lf, text="Show legend", variable=show_var,
        command=_apply_legend, anchor="w", font=("Helvetica", 9),
    )
    show_cb.pack(fill="x")

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
