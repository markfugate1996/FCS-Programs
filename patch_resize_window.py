#!/usr/bin/env python3
"""
Make "Apply size" resize the figure WINDOW, not just the figure.

fig.set_size_inches(..., forward=True) only reaches
FigureManagerTk.resize(), which configures the canvas widget's requested
size and leaves the Toplevel's geometry alone -- so the widget does not
move, no <Configure> fires, the backend's PhotoImage is never rebuilt, and
the next draw paints the new figure over the old one.

Usage:  python3 patch_resize_window.py path/to/fcs_plottools.py
"""
import pathlib
import sys

OLD_1 = '''def place_window(win, x=None, y=None, w=None, h=None) -> tuple:
'''
NEW_1 = '''def resize_figure_window(fig, w_in: float, h_in: float) -> tuple:
    """
    Resize *fig* to *w_in* x *h_in* inches AND the window that displays it.

    ``set_size_inches(..., forward=True)`` is not enough on TkAgg.  The whole
    of its forward path is ``FigureManagerTk.resize``, which configures the
    canvas WIDGET's requested size; the Toplevel keeps the geometry it was
    given, so the widget never actually changes size.  No <Configure> event
    follows -- and that event is what rebuilds the backend's PhotoImage.  The
    next draw then blits the new, smaller figure into the corner of the old
    photo, which is the "new plot drawn on top of the old one" artefact.

    So the window is resized here too.  Clearing the geometry string lets Tk
    shrink-wrap the Toplevel around its children (canvas plus whatever
    toolbar it carries) rather than hold a stale explicit geometry; the
    <Configure> that follows re-derives the figure size from the canvas's
    real pixel size.

    Returns the (width, height) in inches the figure actually ended up with.
    It is read back, not assumed: pixels are whole, so a requested size is
    quantised to 1/dpi, and a window manager may refuse a window bigger than
    the screen.
    """
    fig.set_size_inches(float(w_in), float(h_in), forward=False)

    dpi = float(getattr(fig, "dpi", 0) or 100.0)
    cw = max(1, int(round(float(w_in) * dpi)))
    ch = max(1, int(round(float(h_in) * dpi)))

    canvas = getattr(fig, "canvas", None)
    widget = None
    if canvas is not None and hasattr(canvas, "get_tk_widget"):
        try:
            widget = canvas.get_tk_widget()
        except Exception:
            widget = None

    if widget is None:
        # Non-Tk backend: a batch export on Agg, or a headless test.  There is
        # no window, and set_size_inches above was the whole job.
        try:
            canvas.draw()
        except Exception:
            pass
        return tuple(float(v) for v in fig.get_size_inches())

    try:
        widget.configure(width=cw, height=ch)
        win = canvas.manager.window
        win.geometry("")             # shrink-wrap to the new requested size
        win.update_idletasks()       # let the <Configure> land before measuring
    except Exception as exc:
        # Not fatal: the figure is the requested size even if its window is
        # not, and the caller reports the discrepancy.  Silence would leave a
        # window that disagrees with its own size box and no clue why.
        print(f"[plot] could not resize the figure window ({exc})")

    try:
        canvas.draw()
    except Exception:
        pass
    return tuple(float(v) for v in fig.get_size_inches())


def place_window(win, x=None, y=None, w=None, h=None) -> tuple:
'''

OLD_2 = '''        w = _parse_size_in(w_var.get(), cw)
        h = _parse_size_in(h_var.get(), ch)
        try:
            fig.set_size_inches(w, h, forward=True)
        except Exception as exc:
            print(f"[plot] could not resize figure ({exc})")
            _sync()
            return
        _relayout_after_resize(fig)
        try:
            # Synchronous, not draw_idle: the readouts measure the result.
            fig.canvas.draw()
        except Exception:
            pass
        _sync(requested=(w, h))
'''
NEW_2 = '''        w = _parse_size_in(w_var.get(), cw)
        h = _parse_size_in(h_var.get(), ch)
        try:
            resize_figure_window(fig, w, h)
        except Exception as exc:
            print(f"[plot] could not resize figure ({exc})")
            _sync()
            return
        _relayout_after_resize(fig)
        try:
            # Synchronous, not draw_idle: the readouts measure the result.
            fig.canvas.draw()
        except Exception:
            pass
        _sync(requested=(w, h))
'''

OLD_3 = '''        req_w, req_h = requested
        if abs(got_w - req_w) > 0.01 or abs(got_h - req_h) > 0.01:
'''
NEW_3 = '''        req_w, req_h = requested
        # A canvas is a whole number of pixels, so an achievable size is
        # quantised to 1/dpi inch; warning about that rounding would cry wolf
        # on every apply.  Anything beyond a pixel or so is a real refusal.
        tol = max(0.02, 1.5 / float(getattr(fig, "dpi", 100) or 100))
        if abs(got_w - req_w) > tol or abs(got_h - req_h) > tol:
'''

EDITS = [("resize_figure_window helper", OLD_1, NEW_1),
         ("Apply size resizes the window too", OLD_2, NEW_2),
         ("pixel-aware tolerance for the clamp warning", OLD_3, NEW_3)]


def main(path_str: str) -> int:
    path = pathlib.Path(path_str)
    data = path.read_bytes()
    crlf = data.count(b"\r\n")
    lf_only = data.count(b"\n") - crlf
    if crlf and lf_only:
        print(f"ERROR: mixed line endings ({crlf} CRLF, {lf_only} LF)")
        return 2
    newline = b"\r\n" if crlf else b"\n"
    print(f"line endings: {'CRLF' if crlf else 'LF'}")

    def enc(text):
        return text.encode("utf-8").replace(b"\n", newline)

    for name, old_lf, new_lf in EDITS:
        old, new = enc(old_lf), enc(new_lf)
        if data.count(new) and not data.count(old):
            print(f"ERROR: '{name}' looks already applied")
            return 2
        n = data.count(old)
        if n != 1:
            print(f"ERROR: anchor for '{name}' matched {n} times, expected 1")
            return 2
        data = data.replace(old, new, 1)
        print(f"  applied: {name}")

    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
