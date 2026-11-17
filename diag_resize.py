"""
Diagnostic for the figure-window resize, to run ON YOUR MACHINE if
"Apply size" still misbehaves.  Opens one small figure, resizes it
programmatically, and reports what Tk and matplotlib each think happened.

    python3 diag_resize.py          (from the FCS-Programs directory)

Close the figure window when it finishes.  Paste the output.
"""
import sys

import matplotlib
import matplotlib.pyplot as plt

import fcs_plottools


def report(tag, fig):
    print(f"\n--- {tag} ---")
    w, h = fig.get_size_inches()
    print(f"figure      : {w:.3f} x {h:.3f} in  @ dpi {fig.dpi:g}  "
          f"-> {w * fig.dpi:.0f} x {h * fig.dpi:.0f} px")
    canvas = fig.canvas
    try:
        widget = canvas.get_tk_widget()
        print(f"canvas widget: actual {widget.winfo_width()} x "
              f"{widget.winfo_height()} px, "
              f"requested {widget.winfo_reqwidth()} x "
              f"{widget.winfo_reqheight()} px")
    except Exception as exc:
        print(f"canvas widget: unavailable ({exc})")
        return
    try:
        win = canvas.manager.window
        print(f"window       : geometry {win.winfo_geometry()}  "
              f"(wm {win.geometry()})")
    except Exception as exc:
        print(f"window       : unavailable ({exc})")
    photo = getattr(canvas, "_tkphoto", None)
    if photo is not None:
        try:
            print(f"backend photo: {photo.width()} x {photo.height()} px "
                  "(should match the figure px above; if it is larger, the "
                  "old plot is what shows through)")
        except Exception as exc:
            print(f"backend photo: unreadable ({exc})")
    else:
        print("backend photo: no _tkphoto attribute on this backend")


def main():
    print(f"matplotlib {matplotlib.__version__}  backend "
          f"{matplotlib.get_backend()}  python {sys.version.split()[0]}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title("resize diagnostic")
    plt.show(block=False)
    try:
        fig.canvas.manager.window.update()
    except Exception:
        pass
    fig.canvas.draw()
    report("before resize (9 x 4)", fig)

    got = fcs_plottools.resize_figure_window(fig, 3.35, 2.60)
    print(f"\nresize_figure_window returned {got[0]:.3f} x {got[1]:.3f} in")
    try:
        fig.canvas.manager.window.update()
    except Exception:
        pass
    report("after resize (asked 3.35 x 2.60)", fig)

    print("\nLook at the window: is it 3.35 x 2.60 in, with one clean plot?")
    try:
        plt.show(block=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
