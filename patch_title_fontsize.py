#!/usr/bin/env python3
"""
Add a title font-size control to the Labels section of the plot controls
panel (fcs_plottools._build_label_section).

Line-ending preserving: operates on bytes, anchors are matched in whatever
line ending the working-tree file uses.

Usage:  python3 patch_title_fontsize.py path/to/fcs_plottools.py
"""
import sys
import pathlib

OLD_1_LF = '''    tk.Label(lf, text="Title:", anchor="w", font=("Helvetica", 9)).pack(
        fill="x", pady=(2, 0))
'''

NEW_1_LF = '''    title_row = tk.Frame(lf)
    title_row.pack(fill="x", pady=(2, 0))
    tk.Label(title_row, text="Title:", anchor="w",
             font=("Helvetica", 9)).pack(side="left")
    tk.Label(title_row, text="Size:", anchor="e",
             font=("Helvetica", 9)).pack(side="left", padx=(10, 2))
    title_fs_var = tk.StringVar(
        master=panel, value=_fmt_fontsize(ax_main.title.get_fontsize()))
    tk.Spinbox(title_row, textvariable=title_fs_var,
               from_=4, to=48, increment=1, width=5,
               font=("Helvetica", 9), justify="right").pack(side="left")
'''

OLD_2_LF = '''    def _apply_labels(_event=None):
        new_title = title_box.get("1.0", "end-1c")
        fs = ax_main.title.get_fontsize()
        ax_main.set_title(new_title, fontsize=fs)
'''

NEW_2_LF = '''    def _apply_labels(_event=None):
        new_title = title_box.get("1.0", "end-1c")
        fs = _parse_fontsize(title_fs_var.get(), ax_main.title.get_fontsize())
        title_fs_var.set(_fmt_fontsize(fs))
        ax_main.set_title(new_title, fontsize=fs)
'''

OLD_3_LF = '''def _build_label_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
'''

NEW_3_LF = '''def _fmt_fontsize(value: float) -> str:
    """Render a point size for display: 10.0 -> "10", 10.5 -> "10.5"."""
    return f"{float(value):g}"


def _parse_fontsize(text: str, fallback: float) -> float:
    """
    Parse a point size typed into the title size box.

    Anything that is not a positive number falls back to *fallback* (the
    size currently on the axis), so a stray keystroke cannot silently
    collapse the title.
    """
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return float(fallback)
    if not (0.0 < value <= 200.0):
        return float(fallback)
    return value


def _build_label_section(
    panel,
    fig: plt.Figure,
    axes_list: list,
) -> None:
'''

EDITS = [("title row + size spinbox", OLD_1_LF, NEW_1_LF),
         ("_apply_labels reads the spinbox", OLD_2_LF, NEW_2_LF),
         ("font-size helpers", OLD_3_LF, NEW_3_LF)]


def main(path_str: str) -> int:
    path = pathlib.Path(path_str)
    data = path.read_bytes()

    crlf = data.count(b"\r\n")
    lf_only = data.count(b"\n") - crlf
    if crlf and lf_only:
        print(f"ERROR: mixed line endings ({crlf} CRLF, {lf_only} LF) - "
              "refusing to guess")
        return 2
    newline = b"\r\n" if crlf else b"\n"
    print(f"line endings: {'CRLF' if crlf else 'LF'}")

    def enc(text: str) -> bytes:
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
