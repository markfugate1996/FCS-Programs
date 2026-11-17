#!/usr/bin/env python3
"""
Independent X/Y label font sizes in the Labels section, and the Legend
text-size slider replaced by the same spin box the title uses.

Usage:  python3 patch_label_legend_fonts.py [path/to/fcs_plottools.py]
"""
import pathlib
import sys

# ── Labels: a size box per axis label ─────────────────────────────────────
OLD_1 = '''    for label_text, var in [("X label:", xlabel_var), ("Y label:", ylabel_var)]:
        row = tk.Frame(lf)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label_text, width=9, anchor="e",
                 font=("Helvetica", 9)).pack(side="left")
        tk.Entry(row, textvariable=var, width=20,
                 font=("Helvetica", 9)).pack(side="left", padx=4)
'''
NEW_1 = '''    # Each axis label carries its own size, independent of the title and of
    # each other: a y label that has to fit a tall panel is not the same
    # problem as an x label under a wide one.  Seeded from what the figure
    # actually uses rather than from a constant, so a producer that chose
    # its own size keeps it until the box is changed.
    xlabel_fs_var = tk.StringVar(
        master=panel, value=_fmt_fontsize(ax_bottom.xaxis.label.get_fontsize()))
    ylabel_fs_var = tk.StringVar(
        master=panel, value=_fmt_fontsize(ax_main.yaxis.label.get_fontsize()))

    for label_text, var, fs_var in [("X label:", xlabel_var, xlabel_fs_var),
                                    ("Y label:", ylabel_var, ylabel_fs_var)]:
        row = tk.Frame(lf)
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label_text, width=9, anchor="e",
                 font=("Helvetica", 9)).pack(side="left")
        tk.Entry(row, textvariable=var, width=16,
                 font=("Helvetica", 9)).pack(side="left", padx=4)
        tk.Label(row, text="Size:", anchor="e",
                 font=("Helvetica", 9)).pack(side="left", padx=(4, 2))
        tk.Spinbox(row, textvariable=fs_var, from_=4, to=48, increment=1,
                   width=5, font=("Helvetica", 9),
                   justify="right").pack(side="left")
'''

OLD_2 = '''        ax_main.set_title(new_title, fontsize=fs)
        ax_bottom.set_xlabel(xlabel_var.get(), fontsize=12)
        ax_main.set_ylabel(ylabel_var.get(), fontsize=12)
'''
NEW_2 = '''        ax_main.set_title(new_title, fontsize=fs)
        # Previously both were forced to 12 on every apply, which silently
        # restyled a figure whose producer had chosen otherwise.  They now
        # fall back to the size the axis already has.
        xfs = _parse_fontsize(xlabel_fs_var.get(),
                              ax_bottom.xaxis.label.get_fontsize())
        yfs = _parse_fontsize(ylabel_fs_var.get(),
                              ax_main.yaxis.label.get_fontsize())
        xlabel_fs_var.set(_fmt_fontsize(xfs))
        ylabel_fs_var.set(_fmt_fontsize(yfs))
        ax_bottom.set_xlabel(xlabel_var.get(), fontsize=xfs)
        ax_main.set_ylabel(ylabel_var.get(), fontsize=yfs)
'''

# ── Legend: spin box in place of the slider ───────────────────────────────
OLD_3 = '''    size_var = tk.IntVar(master=panel, value=current_size)
'''
NEW_3 = '''    # Text, not an int var: the box is typed into, and a stray keystroke
    # should fall back rather than raise out of IntVar.get().
    size_var = tk.StringVar(master=panel, value=_fmt_fontsize(current_size))
    _legend_fs_last = {"v": float(current_size)}
'''

OLD_4 = '''    def _apply_legend(*_):
        _refresh_entries_list()
'''
NEW_4 = '''    def _legend_fontsize() -> float:
        """
        The size in the box, falling back to the last one that took.

        Falling back to *current_size* instead would throw away a size the
        user had already applied the moment they fat-fingered the next one.
        """
        fs = _parse_fontsize(size_var.get(), _legend_fs_last["v"])
        _legend_fs_last["v"] = fs
        size_var.set(_fmt_fontsize(fs))
        return fs

    def _apply_legend(*_):
        _refresh_entries_list()
'''

OLD_5 = '''        new = ax.legend(
            live_h, live_l,
            loc=loc_var.get(),
            fontsize=size_var.get(),
            title=title_text,
            title_fontsize=size_var.get(),
'''
NEW_5 = '''        _fs = _legend_fontsize()
        new = ax.legend(
            live_h, live_l,
            loc=loc_var.get(),
            fontsize=_fs,
            title=title_text,
            title_fontsize=_fs,
'''

OLD_6 = '''    tk.Scale(
        size_row, from_=4, to=20, orient="horizontal",
        variable=size_var, command=_apply_legend,
        showvalue=True, length=130, font=("Helvetica", 8),
    ).pack(side="left", padx=4, fill="x", expand=True)
'''
NEW_6 = '''    # A spin box, matching the title and axis-label size controls in the
    # Labels section: one idiom for every font size in the panel.
    size_box = tk.Spinbox(
        size_row, textvariable=size_var, from_=4, to=48, increment=1,
        width=5, font=("Helvetica", 9), justify="right",
        command=_apply_legend,
    )
    size_box.pack(side="left", padx=4)
    tk.Label(size_row, text="pt", font=("Helvetica", 9)).pack(side="left")
    # The arrows apply through command=; typing applies on Return or on
    # leaving the box.  Deliberately NOT a trace on the variable: that fires
    # per keystroke, so typing "14" would rebuild the legend at size 1 first.
    size_box.bind("<Return>", _apply_legend)
    size_box.bind("<FocusOut>", _apply_legend)
'''

EDITS = [("axis label size boxes", OLD_1, NEW_1),
         ("apply labels uses the axis sizes", OLD_2, NEW_2),
         ("legend size held as text", OLD_3, NEW_3),
         ("legend size parser", OLD_4, NEW_4),
         ("legend rebuild uses the parsed size", OLD_5, NEW_5),
         ("legend spin box replaces the slider", OLD_6, NEW_6)]


def _target(argv):
    arg = argv[1] if len(argv) > 1 else "."
    p = pathlib.Path(arg)
    if p.is_dir():
        p = p / "fcs_plottools.py"
    if not p.is_file():
        print(f"ERROR: {p} not found. Run this from the FCS-Programs "
              f"directory, or pass the path to fcs_plottools.py.")
        raise SystemExit(2)
    return p


def main(argv) -> int:
    path = _target(argv)
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
    sys.exit(main(sys.argv))
