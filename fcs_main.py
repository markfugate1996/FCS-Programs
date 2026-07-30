"""
fcs_main.py
===========
Main entry point for the FCS analysis suite.

Run this file to open the main menu:

    python fcs_main.py

The main window contains an embedded Workspace panel for managing files,
and an analysis Tasks panel below it.  All analysis tasks operate on
whichever file is currently marked active (► in the list).
"""

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fcs_reader import load_dataset, FCSData
import fcs_plot
import fcs_lifetime
import fcs_lifetime_fit
import fcs_lifetime_recon
import fcs_corr
import fcs_pch
import fcs_fit
import fcs_calib
import fcs_plotopen


# ── Global workspace state ────────────────────────────────────────────────────

workspace:  dict[str, FCSData] = {}
active_key: str | None = None


# ── Workspace helpers ─────────────────────────────────────────────────────────

def _unique_key(path: Path) -> str:
    name = path.name
    if name not in workspace:
        return name
    if workspace[name].filepath.resolve() == path.resolve():
        return name
    i = 2
    while True:
        candidate = f"{path.stem}({i}){path.suffix}"
        if candidate not in workspace:
            return candidate
        i += 1


def _add_file(path: Path) -> str:
    key = _unique_key(path)
    #if key not in workspace:
    #    workspace[key] = read_fcs(path)
    # changed 20260624
    if key not in workspace:
        workspace[key] = load_dataset(path)
    return key


def _remove_file(key: str):
    global active_key
    if key not in workspace:
        return
    del workspace[key]
    if active_key == key:
        active_key = next(reversed(workspace), None)


def _set_active(key: str):
    global active_key
    if key in workspace:
        active_key = key


def _active_data() -> FCSData | None:
    if active_key and active_key in workspace:
        return workspace[active_key]
    return None


#def _file_summary(d: FCSData) -> str:
#    return (
#        f"{d.filepath.name}\n"
#        f"  Duration : {d.duration_s:.2f} s\n"
#        f"  Ch1      : {len(d.ch1_deltas):,} photons  "
#        f"({d.count_rate_ch1_hz:,.0f} CPS)\n"
#        f"  Ch2      : {len(d.ch2_deltas):,} photons  "
#        f"({d.count_rate_ch2_hz:,.0f} CPS)"
#    )

def _file_summary(d) -> str:
    # Lifetime-decay datasets (.ifx) carry no photon-record fields; summarise
    # them by their decay/IRF shape instead.
    if getattr(d, "kind", None) == "lifetime_decay":
        title = d.params.get("title") or d.filepath.stem
        return (
            f"{d.filepath.name}\n"
            f"  Type     : lifetime decay (.ifx)\n"
            f"  Title    : {title}\n"
            f"  Bins     : {d.n_bins}  (laser period {d.laser_period_ns:.3f} ns)\n"
            f"  IRF      : {'present' if d.has_irf else 'none'}"
        )
    # List only the channels the file actually recorded.  Showing
    # "Ch2 : 0 photons" for a single-channel acquisition would be
    # indistinguishable from a dual-channel one with a dead detector, and
    # those need to stay distinguishable at a glance.
    channels = tuple(getattr(d, "channels", (1, 2)))
    lines = [
        f"{d.filepath.name}",
        f"  Duration : {d.duration_s:.2f} s",
    ]
    for ch in channels:
        n_ph = len(d.ch1_deltas if ch == 1 else d.ch2_deltas)
        cps  = d.count_rate_ch1_hz if ch == 1 else d.count_rate_ch2_hz
        lines.append(f"  Ch{ch}      : {n_ph:,} photons  ({cps:,.0f} CPS)")
    if len(channels) < 2:
        lines.append(f"  Channels : single-channel file "
                     f"({getattr(d, 'channel_summary', 'one channel')})")
    return "\n".join(lines)



# ── Status bar ────────────────────────────────────────────────────────────────

def status_set(message: str):
    status_text.config(state="normal")
    status_text.delete("1.0", tk.END)
    status_text.insert(tk.END, message)
    status_text.config(state="disabled")


def status_refresh():
    if not workspace:
        status_set("No files loaded.  Use 'Add file(s)' to get started.")
        return
    d = _active_data()
    if d is None:
        status_set(f"{len(workspace)} file(s) in workspace.  No active file selected.")
        return
    header = (
        f"Active file  ({len(workspace)} in workspace):\n"
        f"{'─' * 44}\n"
    )
    status_set(header + _file_summary(d))


# ── Workspace panel actions ───────────────────────────────────────────────────

def ws_add():
    paths = filedialog.askopenfilenames(
        title="Add FCS files to workspace",
        #filetypes=[("FCS data files", "*.fcs"), ("All files", "*.*")],
        filetypes=[("All files", "*.*"), ("FCS data files", "*.fcs")],
    )
    if not paths:
        return
    errors = []
    last_key = None
    for p in paths:
        try:
            last_key = _add_file(Path(p))
        except Exception as e:
            errors.append(f"{Path(p).name}: {e}")
    if errors:
        messagebox.showerror("Some files could not be loaded", "\n".join(errors))
    global active_key
    if active_key is None and last_key:
        active_key = last_key
    ws_refresh_list()
    if last_key and last_key in workspace:
        idx = list(workspace.keys()).index(last_key)
        ws_listbox.selection_clear(0, tk.END)
        ws_listbox.selection_set(idx)
        ws_listbox.see(idx)
    status_refresh()


def ws_remove():
    keys = ws_selected_keys()
    if not keys:
        messagebox.showinfo("Nothing selected", "Select one or more files in the list first.")
        return
    if len(keys) == 1:
        prompt = f"Remove '{keys[0]}' from the workspace?\n(The file on disk is not affected.)"
    else:
        listed = "\n".join(f"  • {k}" for k in keys)
        prompt = (
            f"Remove these {len(keys)} files from the workspace?\n"
            f"(The files on disk are not affected.)\n\n{listed}"
        )
    if not messagebox.askyesno("Remove file(s)", prompt):
        return
    for key in keys:
        _remove_file(key)
    ws_refresh_list()
    status_refresh()


def ws_set_active():
    key = ws_selected_key()
    if key is None:
        messagebox.showinfo("Nothing selected", "Select a file in the list first.")
        return
    _set_active(key)
    ws_refresh_list()
    status_refresh()


def _key_from_label(raw: str) -> str:
    """Strip the active-marker / indent prefix from a listbox row label."""
    return raw[2:] if raw.startswith("► ") else raw[3:]


def ws_selected_key() -> str | None:
    """Return the first selected key, or None if nothing is selected."""
    sel = ws_listbox.curselection()
    if not sel:
        return None
    return _key_from_label(ws_listbox.get(sel[0]))


def ws_selected_keys() -> list[str]:
    """Return all currently selected keys, in listbox (workspace) order."""
    return [_key_from_label(ws_listbox.get(i)) for i in ws_listbox.curselection()]


def ws_refresh_list():
    """Repopulate the listbox from the current workspace dict."""
    # Remember which key was selected so we can restore it
    prev_key = ws_selected_key()

    ws_listbox.delete(0, tk.END)
    new_sel_idx = None
    for i, key in enumerate(workspace):
        label = f"► {key}" if key == active_key else f"   {key}"
        ws_listbox.insert(tk.END, label)
        if key == active_key:
            ws_listbox.itemconfig(i, fg="steelblue")
        if key == prev_key:
            new_sel_idx = i

    if new_sel_idx is not None:
        ws_listbox.selection_set(new_sel_idx)
        ws_listbox.see(new_sel_idx)


def ws_on_select(event=None):
    """Update the status bar when the user clicks a row in the list."""
    keys = ws_selected_keys()

    # Multi-selection: show the batch queue instead of a single file's info.
    if len(keys) >= 2:
        listed = "\n".join(f"  • {k}" for k in keys)
        status_set(
            f"{len(keys)} files selected for batch:\n"
            f"{'─' * 44}\n"
            f"{listed}\n\n"
            f"Click 'Compute Correlation' to run each in turn."
        )
        return

    key = keys[0] if keys else None
    if key and key in workspace:
        # Show this file's info in the status bar without changing active_key
        d = workspace[key]
        prefix = (
            f"Selected: {key}"
            + ("  [active]\n" if key == active_key else "\n")
            + f"{'─' * 44}\n"
        )
        status_set(prefix + _file_summary(d))
    else:
        status_refresh()


# ── Analysis task functions ───────────────────────────────────────────────────

def task_plot_intensity():
    datasets = _selected_datasets()
    if len(datasets) >= 2:
        params = _ask_intensity_batch(len(datasets))
        if params is None:
            return
        bin_width_s, mode = params
        if mode == "combined":
            status_set(f"Combined intensity overlay — {len(datasets)} files")
            root.update_idletasks()
            try:
                fcs_plot.plot_intensity_overlay(
                    datasets, bin_width_s=bin_width_s, export=export_var.get())
            except Exception as e:
                messagebox.showerror("Plot error", str(e))
            status_refresh()
        else:
            _run_separate(
                datasets, "intensity",
                lambda d: fcs_plot.plot_intensity(
                    d, bin_width_s=bin_width_s, export=export_var.get()),
            )
        return

    d = _active_data()
    if d is None:
        _no_active_file_warning()
        return
    bin_width_s = _ask_intensity_single()
    if bin_width_s is None:
        return
    fcs_plot.plot_intensity(d, bin_width_s=bin_width_s, export=export_var.get())


def task_plot_lifetime():
    datasets = _selected_datasets()
    if len(datasets) >= 2:
        params = _ask_lifetime_batch(len(datasets))
        if params is None:
            return
        n_bins, mode = params
        if mode == "combined":
            status_set(f"Combined lifetime overlay — {len(datasets)} files")
            root.update_idletasks()
            try:
                fcs_lifetime.plot_lifetime_overlay(
                    datasets, n_bins=n_bins, export=export_var.get())
            except Exception as e:
                messagebox.showerror("Plot error", str(e))
            status_refresh()
        else:
            _run_separate(
                datasets, "lifetime",
                lambda d: fcs_lifetime.plot_lifetime(
                    d, n_bins=n_bins, export=export_var.get()),
            )
        return

    d = _active_data()
    if d is None:
        _no_active_file_warning()
        return

    # .ifx decays are already binned — no bin chooser; plot directly.
    if getattr(d, "kind", None) == "lifetime_decay":
        fcs_lifetime.plot_lifetime(d, export=export_var.get())
        return

    dialog = tk.Toplevel(root)
    dialog.title("Lifetime histogram — options")
    dialog.geometry("280x160")
    dialog.resizable(False, False)
    dialog.grab_set()

    tk.Label(dialog, text="Number of bins:", font=("Helvetica", 11), pady=10).pack()

    bin_var = tk.IntVar(value=4096)
    bin_menu = tk.OptionMenu(dialog, bin_var, *fcs_lifetime._VALID_N_BINS)
    bin_menu.config(width=14)
    bin_menu.pack()

    tk.Label(
        dialog,
        text="Fewer bins → smoother, noisier at high resolution.\n4096 = native (~12 ps / bin).",
        font=("Helvetica", 9), fg="grey", justify="center", pady=6,
    ).pack()

    def _on_ok():
        dialog.destroy()
        fcs_lifetime.plot_lifetime(d, n_bins=bin_var.get(), export=export_var.get())

    tk.Button(dialog, text="Plot", width=12, command=_on_ok, pady=4).pack()


def _no_active_file_warning():
    if workspace:
        messagebox.showinfo(
            "No active file",
            "Select a file in the workspace list and click 'Set as active'.",
        )
    else:
        messagebox.showinfo(
            "No data loaded",
            "Use 'Add file(s)' to load an FCS file first.",
        )


# ── Batch plotting (Intensity / Lifetime / PCH) ───────────────────────────────
#
# When two or more files are selected (ctrl/shift-click) and a batch-capable
# task is run, a single dialog collects that task's parameters plus an output
# mode: "separate" produces one figure per file; "combined" overlays all files
# on one figure.  Parameters are asked once and applied to every file so the
# combined overlay is comparable and the separate run isn't a dialog per file.
# (Correlation keeps its own per-file flow because of its interactive gate.)

_batch_defaults: dict = {
    "mode":               "separate",
    "n_bins":             4096,
    "pch_channel":        "both",
    "pch_bw_label":       None,    # filled from the PCH module default on first use
    "intensity_bw_label": None,    # filled from the intensity module default
}


def _selected_datasets() -> list:
    """Return the FCSData objects for the currently selected workspace rows."""
    return [workspace[k] for k in ws_selected_keys() if k in workspace]


def _add_output_mode_section(dialog) -> "tk.StringVar":
    """Add the shared 'Output: separate / combined' radio block to a dialog."""
    frame = tk.LabelFrame(dialog, text="Output", padx=10, pady=4)
    frame.pack(fill="x", padx=12, pady=6)
    mode_var = tk.StringVar(value=_batch_defaults["mode"])
    tk.Radiobutton(frame, text="Separate plots  (one window per file)",
                   variable=mode_var, value="separate", anchor="w").pack(fill="x")
    tk.Radiobutton(frame, text="Combined plot  (overlay all files)",
                   variable=mode_var, value="combined", anchor="w").pack(fill="x")
    return mode_var


def _add_intensity_bin_section(dialog) -> "tk.StringVar":
    """Add the shared intensity 'Bin width' dropdown; returns its label var."""
    if _batch_defaults["intensity_bw_label"] is None:
        _batch_defaults["intensity_bw_label"] = fcs_plot._DEFAULT_BIN_WIDTH_LABEL
    frame = tk.LabelFrame(dialog, text="Bin width", padx=10, pady=6)
    frame.pack(fill="x", padx=12, pady=6)
    bw_var = tk.StringVar(value=_batch_defaults["intensity_bw_label"])
    tk.OptionMenu(frame, bw_var, *fcs_plot._BIN_WIDTH_OPTIONS.keys()).pack()
    return bw_var


def _add_ok_cancel(dialog, on_ok, ok_text="Plot"):
    """Add a centred OK/Cancel button row; Cancel just destroys the dialog."""
    btns = tk.Frame(dialog)
    btns.pack(pady=10)
    tk.Button(btns, text=ok_text, width=12, command=on_ok,
              pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=dialog.destroy,
              pady=4).pack(side="left", padx=6)


def _ask_intensity_single():
    """Dialog to choose the intensity bin width for one file.

    Returns the bin width in seconds, or None if cancelled.
    """
    dialog = tk.Toplevel(root)
    dialog.title("Intensity — options")
    dialog.resizable(False, False)
    dialog.grab_set()
    tk.Label(dialog, text="Intensity trace",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    bw_var = _add_intensity_bin_section(dialog)
    result = {"bw": None}

    def _ok():
        _batch_defaults["intensity_bw_label"] = bw_var.get()
        result["bw"] = fcs_plot._BIN_WIDTH_OPTIONS[bw_var.get()]
        dialog.destroy()

    _add_ok_cancel(dialog, _ok)
    dialog.wait_window()
    return result["bw"]


def _ask_intensity_batch(n: int):
    """Dialog for a batch intensity plot.  Returns (bin_width_s, mode) or None."""
    dialog = tk.Toplevel(root)
    dialog.title("Batch intensity — options")
    dialog.resizable(False, False)
    dialog.grab_set()
    tk.Label(dialog, text=f"Intensity — {n} files",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    bw_var   = _add_intensity_bin_section(dialog)
    mode_var = _add_output_mode_section(dialog)
    result = {"val": None}

    def _ok():
        _batch_defaults["intensity_bw_label"] = bw_var.get()
        _batch_defaults["mode"] = mode_var.get()
        result["val"] = (fcs_plot._BIN_WIDTH_OPTIONS[bw_var.get()], mode_var.get())
        dialog.destroy()

    _add_ok_cancel(dialog, _ok)
    dialog.wait_window()
    return result["val"]


def _ask_lifetime_batch(n: int):
    """Dialog for a batch lifetime plot.  Returns (n_bins, mode) or None."""
    dialog = tk.Toplevel(root)
    dialog.title("Batch lifetime — options")
    dialog.resizable(False, False)
    dialog.grab_set()
    tk.Label(dialog, text=f"Lifetime decay — {n} files",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    bin_frame = tk.LabelFrame(dialog, text="Number of bins", padx=10, pady=6)
    bin_frame.pack(fill="x", padx=12, pady=6)
    bin_var = tk.IntVar(value=_batch_defaults["n_bins"])
    tk.OptionMenu(bin_frame, bin_var, *fcs_lifetime._VALID_N_BINS).pack()
    tk.Label(bin_frame,
             text="4096 = native (~12 ps / bin).  Shared by all files.",
             font=("Helvetica", 9), fg="grey").pack()

    mode_var = _add_output_mode_section(dialog)
    result = {"val": None}

    def _ok():
        _batch_defaults["n_bins"] = bin_var.get()
        _batch_defaults["mode"]   = mode_var.get()
        result["val"] = (bin_var.get(), mode_var.get())
        dialog.destroy()

    _add_ok_cancel(dialog, _ok)
    dialog.wait_window()
    return result["val"]


# Which detector channels each batch-PCH channel choice needs.
_PCH_NEEDS = {"ch1": (1,), "ch2": (2,), "both": (1, 2), "combined": (1, 2)}


def _ask_pch_batch(n: int, datasets=None):
    """
    Dialog for a batch PCH plot.  Returns (channel, bin_width_s, mode) or None.

    When *datasets* is given, channel options that not every selected file can
    satisfy are disabled.  The rule is the INTERSECTION of the selection's
    channels, not the union: one parameter set is applied to every file in the
    batch, so an option is only offered if it works for all of them.  A
    selection mixing a Ch1-only and a Ch2-only file therefore offers nothing,
    and the dialog says so rather than failing per file at compute time.
    """
    if _batch_defaults["pch_bw_label"] is None:
        _batch_defaults["pch_bw_label"] = fcs_pch._DEFAULT_BIN_WIDTH_LABEL

    dialog = tk.Toplevel(root)
    dialog.title("Batch PCH — options")
    dialog.resizable(False, False)
    dialog.grab_set()
    tk.Label(dialog, text=f"Photon Counting Histogram — {n} files",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    ch_frame = tk.LabelFrame(dialog, text="Channel(s)", padx=10, pady=4)
    ch_frame.pack(fill="x", padx=12, pady=6)

    if datasets:
        common = set((1, 2))
        for _d in datasets:
            common &= set(getattr(_d, "channels", (1, 2)))
    else:
        common = set((1, 2))
    supported = tuple(v for v, need in _PCH_NEEDS.items()
                      if all(c in common for c in need))

    default_ch = _batch_defaults["pch_channel"]
    if supported and default_ch not in supported:
        default_ch = supported[0]
    ch_var = tk.StringVar(value=default_ch)

    for text, value in [
        ("Ch1 only",                              "ch1"),
        ("Ch2 only",                              "ch2"),
        ("Both channels — overlay",               "both"),
        ("Both channels — combined (Ch1 + Ch2)",  "combined"),
    ]:
        rb = tk.Radiobutton(ch_frame, text=text, variable=ch_var, value=value,
                            anchor="w")
        if value not in supported:
            rb.configure(state="disabled")
        rb.pack(fill="x")

    if len(supported) < len(_PCH_NEEDS):
        msg = ("No channel is common to every selected file — deselect the "
               "files that use a different detector."
               if not supported else
               "Some files in this selection are single-channel; options "
               "needing a channel they lack are disabled.")
        tk.Label(ch_frame, text=msg, font=("Helvetica", 8), fg="grey",
                 anchor="w", justify="left", wraplength=260).pack(
                     fill="x", pady=(4, 0))

    bw_frame = tk.LabelFrame(dialog, text="Bin width", padx=10, pady=6)
    bw_frame.pack(fill="x", padx=12, pady=6)
    bw_var = tk.StringVar(value=_batch_defaults["pch_bw_label"])
    tk.OptionMenu(bw_frame, bw_var, *fcs_pch._BIN_WIDTH_OPTIONS.keys()).pack()

    mode_var = _add_output_mode_section(dialog)
    result = {"val": None}

    def _ok():
        # Robustness net: the radios for unavailable channels are disabled
        # above, so this should be unreachable from the GUI.  It is kept
        # because the remembered default is per-session rather than
        # per-selection, and because a future edit could add a channel option
        # without updating _PCH_NEEDS.
        if ch_var.get() not in supported:
            messagebox.showerror(
                "Channel not available",
                "The selected files do not all have the channel this option "
                "needs.  Pick one of the enabled options, or change the "
                "selection.",
                parent=dialog)
            return
        _batch_defaults["pch_channel"]  = ch_var.get()
        _batch_defaults["pch_bw_label"] = bw_var.get()
        _batch_defaults["mode"]         = mode_var.get()
        result["val"] = (
            ch_var.get(),
            fcs_pch._BIN_WIDTH_OPTIONS[bw_var.get()],
            mode_var.get(),
        )
        dialog.destroy()

    _add_ok_cancel(dialog, _ok, ok_text="Compute")
    dialog.wait_window()
    return result["val"]


def _ask_correlation_batch(n: int):
    """
    Ask the output mode for a batch correlation.

    Returns 'separate', 'combined', or None if cancelled.  Unlike the other
    batch tasks the correlation parameters themselves are collected by the
    existing correlation dialog (per file for separate, once for combined),
    so this only chooses the output mode.
    """
    dialog = tk.Toplevel(root)
    dialog.title("Batch correlation — options")
    dialog.resizable(False, False)
    dialog.grab_set()
    tk.Label(dialog, text=f"Correlation — {n} files",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    tk.Label(
        dialog,
        text=("Separate: parameter menu and plot for each file.\n"
              "Combined: choose parameters once, overlay all G(τ)."),
        font=("Helvetica", 9), fg="grey", justify="center",
    ).pack(padx=12)
    mode_var = _add_output_mode_section(dialog)
    result = {"mode": None}

    def _ok():
        _batch_defaults["mode"] = mode_var.get()
        result["mode"] = mode_var.get()
        dialog.destroy()

    _add_ok_cancel(dialog, _ok, ok_text="Continue")
    dialog.wait_window()
    return result["mode"]


# A batch of at least this many files is worth confirming before running it
# without saving anything.
_BATCH_EXPORT_WARN_AT = 3


def _confirm_batch_export(n_files: int, label: str) -> bool:
    """
    Check before running a multi-file batch with exporting switched off.

    Returns True to proceed.  Answering "Yes" turns exporting on for this run
    and every later one, which is almost always what was meant -- the box is
    on by default, so finding it off usually means it was cleared by accident
    rather than deliberately.
    """
    if export_var.get() or n_files < _BATCH_EXPORT_WARN_AT:
        return True

    ans = messagebox.askyesnocancel(
        "Save results?",
        f"About to run {label} on {n_files} files, but "
        f"“Export plotted data to CSV” is not ticked, so nothing will be "
        f"written to disk.\n\n"
        f"Yes\t— turn saving on and run\n"
        f"No\t— run anyway without saving\n"
        f"Cancel\t— go back",
        parent=root,
    )
    if ans is None:
        return False
    if ans:
        export_var.set(True)
    return True


def _run_separate(datasets, label: str, plot_one):
    """Loop the selected files, plotting each into its own window."""
    total = len(datasets)
    if not _confirm_batch_export(total, label):
        status_refresh()
        return
    for i, d in enumerate(datasets, 1):
        status_set(f"Batch {label} — file {i} of {total}:\n{d.filepath.name}")
        root.update_idletasks()   # flush status before each plot opens
        try:
            plot_one(d)
        except Exception as e:
            messagebox.showerror("Plot error", f"{d.filepath.name}:\n{e}")
    status_refresh()


# ── Help ──────────────────────────────────────────────────────────────────────

def show_help():
    help_win = tk.Toplevel(root)
    help_win.title("Help — FCS Analysis Suite")
    help_win.geometry("420x260")
    help_win.resizable(False, False)

    tk.Label(help_win, text="Available documentation",
             font=("Helvetica", 12, "bold"), pady=10).pack()
    tk.Label(help_win, text="Click a file to open it.",
             font=("Helvetica", 10), fg="grey").pack()

    docs = [("fcs_reader — Usage Guide", "fcs_reader_usage.md")]
    for display_name, filename in docs:
        filepath = Path(__file__).parent / filename
        if filepath.exists():
            tk.Button(
                help_win, text=display_name, anchor="w", relief="flat",
                fg="steelblue", cursor="hand2",
                command=lambda p=filepath: open_doc(p),
            ).pack(fill="x", padx=20, pady=4)
        else:
            tk.Label(help_win, text=f"{display_name}  (file not found)",
                     fg="grey", anchor="w").pack(fill="x", padx=20, pady=4)


def open_doc(path: Path):
    import subprocess, platform
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.call(["open", str(path)])
        else:
            subprocess.call(["xdg-open", str(path)])
    except Exception as e:
        messagebox.showerror("Could not open file", str(e))


# ── Build main window ─────────────────────────────────────────────────────────

root = tk.Tk()
root.title("FCS Analysis Suite")
root.geometry("680x480")
root.minsize(520, 400)
root.resizable(True, True)

# Global toggle: when checked, each analysis also writes its plotted data
# to a CSV in an 'analysis' folder beside the source file.
#
# Defaults ON.  Exporting is cheap and non-destructive -- a CSV per analysis
# in a subfolder -- whereas discovering after a long batch that nothing was
# written means running the whole thing again.  The asymmetry is large enough
# that saving by default is the right failure mode; _confirm_batch_export
# catches the case where it has been deliberately switched off.
export_var = tk.BooleanVar(value=True)

# Title
tk.Label(root, text="FCS Analysis Suite",
         font=("Helvetica", 16, "bold"), pady=10).pack()

# ── Middle row: Workspace left, Tasks right ───────────────────────────────────
mid_frame = tk.Frame(root)
mid_frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))

# ── Workspace panel (left) ────────────────────────────────────────────────────
ws_frame = tk.LabelFrame(mid_frame, text="Workspace", padx=8, pady=6)
ws_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

# Listbox + scrollbar
lb_frame = tk.Frame(ws_frame)
lb_frame.pack(fill="both", expand=True)

lb_scroll = tk.Scrollbar(lb_frame, orient="vertical")
ws_listbox = tk.Listbox(
    lb_frame,
    yscrollcommand=lb_scroll.set,
    font=("Courier", 10),
    selectmode="extended",   # allow ctrl/shift-click multi-select for batch tasks
    activestyle="none",
    selectforeground="white",
    height=5,
)
lb_scroll.config(command=ws_listbox.yview)
lb_scroll.pack(side="right", fill="y")
ws_listbox.pack(side="left", fill="both", expand=True)

ws_listbox.bind("<<ListboxSelect>>", ws_on_select)
ws_listbox.bind("<Double-Button-1>", lambda e: ws_set_active())

# Workspace buttons
ws_btn_frame = tk.Frame(ws_frame)
ws_btn_frame.pack(fill="x", pady=(6, 0))

tk.Button(ws_btn_frame, text="Add file(s)…", command=ws_add,
          width=13, pady=3).pack(side="left", padx=(0, 4))
tk.Button(ws_btn_frame, text="Set as active", command=ws_set_active,
          fg="steelblue", width=13, pady=3).pack(side="left", padx=4)
tk.Button(ws_btn_frame, text="Remove", command=ws_remove,
          fg="tomato", width=10, pady=3).pack(side="right")

def task_correlation():
    """Open the correlation dialog for the selected dataset(s).

    With two or more files selected (ctrl/shift-click) a chooser offers:
      • Separate — the parameter menu and result plot appear for each file,
        exactly as in the single-file case (per-file gate supported).
      • Combined — parameters are chosen once and every file's G(τ) curve is
        overlaid on a single plot.
    Otherwise the active ► file is used, as before.
    """
    datasets = _selected_datasets()

    if len(datasets) >= 2:
        mode = _ask_correlation_batch(len(datasets))
        if mode is None:
            return

        if mode == "combined":
            if not _confirm_batch_export(len(datasets), "correlation"):
                return
            # Ask the correlation parameters once (using the first file to
            # populate the dialog), then apply them to every selection.
            params = fcs_corr.run_correlation_dialog(
                datasets[0], export=export_var.get(), collect_only=True)
            if params is None:
                return
            total = len(datasets)
            results = []
            for i, d in enumerate(datasets, 1):
                status_set(f"Correlation overlay — file {i} of {total}:\n"
                           f"{d.filepath.name}")
                root.update_idletasks()
                res = fcs_corr.compute_correlation_for(d, params, parent=root)
                if res is not None:        # None → gate cancelled / too narrow
                    results.append((d, res))
            if results:
                try:
                    fcs_corr.plot_correlation_overlay(
                        results,
                        corr_type=params["corr_type"],
                        method=params["method"],
                        tau_min_s=params["tau_min_ms"] * 1e-3,
                        tau_max_s=params["tau_max_ms"] * 1e-3,
                        export=export_var.get(),
                    )
                except Exception as e:
                    messagebox.showerror("Plot error", str(e))
            status_refresh()
        else:
            _run_separate(
                datasets, "correlation",
                lambda d: fcs_corr.run_correlation_dialog(
                    d, export=export_var.get()),
            )
        return

    # Single-file path (uses the active ► file, as before).
    d = _active_data()
    if d is None:
        _no_active_file_warning()
        return
    fcs_corr.run_correlation_dialog(d, export=export_var.get())


def task_pch():
    """Open the PCH dialog for the active dataset, or batch over a selection."""
    datasets = _selected_datasets()
    if len(datasets) >= 2:
        params = _ask_pch_batch(len(datasets), datasets)
        if params is None:
            return
        channel_choice, bin_width_s, mode = params
        if mode == "combined":
            status_set(f"Combined PCH overlay — {len(datasets)} files")
            root.update_idletasks()
            try:
                fcs_pch.plot_pch_overlay(
                    datasets, channel_choice, bin_width_s, export=export_var.get())
            except Exception as e:
                messagebox.showerror("Computation error", str(e))
            status_refresh()
        else:
            _run_separate(
                datasets, "PCH",
                lambda d: fcs_pch.plot_pch_single(
                    d, channel_choice, bin_width_s, export=export_var.get()),
            )
        return

    d = _active_data()
    if d is None:
        _no_active_file_warning()
        return
    fcs_pch.run_pch_dialog(d, export=export_var.get())

def task_model():
    """Open the matching modelling workflow for the active dataset.

    Photon data (.fcs) → the correlation / lifetime / PCH chooser in fcs_fit.
    Lifetime decays (.ifx) → a tail-fit vs IRF-reconvolution chooser, since
    correlation and PCH need photon records an .ifx file does not carry.
    No active file → the same chooser with only Correlation enabled.  Saved
    correlation CSVs carry their own metadata, so modelling them needs no .fcs
    in the workspace — which matters because the .fcs originals are far larger
    than the exported curves and often are not the copy you have to hand.

    """
    d = _active_data()
    if d is None:
        fcs_fit.run_model_dialog(None, parent=root)
        return
    if getattr(d, "kind", None) == "lifetime_decay":
        _lifetime_method_dialog(d)
        return
    # Pass the workspace file order so global-fit datasets/rows follow it
    # instead of being listed alphabetically.
    order = [fd.filepath.name for fd in workspace.values()]
    fcs_fit.run_model_dialog(d, parent=root, workspace_order=order)


def _lifetime_method_dialog(d):
    """Choose tail fit (default) vs IRF reconvolution for an .ifx decay."""
    dialog = tk.Toplevel(root)
    dialog.title("Model lifetime decay")
    dialog.resizable(False, False)
    dialog.grab_set()

    tk.Label(dialog, text="Model lifetime decay",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    tk.Label(dialog, text=d.filepath.name,
             font=("Helvetica", 9), fg="grey").pack()

    has_irf = bool(getattr(d, "has_irf", False))
    method_var = tk.StringVar(value="tail")

    frame = tk.LabelFrame(dialog, text="Method", padx=12, pady=8)
    frame.pack(fill="x", padx=14, pady=8)
    tk.Radiobutton(frame, text="Tail fit  (sum of exponentials, no IRF)",
                   variable=method_var, value="tail", anchor="w").pack(fill="x")
    recon_label = ("IRF reconvolution" if has_irf
                   else "IRF reconvolution  (no IRF in this file)")
    tk.Radiobutton(frame, text=recon_label,
                   variable=method_var, value="recon", anchor="w",
                   state=("normal" if has_irf else "disabled")).pack(fill="x")

    def _go():
        method = method_var.get()
        dialog.destroy()
        if method == "recon":
            fcs_lifetime_recon.run_reconv_fit_dialog(d, parent=root)
        else:
            fcs_lifetime_fit.run_lifetime_fit_dialog(d, parent=root)

    btns = tk.Frame(dialog)
    btns.pack(pady=10)
    tk.Button(btns, text="Next →", width=12, command=_go, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=dialog.destroy, pady=4).pack(side="left", padx=6)


def task_calibrate():
    """Open the volume-calibration plotter (⟨N⟩ vs known concentration)."""
    d = _active_data()
    init_dir = None
    if d is not None:
        fits = d.filepath.parent / "fits"
        init_dir = fits if fits.exists() else d.filepath.parent
    fcs_calib.run_calibration_dialog(parent=root, init_dir=init_dir)


def task_open_plot():
    """
    Reopen a previously exported analysis as a live, interactive figure.

    Opens the exported CSV, not a PNG: the result comes back as a real
    matplotlib figure with the full plot-controls panel, so it can be
    rescaled, restyled and re-saved.  See fcs_plotopen.
    """
    d = _active_data()
    init_dir = None
    if d is not None:
        analysis = d.filepath.parent / "analysis"
        init_dir = analysis if analysis.exists() else d.filepath.parent
    fcs_plotopen.run_open_dialog(parent=root, initialdir=init_dir)


# ── Tasks panel (right) ───────────────────────────────────────────────────────
tasks = [
    ("Plot Intensity",              task_plot_intensity),
    ("Plot Lifetime Decay",         task_plot_lifetime),
    ("Compute Correlation",         task_correlation),
    ("Photon Counting Histogram",   task_pch),
    ("Model Data",                  task_model),
    ("Calibrate Volume (⟨N⟩ vs C)", task_calibrate),
    ("Reopen Saved Plot…",          task_open_plot),
]

task_frame = tk.LabelFrame(mid_frame, text="Tasks", padx=10, pady=8)
task_frame.pack(side="left", fill="y")

for label, command in tasks:
    tk.Button(task_frame, text=label, command=command,
              pady=4).pack(fill="x", pady=3)

# ── Export toggle ─────────────────────────────────────────────────────────────
tk.Frame(task_frame, height=1, bg="#d9d9d9").pack(fill="x", pady=(8, 4))
tk.Checkbutton(
    task_frame,
    text="Export plotted data to CSV",
    variable=export_var,
    anchor="w",
).pack(fill="x")
tk.Label(
    task_frame,
    text="Saves to an 'analysis' folder\nbeside the source file.",
    font=("Helvetica", 9), fg="grey", anchor="w", justify="left",
).pack(fill="x")

# ── Help button ───────────────────────────────────────────────────────────────
tk.Button(root, text="Help / Documentation", width=28,
          command=show_help, pady=4).pack(pady=(0, 8))

# ── Status bar ────────────────────────────────────────────────────────────────
status_frame = tk.LabelFrame(root, text="Status", padx=10, pady=6)
status_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))

status_text = scrolledtext.ScrolledText(
    status_frame, height=6, state="disabled",
    font=("Courier", 10), bg="#f7f7f7", relief="flat",
)
status_text.pack(fill="both", expand=True)

status_refresh()


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def _on_close():
        plt.close('all')
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()