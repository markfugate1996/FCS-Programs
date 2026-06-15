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

from fcs_reader import read_fcs, FCSData
import fcs_plot
import fcs_lifetime
import fcs_corr
import fcs_pch
import fcs_fit
import fcs_calib


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
    if key not in workspace:
        workspace[key] = read_fcs(path)
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


def _file_summary(d: FCSData) -> str:
    return (
        f"{d.filepath.name}\n"
        f"  Duration : {d.duration_s:.2f} s\n"
        f"  Ch1      : {len(d.ch1_deltas):,} photons  "
        f"({d.count_rate_ch1_hz:,.0f} CPS)\n"
        f"  Ch2      : {len(d.ch2_deltas):,} photons  "
        f"({d.count_rate_ch2_hz:,.0f} CPS)"
    )


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
    key = ws_selected_key()
    if key is None:
        messagebox.showinfo("Nothing selected", "Select a file in the list first.")
        return
    confirm = messagebox.askyesno(
        "Remove file",
        f"Remove '{key}' from the workspace?\n(The file on disk is not affected.)",
    )
    if not confirm:
        return
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


def ws_selected_key() -> str | None:
    sel = ws_listbox.curselection()
    if not sel:
        return None
    raw = ws_listbox.get(sel[0])
    return raw[2:] if raw.startswith("► ") else raw[3:]


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
    key = ws_selected_key()
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
    d = _active_data()
    if d is None:
        _no_active_file_warning()
        return
    fcs_plot.plot_intensity(d, export=export_var.get())


def task_plot_lifetime():
    d = _active_data()
    if d is None:
        _no_active_file_warning()
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
export_var = tk.BooleanVar(value=False)

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
    selectmode="single",
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
    """Open the correlation dialog for the active dataset."""
    d = _active_data()
    if d is None:
        _no_active_file_warning()
        return
    fcs_corr.run_correlation_dialog(d, export=export_var.get())


def task_pch():
    """Open the PCH dialog for the active dataset."""
    d = _active_data()
    if d is None:
        _no_active_file_warning()
        return
    fcs_pch.run_pch_dialog(d, export=export_var.get())


def task_model():
    """Open the data-type chooser, then the matching modelling workflow."""
    d = _active_data()
    if d is None:
        _no_active_file_warning()
        return
    fcs_fit.run_model_dialog(d, parent=root)


def task_calibrate():
    """Open the volume-calibration plotter (⟨N⟩ vs known concentration)."""
    d = _active_data()
    init_dir = None
    if d is not None:
        fits = d.filepath.parent / "fits"
        init_dir = fits if fits.exists() else d.filepath.parent
    fcs_calib.run_calibration_dialog(parent=root, init_dir=init_dir)


# ── Tasks panel (right) ───────────────────────────────────────────────────────
tasks = [
    ("Plot Intensity",              task_plot_intensity),
    ("Plot Lifetime Decay",         task_plot_lifetime),
    ("Compute Correlation",         task_correlation),
    ("Photon Counting Histogram",   task_pch),
    ("Model Data",                  task_model),
    ("Calibrate Volume (⟨N⟩ vs C)", task_calibrate),
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