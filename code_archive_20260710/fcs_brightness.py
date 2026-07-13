"""
fcs_brightness.py
=================
Compare molecular brightness (CPSM — counts per second per molecule) across
several datasets in the workspace, and plot it against any independent variable
the user supplies (concentration, quencher amount, ⟨N⟩, buffer condition, …).

Launched from the "Compare Brightness (CPSM)" task in the main window.  Unlike
the per-file analysis tasks, this one operates on a *set* of workspace files.

Two ways to get brightness
--------------------------
1. CPS / ⟨N⟩   — brightness = (total photons / total time) / ⟨N⟩, where ⟨N⟩ is
   typed in per file (e.g. from your correlation analysis).  This is the simple,
   transparent route: the count rate divided by the number of molecules.  Enter
   ⟨N⟩ = 1 to plot the raw count rate (CPS) itself.

2. Moment estimate — brightness η = Q/(γ₂·T) from the photon-counting statistics
   at a chosen bin width T, with Q = Var(k)/⟨k⟩ − 1 and γ₂ = 2^(−3/2) for a 3D
   Gaussian PSF.  This is self-contained (no ⟨N⟩ needed); it derives the
   occupancy from the size of the intensity fluctuations and is the same CPSM a
   single-species PCH fit reports as ε/T.

Both methods produce a per-molecule brightness in counts/s; the plot is in
kHz/molecule.  The independent variable is entirely the user's — the program
does not assume it is ⟨N⟩ or anything else.

Corrections
-----------
Each file carries an optional correction factor (default 1.0) for differences in
excitation power or ND-filter attenuation between acquisitions:

    corrected brightness = brightness · (correction factor)

The convention is the user's: report whatever factor brings each dataset onto a
common excitation/detection basis.

Spreadsheet paste
-----------------
Independent-variable values, ⟨N⟩ values, and correction factors can be typed per
row or pasted from a spreadsheet.  Choose which column the paste lands in; a
multi-column paste fills consecutive columns (independent variable → ⟨N⟩ →
correction) from where you start.

Dependencies
------------
    pip install numpy matplotlib
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import fcs_plottools
import fcs_pch
import fcs_pch_fit
from fcs_models import _PCH_GAMMA2 as GAMMA2
from fcs_reader import FCSData


# ── Brightness methods ────────────────────────────────────────────────────────

def count_rate(d: FCSData, channel: str) -> float:
    """Total count rate (counts/s) for the channel: total photons / total time."""
    return fcs_pch_fit.channel_cps(d, channel)


def brightness_cps_over_n(d: FCSData, channel: str, N: float) -> dict:
    """
    Brightness via CPS / ⟨N⟩.

    eta = (total photons / total time) / N.  N is supplied by the user.
    With N = 1 this returns the raw count rate.
    """
    cps = count_rate(d, channel)
    ok = np.isfinite(N) and N > 0
    eta = (cps / N) if ok else float("nan")
    return {"cps": cps, "eta": eta, "N": float(N), "ok": ok,
            "mean": float("nan"), "var": float("nan"), "Q": float("nan")}


def brightness_moment(d: FCSData, channel: str, bin_width_s: float) -> dict:
    """
    Brightness via the photon-counting moments: eta = Q/(γ₂·T).

    Returns cps, ⟨k⟩, Var(k), Q, eta (counts/s per molecule), and the occupancy
    estimate N = ⟨k⟩·γ₂/Q.
    """
    times = fcs_pch_fit.channel_times(d, channel)
    _k, _n_k, M, mean, var = fcs_pch_fit.pch_counts(times, bin_width_s)
    Q = (var / mean - 1.0) if mean > 0 else float("nan")
    ok = np.isfinite(Q) and Q > 0
    eps = (Q / GAMMA2) if ok else float("nan")
    eta = (eps / bin_width_s) if ok else float("nan")
    N_est = (mean / eps) if ok else float("nan")
    cps = mean / bin_width_s
    return {"cps": cps, "mean": mean, "var": var, "Q": Q,
            "eps": eps, "eta": eta, "N": N_est, "ok": ok, "M": int(M)}


# Backwards-compatible alias (the moment estimate was the original default).
molecular_brightness = brightness_moment


# ── Clipboard / spreadsheet paste parsing ─────────────────────────────────────

def parse_pasted_table(raw: str) -> List[List[str]]:
    """
    Parse text copied from a spreadsheet into a list of rows, each a list of
    cell strings.  Columns may be tab- or comma-separated; blank lines are
    skipped.
    """
    rows: List[List[str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            cells = line.split("\t")
        elif "," in line:
            cells = line.split(",")
        else:
            cells = line.split()
        rows.append([c.strip() for c in cells])
    return rows


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_brightness(records: List[dict], indep_name: str, channel: str,
                    method_label: str, show: bool = True
                    ) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot corrected molecular brightness (kHz/molecule) for each record.

    ``records`` are dicts with keys name, x (float or None), eta_corr (counts/s).
    Numeric x → brightness vs x (joined in x-order); otherwise one point per
    dataset by name.
    """
    names = [r["name"] for r in records]
    y_khz = np.array([r["eta_corr"] / 1e3 for r in records], dtype=float)
    xs = [r["x"] for r in records]
    numeric = len(xs) > 0 and all(x is not None and np.isfinite(x) for x in xs)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    if numeric:
        xv = np.array(xs, dtype=float)
        order = np.argsort(xv)
        ax.plot(xv[order], y_khz[order], marker="o", markersize=6,
                linewidth=1.4, color="steelblue")
        for i in order:
            ax.annotate(names[i], (xv[i], y_khz[i]), fontsize=7,
                        textcoords="offset points", xytext=(4, 4), color="#666")
        ax.set_xlabel(indep_name or "Independent variable", fontsize=12)
    else:
        xv = np.arange(len(records))
        ax.plot(xv, y_khz, marker="o", markersize=6, linestyle="none",
                color="steelblue")
        ax.set_xticks(xv)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Dataset", fontsize=12)

    ax.set_ylabel("Molecular brightness  (kHz / molecule)", fontsize=12)
    ax.set_title(f"Molecular brightness across datasets\n{channel}  ·  {method_label}",
                 fontsize=11)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()

    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── Export ────────────────────────────────────────────────────────────────────

def export_brightness(records: List[dict], indep_name: str, channel: str,
                      method: str, method_label: str,
                      bin_width_s: Optional[float], out_path: Path) -> Path:
    """Write the brightness comparison table to a CSV."""
    out_path = Path(out_path)
    header = ["file", "channel", "method",
              indep_name or "independent_var", "correction_factor",
              "brightness_cpsm_raw", "brightness_cpsm_corrected",
              "cps", "N", "Q", "mean_k", "bin_width_s"]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# FCS brightness comparison\n")
        fh.write(f"# exported : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(f"# channel : {channel}\n")
        fh.write(f"# method : {method_label}\n")
        if method == "moment" and bin_width_s:
            fh.write(f"# bin_width_s : {bin_width_s:.6g}\n")
            fh.write(f"# moment model : eta = Q/(gamma2*T), gamma2 = 2^-1.5 (3D Gaussian)\n")
        fh.write(",".join(header) + "\n")
        for r in records:
            def g(key):
                v = r.get(key)
                return "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.6g}"
            row = [r["name"], channel, method,
                   ("" if r["x"] is None else f"{r['x']:.6g}"),
                   f"{r['corr']:.6g}",
                   g("eta_raw"), g("eta_corr"), g("cps"), g("N"), g("Q"), g("mean"),
                   (f"{bin_width_s:.6g}" if (method == "moment" and bin_width_s) else "")]
            fh.write(",".join(str(c) for c in row) + "\n")
    print(f"[brightness] wrote {out_path}")
    return out_path


# ── GUI ───────────────────────────────────────────────────────────────────────

# Editable columns, in left-to-right order, for multi-column paste.
_PASTE_ORDER = ["x", "N", "corr"]
_PASTE_LABELS = {"x": "Independent variable", "N": "⟨N⟩", "corr": "Correction"}


def run_brightness_dialog(workspace: Dict[str, FCSData], parent=None):
    """Show the brightness-comparison dialog for the workspace."""
    import tkinter as tk
    from tkinter import messagebox, filedialog

    if not workspace:
        messagebox.showinfo("No data", "Add files to the workspace first.",
                            parent=parent)
        return

    win = tk.Toplevel(parent)
    win.title("Compare brightness (CPSM)")
    win.geometry("720x560")
    win.minsize(640, 460)
    win.grab_set()

    tk.Label(win, text="Compare molecular brightness across datasets",
             font=("Helvetica", 12, "bold"), pady=6).pack()

    # ── Method + global options ───────────────────────────────────────────────
    opts = tk.Frame(win, padx=12, pady=4)
    opts.pack(fill="x")

    tk.Label(opts, text="Method:").grid(row=0, column=0, sticky="w")
    method_var = tk.StringVar(value="cps_over_n")
    mf = tk.Frame(opts); mf.grid(row=0, column=1, columnspan=3, sticky="w")
    tk.Radiobutton(mf, text="CPS / ⟨N⟩   (enter ⟨N⟩; use 1 for raw CPS)",
                   variable=method_var, value="cps_over_n").pack(anchor="w")
    tk.Radiobutton(mf, text="Moment estimate   η = Q/(γ₂·T)",
                   variable=method_var, value="moment").pack(anchor="w")

    tk.Label(opts, text="Channel:").grid(row=1, column=0, sticky="w", pady=(6, 0))
    ch_var = tk.StringVar(value="ch1")
    chf = tk.Frame(opts); chf.grid(row=1, column=1, sticky="w", pady=(6, 0))
    for txt, val in [("Ch1", "ch1"), ("Ch2", "ch2"), ("Combined", "combined")]:
        tk.Radiobutton(chf, text=txt, variable=ch_var, value=val).pack(side="left")

    tk.Label(opts, text="Bin width:").grid(row=1, column=2, sticky="e", padx=(12, 2), pady=(6, 0))
    bw_var = tk.StringVar(value=fcs_pch._DEFAULT_BIN_WIDTH_LABEL)
    bw_menu = tk.OptionMenu(opts, bw_var, *fcs_pch._BIN_WIDTH_OPTIONS.keys())
    bw_menu.grid(row=1, column=3, sticky="w", pady=(6, 0))

    tk.Label(opts, text="Independent variable name:").grid(
        row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))
    indep_var_name = tk.StringVar(value="")
    tk.Entry(opts, textvariable=indep_var_name, width=26).grid(
        row=2, column=2, columnspan=2, sticky="w", pady=(6, 0))

    # ── Scrollable file table ─────────────────────────────────────────────────
    tbl_frame = tk.LabelFrame(win, text="Files  ·  tick to include", padx=6, pady=4)
    tbl_frame.pack(fill="both", expand=True, padx=12, pady=6)

    canvas = tk.Canvas(tbl_frame, highlightthickness=0)
    vsb = tk.Scrollbar(tbl_frame, orient="vertical", command=canvas.yview)
    inner = tk.Frame(canvas)
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=vsb.set)
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    headers = ["use", "file", "indep. var", "⟨N⟩", "correction"]
    for c, h in enumerate(headers):
        tk.Label(inner, text=h, font=("Helvetica", 9, "bold")).grid(
            row=0, column=c, padx=4, pady=(0, 4), sticky="w")
    indep_hdr = inner.grid_slaves(row=0, column=2)[0]
    indep_var_name.trace_add(
        "write", lambda *_: indep_hdr.config(text=indep_var_name.get() or "indep. var"))

    keys = list(workspace.keys())
    inc_vars: Dict[str, tk.BooleanVar] = {}
    x_vars:   Dict[str, tk.StringVar] = {}
    n_vars:   Dict[str, tk.StringVar] = {}
    c_vars:   Dict[str, tk.StringVar] = {}
    n_entries: Dict[str, tk.Entry] = {}
    for r, key in enumerate(keys, start=1):
        iv = tk.BooleanVar(value=True)
        xv = tk.StringVar(value=""); nv = tk.StringVar(value=""); cv = tk.StringVar(value="1.0")
        tk.Checkbutton(inner, variable=iv).grid(row=r, column=0, padx=4)
        tk.Label(inner, text=key, anchor="w", width=24).grid(row=r, column=1, sticky="w", padx=4)
        tk.Entry(inner, textvariable=xv, width=12).grid(row=r, column=2, padx=4)
        ne = tk.Entry(inner, textvariable=nv, width=9)
        ne.grid(row=r, column=3, padx=4)
        tk.Entry(inner, textvariable=cv, width=8).grid(row=r, column=4, padx=4)
        inc_vars[key] = iv; x_vars[key] = xv; n_vars[key] = nv
        c_vars[key] = cv; n_entries[key] = ne

    # Enable/disable widgets that only apply to one method.
    def _sync_method(*_):
        m = method_var.get()
        bw_menu.config(state="normal" if m == "moment" else "disabled")
        for key in keys:
            n_entries[key].config(state="normal" if m == "cps_over_n" else "disabled")
    method_var.trace_add("write", _sync_method)
    _sync_method()

    # ── Paste from clipboard ──────────────────────────────────────────────────
    paste_row = tk.Frame(win)
    paste_row.pack(fill="x", padx=12)
    tk.Label(paste_row, text="Paste into:").pack(side="left")
    paste_target = tk.StringVar(value=_PASTE_LABELS["x"])
    tk.OptionMenu(paste_row, paste_target, *[_PASTE_LABELS[k] for k in _PASTE_ORDER]
                  ).pack(side="left", padx=(2, 8))

    col_for = {v: k for k, v in _PASTE_LABELS.items()}
    var_for = {"x": x_vars, "N": n_vars, "corr": c_vars}

    def _paste():
        try:
            raw = win.clipboard_get()
        except Exception:
            messagebox.showinfo("Nothing to paste",
                                "The clipboard is empty or not text. Copy one or "
                                "more columns from your spreadsheet first.", parent=win)
            return
        pasted = parse_pasted_table(raw)
        if not pasted:
            return
        start = _PASTE_ORDER.index(col_for[paste_target.get()])
        targets = [k for k in keys if inc_vars[k].get()] or keys
        n = min(len(pasted), len(targets))
        for key, cells in zip(targets[:n], pasted[:n]):
            for j, cell in enumerate(cells):
                ci = start + j
                if ci >= len(_PASTE_ORDER) or cell == "":
                    continue
                var_for[_PASTE_ORDER[ci]][key].set(cell)
        note = ""
        if len(pasted) != len(targets):
            note = (f"\n\nNote: pasted {len(pasted)} row(s) for "
                    f"{len(targets)} included file(s); filled {n}.")
        messagebox.showinfo("Pasted", f"Filled {n} row(s) starting at "
                            f"'{paste_target.get()}'.{note}", parent=win)

    tk.Button(paste_row, text="Paste from clipboard", command=_paste, pady=3).pack(side="left")
    tk.Label(paste_row, text="  one column → that field; multiple columns fill "
             "rightward (indep. var → ⟨N⟩ → correction)",
             font=("Helvetica", 8), fg="grey").pack(side="left")

    # ── Collect / compute ─────────────────────────────────────────────────────
    def _collect():
        method = method_var.get()
        channel = ch_var.get()
        bw = fcs_pch._BIN_WIDTH_OPTIONS[bw_var.get()]
        method_label = (f"η = Q/(γ₂·T),  bin {fcs_pch._format_bin_width(bw)}"
                        if method == "moment" else "CPS / ⟨N⟩")
        records: List[dict] = []
        warned_Q, missing_N = [], []
        for key in keys:
            if not inc_vars[key].get():
                continue
            try:
                corr = float(c_vars[key].get())
            except ValueError:
                messagebox.showerror("Invalid correction",
                                     f"Correction factor for '{key}' is not a number.",
                                     parent=win)
                return None
            xtxt = x_vars[key].get().strip()
            x = None
            if xtxt != "":
                try:
                    x = float(xtxt)
                except ValueError:
                    x = None
            try:
                if method == "cps_over_n":
                    ntxt = n_vars[key].get().strip()
                    if ntxt == "":
                        missing_N.append(key); continue
                    Nval = float(ntxt)
                    b = brightness_cps_over_n(workspace[key], channel, Nval)
                else:
                    b = brightness_moment(workspace[key], channel, bw)
            except ValueError:
                messagebox.showerror("Invalid ⟨N⟩",
                                     f"⟨N⟩ for '{key}' is not a number.", parent=win)
                return None
            except Exception as e:
                messagebox.showerror("Brightness error", f"{key}: {e}", parent=win)
                return None
            if not b["ok"]:
                warned_Q.append(key)
            records.append({
                "name": key, "x": x, "corr": corr,
                "eta_raw": b["eta"], "eta_corr": b["eta"] * corr,
                "cps": b["cps"], "mean": b.get("mean", float("nan")),
                "Q": b.get("Q", float("nan")), "N": b["N"],
            })
        if missing_N:
            messagebox.showerror(
                "Missing ⟨N⟩",
                "CPS / ⟨N⟩ needs an ⟨N⟩ for every included file. Missing:\n\n"
                + ", ".join(missing_N), parent=win)
            return None
        if not records:
            messagebox.showinfo("No files selected",
                                "Tick at least one file to include.", parent=win)
            return None
        if warned_Q:
            messagebox.showwarning(
                "Non-positive Q",
                "These files have Q ≤ 0 at this bin width, so the moment "
                "brightness is not meaningful (counts too low / essentially "
                "Poisson):\n\n" + ", ".join(warned_Q)
                + "\n\nTry a larger bin width, or use CPS / ⟨N⟩.", parent=win)
        return records, method, method_label, channel, bw

    def _plot():
        got = _collect()
        if got is None:
            return
        records, method, method_label, channel, bw = got
        plot_brightness(records, indep_var_name.get(), channel, method_label, show=True)

    def _export():
        got = _collect()
        if got is None:
            return
        records, method, method_label, channel, bw = got
        path = filedialog.asksaveasfilename(
            title="Save brightness table", defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")], parent=win,
            initialfile="brightness_comparison.csv")
        if not path:
            return
        bw_use = bw if method == "moment" else None
        export_brightness(records, indep_var_name.get(), channel, method,
                          method_label, bw_use, Path(path))
        try:
            fig, _ax = plot_brightness(records, indep_var_name.get(), channel,
                                       method_label, show=False)
            fig.savefig(Path(path).with_suffix(".png"), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[brightness] could not save figure: {e}")
        messagebox.showinfo("Exported", f"Saved table to:\n{path}", parent=win)

    btns = tk.Frame(win)
    btns.pack(pady=10)
    tk.Button(btns, text="Plot brightness", width=16, command=_plot, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Export CSV…", width=12, command=_export, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Close", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()
