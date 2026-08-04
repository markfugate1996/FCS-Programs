"""
fcs_fitdialogs.py
=================
The dialog pieces every multi-dataset fit needs, shared by all three analyses.

    fcs_fit.py            correlation global fit
    fcs_pch_fit.py        PCH global fit
    fcs_lifetime_fit.py   lifetime global fit

What is here
------------
    DatasetSource         what one analysis calls its files and how to read
                          them -- the only thing the picker needs to differ on
    select_datasets(...)  the curated file list: discover, add, remove, reorder
    dataset_legend(...)   numbered list of the selected datasets
    ParamTable(...)       the Parameter / Link / Guess / Lower / Upper / Fix /
                          Rule grid, with live rule checking
    show_rule_help(...)   the rule syntax reference, in a window

Why shared
----------
The three fitters ask the same question -- for each model parameter, what is
the starting guess, what are its bounds, and is it shared, held or free? -- and
differ only in what a parameter means.  Three copies of that grid would drift
the way ``_fits_dir`` and the parameter tables did before fcs_fitcommon and
fcs_globalfit collected them, and this grid is the one place a mistake is
expensive: it decides what the fit is allowed to do.

Dependencies
------------
tkinter is imported inside functions, not at module scope, so a headless
script can import this module for its constants without needing a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence

import numpy as np

import fcs_globalfit
from fcs_fitcommon import fmt_bound as _fmt, parse_bound as _parse_bound


__all__ = ["dataset_legend", "ParamTable", "show_rule_help",
           "DatasetEntry", "DatasetSource", "select_datasets",
           "session_dir", "reset_session"]


# ── Dataset selection ─────────────────────────────────────────────────────────

class DatasetEntry(NamedTuple):
    """
    One row of the picker: a file, optionally one series within it.

    Rows are entries rather than plain paths because a PCH export can hold Ch1
    and Ch2, and a lifetime export can hold two decays.  Those are separate
    datasets to a fit -- separately linkable, separately removable, separately
    numbered in a rule -- so they have to be separate ROWS, not one row that
    silently becomes two after the picker closes.  A correlation CSV holds one
    curve and leaves ``series`` None.
    """
    path: Path
    series: Optional[str] = None
    label: str = ""

    @property
    def uid(self) -> str:
        """Stable identity for remembering order and removals across a session."""
        return f"{self.path}|{self.series or ''}"

    def shown(self) -> str:
        return self.label or (self.path.name if self.series is None
                              else f"{self.path.name}  [{self.series}]")


class DatasetSource(NamedTuple):
    """
    What one analysis calls its saved datasets, and how to find and read them.

    Everything the picker does -- discovering a folder, adding files, removing
    rows, reordering, remembering the arrangement -- is the same for all three
    analyses.  These are the parts that are not.
    """
    key: str                       # session-state bucket; one per analysis
    title: str                     # window title
    noun: str                      # "correlation datasets", "PCH histograms"
    discover: Callable             # (Path) -> list[DatasetEntry]
    load: Callable                 # (DatasetEntry) -> dataset dict
    filetypes: tuple = (("CSV files", "*.csv"), ("All files", "*.*"))
    empty_hint: str = ""           # shown when a folder yields nothing


# Remembered per analysis, so curating a list of decays does not disturb a
# list of correlations left mid-session in another window.
_SESSION: Dict[str, dict] = {}


def _session(key: str) -> dict:
    return _SESSION.setdefault(
        key, {"order": [], "removed": set(), "dir": None})


def session_dir(key: str) -> Optional[Path]:
    """Folder this analysis's picker last worked in, or None."""
    d = _session(key)["dir"]
    if d and Path(d).exists():
        return Path(d)
    return None


def reset_session(key: Optional[str] = None) -> None:
    """Forget remembered order/removals -- for one analysis, or all of them."""
    if key is None:
        _SESSION.clear()
    else:
        _SESSION.pop(key, None)


def _apply_remembered(entries: list, remembered: list) -> list:
    """
    Re-apply the row order confirmed earlier in this session.

    Entries present last time keep their hand-made positions; anything new
    follows afterwards in the order it was passed in.  Unchanged when nothing
    has been remembered, so a fresh session keeps whatever order it was given.
    """
    if not remembered:
        return list(entries)
    rank = {u: i for i, u in enumerate(remembered)}
    n = len(rank)
    decorated = [(rank.get(e.uid, n), i, e) for i, e in enumerate(entries)]
    return [e for _r, _i, e in sorted(decorated, key=lambda t: (t[0], t[1]))]


def select_datasets(parent, source: DatasetSource, init_dir, on_done,
                    preorder=None):
    """
    Screen — build the list of saved datasets to fit.

    The list is the user's to curate: seeded by discovering *init_dir* (may be
    None, in which case it starts empty), then extended by browsing for files
    or folders anywhere, pruned with Remove, and ordered with Move up / down.
    The arrangement and the removals last for the session; "Reset list" undoes
    both and re-discovers the current folder.

    *preorder* optionally sorts the initial discovery -- correlation passes the
    workspace file order so the fit follows it instead of an alphabetical
    listing.  It is applied before any remembered manual arrangement, which
    always wins.

    Calls ``on_done(datasets)`` with the loaded dataset dicts, in row order.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    st = _session(source.key)
    init_dir = Path(init_dir) if init_dir else None
    removed = set(st["removed"])          # working copy; committed by Next

    entries: list = list(source.discover(init_dir)) if init_dir else []
    entries = [e for e in entries if e.uid not in removed]
    if preorder is not None:
        entries = preorder(entries)
    entries = _apply_remembered(entries, st["order"])

    win = tk.Toplevel(parent)
    win.title(source.title)
    win.geometry("640x520")
    win.minsize(560, 440)
    win.grab_set()

    tk.Label(win, text="Select datasets to include in the fit",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text="Selected (highlighted) rows are included.  "
                       "Ctrl-click toggles one row · Shift-click selects a range · "
                       "click empty space below the list to clear.\n"
                       "Row order sets the dataset order in the fit and its "
                       "output — use Move ↑ / Move ↓ to change it.",
             font=("Helvetica", 9), fg="grey",
             wraplength=600, justify="left").pack()

    folder_var = tk.StringVar(value="")

    def _update_folder():
        folder_var.set(f"Folder: {init_dir}" if init_dir
                       else "Folder: none chosen yet")

    tk.Label(win, textvariable=folder_var, font=("Courier", 8), fg="grey",
             wraplength=600, justify="left").pack()

    lb_frame = tk.Frame(win)
    lb_frame.pack(fill="both", expand=True, padx=12, pady=6)
    scroll = tk.Scrollbar(lb_frame, orient="vertical")
    listbox = tk.Listbox(lb_frame, selectmode="extended",
                         yscrollcommand=scroll.set, activestyle="none",
                         font=("Courier", 9))
    scroll.config(command=listbox.yview)
    scroll.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    info = tk.StringVar(value="")

    def _populate(select_all=True, keep=None, keep_indices=None):
        """Refill the listbox from *entries*.

        Selection is restored either by entry (*keep*) or by row index
        (*keep_indices*, used by the reorder buttons, which know exactly which
        rows moved and are unambiguous even if an entry appears twice)."""
        keep = keep or set()
        keep_indices = set(keep_indices or ())
        listbox.delete(0, tk.END)
        # Rows are numbered because per-dataset rules address datasets by
        # number; the numbering has to be the same one the rule screen shows.
        width = len(str(max(len(entries), 1)))
        for i, e in enumerate(entries):
            listbox.insert(tk.END, f"{i + 1:>{width}}  {e.shown()}")
            if select_all or e in keep or i in keep_indices:
                listbox.selection_set(i)
        _update_info()

    def _update_info(*_):
        if not entries:
            info.set(f"Nothing listed — use 'Add files…' or 'Browse folder…'.")
        else:
            info.set(f"{len(listbox.curselection())} of {len(entries)} included")

    def _selected() -> set:
        return {entries[i] for i in listbox.curselection()}

    def _click_deselect(event):
        """File-explorer feel: clicking the empty space below the rows clears."""
        idx = listbox.nearest(event.y)
        bbox = listbox.bbox(idx) if idx >= 0 else None
        if bbox is None or event.y > bbox[1] + bbox[3]:
            listbox.selection_clear(0, tk.END)
            _update_info()
            return "break"
        return None

    def _select_all(*_):
        listbox.selection_set(0, tk.END)
        _update_info()
        return "break"

    def _select_none(*_):
        listbox.selection_clear(0, tk.END)
        _update_info()

    def _move(delta):
        """Shift the selected row(s) up (delta=-1) or down (+1) by one.

        A multi-row selection moves as a block and keeps its internal order;
        rows that would run off the end, or push into another selected row,
        simply stay put."""
        sel = sorted(listbox.curselection())
        if not sel:
            return
        n = len(entries)
        sel_set = set(sel)
        items = [{"e": e, "sel": i in sel_set} for i, e in enumerate(entries)]
        seq = [items[i] for i in (sel if delta < 0 else reversed(sel))]
        pos = {id(it): k for k, it in enumerate(items)}
        for it in seq:
            cur = pos[id(it)]
            tgt = cur + delta
            if tgt < 0 or tgt >= n or items[tgt]["sel"]:
                continue
            items[cur], items[tgt] = items[tgt], items[cur]
            pos[id(items[cur])] = cur
            pos[id(items[tgt])] = tgt
        entries[:] = [it["e"] for it in items]
        moved = [k for k, it in enumerate(items) if it["sel"]]
        _populate(select_all=False, keep_indices=moved)
        if moved:
            listbox.see(min(moved))
            listbox.see(max(moved))

    def _reset_list():
        """Undo this session's curation: re-discover the folder, drop removals.

        Only local state is touched — the remembered order/removals are
        overwritten wholesale by Next, so Cancel after a Reset still leaves the
        previous session state intact.  Files added by hand from other folders
        are dropped too: this is a full reset to "what is in the current
        folder", not a partial one."""
        if not messagebox.askyesno(
                "Reset list",
                "Rebuild the list from the current folder?\n\n"
                "This undoes the manual order and restores removed rows.\n"
                "Files added by hand from other folders are dropped.\n"
                "(No file on disk is affected.)",
                parent=win):
            return
        removed.clear()
        base = list(source.discover(init_dir)) if init_dir else []
        if preorder is not None:
            base = preorder(base)
        entries[:] = base
        _populate(select_all=True)

    listbox.bind("<<ListboxSelect>>", _update_info)
    listbox.bind("<Button-1>", _click_deselect)
    listbox.bind("<Control-a>", _select_all)
    listbox.bind("<Control-A>", _select_all)
    _populate(select_all=True)

    def _browse_dir_str() -> str:
        return str(init_dir) if init_dir else ""

    def _extend(new_entries, keep):
        """Append entries not already listed; un-remove any re-added."""
        added = 0
        for e in new_entries:
            removed.discard(e.uid)      # an explicit re-add overrides a Remove
            if e not in entries:
                entries.append(e)
                keep.add(e)
                added += 1
        return added

    def _add_files():
        nonlocal init_dir
        new = filedialog.askopenfilenames(
            title=f"Add {source.noun}",
            initialdir=_browse_dir_str(),
            defaultextension=".csv",
            filetypes=list(source.filetypes),
            parent=win)
        if not new:
            return
        keep = _selected()
        found, errors = [], []
        for n in new:
            q = Path(n)
            try:
                # Discovery on the file's own folder is what expands a
                # multi-series file into its rows; picking one file must not
                # give a different result from browsing to its folder.
                got = [e for e in source.discover(q.parent) if e.path == q]
                if not got:
                    errors.append(f"{q.name}: not a readable {source.noun[:-1]}")
                found.extend(got)
            except Exception as e:      # noqa: BLE001 — reported below
                errors.append(f"{q.name}: {e}")
        added = _extend(found, keep)
        init_dir = Path(new[0]).parent
        _update_folder()
        _populate(select_all=False, keep=keep)
        if errors:
            messagebox.showerror("Some files could not be read",
                                 "\n".join(errors), parent=win)
        elif added == 0:
            messagebox.showinfo("Already listed",
                                "Every file picked is already in the list.",
                                parent=win)

    def _browse_folder():
        nonlocal init_dir
        chosen = filedialog.askdirectory(
            title=f"Choose a folder of {source.noun}",
            initialdir=_browse_dir_str(), mustexist=True, parent=win)
        if not chosen:
            return
        folder = Path(chosen)
        found = list(source.discover(folder))
        if not found:
            messagebox.showinfo(
                f"No {source.noun}",
                f"Nothing in '{folder.name}' parsed as one.\n\n"
                + (source.empty_hint or ""),
                parent=win)
            return
        keep = _selected()
        added = _extend(found, keep)
        init_dir = folder
        _update_folder()
        _populate(select_all=False, keep=keep)
        if added == 0:
            messagebox.showinfo("Already listed",
                                f"All {len(found)} in '{folder.name}' are "
                                f"already in the list.", parent=win)

    def _remove_files():
        """Drop the selected row(s) from the list (never from disk).

        Selection does double duty here — it marks both what gets fitted and
        what Remove acts on — so the rows that stay keep whatever inclusion
        state they had; if that would leave nothing included, everything left
        is re-included rather than handing back an empty list."""
        sel = sorted(listbox.curselection(), reverse=True)
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Select the row(s) to remove from the list "
                                "first.", parent=win)
            return
        keep = _selected() - {entries[i] for i in sel}
        for i in sel:
            removed.add(entries[i].uid)
            del entries[i]
        if entries and not keep:
            _populate(select_all=True)
        else:
            _populate(select_all=False, keep=keep)

    lst = tk.Frame(win)
    lst.pack(fill="x", padx=12, pady=(0, 2))
    tk.Label(lst, text="List:", font=("Helvetica", 8), fg="grey",
             width=7, anchor="w").pack(side="left")
    tk.Button(lst, text="Add files…", command=_add_files,
              width=11, pady=3).pack(side="left", padx=(0, 4))
    tk.Button(lst, text="Browse folder…", command=_browse_folder,
              width=13, pady=3).pack(side="left", padx=4)
    tk.Button(lst, text="Remove", command=_remove_files,
              fg="tomato", width=9, pady=3).pack(side="left", padx=4)
    tk.Button(lst, text="Reset list", command=_reset_list,
              width=10, pady=3).pack(side="left", padx=4)

    tools = tk.Frame(win)
    tools.pack(fill="x", padx=12, pady=(0, 2))
    tk.Label(tools, text="Include:", font=("Helvetica", 8), fg="grey",
             width=7, anchor="w").pack(side="left")
    tk.Button(tools, text="Select all", command=_select_all,
              width=9, pady=3).pack(side="left", padx=(0, 4))
    tk.Button(tools, text="Select none", command=_select_none,
              width=9, pady=3).pack(side="left", padx=4)
    tk.Label(tools, text="Order:", font=("Helvetica", 8), fg="grey",
             anchor="w").pack(side="left", padx=(14, 2))
    tk.Button(tools, text="Move ↑", command=lambda: _move(-1),
              width=8, pady=3).pack(side="left", padx=4)
    tk.Button(tools, text="Move ↓", command=lambda: _move(+1),
              width=8, pady=3).pack(side="left", padx=4)

    _update_folder()
    _update_info()

    tk.Label(win, textvariable=info, font=("Helvetica", 9), fg="grey").pack()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("No datasets",
                                "Select at least one dataset (click a row to "
                                "highlight it).", parent=win)
            return
        # Commit the curated list: the arrangement of every row (included or
        # not, so excluded datasets keep their place if re-included later), the
        # rows removed by hand, and the folder to come back to next time.
        st["order"] = [e.uid for e in entries]
        st["removed"] = set(removed)
        if init_dir:
            st["dir"] = str(init_dir)
        chosen = [entries[i] for i in sel]
        loaded, errors = [], []
        for e in chosen:
            try:
                loaded.append(source.load(e))
            except Exception as exc:    # noqa: BLE001 — reported below
                errors.append(f"{e.shown()}: {exc}")
        if errors:
            messagebox.showerror("Some files could not be read",
                                 "\n".join(errors), parent=win)
        if not loaded:
            return
        win.destroy()
        on_done(loaded)

    tk.Button(btns, text="Next →", command=_next,
              width=12, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", command=win.destroy,
              width=10, pady=4).pack(side="left", padx=6)

    win.wait_window()


# ── Dataset legend ────────────────────────────────────────────────────────────

def dataset_legend(parent, names: Sequence[str], height: int = 5):
    """
    A numbered, read-only list of the selected datasets.

    Rules address datasets by number, so the numbering has to be on screen
    while the rule is being typed.  Requiring the user to remember that d7 is
    the third replicate of the 5 uM point is how the wrong dataset gets fixed
    at the wrong value.

    Returns the frame, already packed by the caller's convention (it is not
    packed here, so the caller decides where it goes).
    """
    import tkinter as tk

    frame = tk.LabelFrame(parent, text=f"Datasets ({len(names)})",
                          padx=6, pady=4)
    inner = tk.Frame(frame)
    inner.pack(fill="both", expand=True)

    rows = min(max(len(names), 1), height)
    text = tk.Text(inner, height=rows, width=52, font=("Courier", 8),
                   wrap="none", relief="flat", background=frame["background"])
    scroll = tk.Scrollbar(inner, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=scroll.set)
    if len(names) > rows:
        scroll.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)

    width = len(str(max(len(names), 1)))
    for i, nm in enumerate(names, start=1):
        text.insert("end", f"{i:>{width}}  {nm}\n")
    text.configure(state="disabled")
    return frame


# ── Rule help ─────────────────────────────────────────────────────────────────

def show_rule_help(parent, names: Optional[Sequence[str]] = None):
    """
    Show the rule syntax reference in a scrollable, selectable window.

    The text comes from :data:`fcs_globalfit.RULE_HELP`, so it cannot describe
    a syntax the parser does not accept.  Selectable rather than a message box
    because the most useful thing to do with an example is copy it.
    """
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Per-dataset rules — syntax")
    win.geometry("640x560")

    body = tk.Frame(win)
    body.pack(fill="both", expand=True, padx=8, pady=8)
    scroll = tk.Scrollbar(body, orient="vertical")
    text = tk.Text(body, wrap="word", font=("Courier", 9),
                   yscrollcommand=scroll.set)
    scroll.config(command=text.yview)
    scroll.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    text.insert("1.0", fcs_globalfit.rule_help_for(names))
    text.configure(state="disabled")

    tk.Button(win, text="Close", width=10, command=win.destroy,
              pady=4).pack(pady=(0, 8))
    return win


# ── Parameter table ───────────────────────────────────────────────────────────

class ParamTable:
    """
    The per-parameter grid: Link, Guess, Lower, Upper, Fix, and a rule.

    Build it into *parent*, then call :meth:`read` when the user commits.  The
    caller supplies the model and the dataset names; everything else -- widget
    state, live rule checking, validation -- is handled here.

    The Rule column
    ---------------
    Ticking Rule for a parameter disables that row's Link and Fix boxes.  Both
    describe the same thing, and with both live there is no defined answer to
    "the box says linked and the rule says F(2=0), which wins?".  This is not a
    judgement about what should be linked -- any rule the parser accepts is
    allowed -- only about not having two controls for one setting.

    Rules are checked as they are typed and the result shown in plain words
    underneath, because a rule is a small program and the failure mode of a
    small program is doing something other than what was meant.  Seeing
    "linked {d1, d3, d4}; fixed d2 = 1e-06; free d5, d6" is what catches the
    off-by-one before the fit runs, not after.
    """

    def __init__(self, parent, model, dataset_names: Sequence[str],
                 guesses0: Optional[Dict[str, float]] = None,
                 allow_rules: bool = True):
        import tkinter as tk

        self.model = model
        self.names = [p.name for p in model.params]
        self.dataset_names = list(dataset_names)
        self.D = len(self.dataset_names)
        self.multi = self.D > 1
        guesses0 = guesses0 or {}

        self.link_vars: Dict[str, tk.BooleanVar] = {}
        self.guess_vars: Dict[str, tk.StringVar] = {}
        self.lower_vars: Dict[str, tk.StringVar] = {}
        self.upper_vars: Dict[str, tk.StringVar] = {}
        self.fixed_vars: Dict[str, tk.BooleanVar] = {}
        self.rule_on: Dict[str, tk.BooleanVar] = {}
        self.rule_text: Dict[str, tk.StringVar] = {}
        self._link_widgets: Dict[str, object] = {}
        self._fix_widgets: Dict[str, object] = {}
        self._rule_widgets: Dict[str, object] = {}

        self.frame = tk.Frame(parent)
        table = tk.Frame(self.frame, padx=12, pady=4)
        table.pack(fill="x")

        headers = ["Parameter", "Link", "Guess", "Lower", "Upper", "Fix"]
        if allow_rules and self.multi:
            headers += ["Rule?", "Rule"]
        for c, h in enumerate(headers):
            tk.Label(table, text=h, font=("Helvetica", 10, "bold")).grid(
                row=0, column=c, padx=4, pady=(0, 4))

        for r, p in enumerate(model.params, start=1):
            label = p.name + (f" ({p.unit})" if p.unit else "")
            tk.Label(table, text=label, anchor="w", width=12).grid(
                row=r, column=0, sticky="w", padx=4, pady=2)

            lkv = tk.BooleanVar(value=p.link_default if self.multi else False)
            gv  = tk.StringVar(value=f"{guesses0.get(p.name, p.default):.6g}")
            lv  = tk.StringVar(value=_fmt(p.lower))
            uv  = tk.StringVar(value=_fmt(p.upper))
            fv  = tk.BooleanVar(value=p.fixed)

            lkw = tk.Checkbutton(table, variable=lkv,
                                 state="normal" if self.multi else "disabled")
            lkw.grid(row=r, column=1, padx=4)
            tk.Entry(table, textvariable=gv, width=12).grid(row=r, column=2, padx=4)
            tk.Entry(table, textvariable=lv, width=10).grid(row=r, column=3, padx=4)
            tk.Entry(table, textvariable=uv, width=10).grid(row=r, column=4, padx=4)
            fw = tk.Checkbutton(table, variable=fv)
            fw.grid(row=r, column=5, padx=4)

            self.link_vars[p.name]  = lkv
            self.guess_vars[p.name] = gv
            self.lower_vars[p.name] = lv
            self.upper_vars[p.name] = uv
            self.fixed_vars[p.name] = fv
            self._link_widgets[p.name] = lkw
            self._fix_widgets[p.name]  = fw

            if allow_rules and self.multi:
                rv = tk.BooleanVar(value=False)
                rt = tk.StringVar(value="")
                rw = tk.Entry(table, textvariable=rt, width=26,
                              state="disabled")
                tk.Checkbutton(
                    table, variable=rv,
                    command=lambda n=p.name: self._toggle_rule(n),
                ).grid(row=r, column=6, padx=4)
                rw.grid(row=r, column=7, padx=4)
                rt.trace_add("write", lambda *_a: self._refresh_status())
                self.rule_on[p.name] = rv
                self.rule_text[p.name] = rt
                self._rule_widgets[p.name] = rw

        if allow_rules and self.multi:
            bar = tk.Frame(self.frame)
            bar.pack(fill="x", padx=12, pady=(4, 0))
            tk.Button(bar, text="Rule syntax…", pady=2,
                      command=lambda: show_rule_help(
                          self.frame, self.dataset_names)).pack(side="left")
            tk.Label(bar, text="  e.g.  L(1,3,4); F(2=1e-6)    or    "
                              "L(1-10); L(11-20)",
                     font=("Courier", 8), fg="grey").pack(side="left")

            self.status = tk.StringVar(value="")
            self._status_label = tk.Label(
                self.frame, textvariable=self.status, font=("Courier", 8),
                fg="grey", anchor="w", justify="left", wraplength=620)
            self._status_label.pack(fill="x", padx=12, pady=(2, 0))
        else:
            self.status = None
            self._status_label = None

    # ── internals ────────────────────────────────────────────────────────────

    def _toggle_rule(self, name: str):
        """Enable the rule entry and stand the Link/Fix boxes down."""
        on = self.rule_on[name].get()
        self._rule_widgets[name].configure(state="normal" if on else "disabled")
        for widgets in (self._link_widgets, self._fix_widgets):
            widgets[name].configure(state="disabled" if on else "normal")
        if not on:
            # Leaving the text in place would make an unticked rule look active.
            self.rule_text[name].set("")
        self._refresh_status()

    def _parse_rules(self):
        """``(plans, errors)`` for every parameter whose rule box is ticked."""
        plans: Dict[str, tuple] = {}
        errors: Dict[str, str] = {}
        for name in self.names:
            if name not in self.rule_on or not self.rule_on[name].get():
                continue
            text = self.rule_text[name].get().strip()
            if not text:
                errors[name] = "rule is empty"
                continue
            try:
                default = float(self.guess_vars[name].get())
            except (TypeError, ValueError):
                default = 0.0
            try:
                plan, _summary = fcs_globalfit.parse_dataset_rule(
                    text, self.D, default_value=default)
            except fcs_globalfit.RuleError as exc:
                errors[name] = str(exc).replace("\n", " ")
            else:
                plans[name] = plan
        return plans, errors

    def _refresh_status(self):
        if self.status is None:
            return
        plans, errors = self._parse_rules()
        lines: List[str] = []
        for name in self.names:
            if name in errors:
                lines.append(f"✗ {name}: {errors[name]}")
            elif name in plans:
                lines.append(f"✓ {name}: " + fcs_globalfit.describe_plan(
                    plans[name], self.dataset_names))
        self.status.set("\n".join(lines))
        if self._status_label is not None:
            self._status_label.configure(fg="#a00" if errors else "#050")

    # ── public ───────────────────────────────────────────────────────────────

    def pack(self, **kw):
        self.frame.pack(**kw)
        return self.frame

    def read(self) -> dict:
        """
        Collect and validate the table.

        Returns ``{guesses, lowers, uppers, linked, fixed, plans}`` ready for
        ``fit_linked`` / ``fit_global``.

        Raises
        ------
        ValueError
            With a message meant for a message box.  Bad numbers, crossed
            bounds and unusable rules are all caught here rather than surfacing
            from inside the fit, where the message would be about arrays.
        """
        try:
            guesses = {n: float(self.guess_vars[n].get()) for n in self.names}
        except ValueError:
            raise ValueError("Guesses must be numbers.")
        try:
            lowers = {n: _parse_bound(self.lower_vars[n].get(), -np.inf)
                      for n in self.names}
            uppers = {n: _parse_bound(self.upper_vars[n].get(), np.inf)
                      for n in self.names}
        except ValueError:
            raise ValueError("Bounds must be numbers (use 'inf' / '-inf').")

        for n in self.names:
            if lowers[n] >= uppers[n]:
                raise ValueError(f"For '{n}', lower must be < upper.")

        plans, errors = self._parse_rules()
        if errors:
            raise ValueError(
                "These rules could not be used:\n\n"
                + "\n".join(f"  {n}:  {msg}" for n, msg in errors.items())
                + "\n\nUse the 'Rule syntax…' button for the accepted forms.")

        # A rule supersedes the row's Link/Fix boxes, which are disabled while
        # it is active; sending them anyway would be harmless (fit_linked
        # prefers the plan) but would leave the result's flags describing
        # controls the user could not reach.
        linked = {n: (self.link_vars[n].get()
                      and self.multi and n not in plans)
                  for n in self.names}
        fixed = {n: (self.fixed_vars[n].get() and n not in plans)
                 for n in self.names}

        # A held value is held at its guess, so a bound that excludes the guess
        # is a contradiction the fit cannot report -- it never sees the
        # parameter at all.
        for n in self.names:
            if fixed[n] and not (lowers[n] <= guesses[n] <= uppers[n]):
                raise ValueError(
                    f"'{n}' is fixed at {guesses[n]:g}, which is outside its "
                    f"bounds [{_fmt(lowers[n])}, {_fmt(uppers[n])}].")

        return {"guesses": guesses, "lowers": lowers, "uppers": uppers,
                "linked": linked, "fixed": fixed,
                "plans": plans or None}
