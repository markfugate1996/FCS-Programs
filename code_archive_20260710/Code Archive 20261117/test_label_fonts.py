"""
Standalone test for the Labels section font-size controls: title, X label
and Y label, each independent.

Supersedes test_title_fontsize.py (same checks, plus the axis labels).
No tkinter needed; the three functions are pulled out with `ast` and run
against a stub tkinter and a real matplotlib Figure.

Usage: python3 test_label_fonts.py [path/to/fcs_plottools.py]
"""
import ast
import sys
import types

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WANTED = {"_fmt_fontsize", "_parse_fontsize", "_build_label_section",
          "_tick_fontsize"}
ok = 0


def check(label, cond, detail=""):
    global ok
    assert cond, f"{label}{(': ' + str(detail)) if detail else ''}"
    ok += 1
    print(f"PASS  {label}")


class _Widget:
    def __init__(self, master=None, **kw):
        self.master, self.kw, self.children = master, kw, []
        if isinstance(master, _Widget):
            master.children.append(self)
    def pack(self, **kw): pass
    def bind(self, *a): pass
    def _walk(self):
        yield self
        for c in self.children:
            yield from c._walk()


class _Text(_Widget):
    def __init__(self, master=None, **kw):
        super().__init__(master, **kw); self._buf = ""
    def insert(self, _i, text): self._buf += text
    def get(self, *_a): return self._buf
    def set_text(self, text): self._buf = text


class _StringVar:
    def __init__(self, master=None, value=""): self._v = str(value)
    def get(self): return self._v
    def set(self, v): self._v = str(v)


def _tk_stub():
    m = types.ModuleType("tkinter")
    for n in ("Frame", "Label", "LabelFrame", "Entry", "Button", "Spinbox"):
        setattr(m, n, type(n, (_Widget,), {}))
    m.Text, m.StringVar = _Text, _StringVar
    return m


def _load(path):
    src = open(path, encoding="utf-8").read()
    picked = [n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    missing = WANTED - {n.name for n in picked}
    assert not missing, f"not found: {sorted(missing)}"
    ns = {"plt": plt}
    exec(compile(ast.Module(body=picked, type_ignores=[]), path, "exec"), ns)
    return ns, src


def _find(root, name):
    return [w for w in root._walk() if type(w).__name__ == name]


def main(path):
    sys.modules["tkinter"] = _tk_stub()
    ns, src = _load(path)
    build = ns["_build_label_section"]

    fig, ax = plt.subplots()
    ax.set_title("Lifetime fit — 2-exp\nsample A", fontsize=11)
    ax.set_xlabel("tau (us)", fontsize=12)
    ax.set_ylabel("G(tau)", fontsize=9)       # a producer that chose its own

    panel = _Widget()
    build(panel, fig, [ax])

    spins = _find(panel, "Spinbox")
    check("four size boxes: title, X label, Y label, ticks", len(spins) == 4,
          len(spins))
    t_var, x_var, y_var, k_var = (s.kw["textvariable"] for s in spins)
    check("each box starts from its own artist's size",
          (t_var.get(), x_var.get(), y_var.get()) == ("11", "12", "9"),
          (t_var.get(), x_var.get(), y_var.get()))
    check("the tick box starts from the size the ticks are drawn at",
          float(k_var.get()) == ns["_tick_fontsize"](ax),
          (k_var.get(), ns["_tick_fontsize"](ax)))

    apply_labels = next(b.kw["command"] for b in _find(panel, "Button")
                        if b.kw.get("text", "").startswith("Apply labels"))

    # applying without touching anything must not restyle the figure
    apply_labels()
    check("a no-op apply leaves every size alone (was forced to 12 before)",
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()) == (11, 12, 9),
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()))

    # independence, one at a time
    t_var.set("18"); apply_labels()
    check("title size moves alone",
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()) == (18, 12, 9))

    x_var.set("7"); apply_labels()
    check("X label size moves alone",
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()) == (18, 7, 9))

    y_var.set("14.5"); apply_labels()
    check("Y label size moves alone, fractional accepted",
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()) == (18, 7, 14.5))

    # text still applies, and carries the chosen sizes
    boxes = _find(panel, "_Text")
    boxes[0].set_text("New title")
    entries = _find(panel, "Entry")
    entries[0].kw["textvariable"].set("lag (ms)")
    entries[1].kw["textvariable"].set("G")
    apply_labels()
    check("label text still applies",
          (ax.get_title(), ax.get_xlabel(), ax.get_ylabel())
          == ("New title", "lag (ms)", "G"),
          (ax.get_title(), ax.get_xlabel(), ax.get_ylabel()))
    check("sizes survive a text edit",
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()) == (18, 7, 14.5))

    # junk in one box must not disturb the other two
    for var, junk in ((t_var, "abc"), (x_var, ""), (y_var, "0")):
        var.set(junk)
    apply_labels()
    check("junk in all three falls back to each axis's own size",
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()) == (18, 7, 14.5),
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()))
    check("the boxes are restored to the real sizes",
          (t_var.get(), x_var.get(), y_var.get()) == ("18", "7", "14.5"),
          (t_var.get(), x_var.get(), y_var.get()))

    # ── tick labels ───────────────────────────────────────────────────────
    k_var.set("7"); apply_labels()
    sizes = {lbl.get_fontsize() for lbl in
             list(ax.get_xticklabels()) + list(ax.get_yticklabels())}
    check("tick labels follow the box, x and y together",
          sizes == {7.0}, sizes)
    check("and the other three sizes are left alone",
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()) == (18, 7, 14.5),
          (ax.title.get_fontsize(), ax.xaxis.label.get_fontsize(),
           ax.yaxis.label.get_fontsize()))
    check("the exponent offset text follows too",
          ax.xaxis.get_offset_text().get_fontsize() == 7.0
          and ax.yaxis.get_offset_text().get_fontsize() == 7.0,
          (ax.xaxis.get_offset_text().get_fontsize(),
           ax.yaxis.get_offset_text().get_fontsize()))
    check("minor tick labels are included",
          all(t.label1.get_fontsize() == 7.0
              for t in ax.xaxis.get_minor_ticks()) or
          not list(ax.xaxis.get_minor_ticks()))

    k_var.set("junk"); apply_labels()
    sizes = {lbl.get_fontsize() for lbl in ax.get_xticklabels()}
    check("junk in the tick box falls back to the live size",
          sizes == {7.0} and k_var.get() == "7", (sizes, k_var.get()))

    # every panel, not just the one being titled
    fig3, (axt, axb) = plt.subplots(2, 1)
    axt.set_title("t", fontsize=10)
    axt.set_ylabel("G", fontsize=10)
    axb.set_xlabel("tau", fontsize=10)
    panel3 = _Widget()
    build(panel3, fig3, [axt, axb])
    spins3 = _find(panel3, "Spinbox")
    spins3[3].kw["textvariable"].set("6")
    next(b.kw["command"] for b in _find(panel3, "Button")
         if b.kw.get("text", "").startswith("Apply labels"))()
    both = {lbl.get_fontsize() for a in (axt, axb)
            for lbl in list(a.get_xticklabels()) + list(a.get_yticklabels())}
    check("a residual strip gets the same tick size as the main panel",
          both == {6.0}, both)

    # stacked figures: x label belongs to the bottom axis, y to the first
    fig2, (axa, axb) = plt.subplots(2, 1, sharex=True)
    axa.set_title("top", fontsize=10)
    axa.set_ylabel("G", fontsize=10)
    axb.set_xlabel("tau", fontsize=10)
    panel2 = _Widget()
    build(panel2, fig2, [axa, axb])
    s2 = _find(panel2, "Spinbox")
    v2 = [s.kw["textvariable"] for s in s2]
    v2[1].set("8"); v2[2].set("16")
    next(b.kw["command"] for b in _find(panel2, "Button")
         if b.kw.get("text", "").startswith("Apply labels"))()
    check("stacked panels: X size hits the bottom axis, Y the main one",
          (axb.xaxis.label.get_fontsize(), axa.yaxis.label.get_fontsize())
          == (8, 16),
          (axb.xaxis.label.get_fontsize(), axa.yaxis.label.get_fontsize()))

    check("no hard-coded fontsize=12 left in the label apply",
          "set_xlabel(xlabel_var.get(), fontsize=12)" not in src)

    print(f"\nAll {ok} checks passed.")
    return 0


def _target_path(argv):
    import pathlib
    p = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    if p.is_dir():
        p = p / "fcs_plottools.py"
    if not p.is_file():
        print(f"ERROR: {p} not found.\n"
              f"Run this from the FCS-Programs directory, or pass the path "
              f"to fcs_plottools.py.")
        raise SystemExit(2)
    return str(p)


if __name__ == "__main__":
    sys.exit(main(_target_path(sys.argv)))
