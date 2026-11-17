"""
Edit table for the "Plot defaults" preferences tab.

Imported by patch_plot_defaults.py; kept separate so the edits are
readable on their own.  Every entry is (description, old_text, new_text)
with LF newlines; the patcher re-encodes to each file's own endings.
"""

PLOTTOOLS = [
    # ── residual style: constants + helper ────────────────────────────────
    ("residual style constants and helper",
'''# ── Gridlines ─────────────────────────────────────────────────────────────────
''',
'''# ── Residual traces ───────────────────────────────────────────────────────────

# How residual panels are drawn by default, suite-wide.  A thin line makes a
# trend in the residuals legible: a systematic wander that reads as a vague
# thickening in a cloud of dots is obvious as a curve.  Points remain the
# honest choice when the residuals are genuinely independent samples and a
# connecting line would imply continuity that is not there, so this is a
# preference rather than a fixed change.
#
# "points" | "line" | "line+points"
_RESID_STYLE = "line"
_RESID_LINEWIDTH = 0.8

_RESID_STYLES = ("points", "line", "line+points")


def residual_style(marker: str = ".", markersize: float = 2.5,
                   **overrides) -> dict:
    """
    Matplotlib keyword arguments for one residual trace.

    Call sites pass the marker and size they would otherwise have hard-coded
    (so switching back to "points" reproduces the figure exactly as it was
    drawn before) and keep passing their own colour, alpha and label::

        axr.plot(x, res, color=c, alpha=0.8, label="_resid",
                 **fcs_plottools.residual_style(markersize=2.5))

    *overrides* win over everything, for a call site with a real reason to
    differ.  An unrecognised _RESID_STYLE falls back to points rather than
    raising: a bad preference value must not take down a fit plot.
    """
    style = _RESID_STYLE if _RESID_STYLE in _RESID_STYLES else "points"
    if style == "points":
        kw = {"linestyle": "none", "marker": marker, "markersize": markersize}
    elif style == "line":
        kw = {"linestyle": "-", "linewidth": _RESID_LINEWIDTH, "marker": "None"}
    else:
        kw = {"linestyle": "-", "linewidth": _RESID_LINEWIDTH,
              "marker": marker, "markersize": markersize}
    kw.update(overrides)
    return kw


# ── Gridlines ─────────────────────────────────────────────────────────────────
'''),

    # ── grid: scalars replace the two dicts ───────────────────────────────
    ("grid style scalars and helpers",
'''_GRID_MAJOR_KW = {"linestyle": "--", "linewidth": 0.4, "alpha": 0.5}
# Minor lines are deliberately lighter.  A semilog correlation axis carries
# eight minor lines per decade, and at major weight they read as a hatch over
# the data rather than as a reference to measure against.
_GRID_MINOR_KW = {"linestyle": ":", "linewidth": 0.3, "alpha": 0.3}
''',
'''# Scalars rather than one dict because a preference targets a module
# attribute: a dict would have to be replaced wholesale to change one value.
_GRID_MAJOR_LINESTYLE = "--"
_GRID_MAJOR_LINEWIDTH = 0.4
_GRID_MAJOR_ALPHA = 0.5
# Minor lines are deliberately lighter.  A semilog correlation axis carries
# eight minor lines per decade, and at major weight they read as a hatch over
# the data rather than as a reference to measure against.
_GRID_MINOR_LINESTYLE = ":"
_GRID_MINOR_LINEWIDTH = 0.3
_GRID_MINOR_ALPHA = 0.3

# Gridlines on a freshly drawn figure: "as drawn" leaves each plotter's own
# choice alone, which is the shipped behaviour; "on" and "off" override every
# axis at display time.
_GRID_DEFAULT = "as drawn"
_GRID_DEFAULTS = ("as drawn", "on", "off")

# Whether "on" brings minor gridlines with it.
_GRID_SHOW_MINOR = True


def grid_major_kw() -> dict:
    """Style for major gridlines, read fresh so preferences take effect."""
    return {"linestyle": _GRID_MAJOR_LINESTYLE,
            "linewidth": _GRID_MAJOR_LINEWIDTH,
            "alpha": _GRID_MAJOR_ALPHA}


def grid_minor_kw() -> dict:
    """Style for minor gridlines, read fresh so preferences take effect."""
    return {"linestyle": _GRID_MINOR_LINESTYLE,
            "linewidth": _GRID_MINOR_LINEWIDTH,
            "alpha": _GRID_MINOR_ALPHA}


def apply_grid_default(axes_list) -> None:
    """
    Impose the _GRID_DEFAULT preference on freshly drawn axes.

    Called from show_figure, after the plotter has drawn and before the
    controls panel is built, so the Grid checkboxes reflect what is on
    screen.  "as drawn" returns immediately, which is why a plotter that
    deliberately draws no grid keeps its appearance unless the user asks
    otherwise.

    Note that "on" calls minorticks_on() where minor gridlines are wanted
    but no minor ticks exist: gridlines are drawn at ticks, so without them
    ax.grid(which="minor") silently draws nothing.
    """
    if _GRID_DEFAULT not in _GRID_DEFAULTS or _GRID_DEFAULT == "as drawn":
        return
    want = _GRID_DEFAULT == "on"
    for ax in axes_list:
        try:
            if want:
                ax.grid(True, which="major", **grid_major_kw())
            else:
                ax.grid(False, which="major")
        except Exception:
            pass
        try:
            if want and _GRID_SHOW_MINOR:
                if not (len(ax.xaxis.get_minorticklocs()) or
                        len(ax.yaxis.get_minorticklocs())):
                    ax.minorticks_on()
                ax.grid(True, which="minor", **grid_minor_kw())
            else:
                ax.grid(False, which="minor")
        except Exception:
            pass
'''),

    ("grid section reads the helpers (major)",
'''                    ax.grid(True, which="major", **dict(_GRID_MAJOR_KW))
''',
'''                    ax.grid(True, which="major", **grid_major_kw())
'''),

    ("grid section reads the helpers (minor)",
'''                    ax.grid(True, which="minor", **dict(_GRID_MINOR_KW))
''',
'''                    ax.grid(True, which="minor", **grid_minor_kw())
'''),

    ("show_figure applies the grid default",
'''    # Try to get the Tk window that backs this figure
''',
'''    # Gridlines before the panel is built, so the Grid checkboxes open
    # showing what is actually on screen.
    apply_grid_default(axes_list)

    # Try to get the Tk window that backs this figure
'''),

    # ── save: tight crop default ──────────────────────────────────────────
    ("tight-crop default constant",
'''# Raster resolution.  Ignored for svg/pdf, which have no pixel grid.
_FIGURE_DPI = 150
''',
'''# Raster resolution.  Ignored for svg/pdf, which have no pixel grid.
_FIGURE_DPI = 150

# Crop saved figures to the drawn artists (plus savefig.pad_inches) rather
# than writing them at the figure's own size.  Convenient for a figure that
# will be dropped into a document, wrong when the file has to hit a column
# width exactly -- see the Figure size section of the controls panel, whose
# checkbox starts from this value.
_SAVE_TIGHT = True
'''),

    ("save_figure honours the default",
'''def save_figure(fig, path, *, fmt: str = None, dpi: int = None,
                tight: bool = True):
''',
'''def save_figure(fig, path, *, fmt: str = None, dpi: int = None,
                tight: bool = None):
'''),

    ("save_figure docstring and tight resolution",
'''    kw = {}
    if tight:
        kw["bbox_inches"] = "tight"
''',
'''    kw = {}
    if _SAVE_TIGHT if tight is None else tight:
        kw["bbox_inches"] = "tight"
'''),

    ("size section checkbox starts from the preference",
'''    tight_var = tk.BooleanVar(master=panel, value=True)
''',
'''    tight_var = tk.BooleanVar(master=panel, value=bool(_SAVE_TIGHT))
'''),
]


# ── Residual call sites ───────────────────────────────────────────────────
# Each keeps its own colour, alpha and label; only the marker/line styling
# moves into residual_style(), seeded with what the site drew before.

RESID_SITES = {
    "fcs_fit.py": [
        ("global fit residual panel",
'''        axr.semilogx(tau_ms, res, linestyle="none", marker=".",
                     markersize=2.5, color=c, alpha=0.8,
                     label=f"_{ds['name']}  (residual)")
''',
'''        axr.semilogx(tau_ms, res, color=c, alpha=0.8,
                     label=f"_{ds['name']}  (residual)",
                     **fcs_plottools.residual_style(markersize=2.5))
'''),
    ],
    "fcs_lifetime_fit.py": [
        ("single lifetime fit residuals",
'''    axr.plot(x, res, linestyle="none", marker=".",
             markersize=3, color="steelblue")
''',
'''    axr.plot(x, res, color="steelblue",
             **fcs_plottools.residual_style(markersize=3))
'''),
        ("lifetime global fit residuals",
'''        axr.plot(d["x"], d["resid"] / sig, linestyle="none", marker=".",
                 markersize=2, color=colour)
''',
'''        axr.plot(d["x"], d["resid"] / sig, color=colour,
                 **fcs_plottools.residual_style(markersize=2))
'''),
    ],
    "fcs_pch_fit.py": [
        ("single PCH fit residuals",
'''    axr.plot(k, res, linestyle="none", marker="o", markersize=4, color="steelblue")
''',
'''    axr.plot(k, res, color="steelblue",
             **fcs_plottools.residual_style(marker="o", markersize=4))
'''),
        ("PCH global fit residuals",
'''        axr.plot(ds["x"], ds["resid"] / sig, marker="o", markersize=3,
                 linestyle="none", color=colour)
''',
'''        axr.plot(ds["x"], ds["resid"] / sig, color=colour,
                 **fcs_plottools.residual_style(marker="o", markersize=3))
'''),
    ],
    "fcs_matrixfit.py": [
        ("matrix fit residuals",
'''    axr.plot(tau_ms, res, "o", ms=2.5, color="#2c6fbb", label="_residuals")
''',
'''    axr.plot(tau_ms, res, color="#2c6fbb", label="_residuals",
             **fcs_plottools.residual_style(marker="o", markersize=2.5))
'''),
        ("matrix overlay residuals",
'''        axr.plot(tau_ms, res, linestyle="none", marker=".", ms=2.5,
                 color=c, alpha=0.8, label="_%s  (residual)" % f["name"])
''',
'''        axr.plot(tau_ms, res, color=c, alpha=0.8,
                 label="_%s  (residual)" % f["name"],
                 **fcs_plottools.residual_style(markersize=2.5))
'''),
    ],
    "fcs_basisfit.py": [
        ("curve-basis residuals",
'''    axr.plot(tau_ms, res, "o", ms=2.5, color="#2c6fbb", label="_residuals")
''',
'''    axr.plot(tau_ms, res, color="#2c6fbb", label="_residuals",
             **fcs_plottools.residual_style(marker="o", markersize=2.5))
'''),
    ],
    "fcs_lifetime_recon.py": [
        ("reconvolution weighted residuals",
'''    axr.plot(t, result["wresid"], linestyle="none", marker=".",
             markersize=2.5, color="steelblue")
''',
'''    axr.plot(t, result["wresid"], color="steelblue",
             **fcs_plottools.residual_style(markersize=2.5))
'''),
    ],
}


# ── Registry entries ──────────────────────────────────────────────────────

PREFS_EDIT = [
    ("Plot defaults registry entries",
'''    Pref(
        key="linsolve_cd_target_chi2_red",
''',
'''    Pref(
        key="resid_style",
        label="Residual trace style",
        section="Plot defaults",
        kind="choice",
        default="line",
        target=("fcs_plottools", "_RESID_STYLE"),
        choices=("points", "line", "line+points"),
        help="A thin line shows a trend in the residuals that a cloud of "
             "dots hides. Applies to FCS, lifetime, PCH and matrix/basis "
             "fit panels; the Traces section still overrides one figure.",
    ),
    Pref(
        key="resid_linewidth",
        label="Residual line width",
        section="Plot defaults",
        kind="float",
        default=0.8,
        target=("fcs_plottools", "_RESID_LINEWIDTH"),
        lo=0.1, hi=5.0,
        help="Points, not pixels. Ignored when the style above is points.",
    ),
    Pref(
        key="save_tight",
        label="Crop saved figures to content",
        section="Plot defaults",
        kind="bool",
        default=True,
        target=("fcs_plottools", "_SAVE_TIGHT"),
        help="On, the file is cropped to the drawn artists and is NOT the "
             "figure size. Off, it is exactly the size set in the controls "
             "panel -- what a journal column width needs.",
    ),
    Pref(
        key="svg_editable_text",
        label="Keep SVG text editable",
        section="Plot defaults",
        kind="bool",
        default=True,
        target=("fcs_plottools", "_SVG_EDITABLE_TEXT"),
        help="On, labels stay as text for Illustrator/Inkscape. Off, every "
             "glyph becomes a path: renders anywhere, editable nowhere.",
    ),
    Pref(
        key="grid_default",
        label="Gridlines on new figures",
        section="Plot defaults",
        kind="choice",
        default="as drawn",
        target=("fcs_plottools", "_GRID_DEFAULT"),
        choices=("as drawn", "on", "off"),
        help="'as drawn' keeps each plot's own choice. The others override "
             "every axis when the figure is shown; the Grid section still "
             "toggles one figure.",
    ),
    Pref(
        key="grid_show_minor",
        label="Include minor gridlines",
        section="Plot defaults",
        kind="bool",
        default=True,
        target=("fcs_plottools", "_GRID_SHOW_MINOR"),
        help="Only consulted when the setting above is 'on'.",
    ),
    Pref(
        key="grid_major_linestyle",
        label="Major gridline style",
        section="Plot defaults",
        kind="choice",
        default="--",
        target=("fcs_plottools", "_GRID_MAJOR_LINESTYLE"),
        choices=("--", ":", "-.", "-"),
        help="Also used by the Grid section when it switches gridlines on.",
    ),
    Pref(
        key="linsolve_cd_target_chi2_red",
'''),
]
