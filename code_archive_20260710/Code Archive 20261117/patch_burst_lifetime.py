#!/usr/bin/env python3
"""
Make the lifetime fitter accept decays exported by burst excision.

An ordinary decay export carries one histogram per channel
(ch1_counts / ch2_counts).  A burst-excision export carries three per
channel -- every photon, the burst photons, and what survived excision --
as chN_all / chN_burst / chN_clean, and the loader rejected the file for
having no column it recognised.

All of them are now offered for fitting: six decays for a two-channel
file, listed in the fit setup screen where the Ch1/Ch2 choice normally
sits.  Whichever is chosen is presented downstream as that channel's
curve, so gating, plotting, the fitters and the reconvolution path need
no changes.

A script rather than a diff because fcs_lifetime_fit.py may be CRLF in
your working tree, and git apply refuses this patch against a CRLF copy.
Every anchor is validated before anything is written.

Usage:  python3 patch_burst_lifetime.py [path/to/fcs_lifetime_fit.py]
"""
import pathlib
import sys

EDITS = [
# ── 1. loader ────────────────────────────────────────────────────────────
("burst decay columns recognised",
'''    series: Dict[str, np.ndarray] = {}
    for ch in (1, 2):
        col = columns.get(f"ch{ch}_counts")
        if col is not None and _has_data(col):
            series[f"Ch{ch}"] = np.asarray(col, dtype=np.float64)
    if not series and "intensity" in columns and _has_data(columns["intensity"]):
        series["decay"] = np.asarray(columns["intensity"], dtype=np.float64)

    if not series:
        raise ValueError(
            f"{path.name} has a time axis but no usable counts column "
            f"(expected ch1_counts / ch2_counts, or intensity).  If this came "
            f"from a single-channel file, the channel it did record should "
            f"still hold data."
        )
''',
'''    series: Dict[str, np.ndarray] = {}
    for ch in (1, 2):
        col = columns.get(f"ch{ch}_counts")
        if col is not None and _has_data(col):
            series[f"Ch{ch}"] = np.asarray(col, dtype=np.float64)
    if not series and "intensity" in columns and _has_data(columns["intensity"]):
        series["decay"] = np.asarray(columns["intensity"], dtype=np.float64)

    # A burst-excision export holds THREE decays per channel instead of one --
    # every photon, the burst photons alone, and what survived excision --
    # written as chN_all / chN_burst / chN_clean.  Each is a decay in its own
    # right (the whole point of splitting them is that the burst population
    # need not share the bulk lifetime), so they are offered as separate
    # series rather than collapsed into one.
    if not series:
        for ch in (1, 2):
            for part, label in BURST_DECAY_PARTS:
                col = columns.get(f"ch{ch}_{part}")
                if col is not None and _has_data(col):
                    series[f"Ch{ch} {label}"] = np.asarray(col,
                                                           dtype=np.float64)

    if not series:
        raise ValueError(
            f"{path.name} has a time axis but no usable counts column "
            f"(expected ch1_counts / ch2_counts, chN_all / chN_burst / "
            f"chN_clean from a burst excision, or intensity).  If this came "
            f"from a single-channel file, the channel it did record should "
            f"still hold data."
        )
'''),

# ── 2. module constants + helpers ────────────────────────────────────────
("burst decay vocabulary",
'''class LoadedDecay:
    """
    A decay read back from an exported CSV, shaped like the piece of FCSData
''',
'''# The three decays a burst-excision export carries per channel, as
# (column suffix, label).  Order is the order they are offered in the fit
# screen.  "all" is every photon, so it is what an ordinary export would
# have held; "clean" is what survived excision and is the one a burst run
# is usually done for, which is why it is the default.
BURST_DECAY_PARTS = (("all", "all photons"),
                     ("burst", "bursts"),
                     ("clean", "burst-free"))

BURST_DECAY_LABELS = tuple(label for _part, label in BURST_DECAY_PARTS)

_BURST_DEFAULT_COMPONENT = "burst-free"


def _split_burst_key(key: str):
    """('Ch1 burst-free') -> (1, 'burst-free'); None for an ordinary key."""
    if not key.startswith("Ch") or " " not in key:
        return None
    head, _, label = key.partition(" ")
    if not head[2:].isdigit() or label not in BURST_DECAY_LABELS:
        return None
    return int(head[2:]), label


class LoadedDecay:
    """
    A decay read back from an exported CSV, shaped like the piece of FCSData
'''),

# ── 3. the adapter ───────────────────────────────────────────────────────
("component-aware adapter",
'''    def __init__(self, path, t_ns, series, irf=None, meta=None):
        self.filepath = Path(path)
        self.t_ns     = np.asarray(t_ns, dtype=np.float64)
        self.series   = {k: np.asarray(v, dtype=np.float64)
                         for k, v in series.items()}
        self.irf      = irf
        self.meta     = dict(meta or {})
        self.params   = dict(self.meta)

        # Channel identity comes from the column names: a Ch2-only export has
        # only a ch2_counts column, so its single decay is Ch2 and must not be
        # relabelled Ch1 just because it is the only one present.
        self._single  = "decay" in self.series
        self.channels = tuple(
            int(k[2:]) for k in ("Ch1", "Ch2") if k in self.series)
        if self._single:
            # An unlabelled 'decay' column carries no channel identity to
            # preserve, but fcs_lifetime indexes colours, labels and
            # require_channels() by channel number, and an empty tuple would
            # draw an empty gate picker.  Presenting it as channel 1 is the
            # convention an .ifx decay already gets: LifetimeData exposes no
            # channels attribute, so callers default it to Ch1.
            # channel_summary still reports "decay", so nothing in the UI
            # claims this curve came off detector 1.
            self.channels = (1,)
''',
'''    def __init__(self, path, t_ns, series, irf=None, meta=None):
        self.filepath = Path(path)
        self.t_ns     = np.asarray(t_ns, dtype=np.float64)
        self.irf      = irf
        self.meta     = dict(meta or {})
        self.params   = dict(self.meta)

        stored = {k: np.asarray(v, dtype=np.float64)
                  for k, v in series.items()}

        # A burst export carries several decays per channel.  Everything
        # downstream of here -- gating, plotting, both fitters, the
        # reconvolution path -- is written against a decay with ONE curve per
        # channel, so rather than teach all of that about components, the
        # SELECTED component is presented as Ch1/Ch2 and the rest are held
        # aside.  Once the choice is made a burst decay is handled exactly
        # like an ordinary one.
        self._by_part: dict = {}
        for key, arr in stored.items():
            split = _split_burst_key(key)
            if split is not None:
                self._by_part[split] = arr
        self.components = tuple(
            label for label in BURST_DECAY_LABELS
            if any(c == label for _ch, c in self._by_part))
        self.component = None

        if self.components:
            self._stored_other = {}
            default = (_BURST_DEFAULT_COMPONENT
                       if _BURST_DEFAULT_COMPONENT in self.components
                       else self.components[0])
            self.select_component(default)
        else:
            self.series = stored
            self._refresh_identity()

    # ── Burst components ─────────────────────────────────────────────────────

    def select_component(self, component: str) -> None:
        """
        Choose which of a burst export's decays this object presents as its
        Ch1/Ch2 curves.

        Raises for an unknown component rather than falling back: silently
        fitting "all photons" when "burst-free" was asked for is a wrong
        number with nothing on the figure to show for it.
        """
        if not self.components:
            raise ValueError(f"{self.filepath.name} is not a burst decay "
                             f"export; it has no components to choose.")
        if component not in self.components:
            raise ValueError(
                f"{self.filepath.name} has no '{component}' decay; it holds "
                f"{', '.join(self.components)}.")
        self.component = component
        self.series = {f"Ch{ch}": arr
                       for (ch, comp), arr in self._by_part.items()
                       if comp == component}
        self._refresh_identity()

    def component_channels(self, component: str) -> tuple:
        """Channels that actually hold data for *component*."""
        return tuple(sorted(ch for (ch, comp) in self._by_part
                            if comp == component))

    def decay_options(self) -> list:
        """
        Every decay this file offers, as (channel, component, label).

        An ordinary export yields one entry per recorded channel and no
        component; a burst export yields one per channel per component --
        six for a two-channel file -- which is what the fit screen lists.
        """
        if not self.components:
            if self._single:
                return [(None, None, "decay")]
            return [(ch, None, f"Ch{ch}") for ch in self.channels]
        out = []
        for ch in (1, 2):
            for label in self.components:
                if (ch, label) in self._by_part:
                    out.append((ch, label, f"Ch{ch} {label}"))
        return out

    def _refresh_identity(self) -> None:
        """Re-derive what the currently presented series means."""
        self._single = "decay" in self.series
''',
),

("identity derivation restored inside the helper",
'''    def _refresh_identity(self) -> None:
        """Re-derive what the currently presented series means."""
        self._single = "decay" in self.series

''',
'''    def _refresh_identity(self) -> None:
        """
        Re-derive what the currently presented series means.

        Called again whenever a burst component is selected, because the
        channels that hold data can differ between components: a channel
        with no bursts detected has an empty 'bursts' decay while its
        'all' and 'burst-free' decays are perfectly good.
        """
        # Channel identity comes from the column names: a Ch2-only export has
        # only a ch2_counts column, so its single decay is Ch2 and must not be
        # relabelled Ch1 just because it is the only one present.
        self._single  = "decay" in self.series
        self.channels = tuple(
            int(k[2:]) for k in ("Ch1", "Ch2") if k in self.series)
        if self._single:
            # An unlabelled 'decay' column carries no channel identity to
            # preserve, but fcs_lifetime indexes colours, labels and
            # require_channels() by channel number, and an empty tuple would
            # draw an empty gate picker.  Presenting it as channel 1 is the
            # convention an .ifx decay already gets: LifetimeData exposes no
            # channels attribute, so callers default it to Ch1.
            # channel_summary still reports "decay", so nothing in the UI
            # claims this curve came off detector 1.
            self.channels = (1,)

'''),

# channel_summary should say which component is showing
("channel summary names the component",
'''    @property
    def channel_summary(self) -> str:
        if self._single:
            return "decay"
        if len(self.channels) == 1:
            return f"Ch{self.channels[0]} only"
        return " + ".join(f"Ch{c}" for c in self.channels)
''',
'''    @property
    def channel_summary(self) -> str:
        if self._single:
            return "decay"
        if len(self.channels) == 1:
            base = f"Ch{self.channels[0]} only"
        else:
            base = " + ".join(f"Ch{c}" for c in self.channels)
        # Which decay of a burst export is on show.  Without this the picker
        # and the fit screen would say "Ch1 + Ch2" for three different pairs
        # of curves and look identical for each.
        return f"{base}, {self.component}" if self.component else base
'''),

# the fit setup screen
("fit screen lists every decay a burst export holds",
'''    else:
        ch_frame = tk.Frame(body)
        ch_frame.grid(row=0, column=1, sticky="w")
        for _ch in (1, 2):
            _rb = tk.Radiobutton(ch_frame, text=f"Ch{_ch}",
                                 variable=ch_var, value=_ch)
            if _ch not in available_ch:
                _rb.configure(state="disabled")
            _rb.pack(side="left")
        if len(available_ch) < 2:
            tk.Label(ch_frame,
                     text=f"({getattr(fcs_data, 'channel_summary', 'one channel')})",
                     font=("Helvetica", 8), fg="grey").pack(side="left", padx=(6, 0))
''',
'''    elif getattr(fcs_data, "components", ()):
        # A burst-excision export: three decays per channel rather than one.
        # They are listed individually instead of as a channel row plus a
        # component row, so what is on offer is the same flat list of decays
        # the file actually holds, and an absent combination simply is not
        # there to pick.
        tk.Label(body, text="", anchor="w").grid(row=0, column=1, sticky="w")
        opt_frame = tk.Frame(body)
        opt_frame.grid(row=0, column=1, sticky="w")
        options = fcs_data.decay_options()
        pick_var = tk.StringVar(
            value=f"{fcs_data.channels[0]}|{fcs_data.component}")

        def _pick_decay(*_a):
            ch_text, _, comp = pick_var.get().partition("|")
            try:
                fcs_data.select_component(comp)
            except ValueError:
                return
            ch_var.set(int(ch_text))

        for _i, (_ch, _comp, _label) in enumerate(options):
            tk.Radiobutton(opt_frame, text=_label, variable=pick_var,
                           value=f"{_ch}|{_comp}", command=_pick_decay,
                           anchor="w", width=16).grid(
                row=_i // 3, column=_i % 3, sticky="w")
        tk.Label(body, text="burst excision", font=("Helvetica", 8),
                 fg="grey").grid(row=0, column=0, sticky="sw")
    else:
        ch_frame = tk.Frame(body)
        ch_frame.grid(row=0, column=1, sticky="w")
        for _ch in (1, 2):
            _rb = tk.Radiobutton(ch_frame, text=f"Ch{_ch}",
                                 variable=ch_var, value=_ch)
            if _ch not in available_ch:
                _rb.configure(state="disabled")
            _rb.pack(side="left")
        if len(available_ch) < 2:
            tk.Label(ch_frame,
                     text=f"({getattr(fcs_data, 'channel_summary', 'one channel')})",
                     font=("Helvetica", 8), fg="grey").pack(side="left", padx=(6, 0))
'''),

# provenance on the result
("the fitted component travels with the result",
'''        result["origin"] = getattr(fcs_data, "source_name", None)
''',
'''        result["origin"] = getattr(fcs_data, "source_name", None)
        # Which of a burst export's decays this was.  Three fits of one file
        # differ in nothing else a reader could see.
        result["decay_component"] = getattr(fcs_data, "component", None)
'''),

("report names the component",
'''    if ch is not None:
        L.append(f"channel    : Ch{ch}")
''',
'''    if ch is not None:
        L.append(f"channel    : Ch{ch}")
    if result.get("decay_component"):
        L.append(f"decay      : {result['decay_component']}  "
                 f"(burst excision)")
'''),

("curve header names the component",
'''        if ch is not None:
            fh.write(f"# channel : Ch{ch}\\n")
''',
'''        if ch is not None:
            fh.write(f"# channel : Ch{ch}\\n")
        if result.get("decay_component"):
            fh.write(f"# decay_component : {result['decay_component']}\\n")
'''),

# batch / global picker
("batch picker offers each component",
'''            if d._single:
                out.append(fcs_fitdialogs.DatasetEntry(
                    path=path, series="decay", label=f"{path.name}  [decay]"))
            else:
                for ch in d.channels:
                    out.append(fcs_fitdialogs.DatasetEntry(
                        path=path, series=f"Ch{ch}",
                        label=f"{path.name}  [Ch{ch}]"))
''',
'''            if d._single:
                out.append(fcs_fitdialogs.DatasetEntry(
                    path=path, series="decay", label=f"{path.name}  [decay]"))
            else:
                # decay_options() covers both shapes: one entry per channel
                # for an ordinary export, one per channel per component for
                # a burst one, so a burst file contributes all of its decays
                # to a global fit instead of only the default component.
                for _ch, _comp, _label in d.decay_options():
                    out.append(fcs_fitdialogs.DatasetEntry(
                        path=path, series=_label,
                        label=f"{path.name}  [{_label}]"))
'''),

("batch loader honours the chosen component",
'''    def _load(entry) -> dict:
        d = load_decay_object(entry.path)
        channel = (None if entry.series == "decay"
                   else int(str(entry.series)[2:]))
''',
'''    def _load(entry) -> dict:
        d = load_decay_object(entry.path)
        if entry.series == "decay":
            channel = None
        else:
            # "Ch1" or, from a burst export, "Ch1 burst-free".
            head, _, comp = str(entry.series).partition(" ")
            channel = int(head[2:])
            if comp:
                d.select_component(comp)
'''),
]


def _target(argv):
    p = pathlib.Path(argv[1] if len(argv) > 1 else ".")
    if p.is_dir():
        p = p / "fcs_lifetime_fit.py"
    if not p.is_file():
        print(f"ERROR: {p} not found. Run this from the FCS-Programs "
              f"directory, or pass the path to fcs_lifetime_fit.py.")
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

    staged = data
    for name, old_lf, new_lf in EDITS:
        old, new = enc(old_lf), enc(new_lf)
        if staged.count(new) and not staged.count(old):
            print(f"ERROR: '{name}' looks already applied. Nothing written.")
            return 2
        n = staged.count(old)
        if n != 1:
            print(f"ERROR: anchor for '{name}' matched {n} times, expected 1."
                  f"  Nothing written.")
            return 2
        staged = staged.replace(old, new, 1)
        print(f"  staged: {name}")

    path.write_bytes(staged)
    print(f"wrote {path} ({len(staged)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
