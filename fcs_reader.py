"""
fcs_reader.py
=============
Reader for binary FCS (Fluorescence Correlation Spectroscopy) data files
as produced by ISS VistaVision / Alba FCS and similar confocal microscopy
acquisition software.

Data format
-----------
The binary .fcs file stores individual photon arrival records in
Time-Tagged Time-Resolved (TTTR) T3 mode.  Each photon has two
time coordinates:

  macrotime  — which laser clock cycle the photon arrived in,
               as an absolute count of cycles since t=0
  microtime  — which of 4096 bins within that cycle it arrived in
               (enables ~12 ps lifetime resolution via TCSPC)

Macrotimes are stored as the difference between each photon's macrotime
and the previous photon's macrotime on the same channel.  Cumulative-
summing these differences recovers the absolute macrotimes.  Microtimes
are stored as absolute bin indices.  The four blocks are:

    [Ch0 macrotime differences]  [Ch1 macrotime differences]
    [Ch0 microtimes]             [Ch1 microtimes]

Each block is preceded by a 2-word preamble: a length marker giving the
block size in bytes (= 4 x photon count) followed by one zero pad word.
The length marker locates each block deterministically, independent of
count rate.

Single-channel files
--------------------
An acquisition with only one detector enabled writes only TWO blocks —
that channel's macrotimes followed by its microtimes — and its metadata
lists only the one channel under [Detection Channels].  The reader detects
this from the block structure and reports it as ``FCSData.channels``,
which is ``(1, 2)``, ``(1,)`` or ``(2,)``.  Arrays belonging to a channel
that was not recorded are empty rather than absent, so attribute access
still works; gate channel-dependent features on ``has_ch1`` / ``has_ch2``
or :func:`require_channels`.

A DEAD channel is a different thing from an ABSENT one and is read
differently: a dual-channel acquisition in which one detector saw no
photons still writes all four blocks, one of them empty, and is reported
as ``channels == (1, 2)`` with zero photons on that channel.  The reader
does not downgrade it to single-channel — whether a silent detector means
a broken PMT or an empty sample is the user's judgement, not the parser's.

Binary layout
-------------
Offset              Content
------------------  ----------------------------------------------------------
0x000–0x3FF         Binary header (1024 bytes)
                    -- dual-channel payload (single-channel omits the Ch1
                       blocks entirely; see "Single-channel files" above) --
0x400               uint32: 4*n0  (byte length of Ch0 macro block)
0x404               uint32: 0 (padding)
0x408               uint32[n0]: Ch0 macrotime differences (laser clock cycles)
...                 uint32: 4*n1  (byte length of Ch1 macro block)
...                 uint32: 0 (padding)
...                 uint32[n1]: Ch1 macrotime differences
...                 uint32: 4*n0  (byte length of Ch0 micro block)
...                 uint32: 0 (padding)
...                 uint32[n0]: Ch0 microtimes (0–4095)
...                 uint32: 4*n1  (byte length of Ch1 micro block)
...                 uint32: 0 (padding)
...                 uint32[n1]: Ch1 microtimes (0–4095)
...                 footer + UTF-8 metadata block

Key header fields
-----------------
0x50  float64   Nominal laser clock frequency in Hz (~20 MHz).
                MacroTime period = 1 / clock_hz.
                The binary stores a rounded value; pass the precise clock
                from the plain-text export for accurate lifetime/diffusion
                measurements (~1% correction).

Usage
-----
    from fcs_reader import read_fcs

    d = read_fcs("experiment.fcs")
    print(d)                               # formatted summary

    # Photon arrival times (macrotime, seconds)
    d.ch0_times_s                          # absolute arrival times, Ch0
    d.ch1_times_s                          # absolute arrival times, Ch1

    # Microtimes (TCSPC, 0-4095 bins within each laser period)
    d.ch0_micro                            # shape (n0,), uint32
    d.ch1_micro                            # shape (n1,), uint32

    # Convert microtime bins to nanoseconds
    laser_period_ns = 1e9 / d.params["clock_hz"]
    d.ch0_micro * laser_period_ns / 4096   # Ch0 arrival time within cycle (ns)

    # Binned intensity trace
    t, I0, I1 = d.bin_intensity(bin_width_s=1e-3)

    # pandas DataFrame of binned intensity
    df = d.to_dataframe(bin_width_s=1e-3)

    # Supply accurate clock from text export header
    d2 = read_fcs("experiment.fcs", clock_hz=20_194_704.968582)
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


# ── File layout constants ────────────────────────────────────────────────────

_HEADER_SIZE       = 0x0400      # 1024 bytes
_DATA_START        = _HEADER_SIZE
_HDR_CLOCK_OFF     = 0x50       # float64: nominal (rounded) laser clock frequency in Hz
_HDR_TRUE_CLOCK_OFF = 0x22       # float64: true measured laser clock frequency in Hz (unaligned)

# Channel index mapping: the binary file uses 0-based indexing (ch0, ch1).
# All public-facing attributes use 1-based naming (ch1, ch2) to match
# the instrument labelling.
# NOTE: Blocks are NOT delimited by magnitude sentinels.  Each block carries
# an explicit length prefix (4 * photon_count bytes); see _extract_four_blocks.
# The old _MARKER_THRESHOLD heuristic is removed because real macrotime
# differences exceed any fixed threshold at low count rates.
_MICROTIME_BINS    = 4096       # fixed by instrument (MicroTime Resolution)


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class FCSData:
    """
    Container returned by :func:`read_fcs`.

    Attributes
    ----------
    filepath : Path
    params : dict
        Experimental parameters.  Keys include ``clock_hz``, ``timestamp``,
        ``objective_mag``, ``excitation_laser``, ``excitation_dichroic``,
        ``emission_dichroic``, ``channel_ch1``, ``channel_ch2``.
        ``clock_hz`` may be overwritten with a more accurate value.
    ch1_deltas : np.ndarray (uint32)
        Inter-photon macrotime differences for Ch1, in laser clock cycles.
    ch2_deltas : np.ndarray (uint32)
        Inter-photon macrotime differences for Ch2, in laser clock cycles.
    ch1_micro : np.ndarray (uint32)
        TCSPC microtime bin for each Ch1 photon (0 – 4095).
    ch2_micro : np.ndarray (uint32)
        TCSPC microtime bin for each Ch2 photon (0 – 4095).
    channels : tuple of int
        Which detector channels the file actually RECORDED: ``(1, 2)`` for a
        normal dual-channel acquisition, ``(1,)`` or ``(2,)`` for a
        single-channel one.  Defaults to ``(1, 2)`` so existing construction
        sites and pickled/cached objects keep working unchanged.

        This describes the FILE LAYOUT, not the photon yield.  A dual-channel
        acquisition in which one detector saw nothing still reports
        ``channels == (1, 2)`` with an empty array for that channel — a dead
        PMT is the user's call to make, not the reader's, and it is a
        different situation from a channel that was never recorded at all.
        Arrays for a channel that is absent from *channels* are empty, so
        attribute access never fails; use :attr:`has_ch1` / :attr:`has_ch2`
        (or :func:`require_channels`) to gate features rather than testing
        array length, which cannot tell "absent" from "dead".
    """
    filepath   : Path
    params     : dict
    ch1_deltas : np.ndarray
    ch2_deltas : np.ndarray
    ch1_micro  : np.ndarray
    ch2_micro  : np.ndarray
    channels   : Tuple[int, ...] = (1, 2)

    # ── Channel availability ─────────────────────────────────────────────────

    @property
    def n_channels(self) -> int:
        """Number of detector channels recorded in the file (1 or 2)."""
        return len(self.channels)

    @property
    def has_ch1(self) -> bool:
        """True when Ch1 was recorded (regardless of how many photons it saw)."""
        return 1 in self.channels

    @property
    def has_ch2(self) -> bool:
        """True when Ch2 was recorded (regardless of how many photons it saw)."""
        return 2 in self.channels

    @property
    def is_single_channel(self) -> bool:
        """True when only one detector channel was recorded."""
        return len(self.channels) == 1

    @property
    def channel_summary(self) -> str:
        """Human-readable channel list, e.g. ``'Ch1 + Ch2'`` or ``'Ch1 only'``."""
        if len(self.channels) == 1:
            return f"Ch{self.channels[0]} only"
        return " + ".join(f"Ch{c}" for c in self.channels)

    def deltas(self, channel: int) -> np.ndarray:
        """Macrotime deltas for *channel* (1 or 2), channel-agnostically."""
        return self.ch1_deltas if channel == 1 else self.ch2_deltas

    def micro(self, channel: int) -> np.ndarray:
        """Microtimes for *channel* (1 or 2), channel-agnostically."""
        return self.ch1_micro if channel == 1 else self.ch2_micro

    def times_s(self, channel: int) -> np.ndarray:
        """Absolute arrival times (s) for *channel* (1 or 2), channel-agnostically."""
        return self.ch1_times_s if channel == 1 else self.ch2_times_s

    def count_rate_hz(self, channel: int) -> float:
        """Mean count rate (Hz) for *channel* (1 or 2), channel-agnostically."""
        return self.count_rate_ch1_hz if channel == 1 else self.count_rate_ch2_hz

    # ── Timing ───────────────────────────────────────────────────────────────

    @property
    def macrotime_period_s(self) -> float:
        """Duration of one laser clock tick in seconds (= 1 / clock_hz)."""
        return 1.0 / float(self.params["clock_hz"])

    @property
    def laser_period_ns(self) -> float:
        """Duration of one laser clock cycle in nanoseconds."""
        return 1e9 / float(self.params["clock_hz"])

    @property
    def microtime_resolution_ns(self) -> float:
        """Width of one TCSPC microtime bin in nanoseconds."""
        return self.laser_period_ns / _MICROTIME_BINS

    @property
    def ch1_times_s(self) -> np.ndarray:
        """Absolute photon arrival times for Ch1, in seconds from t=0."""
        return np.cumsum(self.ch1_deltas.astype(np.float64)) * self.macrotime_period_s

    @property
    def ch2_times_s(self) -> np.ndarray:
        """Absolute photon arrival times for Ch2, in seconds from t=0."""
        return np.cumsum(self.ch2_deltas.astype(np.float64)) * self.macrotime_period_s

    @property
    def ch1_micro_ns(self) -> np.ndarray:
        """Ch1 microtime in nanoseconds (arrival time within laser cycle)."""
        return self.ch1_micro.astype(np.float64) * self.microtime_resolution_ns

    @property
    def ch2_micro_ns(self) -> np.ndarray:
        """Ch2 microtime in nanoseconds (arrival time within laser cycle)."""
        return self.ch2_micro.astype(np.float64) * self.microtime_resolution_ns

    @property
    def duration_s(self) -> float:
        """Total measurement duration in seconds."""
        t0 = float(self.ch1_deltas.sum()) * self.macrotime_period_s
        t1 = float(self.ch2_deltas.sum()) * self.macrotime_period_s
        return max(t0, t1)

    @property
    def count_rate_ch1_hz(self) -> float:
        """Mean count rate on Ch1 in Hz."""
        d = self.duration_s
        return len(self.ch1_deltas) / d if d else float("nan")

    @property
    def count_rate_ch2_hz(self) -> float:
        """Mean count rate on Ch2 in Hz."""
        d = self.duration_s
        return len(self.ch2_deltas) / d if d else float("nan")

    # ── Analysis helpers ─────────────────────────────────────────────────────

    def bin_intensity(
        self,
        bin_width_s: float = 1e-3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Bin photon arrivals into a regular intensity time trace.

        Parameters
        ----------
        bin_width_s : float
            Width of each time bin in seconds (default: 1 ms).

        Returns
        -------
        time_s : np.ndarray (float64)
            Left edge of each bin in seconds.
        I0 : np.ndarray (uint32)
            Photon counts per bin, channel 1.
        I2 : np.ndarray (uint32)
            Photon counts per bin, channel 2.
        """
        t0 = self.ch1_times_s
        t1 = self.ch2_times_s
        duration = max(t0[-1] if len(t0) else 0.0,
                       t1[-1] if len(t1) else 0.0)
        n_bins = int(np.ceil(duration / bin_width_s))
        edges  = np.arange(n_bins + 1) * bin_width_s
        I1, _ = np.histogram(t0, bins=edges)
        I2, _ = np.histogram(t1, bins=edges)
        return edges[:-1], I1.astype(np.uint32), I2.astype(np.uint32)

    def lifetime_histogram(
        self,
        channel: int = 0,
        n_bins: int = _MICROTIME_BINS,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a TCSPC lifetime decay histogram from the microtime data.

        Parameters
        ----------
        channel : int
            1 or 2.
        n_bins : int
            Number of histogram bins (default: 4096, one per microtime bin).

        Returns
        -------
        bin_times_ns : np.ndarray (float64)
            Left edge of each bin in nanoseconds (within one laser period).
        counts : np.ndarray (uint32)
            Photon counts per bin.
        """
        # NB: the existing mapping is preserved exactly — anything that is not
        # literally 1 selects Ch2, including the legacy ``channel=0`` default.
        ch = 1 if channel == 1 else 2
        if ch not in self.channels:
            raise ValueError(
                f"{self.filepath.name} recorded {self.channel_summary}; "
                f"there are no Ch{ch} microtimes to histogram."
            )
        micro = self.ch1_micro if channel == 1 else self.ch2_micro
        counts, edges = np.histogram(micro, bins=n_bins,
                                     range=(0, _MICROTIME_BINS))
        bin_times_ns = edges[:-1] * self.microtime_resolution_ns
        return bin_times_ns, counts.astype(np.uint32)

    def to_dataframe(self, bin_width_s: float = 1e-3):
        """
        Return a :class:`pandas.DataFrame` of binned intensity with columns
        ``time_s``, ``ch0``, ``ch1``.
        """
        if not _PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for to_dataframe().  "
                "Install with:  pip install pandas"
            )
        t, I1, I2 = self.bin_intensity(bin_width_s)
        return pd.DataFrame({"time_s": t, "ch1": I1, "ch2": I2})

    # ── Display ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        clk = self.params.get("clock_hz", float("nan"))
        lines = [
            f"FCSData — {self.filepath.name}",
            "─" * 52,
            "[ Measurement ]",
            f"  Timestamp              : {self.params.get('timestamp', 'unknown')}",
            f"  Duration               : {self.duration_s:.3f} s  ({self.duration_s/60:.3f} min)",
            f"  Clock frequency        : {clk/1e6:.6f} MHz",
            f"  Laser period           : {self.laser_period_ns:.4f} ns",
            f"  Microtime bin width    : {self.microtime_resolution_ns*1000:.4f} ps",
            f"  Channels recorded      : {self.channel_summary}",
            "",
            "[ Photon Statistics ]",
        ]
        # Only list channels the file actually recorded.  Printing "Ch2
        # photons: 0" for a single-channel file would be indistinguishable
        # from a dead detector, which is exactly the distinction this reader
        # now preserves.
        for ch in self.channels:
            lines.append(f"  Ch{ch} photons            : {len(self.deltas(ch)):,}")
        for ch in self.channels:
            lines.append(f"  Ch{ch} count rate         : {self.count_rate_hz(ch)/1e3:.2f} kHz")
        if self.params.get("_channel_note"):
            lines.append(f"  Note                   : {self.params['_channel_note']}")
        lines += [
            "",
            "[ Instrument ]",
        ]
        for key in ("objective_mag", "excitation_laser",
                    "excitation_dichroic", "emission_dichroic"):
            val = self.params.get(key)
            if val:
                lines.append(f"  {key.replace('_',' ').title():<26}: {val}")
        for ch_num in self.channels:
            ch_info = self.params.get(f"channel_ch{ch_num}", {})
            if ch_info:
                lines.append(f"  Ch{ch_num}                        :")
                for k, v in ch_info.items():
                    lines.append(f"    {k:<26}: {v}")
        return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def read_fcs(path: str | Path, clock_hz: Optional[float] = None) -> FCSData:
    """
    Parse a binary FCS data file and return an :class:`FCSData` object.

    Parameters
    ----------
    path : str or Path
        Path to the ``.fcs`` file.
    clock_hz : float, optional
        Override the laser clock frequency (Hz).  The binary header stores
        only a rounded nominal value (20,000,000 Hz), causing ~1% timing
        error.  Supply the precise value from the plain-text export's
        "Laser Clock (Hz)" header line for accurate results.
        Example: ``read_fcs("data.fcs", clock_hz=20_194_704.968582)``

    Returns
    -------
    FCSData

    Raises
    ------
    FileNotFoundError
    ValueError
        If the file is too small, or the data blocks match neither the
        dual-channel (4 block) nor the single-channel (2 block) layout.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    raw_bytes = path.read_bytes()
    if len(raw_bytes) < _HEADER_SIZE + 16:
        raise ValueError(f"File too small ({len(raw_bytes)} bytes).")

    # 1. Binary header
    header        = raw_bytes[:_HEADER_SIZE]
    nominal_clock = float(struct.unpack_from("<d", header, _HDR_CLOCK_OFF)[0])
    true_clock    = float(struct.unpack_from("<d", header, _HDR_TRUE_CLOCK_OFF)[0])

    # 2. ASCII metadata
    meta_offset = _find_metadata_offset(raw_bytes)
    meta_params = _parse_metadata(raw_bytes[meta_offset:].decode("utf-8", errors="replace"))

    # 3. Load all uint32 words from data region
    data_end  = meta_offset - (meta_offset % 4)
    all_words = np.frombuffer(raw_bytes[_DATA_START:data_end], dtype="<u4")

    # 4. Read the length-prefixed data blocks and work out the layout.
    #    Dual-channel:   [marker,pad, Ch0_macro, marker,pad, Ch1_macro,
    #                     marker,pad, Ch0_micro, marker,pad, Ch1_micro]
    #    Single-channel: [marker,pad, macro,     marker,pad, micro]
    n_channels, blocks = _classify_blocks(_scan_blocks(all_words))

    params: dict = {
        "clock_hz": clock_hz if clock_hz is not None else true_clock,
        **meta_params,
    }
    empty = np.empty(0, dtype=all_words.dtype)

    if n_channels == 2:
        ch0_macro, ch1_macro, ch0_micro, ch1_micro = blocks
        channels = (1, 2)
    else:
        macro, micro = blocks
        # Which detector this is comes from the metadata, not the layout:
        # a Ch2-only file has exactly the same two-block payload as a
        # Ch1-only one.  Put the data on the channel it belongs to and
        # leave the other one empty, so nothing downstream has to special-
        # case "the single channel" — it is always Ch1 or always Ch2.
        ch_id, note = _detect_single_channel_id(meta_params)
        if note:
            params["_channel_note"] = note
        channels = (ch_id,)
        if ch_id == 1:
            ch0_macro, ch0_micro = macro, micro
            ch1_macro, ch1_micro = empty, empty
        else:
            ch0_macro, ch0_micro = empty, empty
            ch1_macro, ch1_micro = macro, micro

    # 5. Assemble result
    return FCSData(
        filepath   = path,
        params     = params,
        ch1_deltas = ch0_macro,
        ch2_deltas = ch1_macro,
        ch1_micro  = ch0_micro,
        ch2_micro  = ch1_micro,
        channels   = channels,
    )

def load_dataset(path: str | Path):
    """
    Load any supported dataset file and return the matching container.
    Added 24 june 2026
    Dispatches by file type:
      * ISS time-domain lifetime decays (``.ifx``)  -> fcs_ifx.LifetimeData
        (an already-binned decay + IRF)
      * binary photon-record files (``.fcs``, default) -> FCSData

    This is the single entry point the GUI uses when adding a file to the
    workspace, so a new file type only has to be wired in here.
    """
    p = Path(path)
    # Lazy import keeps fcs_ifx optional and avoids any import-time coupling.
    try:
        import fcs_ifx
    except Exception:
        fcs_ifx = None
    if fcs_ifx is not None and fcs_ifx.is_ifx_file(p):
        return fcs_ifx.read_ifx(p)
    return read_fcs(p)

# ── Internal helpers ─────────────────────────────────────────────────────────

def _scan_blocks(words: np.ndarray, max_blocks: int = 4) -> list[np.ndarray]:
    """
    Read up to *max_blocks* length-prefixed data blocks from *words*.

    Block structure
    ---------------
    Each block is preceded by a 2-word preamble::

        [ length_marker ][ 0 pad ][ block data ... ]

    where ``length_marker`` is the block's size **in bytes** — i.e.
    ``4 * (number of photons in the block)`` — and the pad word is 0.
    Reading the marker gives the block length deterministically, so the
    parser never has to guess boundaries from value magnitudes.  This is
    essential: at low count rates ordinary inter-photon macrotime
    differences routinely exceed any fixed magnitude threshold, so the
    old "scan for words > threshold" approach mis-split the blocks and
    desynchronised the two channels (a spurious Ch1/Ch2 offset that
    corrupted cross-correlation).  The length prefix is count-rate
    independent and recovers the photon counts exactly, including the
    normal case where the two channels differ.

    A marker of 0 is a legitimate EMPTY block: a dual-channel acquisition
    in which one detector recorded nothing still writes all four preambles,
    just with zero payload.  That file is read as an ordinary two-channel
    dataset with an empty channel, so the user sees the missing photons and
    judges for themselves; it is not silently downgraded to single-channel.

    This function does not decide how many channels the file has — it only
    reports how many well-formed blocks it could read before the data ran
    out or stopped looking like a preamble.  :func:`_classify_blocks` makes
    the layout decision.  Scanning is therefore tolerant: it STOPS rather
    than raising, because a stop is the normal way a two-block
    (single-channel) file ends.
    """
    blocks: list[np.ndarray] = []
    pos = 0
    n   = len(words)

    while len(blocks) < max_blocks:
        if pos + 2 > n:
            break                      # no room for another preamble
        marker = int(words[pos])
        pad    = int(words[pos + 1])
        if marker % 4 != 0 or pad != 0:
            break                      # not a preamble — end of the data region
        count = marker // 4
        start = pos + 2
        end   = start + count
        if end > n:
            break                      # declared length overruns; not a real block
        blocks.append(words[start:end].copy())
        pos = end

    return blocks


def _looks_like_microtimes(block: np.ndarray) -> bool:
    """
    True when *block* could be a microtime block.

    Microtimes are TCSPC bin indices bounded by ``_MICROTIME_BINS`` (4096),
    i.e. they fit in 12 bits.  Macrotime deltas are not so bounded — in a
    typical measurement they run to tens of thousands of laser cycles — so
    this cheaply distinguishes the two-block single-channel layout
    (macro, micro) from a truncated read of the four-block layout
    (macro Ch0, macro Ch1, ...).  An empty block carries no evidence either
    way and is accepted.
    """
    return block.size == 0 or int(block.max()) < _MICROTIME_BINS


def _classify_blocks(blocks: list[np.ndarray]) -> Tuple[int, list[np.ndarray]]:
    """
    Decide whether *blocks* describe a two-channel or a single-channel file.

    Returns ``(n_channels, blocks)`` with the block list trimmed to the ones
    that belong to the detected layout.

    Two-channel layout (4 blocks + trailing footer)::

        [4*n0][0]  Ch0 macrotime diffs  (n0 words)
        [4*n1][0]  Ch1 macrotime diffs  (n1 words)
        [4*n0][0]  Ch0 microtimes       (n0 words)
        [4*n1][0]  Ch1 microtimes       (n1 words)
        [footer / metadata tag ...]

    Single-channel layout (2 blocks + trailing footer)::

        [4*n][0]   macrotime diffs      (n words)
        [4*n][0]   microtimes           (n words)
        [footer / metadata tag ...]

    The four-block reading is tried FIRST, so a genuine dual-channel file can
    never be mistaken for a single-channel one.  It is accepted only when the
    macro and micro counts agree per channel; that per-channel consistency
    check is what rejects a spurious four-block read of a single-channel file
    whose footer happened to parse as a preamble.  (This is not hypothetical:
    the footer of a real single-channel file starts with the word 3072, which
    is a positive multiple of 4 followed by a zero — a valid-looking preamble
    that only the length and consistency checks reject.)
    """
    lens = [len(b) for b in blocks]

    # ── Two channels: [macro0, macro1, micro0, micro1] ───────────────────────
    if len(blocks) >= 4 and lens[0] == lens[2] and lens[1] == lens[3]:
        return 2, blocks[:4]

    # ── One channel: [macro, micro] ──────────────────────────────────────────
    if len(blocks) >= 2 and lens[0] == lens[1] and _looks_like_microtimes(blocks[1]):
        return 1, blocks[:2]

    # ── Neither ──────────────────────────────────────────────────────────────
    if len(blocks) < 2:
        raise ValueError(
            f"Found only {len(blocks)} length-prefixed data block(s); a file "
            f"needs 2 (single-channel) or 4 (dual-channel).  The file may be "
            f"truncated, or not in the expected length-prefixed format."
        )
    raise ValueError(
        f"Found {len(blocks)} data blocks with photon counts {lens}, which "
        f"match neither the dual-channel layout (4 blocks, counts "
        f"[n0, n1, n0, n1]) nor the single-channel layout (2 blocks, counts "
        f"[n, n] with microtimes below {_MICROTIME_BINS}).  File may be "
        f"corrupt."
    )


def _detect_single_channel_id(meta_params: dict) -> Tuple[int, Optional[str]]:
    """
    Work out WHICH detector a single-channel file used.

    The binary payload of a Ch2-only acquisition is byte-for-byte the same
    shape as a Ch1-only one — two blocks, macro then micro — so the layout
    alone cannot say which detector produced it.  The trailing ASCII metadata
    can: ``[Detection Channels]`` lists only the channels that were actually
    enabled, which :func:`_parse_metadata` turns into ``channel_ch1`` /
    ``channel_ch2`` keys.

    Returns ``(channel_id, note)`` where *note* is None when the metadata was
    decisive and a short explanation otherwise.  The fallback is Ch1, but it
    is a REPORTED fallback: the note is stored in ``params['_channel_note']``
    and printed in the summary, rather than being assumed silently.
    """
    has1 = "channel_ch1" in meta_params
    has2 = "channel_ch2" in meta_params

    if has1 and not has2:
        return 1, None
    if has2 and not has1:
        return 2, None
    if has1 and has2:
        return 1, ("File contains one data channel but its metadata lists both "
                   "Ch1 and Ch2; assuming the data is Ch1.")
    return 1, ("File contains one data channel and its metadata names no "
               "detection channel; assuming the data is Ch1.")


# ── Feature gating ────────────────────────────────────────────────────────────

def require_channels(data, channels, feature: str) -> None:
    """
    Raise a descriptive ValueError unless *data* recorded every channel in
    *channels*.

    This is the single correctness gate for channel-dependent features.  The
    GUI should ALSO disable the relevant controls for a file that lacks a
    channel, so in normal use this never fires; it is here so that a code
    path which slips past the greying-out fails with an explanation instead
    of an IndexError, a division by zero, or — worst of all — a correlation
    curve computed against an empty array.

    Parameters
    ----------
    data : FCSData
        The dataset to check.  Objects without a ``channels`` attribute are
        treated as dual-channel, so this is safe to call on anything the
        workspace holds.
    channels : iterable of int
        The channel numbers the feature needs, e.g. ``(1, 2)`` for
        cross-correlation or ``(2,)`` for a Ch2 autocorrelation.
    feature : str
        Human-readable feature name, used in the message.
    """
    available = tuple(getattr(data, "channels", (1, 2)))
    missing   = [c for c in channels if c not in available]
    if not missing:
        return
    name    = getattr(getattr(data, "filepath", None), "name", "This dataset")
    have    = " + ".join(f"Ch{c}" for c in available) or "no channels"
    need    = " and ".join(f"Ch{c}" for c in missing)
    raise ValueError(
        f"{feature} needs {need}, but {name} recorded {have}.\n\n"
        f"This is a single-channel acquisition: the missing channel was never "
        f"recorded, so there are no photons to correlate, plot or histogram "
        f"for it.  Choose an analysis that uses {have}."
    )


def require_two_channels(data, feature: str) -> None:
    """Convenience wrapper: raise unless *data* recorded both Ch1 and Ch2."""
    require_channels(data, (1, 2), feature)


def _find_metadata_offset(raw: bytes) -> int:
    window = raw[-4096:]
    for tag in (b"[Excitation", b"[Experiment", b"[Detection", b"[Microscope"):
        idx = window.find(tag)
        if idx != -1:
            return len(raw) - 4096 + idx
    return len(raw)


def _parse_metadata(meta_text: str) -> dict:
    params: dict = {}
    current_section: Optional[str] = None
    current_channel: Optional[str] = None
    channel_data: dict = {}

    def flush_channel():
        nonlocal current_channel, channel_data
        if current_channel and channel_data:
            params[f"channel_{current_channel.lower()}"] = dict(channel_data)
        current_channel, channel_data = None, {}

    for line in meta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\[([^\]]+)\](?:\s*=\s*(.+))?$", line)
        if m:
            name, inline_val = m.group(1).strip(), m.group(2)
            flush_channel()
            if re.match(r"^Ch\d+$", name):
                current_channel = name
            else:
                current_section = name
            if inline_val:
                params[_norm(name)] = inline_val.strip()
            continue
        m = re.match(r"^(.+?)\s*[-–]?\s*:\s*(.+)$", line)
        if m:
            value = m.group(2).strip()
            key   = _norm(m.group(1).strip())
            if current_channel:
                channel_data[key] = value
            elif current_section == "Excitation Laser":
                params["excitation_laser"] = value
            elif current_section == "Excitation Dichroic":
                params["excitation_dichroic"] = value
            elif current_section == "Emission Dichroic":
                params["emission_dichroic"] = value
            else:
                params[key] = value

    flush_channel()

    for old, new in {
        "experiment_time_stamp"             : "timestamp",
        "microscope_objective_magnification": "objective_mag",
    }.items():
        if old in params:
            params[new] = params.pop(old)

    return params


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fcs_reader.py <file.fcs> [clock_hz]")
        sys.exit(1)
    clock = float(sys.argv[2]) if len(sys.argv) > 2 else None
    d = read_fcs(sys.argv[1], clock_hz=clock)
    print(d)
    if "_clock_note" in d.params:
        print("\nNOTE:", d.params["_clock_note"])
    if "_channel_note" in d.params:
        print("\nNOTE:", d.params["_channel_note"])
    print()
    print(f"channels   : {d.channels}  ({d.channel_summary})")
    for _ch in d.channels:
        print(f"ch{_ch}_deltas : shape={d.deltas(_ch).shape}, dtype={d.deltas(_ch).dtype}")
        print(f"ch{_ch}_micro  : shape={d.micro(_ch).shape},  dtype={d.micro(_ch).dtype}")
    print()
    _ch = d.channels[0]
    print(f"Microtime bin width      : {d.microtime_resolution_ns*1000:.4f} ps")
    print(f"First 5 Ch{_ch} times (s)  : {d.times_s(_ch)[:5].tolist()}")
    print(f"First 5 Ch{_ch} micro (ns) : "
          f"{(d.micro(_ch)[:5] * d.microtime_resolution_ns).tolist()}")
