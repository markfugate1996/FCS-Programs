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

Binary layout
-------------
Offset              Content
------------------  ----------------------------------------------------------
0x000–0x3FF         Binary header (1024 bytes)
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
    """
    filepath   : Path
    params     : dict
    ch1_deltas : np.ndarray
    ch2_deltas : np.ndarray
    ch1_micro  : np.ndarray
    ch2_micro  : np.ndarray

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
            "",
            "[ Photon Statistics ]",
            f"  Ch1 photons            : {len(self.ch1_deltas):,}",
            f"  Ch2 photons            : {len(self.ch2_deltas):,}",
            f"  Ch1 count rate         : {self.count_rate_ch1_hz/1e3:.2f} kHz",
            f"  Ch2 count rate         : {self.count_rate_ch2_hz/1e3:.2f} kHz",
            "",
            "[ Instrument ]",
        ]
        for key in ("objective_mag", "excitation_laser",
                    "excitation_dichroic", "emission_dichroic"):
            val = self.params.get(key)
            if val:
                lines.append(f"  {key.replace('_',' ').title():<26}: {val}")
        for ch_num in (1, 2):
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
        If the file is too small or four data blocks cannot be found.
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

    # 4. Find all four data blocks via sentinel markers
    #    Layout: [marker,pad, Ch0_macro, marker,pad, Ch1_macro,
    #             marker,pad, Ch0_micro, marker,pad, Ch1_micro]
    blocks = _extract_four_blocks(all_words)
    if len(blocks) != 4:
        raise ValueError(
            f"Expected 4 data blocks, found {len(blocks)}.  "
            f"File may be corrupt or in an unexpected format."
        )
    ch0_macro, ch1_macro, ch0_micro, ch1_micro = blocks

    # 5. Assemble result
    params: dict = {
        "clock_hz": clock_hz if clock_hz is not None else true_clock,
        **meta_params,
    }
    return FCSData(
        filepath   = path,
        params     = params,
        ch1_deltas = ch0_macro,
        ch2_deltas = ch1_macro,
        ch1_micro  = ch0_micro,
        ch2_micro  = ch1_micro,
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

def _extract_four_blocks(words: np.ndarray) -> list[np.ndarray]:
    """
    Extract the four data blocks (Ch0 macro, Ch1 macro, Ch0 micro, Ch1 micro)
    using the explicit length prefix that precedes every block.

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
    independent and recovers n0 and n1 exactly, including the normal case
    where n0 != n1.

    Layout (4 blocks + trailing footer)::

        [4*n0][0]  Ch0 macrotime diffs  (n0 words)
        [4*n1][0]  Ch1 macrotime diffs  (n1 words)
        [4*n0][0]  Ch0 microtimes       (n0 words)
        [4*n1][0]  Ch1 microtimes       (n1 words)
        [footer / metadata tag ...]

    Returns the four blocks as copies.  Raises ValueError if the markers
    are internally inconsistent (e.g. the two macrotime counts do not
    match the two microtime counts), which would indicate a corrupt file
    or a layout this reader does not understand.
    """
    blocks: list[np.ndarray] = []
    pos = 0
    n = len(words)

    for i in range(4):
        if pos + 2 > n:
            raise ValueError(
                f"Truncated file: ran out of data locating block {i} "
                f"(offset word {pos} of {n})."
            )
        marker = int(words[pos])
        pad    = int(words[pos + 1])
        if marker == 0 or marker % 4 != 0:
            raise ValueError(
                f"Block {i}: length marker {marker} at word {pos} is not a "
                f"positive multiple of 4; file is not in the expected "
                f"length-prefixed format."
            )
        if pad != 0:
            # Non-fatal in principle, but a non-zero pad means our
            # understanding of the preamble is off — fail loudly rather
            # than silently misalign the channels.
            raise ValueError(
                f"Block {i}: expected 0 pad word after length marker, "
                f"got {pad} at word {pos + 1}."
            )
        count = marker // 4
        start = pos + 2
        end   = start + count
        if end > n:
            raise ValueError(
                f"Block {i}: declared length {count} words overruns the "
                f"data region (need up to word {end}, have {n})."
            )
        blocks.append(words[start:end].copy())
        pos = end

    # Integrity check: macro/micro counts must agree per channel.
    n0_macro, n1_macro, n0_micro, n1_micro = (len(b) for b in blocks)
    if n0_macro != n0_micro or n1_macro != n1_micro:
        raise ValueError(
            f"Inconsistent block lengths: Ch0 macro/micro = "
            f"{n0_macro}/{n0_micro}, Ch1 macro/micro = {n1_macro}/{n1_micro}. "
            f"File may be corrupt."
        )

    return blocks


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
    print()
    print(f"ch1_deltas : shape={d.ch1_deltas.shape}, dtype={d.ch1_deltas.dtype}")
    print(f"ch2_deltas : shape={d.ch2_deltas.shape}, dtype={d.ch2_deltas.dtype}")
    print(f"ch1_micro  : shape={d.ch1_micro.shape},  dtype={d.ch1_micro.dtype}")
    print(f"ch2_micro  : shape={d.ch2_micro.shape},  dtype={d.ch2_micro.dtype}")
    print()
    print(f"Microtime bin width   : {d.microtime_resolution_ns*1000:.4f} ps")
    print(f"First 5 Ch1 times (s) : {d.ch1_times_s[:5].tolist()}")
    print(f"First 5 Ch1 micro (ns): {d.ch1_micro_ns[:5].tolist()}")
