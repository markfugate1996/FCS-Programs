"""
fcs_ifx.py
==========
Reader for ISS time-domain lifetime decay files (``.ifx``) as exported by
ISS Vinci (VistaVision's fluorimeter companion) software.

How this differs from .fcs
--------------------------
The binary ``.fcs`` files read by :mod:`fcs_reader` store *individual photon
records* (macrotime + microtime) from the confocal FCS instrument.  An
``.ifx`` file is something else entirely: it is an **already-binned TCSPC
lifetime decay** measured on the fluorimeter.  There are no photon-level
records, no second channel and no macrotime axis, so correlation / PCH /
diffusion analyses do not apply.  What it *does* carry is exactly what the
lifetime tasks need:

  * a time axis (ns, one laser period wide),
  * the fluorescence decay (counts per time bin), and
  * the instrument response function (IRF) measured on the same axis.

The IRF makes proper iterative-reconvolution lifetime fitting possible
(see :mod:`fcs_lifetime_fit`).

File format
-----------
Plain ASCII (CRLF), two parts::

    Key=Value           header lines (and blank separators)
    ...
    [Data]
    <time_ns>  <intensity>  <irf>     whitespace-delimited, one row per bin

The columns present are named on the ``Columns=`` header line; this reader
locates the time / intensity / IRF columns by that line so it is robust to
column-order changes.  Representative header fields::

    Title=AttoBiotin
    Timestamp=Mon Jun 22 15:20:09 2026
    Product=Vinci 3
    AcquisitionType=Time Domain
    PulseRate=20194600                      laser rep-rate (Hz)
    TimeDomainTime=...,unit:ns,timeRange:49.49,timeBins:1010
    Columns=TimeDomainTime,Intensity,IRF

Usage
-----
    from fcs_ifx import read_ifx

    d = read_ifx("20260622_-_attobiotin.ifx")
    print(d)                       # formatted summary

    t, I, irf = d.decay_curve()    # native-resolution arrays (ns, counts, counts)
    t, I, irf = d.decay_curve(rebin=2)   # 2x coarser

    d.laser_period_ns              # 1 / PulseRate, in ns
    d.has_irf                      # True when a usable IRF column is present
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# The string other modules check to recognise a lifetime-decay dataset.
# (kept here so there is a single source of truth)
LIFETIME_KIND = "lifetime_decay"


# ── Result container ─────────────────────────────────────────────────────────

@dataclass
class LifetimeData:
    """
    Container returned by :func:`read_ifx`.

    Deliberately mirrors the small slice of the :class:`fcs_reader.FCSData`
    surface that the lifetime tasks use (``filepath``, ``params``,
    ``laser_period_ns``, ``lifetime_histogram``) so it can flow through the
    existing "Plot Lifetime Decay" path, while adding the decay/IRF arrays the
    lifetime *fitting* needs.

    Attributes
    ----------
    filepath : Path
    params : dict
        Parsed header fields plus derived values (``pulse_rate_hz``,
        ``clock_hz``, ``time_range_ns``, ``time_bins``, ``bin_width_ns``,
        ``title``, ``timestamp`` …).
    time_ns : np.ndarray (float64)
        Sample time of each bin within the laser period (ns).
    intensity : np.ndarray (float64)
        Fluorescence decay, counts per bin.
    irf : np.ndarray (float64)
        Instrument response function, counts per bin (zeros if not recorded).
    """
    filepath:  Path
    params:    dict
    time_ns:   np.ndarray
    intensity: np.ndarray
    irf:       np.ndarray
    kind:      str = field(default=LIFETIME_KIND)

    # ── Basic timing / shape ─────────────────────────────────────────────────

    @property
    def n_bins(self) -> int:
        return int(len(self.time_ns))

    @property
    def bin_width_ns(self) -> float:
        """Median spacing of the time axis (ns)."""
        if len(self.time_ns) < 2:
            return float("nan")
        return float(np.median(np.diff(self.time_ns)))

    @property
    def pulse_rate_hz(self) -> float:
        return float(self.params.get("pulse_rate_hz", float("nan")))

    @property
    def laser_period_ns(self) -> float:
        """One laser cycle in nanoseconds (= 1e9 / PulseRate)."""
        pr = self.pulse_rate_hz
        if pr and np.isfinite(pr):
            return 1e9 / pr
        # Fall back to the spanned time range if the rep-rate is missing.
        return float(self.time_ns[-1] - self.time_ns[0] + self.bin_width_ns)

    @property
    def has_irf(self) -> bool:
        """True when an IRF column is present and not flat/empty."""
        if self.irf is None or len(self.irf) == 0:
            return False
        finite = np.isfinite(self.irf)
        if not finite.any():
            return False
        vals = self.irf[finite]
        return bool(vals.max() > 0 and np.ptp(vals) > 0)

    # ── Curve accessors ──────────────────────────────────────────────────────

    def decay_curve(self, rebin: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(time_ns, intensity, irf)`` at native resolution, or coarsened
        by an integer ``rebin`` factor (block-summing counts, averaging time).

        ``rebin`` need not divide the bin count evenly; any trailing bins that
        do not fill a full block are dropped.
        """
        rebin = max(1, int(rebin))
        t, I, r = self.time_ns, self.intensity, self.irf
        if rebin == 1:
            return t.copy(), I.copy(), r.copy()

        n = (len(t) // rebin) * rebin
        if n == 0:
            return t.copy(), I.copy(), r.copy()
        t = t[:n].reshape(-1, rebin).mean(axis=1)
        I = I[:n].reshape(-1, rebin).sum(axis=1)
        r = r[:n].reshape(-1, rebin).sum(axis=1)
        return t, I, r

    def lifetime_histogram(
        self,
        channel: int = 1,
        n_bins: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compatibility shim mirroring :meth:`fcs_reader.FCSData.lifetime_histogram`.

        For lifetime-decay data there is a single channel (the recorded
        decay).  ``channel`` is accepted for interface compatibility:

          * channel 1  -> the fluorescence decay (default)
          * channel 2  -> the IRF, if present (else an empty curve)

        ``n_bins`` may request integer-factor coarsening relative to the native
        bin count; if ``None`` the native resolution is returned.  Values that
        do not correspond to a clean down-sampling are snapped to the nearest
        integer rebin factor.
        """
        rebin = 1
        if n_bins:
            rebin = max(1, round(self.n_bins / int(n_bins)))
        t, I, r = self.decay_curve(rebin=rebin)
        if channel == 2:
            return t, (r if self.has_irf else np.zeros_like(r))
        return t, I

    # ── Display ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        I = self.intensity
        peak_i = int(np.argmax(I)) if len(I) else 0
        lines = [
            f"LifetimeData — {self.filepath.name}",
            "─" * 52,
            "[ Measurement ]",
            f"  Title                  : {self.params.get('title', 'unknown')}",
            f"  Timestamp              : {self.params.get('timestamp', 'unknown')}",
            f"  Instrument             : {self.params.get('product', 'unknown')}",
            f"  Acquisition            : {self.params.get('acquisitiontype', 'Time Domain')}",
            "",
            "[ Time axis ]",
            f"  Pulse rate             : {self.pulse_rate_hz/1e6:.6f} MHz",
            f"  Laser period           : {self.laser_period_ns:.4f} ns",
            f"  Bins                   : {self.n_bins}",
            f"  Bin width              : {self.bin_width_ns*1000:.2f} ps",
            f"  Span                   : {self.time_ns[0]:.3f} – {self.time_ns[-1]:.3f} ns",
            "",
            "[ Decay ]",
            f"  Total counts           : {int(np.nansum(I)):,}",
            f"  Peak counts            : {int(np.nanmax(I)):,} at {self.time_ns[peak_i]:.3f} ns",
            f"  IRF recorded           : {'yes' if self.has_irf else 'no'}",
        ]
        if self.has_irf:
            r = self.irf
            rp = int(np.argmax(r))
            lines.append(f"  IRF peak               : {int(np.nanmax(r)):,} at {self.time_ns[rp]:.3f} ns")
        return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def read_ifx(path: str | Path) -> LifetimeData:
    """
    Parse an ISS ``.ifx`` time-domain lifetime file and return a
    :class:`LifetimeData`.

    Raises
    ------
    FileNotFoundError
    ValueError
        If no ``[Data]`` block or recognisable time/intensity columns are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")

    header, data_lines = _split_header_data(text)
    params = _parse_header(header)

    col_names = _column_names(params, header)
    time_arr, intensity_arr, irf_arr = _parse_data_block(data_lines, col_names)

    # Derived timing parameters
    pulse_rate = _to_float(params.get("pulserate"))
    params["pulse_rate_hz"] = pulse_rate if pulse_rate is not None else float("nan")
    # Provide clock_hz too so downstream code that speaks the FCS vocabulary
    # ("laser clock") finds a value.
    params["clock_hz"] = params["pulse_rate_hz"]

    trange, tbins = _time_domain_spec(params)
    if trange is not None:
        params["time_range_ns"] = trange
    if tbins is not None:
        params["time_bins"] = tbins
    if len(time_arr) > 1:
        params["bin_width_ns"] = float(np.median(np.diff(time_arr)))

    # Friendly aliases
    if "title" not in params and "title" in {k.lower() for k in params}:
        pass  # already lower-cased by _parse_header

    return LifetimeData(
        filepath  = path,
        params    = params,
        time_ns   = time_arr,
        intensity = intensity_arr,
        irf       = irf_arr,
    )


def is_ifx_file(path: str | Path) -> bool:
    """
    Cheap sniff test: ``.ifx`` extension, or an ISS_Experiment signature /
    ``[Data]`` block near the top of the file.  Used by the loader dispatch.
    """
    path = Path(path)
    if path.suffix.lower() == ".ifx":
        return True
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:2048]
    except Exception:
        return False
    return ("ISS_Experiment" in head) or ("AcquisitionType=Time Domain" in head)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _split_header_data(text: str) -> Tuple[list[str], list[str]]:
    """Split the file at the ``[Data]`` marker into header and data lines."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "[data]":
            return lines[:i], lines[i + 1:]
    raise ValueError("No '[Data]' section found — is this an ISS .ifx file?")


def _parse_header(header_lines: list[str]) -> dict:
    """
    Turn ``Key=Value`` header lines into a dict with lower-cased keys.

    A handful of frequently used fields are also exposed under tidy aliases
    (``title``, ``timestamp``, ``product``) — these already match the
    lower-cased keys, listed here for clarity.
    """
    params: dict = {}
    for line in header_lines:
        s = line.strip()
        if not s or "=" not in s:
            continue
        key, val = s.split("=", 1)
        params[key.strip().lower()] = val.strip()
    return params


def _column_names(params: dict, header_lines: list[str]) -> list[str]:
    """
    Return the ordered column names from the ``Columns=`` header line, e.g.
    ``["TimeDomainTime", "Intensity", "IRF"]``.  Falls back to a sensible
    default if the line is absent.
    """
    cols = params.get("columns")
    if cols:
        return [c.strip() for c in cols.split(",") if c.strip()]
    return ["TimeDomainTime", "Intensity", "IRF"]


def _parse_data_block(
    data_lines: list[str],
    col_names: list[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse the whitespace-delimited numeric rows after ``[Data]`` and pull out
    the time, intensity and IRF columns by matching ``col_names``.
    """
    rows: list[list[float]] = []
    for line in data_lines:
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            # Stop at the first non-numeric line (e.g. a trailing footer).
            break

    if not rows:
        raise ValueError("The [Data] block contained no numeric rows.")

    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=np.float64)

    # Map column names (case-insensitive) to indices.
    lower = [c.lower() for c in col_names]

    def _find(*aliases: str) -> Optional[int]:
        for a in aliases:
            if a in lower:
                idx = lower.index(a)
                if idx < width:
                    return idx
        return None

    t_idx = _find("timedomaintime", "time", "time_ns", "t")
    i_idx = _find("intensity", "counts", "decay", "signal")
    r_idx = _find("irf", "prompt", "instrumentresponse")

    # Reasonable positional fallbacks if names were unexpected.
    if t_idx is None:
        t_idx = 0
    if i_idx is None:
        i_idx = 1 if width > 1 else 0

    time_arr = arr[:, t_idx]
    intensity_arr = arr[:, i_idx]
    irf_arr = arr[:, r_idx] if (r_idx is not None) else np.zeros_like(intensity_arr)

    return time_arr, intensity_arr, irf_arr


def _time_domain_spec(params: dict) -> Tuple[Optional[float], Optional[int]]:
    """
    Parse ``timeRange`` (ns) and ``timeBins`` out of the ``TimeDomainTime=``
    descriptor, e.g. ``type:numeric,unit:ns,timeRange:49.49,timeBins:1010``.
    """
    spec = params.get("timedomaintime", "")
    trange = tbins = None
    m = re.search(r"timeRange\s*:\s*([0-9.eE+-]+)", spec)
    if m:
        trange = _to_float(m.group(1))
    m = re.search(r"timeBins\s*:\s*([0-9]+)", spec)
    if m:
        try:
            tbins = int(m.group(1))
        except ValueError:
            tbins = None
    return trange, tbins


def _to_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    try:
        return float(str(text).strip())
    except (TypeError, ValueError):
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fcs_ifx.py <file.ifx>")
        sys.exit(1)
    d = read_ifx(sys.argv[1])
    print(d)
    print()
    t, I, r = d.decay_curve()
    print(f"time_ns   : shape={t.shape}, {t[0]:.3f} … {t[-1]:.3f} ns")
    print(f"intensity : shape={I.shape}, peak {I.max():.0f}")
    print(f"irf       : shape={r.shape}, peak {r.max():.0f}, usable={d.has_irf}")
