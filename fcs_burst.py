"""
fcs_burst.py
============
Burst detection and excision for fluorescence fluctuation data.

The target is not the single-molecule burst of classical BIFL work but the
RARE BRIGHT CONTAMINANT in an otherwise homogeneous sample -- a dye aggregate
among monomers, a dust particle, a vesicle.  Such a species is far too scarce
to fit as a second component yet bright enough to dominate the long-lag end of
G(tau), so the practical question is whether it is present and what the
correlation looks like without it.

Detection
---------
A sliding window of M consecutive photons is swept along each channel's
arrival-time record.  For the window starting at photon i,

    T_i = t[i + M - 1] - t[i]          (span of the window, seconds)
    r_i = M / T_i                      (instantaneous rate, Hz)

and the window is flagged when ``r_i >= F * <r>``, with ``<r>`` the channel's
mean count rate.  Every photon in a flagged window is marked, and overlapping
flagged windows merge into one burst.

This is an interphoton-time criterion rather than a fixed-bin threshold, which
matters: a burst straddling a bin boundary is split by binning and can fall
below threshold on both sides, while a sliding window has no grid to straddle.

Two properties of ``r_i = M / T_i`` are worth knowing before the numbers are
interpreted (see RATE ESTIMATOR below): it is a biased estimator of the
underlying rate, and ``<r>`` is itself contaminated by the bursts being
searched for.

Coincidence
-----------
Detection runs independently on Ch1 and Ch2, and only bursts whose intervals
OVERLAP IN TIME between the two channels are kept.  A real particle crossing
the focus is seen by both detectors; detector afterpulsing, an electrical
spike, and a cosmic-ray hit are seen by one.  Requiring coincidence therefore
rejects instrument artifacts at the cost of missing any genuine burst too dim
to trigger the weaker channel -- detection sensitivity is set by whichever
channel is worse.

Correlation
-----------
Excising bursts cuts the record into burst-free REGIONS.  Each region is then
segmented by the suite's ordinary fixed-duration rule
(:func:`fcs_corr.segment_bounds`), every segment is correlated on its own, and
the segments from all regions are pooled into one segment-and-average.  A 10 s
region at a 1 s segment length contributes ten segments, exactly as an
unbroken 10 s record would.

The photon stream is never stitched back together across a cut.  Concatenation
would place photons from either side of an excised burst at an artificially
short lag and manufacture correlation that was never measured.  Segment
boundaries are never allowed to straddle a cut either, so no segment contains
the discontinuity.

A region shorter than one segment is dropped whole; within a region the
remainder goes into the final segment, which therefore runs between 1x and 2x
the nominal length.  Both behaviours are inherited from segment_bounds rather
than reimplemented, so segmentation here matches the rest of the suite --
including, deliberately, the over-long terminal segment that already occurs at
the end of any photon stream.

Public API
----------
    false_alarm_probability(M, F)                   -> float
    expected_false_windows(n_photons, M, F)         -> float
    detect_bursts(times_s, duration_s, M, F, ...)   -> ChannelBursts
    coincident_intervals(iv_a, iv_b)                -> np.ndarray (n, 2)
    complement_intervals(iv, t_start, t_end)        -> np.ndarray (n, 2)
    photons_in_intervals(times_s, iv)               -> np.ndarray (bool)
    analyse(fcs_data, M, F, ...)                    -> BurstResult
    plot_intensity_bursts(fcs_data, result, ...)    -> (fig, ax)
    plot_burst_lifetimes(fcs_data, result, ...)     -> (fig, ax)
    correlate_clean(fcs_data, result, params)       -> dict
    retention_summary(corr)                         -> str
    required_segment_s(tau_max_s, requested_s)      -> float
    plot_burst_correlation(fcs_data, result, corr)  -> (fig, ax)
    register(plotopen=None)                         -> None
    rebuild_plot(meta, columns, show, path)         -> (fig, ax)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_M = 5          # photons per sliding window
_DEFAULT_F = 5.0        # rate multiplier: flag when r >= F * mean rate

# Fallback segment length, used only if fcs_corr does not expose its own
# default.  The segment length is really set by the longest lag wanted:
# a segment of duration D only supports lags well below D, which is what
# fcs_corr._MIN_SEGMENT_FACTOR encodes.  Callers should pass seg_duration_s
# explicitly rather than lean on this.
_FALLBACK_SEG_DURATION_S = 1.0


# ── Containers ────────────────────────────────────────────────────────────────

@dataclass
class ChannelBursts:
    """Detection result for ONE channel."""
    channel:        int
    n_photons:      int
    mean_rate_hz:   float
    threshold_hz:   float
    in_burst:       np.ndarray        # bool, one per photon
    intervals:      np.ndarray        # (n, 2) float, [start_s, end_s]

    @property
    def n_bursts(self) -> int:
        return int(self.intervals.shape[0])

    @property
    def n_burst_photons(self) -> int:
        return int(self.in_burst.sum())

    @property
    def burst_photon_fraction(self) -> float:
        return (self.n_burst_photons / self.n_photons) if self.n_photons else 0.0

    @property
    def burst_time_s(self) -> float:
        if self.intervals.size == 0:
            return 0.0
        return float((self.intervals[:, 1] - self.intervals[:, 0]).sum())


@dataclass
class BurstResult:
    """Detection, coincidence and excision over both channels."""
    M:              int
    F:              float
    duration_s:     float
    ch1:            Optional[ChannelBursts]
    ch2:            Optional[ChannelBursts]
    coincident:     np.ndarray        # (n, 2) burst intervals kept
    clean:          np.ndarray        # (n, 2) complement of `coincident`
    ch1_excised:    np.ndarray        # bool per Ch1 photon: inside a kept burst
    ch2_excised:    np.ndarray        # bool per Ch2 photon
    notes:          List[str] = field(default_factory=list)

    @property
    def n_coincident(self) -> int:
        return int(self.coincident.shape[0])

    @property
    def burst_time_s(self) -> float:
        if self.coincident.size == 0:
            return 0.0
        return float((self.coincident[:, 1] - self.coincident[:, 0]).sum())

    @property
    def clean_time_s(self) -> float:
        if self.clean.size == 0:
            return 0.0
        return float((self.clean[:, 1] - self.clean[:, 0]).sum())

    def summary(self) -> str:
        """Multi-line report suitable for the status bar or a console."""
        lines = [
            f"Burst detection  M = {self.M} photons,  F = {self.F:g} x mean rate",
            f"{'-' * 56}",
        ]
        for cb in (self.ch1, self.ch2):
            if cb is None:
                continue
            lines.append(
                f"  Ch{cb.channel}:  {cb.n_bursts:,} bursts   "
                f"threshold {cb.threshold_hz:,.0f} cps   "
                f"({cb.mean_rate_hz:,.0f} cps mean)"
            )
            lines.append(
                f"        {cb.n_burst_photons:,} / {cb.n_photons:,} photons "
                f"({100 * cb.burst_photon_fraction:.3f} %)"
            )
        lines.append(
            f"  Coincident (Ch1 AND Ch2):  {self.n_coincident:,} bursts"
        )
        if self.duration_s:
            lines.append(
                f"  Excised {self.burst_time_s * 1e3:.2f} ms of "
                f"{self.duration_s:.2f} s "
                f"({100 * self.burst_time_s / self.duration_s:.4f} %)  "
                f"in {len(self.clean):,} clean intervals"
            )
        n1 = int(self.ch1_excised.sum()) if self.ch1_excised.size else 0
        n2 = int(self.ch2_excised.sum()) if self.ch2_excised.size else 0
        lines.append(f"  Photons removed:  Ch1 {n1:,}   Ch2 {n2:,}")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


# ── Detection ─────────────────────────────────────────────────────────────────

def sliding_rates(times_s: np.ndarray, M: int) -> np.ndarray:
    """
    Instantaneous rate r_i = M / (t[i+M-1] - t[i]) for every window start i.

    Returns an array of length ``len(times_s) - M + 1`` (empty if the record
    is shorter than M photons).  A window whose M photons carry identical
    timestamps -- possible when the macrotime clock is coarse relative to the
    burst -- spans zero time and yields +inf, which is the physically right
    answer and compares correctly against any finite threshold.
    """
    t = np.asarray(times_s, dtype=np.float64)
    M = int(M)
    if M < 2:
        raise ValueError(f"M must be at least 2 photons (got {M}).")
    if t.size < M:
        return np.empty(0, dtype=np.float64)

    span = t[M - 1:] - t[: t.size - M + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return M / span


def false_alarm_probability(M: int, F: float) -> float:
    """
    Probability that one window fires on PURE POISSON background.

    A window starting at a photon spans M-1 interphoton intervals, so its
    span T is Gamma(M-1, 1/lambda) distributed.  The window fires when
    r = M/T >= F*lambda, i.e. when T <= M/(F*lambda), and lambda cancels:

        P = gammainc(M - 1, M / F)          (regularised lower incomplete)

    The rate is therefore a property of M and F ALONE -- independent of the
    count rate, the acquisition length, and the sample.  That is what makes
    it usable as a design rule rather than something to tune per dataset.

    Two things to remember when reading the number.  It is PER WINDOW, and
    there is one window per photon, so a record of N photons yields about
    N*P false windows.  And each flagged window marks M photons, so the
    fraction of DATA affected is larger again, by up to a factor of M once
    merging is accounted for.

    Returns NaN if scipy is unavailable; it is a diagnostic, never a
    dependency of detection.
    """
    try:
        from scipy.special import gammainc
    except ImportError:
        return float("nan")
    if int(M) < 2 or float(F) <= 0:
        return float("nan")
    return float(gammainc(int(M) - 1, int(M) / float(F)))


def expected_false_windows(n_photons: int, M: int, F: float) -> float:
    """Expected number of background windows flagged in *n_photons* records."""
    p = false_alarm_probability(M, F)
    return float(max(0, int(n_photons) - int(M) + 1) * p)


def _runs_to_intervals(times_s: np.ndarray,
                       in_burst: np.ndarray) -> np.ndarray:
    """Contiguous True runs of *in_burst* -> (n, 2) [first_t, last_t]."""
    if not in_burst.any():
        return np.empty((0, 2), dtype=np.float64)
    edges = np.diff(in_burst.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    if in_burst[0]:
        starts = np.concatenate(([0], starts))
    if in_burst[-1]:
        ends = np.concatenate((ends, [in_burst.size - 1]))
    return np.column_stack((times_s[starts], times_s[ends])).astype(np.float64)


def detect_bursts(
    times_s: np.ndarray,
    duration_s: float,
    M: int = _DEFAULT_M,
    F: float = _DEFAULT_F,
    channel: int = 1,
    mean_rate_hz: Optional[float] = None,
) -> ChannelBursts:
    """
    Flag bursts in one channel's arrival-time record.

    Parameters
    ----------
    times_s : array
        Photon arrival times in seconds, ascending.
    duration_s : float
        Acquisition length, used for the mean rate.
    M : int
        Photons per sliding window.
    F : float
        Rate multiplier.  A window is flagged when its instantaneous rate is
        at least ``F`` times the mean count rate.
    mean_rate_hz : float, optional
        Reference rate for the threshold.  Defaults to ``len(times) /
        duration_s`` -- the WHOLE-TRACE mean, which includes the burst photons
        themselves and is therefore biased upward by exactly the contaminant
        being searched for.  Pass a burst-free reference (an earlier pass, or
        a monomer-only control measured under identical conditions) to remove
        that bias.

    Returns
    -------
    ChannelBursts
    """
    t = np.asarray(times_s, dtype=np.float64)
    n = t.size

    if mean_rate_hz is None:
        mean_rate_hz = (n / duration_s) if duration_s > 0 else float("nan")
    threshold = float(F) * float(mean_rate_hz)

    in_burst = np.zeros(n, dtype=bool)
    if n >= int(M) and np.isfinite(threshold) and threshold > 0:
        r = sliding_rates(t, M)
        hot = r >= threshold

        # Mark all M photons of every flagged window, merging overlaps.  A
        # difference array keeps this O(N): naive per-window slicing is
        # O(N*M) and, at the count rates this is used on, that is the
        # difference between instant and a visible stall.
        if hot.any():
            delta = np.zeros(n + 1, dtype=np.int32)
            idx = np.flatnonzero(hot)
            np.add.at(delta, idx, 1)
            np.add.at(delta, idx + int(M), -1)
            in_burst = np.cumsum(delta)[:n] > 0

    return ChannelBursts(
        channel=int(channel),
        n_photons=int(n),
        mean_rate_hz=float(mean_rate_hz),
        threshold_hz=float(threshold),
        in_burst=in_burst,
        intervals=_runs_to_intervals(t, in_burst),
    )


# ── Interval algebra ──────────────────────────────────────────────────────────

def merge_intervals(iv: np.ndarray) -> np.ndarray:
    """Sort and union a set of [start, end] intervals."""
    iv = np.asarray(iv, dtype=np.float64).reshape(-1, 2)
    if iv.shape[0] == 0:
        return iv
    iv = iv[np.argsort(iv[:, 0], kind="stable")]
    out = [iv[0].copy()]
    for lo, hi in iv[1:]:
        if lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append(np.array([lo, hi]))
    return np.array(out, dtype=np.float64)


def coincident_intervals(iv_a: np.ndarray, iv_b: np.ndarray) -> np.ndarray:
    """
    Intervals from *iv_a* that overlap some interval of *iv_b*, unioned with
    the partner they overlap.

    The UNION of the overlapping pair is returned, not the intersection: the
    two detectors see the same particle over slightly different windows, so
    the event's true extent is the outer envelope.  Excising the intersection
    would leave the leading and trailing photons of the burst in the data,
    which is the part of it most easily mistaken for signal.

    Touching intervals count as overlapping; a shared endpoint is a real
    coincidence at this timing resolution, not a boundary accident.
    """
    a = merge_intervals(iv_a)
    b = merge_intervals(iv_b)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)

    kept: List[np.ndarray] = []
    i = j = 0
    while i < a.shape[0] and j < b.shape[0]:
        lo = max(a[i, 0], b[j, 0])
        hi = min(a[i, 1], b[j, 1])
        if lo <= hi:                                  # they overlap
            kept.append(np.array([min(a[i, 0], b[j, 0]),
                                  max(a[i, 1], b[j, 1])]))
        # Advance whichever interval ends first; the other may still overlap
        # the next one along.
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1

    if not kept:
        return np.empty((0, 2), dtype=np.float64)
    return merge_intervals(np.array(kept, dtype=np.float64))


def complement_intervals(iv: np.ndarray, t_start: float,
                         t_end: float) -> np.ndarray:
    """The parts of [t_start, t_end] not covered by *iv*."""
    iv = merge_intervals(iv)
    out: List[Tuple[float, float]] = []
    cursor = float(t_start)
    for lo, hi in iv:
        if hi <= cursor:
            continue
        if lo > cursor:
            out.append((cursor, min(lo, t_end)))
        cursor = max(cursor, hi)
        if cursor >= t_end:
            break
    if cursor < t_end:
        out.append((cursor, float(t_end)))
    return np.array([p for p in out if p[1] > p[0]],
                    dtype=np.float64).reshape(-1, 2)


def photons_in_intervals(times_s: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """
    Boolean mask: True where a photon falls inside any interval of *iv*.

    Intervals are CLOSED at both ends.  That is not a detail: burst interval
    ends come from _runs_to_intervals, which takes them from the photon times
    themselves, so the last photon of a burst sits exactly on its end.  A
    half-open [lo, hi) test therefore leaves one photon of every burst in the
    data -- the trailing photon, which is the part of a burst most easily
    mistaken for signal -- and leaves a zero-width interval excising nothing
    at all.  A zero-width interval is reachable whenever M photons share one
    macrotime tick, which a bright enough spike can do.

    Intervals are disjoint and ascending after merging, so each photon has at
    most one candidate: the last interval whose start does not exceed it.
    """
    t = np.asarray(times_s, dtype=np.float64)
    iv = merge_intervals(iv)
    if t.size == 0 or iv.shape[0] == 0:
        return np.zeros(t.size, dtype=bool)
    idx = np.searchsorted(iv[:, 0], t, side="right") - 1
    ok = idx >= 0
    # Clip only to keep the gather in range; `ok` masks the result anyway.
    return ok & (t <= iv[np.clip(idx, 0, iv.shape[0] - 1), 1])


# ── Top-level analysis ────────────────────────────────────────────────────────

def analyse(
    fcs_data,
    M: int = _DEFAULT_M,
    F: float = _DEFAULT_F,
    mean_rate_ch1_hz: Optional[float] = None,
    mean_rate_ch2_hz: Optional[float] = None,
) -> BurstResult:
    """
    Run detection on both channels and keep only coincident bursts.

    A single-channel acquisition has no second detector to confirm against.
    Rather than silently returning nothing, every detected burst is kept and
    a note records that the coincidence test could not be applied -- so the
    result is still usable, and the reader can see on what terms.
    """
    d = fcs_data
    duration = float(d.duration_s)
    notes: List[str] = []

    t1 = d.ch1_times_s if getattr(d, "has_ch1", True) else np.empty(0)
    t2 = d.ch2_times_s if getattr(d, "has_ch2", True) else np.empty(0)

    cb1 = (detect_bursts(t1, duration, M, F, 1, mean_rate_ch1_hz)
           if getattr(d, "has_ch1", True) else None)
    cb2 = (detect_bursts(t2, duration, M, F, 2, mean_rate_ch2_hz)
           if getattr(d, "has_ch2", True) else None)

    if cb1 is not None and cb2 is not None:
        coincident = coincident_intervals(cb1.intervals, cb2.intervals)
        n_single = (cb1.n_bursts + cb2.n_bursts)
        if n_single and coincident.shape[0] == 0:
            notes.append(
                f"{n_single:,} single-channel bursts were found but none "
                f"coincided; nothing was excised."
            )
    else:
        present = cb1 if cb1 is not None else cb2
        coincident = (merge_intervals(present.intervals)
                      if present is not None
                      else np.empty((0, 2), dtype=np.float64))
        notes.append(
            "Single-channel data: the cross-channel coincidence test could "
            "not be applied, so instrument artifacts (afterpulsing, spikes) "
            "are NOT rejected and every detected burst was excised."
        )

    clean = complement_intervals(coincident, 0.0, duration)

    return BurstResult(
        M=int(M), F=float(F), duration_s=duration,
        ch1=cb1, ch2=cb2,
        coincident=coincident, clean=clean,
        ch1_excised=photons_in_intervals(t1, coincident),
        ch2_excised=photons_in_intervals(t2, coincident),
        notes=notes,
    )


# ── Correlation over the clean intervals ──────────────────────────────────────

#: Correlation backends this module can average over burst-free regions.
#: Both produce a lag grid that can be reconciled across segments of unequal
#: length -- 'multitau' because its grid is a pure function of (tau_min, m,
#: coarsen), and 'multitau_pkg' because fcs_corr._align_to_grid matches lags
#: by VALUE, so the extra octaves an over-long segment earns simply go
#: unfilled instead of misaligning the average.  The tau-edge methods are
#: excluded only because they were not asked for.
_METHODS = ("multitau", "multitau_pkg")

#: What burst excision cannot vary.  A collection names the values that stay
#: ALLOWED; a scalar pins the setting.  Passed straight to
#: fcs_corr.run_correlation_dialog as `force`, so the matching controls show
#: disabled at exactly these values rather than being overridden after the
#: user has set them.  Re-checked in correlate_clean, which can also be
#: called without the dialog.
_FORCED = {
    "corr_type":    "cross",     # the spec is Ch1 x Ch2
    "method":       _METHODS,
    "segment":      True,        # segmenting is the whole uncertainty model
    "full_trace_G": False,       # no whole stream survives excision
    "thin_factor":  1,
}


def default_params(**overrides) -> dict:
    """An fcs_corr-style parameter dict for burst correlation.

    Starts from the suite's correlation defaults, applies :data:`_FORCED` --
    the same constraints the GUI locks into the dialog -- and then any
    overrides.  Applying _FORCED matters: fcs_corr's own default method is
    'perbin', so a dict that merely copied the defaults would be rejected on
    arrival.  Where _FORCED names a set, the first entry is taken.
    """
    import fcs_corr
    out = dict(fcs_corr._defaults)
    for key, spec in _FORCED.items():
        out[key] = (spec[0] if isinstance(spec, (set, list, tuple))
                    else spec)
    out.update(overrides)
    return out


def required_segment_s(tau_max_s: float, requested_s: float = None) -> float:
    """The segment length actually used, given the longest lag wanted.

    A segment must be at least ``fcs_corr._MIN_SEGMENT_FACTOR`` times the
    longest lag, or the correlation at that lag is estimated from a handful
    of photon pairs.  The usual 1 s segment therefore covers lags up to
    100 ms; beyond that the segment has to grow with tau_max.  The floor is
    enforced rather than warned about, because a segment too short for the
    requested lag does not merely widen the error bars -- it biases the
    long-lag end low.
    """
    import fcs_corr
    floor = float(getattr(fcs_corr, "_MIN_SEGMENT_FACTOR", 10.0)) * float(tau_max_s)
    base = float(getattr(fcs_corr, "_SEGMENT_DURATION_S", _FALLBACK_SEG_DURATION_S)
                 if requested_s is None else requested_s)
    return max(base, floor)


def correlate_clean(fcs_data, result: BurstResult, params: Optional[dict] = None,
                    progress_cb=None) -> dict:
    """
    Ch1 x Ch2 multi-tau cross-correlation of the burst-free data.

    EVERY burst-free region is segmented at the fixed segment length, and the
    segments from all regions are pooled into one ensemble -- three clean
    regions of 20 s at a 1 s segment give 60 segments, exactly as 60 s of
    unbroken record would.  Segmentation uses
    :func:`fcs_corr.segment_bounds`, so the remainder of each region lands in
    that region's final segment (1x to 2x nominal), matching the rest of the
    suite.  Segment boundaries never straddle an excised burst, and the
    photon stream is never concatenated across a cut.

    *params* is the dict :func:`fcs_corr.run_correlation_dialog` returns with
    ``collect_only=True``.  Honoured: ``tau_min_ms``, ``tau_max_ms``,
    ``method`` (either entry of :data:`_METHODS`), ``mt_channels``,
    ``mt_coarsen`` (``multitau`` only -- multipletau's grid is hard-wired
    base-2), ``mt_compress`` (``multitau_pkg`` only), ``seg_duration_s``.
    Anything in :data:`_FORCED` is rejected rather than silently ignored.

    The requested segment length is raised if it is below
    :func:`required_segment_s`; ``gate`` is ignored, because a microtime gate
    changes the count rate and the detection threshold is F times the mean
    count rate.
    """
    import warnings
    import fcs_corr

    d = fcs_data
    params = dict(default_params() if params is None else params)
    notes: List[str] = []

    for key, spec in _FORCED.items():
        if key not in params:
            continue
        got = params[key]
        if key in ("segment", "full_trace_G"):
            if bool(got) != spec:
                notes.append(f"{key} was forced to {spec}")
            continue
        if key == "thin_factor":
            if int(got) != 1:
                raise ValueError(
                    f"Photon thinning (x{int(got)}) is not supported with "
                    f"burst excision.  Set the thinning factor to 1."
                )
            continue
        allowed = spec if isinstance(spec, (set, list, tuple)) else (spec,)
        if got not in allowed:
            raise ValueError(
                f"Burst excision supports {key} = "
                f"{' or '.join(repr(a) for a in allowed)} only "
                f"(got {got!r})."
            )

    if params.get("gate"):
        notes.append("the microtime gate was not applied (see correlate_clean)")

    method = params.get("method", "multitau")
    tau_min_s = float(params.get("tau_min_ms", 0.01)) * 1e-3
    tau_max_s = float(params.get("tau_max_ms", 10.0)) * 1e-3
    m = max(2, int(params.get("mt_channels", 8)))
    coarsen = max(2, int(params.get("mt_coarsen", 8)))
    compress = params.get("mt_compress", "average")

    requested_seg = float(params.get("seg_duration_s",
                                     getattr(fcs_corr, "_SEGMENT_DURATION_S",
                                             _FALLBACK_SEG_DURATION_S)))
    if requested_seg <= 0:
        raise ValueError("Segment duration must be positive.")
    seg_duration_s = required_segment_s(tau_max_s, requested_seg)
    if seg_duration_s > requested_seg:
        notes.append(
            f"segment length raised from {requested_seg:g} s to "
            f"{seg_duration_s:g} s "
            f"({getattr(fcs_corr, '_MIN_SEGMENT_FACTOR', 10)} x tau_max)")
    if method == "multitau_pkg" and int(params.get("mt_coarsen", 8)) != 8:
        notes.append("mt_coarsen is ignored by multipletau (its grid is base-2)")

    fcs_corr.require_usable_channels(d, "cross")
    t1 = np.asarray(d.ch1_times_s, dtype=np.float64)
    t2 = np.asarray(d.ch2_times_s, dtype=np.float64)

    # Filter regions BEFORE segmenting.  segment_bounds returns one whole-span
    # segment when the span is shorter than seg_duration_s, so handing it a
    # short region would silently produce an undersized segment rather than
    # the exclusion that is wanted.
    regions = [(lo, hi) for lo, hi in result.clean
               if (hi - lo) >= seg_duration_s]
    n_regions_dropped = int(result.clean.shape[0] - len(regions))
    T_dropped = float(sum(hi - lo for lo, hi in result.clean
                          if (hi - lo) < seg_duration_s))

    if not regions:
        longest = (float((result.clean[:, 1] - result.clean[:, 0]).max())
                   if result.clean.size else 0.0)
        raise ValueError(
            f"No burst-free region is as long as one {seg_duration_s:.4g} s "
            f"segment (longest is {longest:.4g} s, from "
            f"{result.clean.shape[0]} regions).  Shorten the longest lag, or "
            f"tighten the detection settings so less is excised."
        )

    bounds: List[Tuple[float, float]] = []
    for lo, hi in regions:
        bounds.extend(fcs_corr.segment_bounds(float(lo), float(hi),
                                              seg_duration_s))

    seg_curves: List[Tuple[np.ndarray, np.ndarray]] = []
    kept: List[Tuple[float, float]] = []
    n_skipped = 0
    n1_used = n2_used = 0
    T_used = 0.0

    for k, (lo, hi) in enumerate(bounds):
        a = t1[(t1 >= lo) & (t1 < hi)] - lo
        b = t2[(t2 >= lo) & (t2 < hi)] - lo
        if a.size < 2 or b.size < 2:
            warnings.warn(
                f"Segment {k + 1}/{len(bounds)} ({lo:.4g}-{hi:.4g} s) has too "
                f"few photons (Ch1 {a.size}, Ch2 {b.size}) and was skipped; "
                f"the quoted uncertainty uses the remaining segments.",
                RuntimeWarning,
            )
            n_skipped += 1
            continue
        span = float(hi - lo)
        if method == "multitau_pkg":
            tau_k, G, _, _ = fcs_corr.compute_multitau_pkg(
                a, b, tau_min_s, tau_max_s, m=m, segment=False,
                duration_s=span, symmetric=True, compress=compress)
        else:
            tau_k, G, _, _ = fcs_corr.compute_multitau(
                a, b, tau_min_s, tau_max_s, m=m, coarsen=coarsen,
                segment=False, duration_s=span, symmetric=True)
        seg_curves.append((np.asarray(tau_k, dtype=np.float64),
                           np.asarray(G, dtype=np.float64)))
        kept.append((float(lo), float(hi)))
        n1_used += int(a.size)
        n2_used += int(b.size)
        T_used += span
        if progress_cb is not None:
            progress_cb(k + 1, f"segment {k + 1} / {len(bounds)}")

    if not seg_curves:
        raise ValueError(
            "Every segment was empty on at least one channel; no correlation "
            "could be computed."
        )

    # Reference grid = the longest segment's.  multipletau's grid grows with
    # trace length, so an over-long terminal segment carries more octaves
    # than the nominal ones; _align_to_grid matches by lag VALUE and leaves
    # the rest NaN, which nanmean/nanstd then skip.  For 'multitau' every
    # grid is already identical and this reduces to a plain stack.
    ref_i = int(np.argmax([t.size for t, _ in seg_curves]))
    tau_ref = seg_curves[ref_i][0]
    stack = np.vstack([fcs_corr._align_to_grid(tau_ref, t, G)
                       for t, G in seg_curves])

    G_mean = np.nanmean(stack, axis=0)
    if stack.shape[0] > 1:
        with np.errstate(invalid="ignore"):
            G_std = (np.nanstd(stack, axis=0, ddof=1)
                     / np.sqrt(float(stack.shape[0])))
    else:
        # One segment carries no scatter.  NaN, not zero, so a downstream fit
        # treats the point as unweighted rather than infinitely well known.
        G_std = np.full(tau_ref.size, np.nan)

    n1_tot, n2_tot = int(t1.size), int(t2.size)
    n1_clean = n1_tot - int(np.asarray(result.ch1_excised, bool).sum())
    n2_clean = n2_tot - int(np.asarray(result.ch2_excised, bool).sum())

    return {
        "tau":                tau_ref,
        "G":                  G_mean,
        "G_std":              G_std,
        "n_segments":         int(stack.shape[0]),
        "method":             method,
        "m":                  m,
        "coarsen":            coarsen if method == "multitau" else None,
        "compress":           compress if method == "multitau_pkg" else None,
        "tau_min_s":          tau_min_s,
        "tau_max_s":          tau_max_s,
        "segment_bounds":     np.asarray(kept, dtype=np.float64),
        "seg_duration_s":     seg_duration_s,
        "seg_duration_requested_s": requested_seg,
        "n_regions_used":     len(regions),
        "n_regions_dropped":  n_regions_dropped,
        "T_dropped_fragments": T_dropped,
        "n_segments_skipped": int(n_skipped),
        "T_used":             float(T_used),
        "n_ch1":              int(n1_used),
        "n_ch2":              int(n2_used),
        "n_ch1_total":        n1_tot,
        "n_ch2_total":        n2_tot,
        "n_ch1_clean":        int(n1_clean),
        "n_ch2_clean":        int(n2_clean),
        "notes":              notes,
    }


def retention_summary(corr: dict) -> str:
    """How much of the data actually reached the correlation, and why not more.

    Two separate losses, reported separately because they have different
    causes and different fixes: photons removed as burst (lower F or M to
    remove less), and clean photons stranded in a burst-free region too short
    to hold one segment (shorten the longest lag, or accept the loss).  The
    second is invisible in the burst statistics and can easily be the larger
    of the two.
    """
    lines = []
    for ch in (1, 2):
        tot = corr.get(f"n_ch{ch}_total", 0)
        if not tot:
            continue
        clean = corr.get(f"n_ch{ch}_clean", 0)
        used = corr.get(f"n_ch{ch}", 0)
        lines.append(
            f"  Ch{ch}: of {tot:,} photons, "
            f"{100.0 * clean / tot:.2f} % remain after burst excision, "
            f"{100.0 * used / tot:.2f} % remain suitable for segmenting"
        )
    if corr.get("n_regions_dropped"):
        lines.append(
            f"  {corr['n_regions_dropped']:,} burst-free regions "
            f"({corr['T_dropped_fragments']:.2f} s) were shorter than one "
            f"{corr['seg_duration_s']:g} s segment and were not used"
        )
    return "\n".join(lines)


# ── Metadata shared by the exports ────────────────────────────────────────────

def _burst_meta(result: BurstResult) -> dict:
    meta = {
        "burst_M_photons":   result.M,
        "burst_F_multiplier": f"{result.F:g}",
        "burst_rate_rule":   "r = M / (t[i+M-1] - t[i]);  flagged when r >= F * mean rate",
        "burst_coincidence": "union of Ch1 and Ch2 burst intervals that overlap in time",
        "duration_s":        f"{result.duration_s:.6g}",
        "n_bursts_coincident": result.n_coincident,
        "burst_time_s":      f"{result.burst_time_s:.6g}",
        "clean_time_s":      f"{result.clean_time_s:.6g}",
        "n_clean_intervals": int(result.clean.shape[0]),
    }
    for cb in (result.ch1, result.ch2):
        if cb is None:
            continue
        c = cb.channel
        meta[f"ch{c}_mean_rate_hz"] = f"{cb.mean_rate_hz:.6g}"
        meta[f"ch{c}_threshold_hz"] = f"{cb.threshold_hz:.6g}"
        meta[f"ch{c}_n_bursts"] = cb.n_bursts
        meta[f"ch{c}_burst_photons"] = cb.n_burst_photons
    return meta


# ── Plot: intensity with bursts marked ────────────────────────────────────────

def _draw_intensity_bursts(time_s, I1_cps, I2_cps, spans, bin_width_s,
                           title, solo=None):
    """Draw the intensity trace with burst spans shaded.

    Shared by the live plot and by :func:`rebuild_plot`, so a reopened
    figure is drawn by the same code that drew the original rather than by
    a second renderer that can drift away from it.

    *spans* is an (n, 2) array of burst intervals; *solo* maps a channel
    number to the midpoints of its rejected single-channel detections, and
    is absent on a reopen because the CSV records only what survived.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    time_s = np.asarray(time_s, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11, 5))
    for arr, colour, label in ((I1_cps, "steelblue", "Ch1"),
                               (I2_cps, "tomato", "Ch2")):
        if arr is None or not np.isfinite(np.asarray(arr)).any():
            continue
        ax.plot(time_s, arr, color=colour, linewidth=0.8, label=label,
                alpha=0.9)

    # At typical burst widths a span is far narrower than one pixel, so a
    # minimum visual width is enforced -- the alternative is a figure that
    # truthfully shows nothing at all.
    spans = np.asarray(spans, dtype=np.float64).reshape(-1, 2)
    if time_s.size:
        min_vis = max(bin_width_s, (time_s[-1] - time_s[0]) / 2000.0)
    else:
        min_vis = bin_width_s
    for k, (lo, hi) in enumerate(spans):
        pad = max(0.0, (min_vis - (hi - lo)) / 2.0)
        ax.axvspan(lo - pad, hi + pad, color="goldenrod", alpha=0.35,
                   linewidth=0,
                   label="coincident burst" if k == 0 else None)

    # Rejected single-channel detections as a rug tick: the reader needs to
    # see how many there were to judge whether the coincidence requirement
    # is doing real work.  At permissive (M, F) there can be thousands, so
    # these arrive pre-computed by one vectorised test, not a loop.
    for ch, colour in ((1, "navy"), (2, "darkred")):
        xs = np.asarray((solo or {}).get(ch, ()), dtype=np.float64)
        if xs.size == 0:
            continue
        ax.plot(xs, np.full(xs.size, ax.get_ylim()[1]), linestyle="none",
                marker="|", markersize=5, color=colour, alpha=0.5,
                label=f"Ch{ch} only, rejected ({xs.size:,})")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Intensity (CPS)", fontsize=12)
    ax.set_title(title, fontsize=11)
    finite = [np.asarray(a, dtype=np.float64) for a in (I1_cps, I2_cps)
              if a is not None and np.isfinite(np.asarray(a)).any()]
    if finite:
        combined = np.concatenate(finite)
        ax.set_ylim(0, max(np.nanpercentile(combined, 99.9) * 1.25, 1.0))
    if time_s.size:
        ax.set_xlim(time_s[0], time_s[-1])
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
    fig.tight_layout()
    return fig, ax


def plot_intensity_bursts(
    fcs_data,
    result: BurstResult,
    bin_width_s: Optional[float] = None,
    show: bool = True,
    export: bool = False,
):
    """
    Intensity trace with the excised regions shaded and the burst photons
    drawn as a separate overlay.

    The ORIGINAL trace is what is plotted -- nothing is removed from the
    figure.  Excision is shown, not performed, so the bursts can be judged
    against the baseline they were pulled out of.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import fcs_plot
    import fcs_export

    d = fcs_data
    if bin_width_s is None:
        bin_width_s = fcs_plot._choose_bin_width(d.duration_s)

    time_s, I1, I2 = d.bin_intensity(bin_width_s=bin_width_s)
    I1_cps = I1 / bin_width_s
    I2_cps = I2 / bin_width_s

    solo = {}
    for cb in (result.ch1, result.ch2):
        if cb is None or cb.n_bursts == 0:
            continue
        mids = 0.5 * (cb.intervals[:, 0] + cb.intervals[:, 1])
        solo[cb.channel] = mids[~photons_in_intervals(mids,
                                                      result.coincident)]

    fig, ax = _draw_intensity_bursts(
        time_s, I1_cps, I2_cps, result.coincident, bin_width_s,
        title=(f"Intensity with burst detection — {d.filepath.name}\n"
               f"bin {fcs_plot._format_bin_width(bin_width_s)}  ·  "
               f"M = {result.M}, F = {result.F:g}  ·  "
               f"{result.n_coincident:,} coincident bursts"),
        solo=solo)

    if export:
        in_burst_bin = np.zeros(time_s.size, dtype=np.float64)
        if result.coincident.size:
            centres = time_s + 0.5 * bin_width_s
            in_burst_bin = photons_in_intervals(
                centres, result.coincident).astype(np.float64)
        meta = dict(_burst_meta(result))
        meta["bin_width_s"] = f"{bin_width_s:.6g}"
        # The keys fcs_plot's intensity rebuilder reads, so this file is
        # interchangeable with an ordinary intensity export apart from the
        # extra bin_in_burst column.
        _chans = getattr(d, "channels", (1, 2))
        meta["channels_recorded"] = "+".join(f"Ch{c}" for c in _chans)
        meta["ch1_photons"] = len(d.ch1_times_s) if 1 in _chans else ""
        meta["ch2_photons"] = len(d.ch2_times_s) if 2 in _chans else ""
        fcs_export.safe_export(
            d, "burst intensity",
            {
                "time_s":      time_s,
                "ch1_counts":  I1,
                "ch2_counts":  I2,
                "ch1_cps":     I1_cps,
                "ch2_cps":     I2_cps,
                "bin_in_burst": in_burst_bin,
            },
            meta=meta,
        )
        _export_burst_list(d, result)

    if show:
        import fcs_plottools
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


def _export_burst_list(fcs_data, result: BurstResult) -> None:
    """Write one row per coincident burst: when it was and how big it was."""
    import fcs_export

    iv = result.coincident
    if iv.shape[0] == 0:
        return
    t1 = np.asarray(fcs_data.ch1_times_s, dtype=np.float64)
    t2 = np.asarray(fcs_data.ch2_times_s, dtype=np.float64)
    start, end = iv[:, 0], iv[:, 1]
    dur = end - start
    n1 = np.searchsorted(t1, end, "right") - np.searchsorted(t1, start, "left")
    n2 = np.searchsorted(t2, end, "right") - np.searchsorted(t2, start, "left")
    with np.errstate(divide="ignore", invalid="ignore"):
        fcs_export.safe_export(
            fcs_data, "burst list",
            {
                "burst_index":  np.arange(1, iv.shape[0] + 1, dtype=float),
                "start_s":      start,
                "end_s":        end,
                "duration_s":   dur,
                "ch1_photons":  n1.astype(float),
                "ch2_photons":  n2.astype(float),
                "ch1_rate_hz":  n1 / dur,
                "ch2_rate_hz":  n2 / dur,
            },
            meta=_burst_meta(result),
        )


# ── Plot: lifetime decays, burst vs. burst-free ───────────────────────────────

def _draw_burst_lifetimes(series, title):
    """Draw the burst and burst-free decays on ONE panel.

    Shared by the live plot and :func:`rebuild_plot`.  *series* is the
    exported column mapping itself -- ``time_ns`` plus ``chN_burst`` and
    ``chN_clean`` -- so the figure is drawn from exactly what the CSV holds,
    and a reopened decay cannot disagree with the file it came from.

    One panel, not one per channel, to match fcs_lifetime: the reopen
    overlay composites only a figure's FIRST axes and treats the rest as
    residual strips, so a two-panel layout would have had Ch2 silently
    dropped whenever a burst decay was overlaid with anything else.  Colour
    therefore distinguishes channel (as in the intensity plot) and line
    style distinguishes population.

    Counts are NOT normalised: the burst population is orders of magnitude
    smaller, and scaling the two to a common peak would hide precisely that.
    """
    import matplotlib.pyplot as plt

    time_ns = np.asarray(series["time_ns"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    drew = False
    for ch, colour in ((1, "steelblue"), (2, "tomato")):
        for part, style, label in (("clean", "-", "burst-free"),
                                   ("burst", "--", "burst")):
            col = series.get(f"ch{ch}_{part}")
            if col is None:
                continue
            c = np.asarray(col, dtype=np.float64)
            if not np.isfinite(c).any():
                continue
            # Zeros are masked to NaN for the LOG axis only; the exported
            # columns keep the real zero counts.
            ax.semilogy(time_ns, np.where(c > 0, c, np.nan), color=colour,
                        linestyle=style, linewidth=0.9,
                        label=f"Ch{ch} {label} ({int(np.nansum(c)):,})")
            drew = True
    if drew:
        ax.legend(fontsize=9, framealpha=0.85, ncol=2)
    else:
        ax.text(0.5, 0.5, "no decay data in this export",
                transform=ax.transAxes, ha="center", va="center",
                color="grey")
    ax.set_xlabel("Time (ns)", fontsize=11)
    ax.set_ylabel("Counts", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_burst_lifetimes(
    fcs_data,
    result: BurstResult,
    n_bins: int = 4096,
    show: bool = True,
    export: bool = False,
):
    """
    TCSPC decays for the burst photons and the burst-free photons, per channel.

    Splitting the decay this way is the point of doing it at all: if the
    bursts come from a distinct species, its decay need not match the bulk,
    and a difference here is independent evidence that the detection found
    something real rather than the upper tail of ordinary shot noise.

    Binning follows FCSData.lifetime_histogram exactly -- a histogram of the
    RAW microtime bins over range (0, 4096), with the left edge of each bin
    reported in ns.  This module cannot call that method directly because it
    needs the histogram of a photon SUBSET, but the arithmetic is reproduced
    rather than reinvented so a burst decay is directly comparable with an
    ordinary one, and a column of either can be handed to the same fitter.

    Counts are plotted on a log axis and NOT normalised: the burst population
    is orders of magnitude smaller, and scaling the two to a common peak
    would hide exactly that.
    """
    import matplotlib.pyplot as plt
    import fcs_export
    import fcs_plottools
    from fcs_reader import _MICROTIME_BINS

    d = fcs_data
    n_bins = int(n_bins)
    counts, edges = np.histogram(np.empty(0), bins=n_bins,
                                 range=(0, _MICROTIME_BINS))
    time_ns = edges[:-1] * d.microtime_resolution_ns
    blank = np.full(n_bins, np.nan, dtype=np.float64)

    # Fixed schema, as for the ordinary lifetime export: every column is
    # present for both channels whatever the file recorded, with NaN (not
    # zero, which would read back as a real measurement of no photons) for
    # a channel that is absent.
    series: dict = {"time_ns": time_ns}
    for ch in (1, 2):
        for part in ("all", "burst", "clean"):
            series[f"ch{ch}_{part}"] = blank.copy()

    for ch in (1, 2):
        if ch not in getattr(d, "channels", (1, 2)):
            continue
        micro = np.asarray(d.ch1_micro if ch == 1 else d.ch2_micro)
        if micro.size == 0:
            continue
        excised = result.ch1_excised if ch == 1 else result.ch2_excised
        mask = (np.asarray(excised, dtype=bool)
                if np.size(excised) == micro.size
                else np.zeros(micro.size, dtype=bool))
        rng = (0, _MICROTIME_BINS)
        series[f"ch{ch}_all"] = np.histogram(
            micro, bins=n_bins, range=rng)[0].astype(np.float64)
        series[f"ch{ch}_burst"] = np.histogram(
            micro[mask], bins=n_bins, range=rng)[0].astype(np.float64)
        series[f"ch{ch}_clean"] = np.histogram(
            micro[~mask], bins=n_bins, range=rng)[0].astype(np.float64)

    fig, ax = _draw_burst_lifetimes(
        series,
        title=(f"Lifetime decay, burst vs. burst-free \u2014 "
               f"{d.filepath.name}   "
               f"(M = {result.M}, F = {result.F:g}, {n_bins} bins)"))

    if export:
        meta = dict(_burst_meta(result))
        meta["n_bins"] = n_bins
        meta["laser_period_ns"] = f"{d.laser_period_ns:.6g}"
        meta["channels_recorded"] = "+".join(
            f"Ch{c}" for c in getattr(d, "channels", (1, 2)))
        fcs_export.safe_export(d, "burst lifetime", series, meta=meta,
                               suffix=f"{n_bins}bins")

    if show:
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── Plot: correlation of the clean intervals ──────────────────────────────────

def plot_burst_correlation(
    fcs_data,
    result: BurstResult,
    corr: dict,
    show: bool = True,
    export: bool = False,
):
    """
    Draw the burst-excised cross-correlation using the suite's own plotter,
    then export under a distinct analysis name.

    fcs_corr.plot_correlation does the drawing so this curve looks like every
    other G(tau) in the suite, but its export is bypassed: it would write to
    the ``correlation`` analysis slot and overwrite the ordinary correlation
    of the same file, which is precisely the curve this one is meant to be
    compared against.
    """
    import fcs_corr
    import fcs_export

    # show=False here and shown below: plot_correlation would otherwise put
    # the figure on screen under its own generic title, which is then
    # rewritten underneath the user.
    tau = corr["tau"]
    fig, ax = fcs_corr.plot_correlation(
        tau, corr["G"], corr["G_std"], "cross", fcs_data,
        corr.get("tau_min_s", float(tau.min())),
        corr.get("tau_max_s", float(tau.max())),
        corr["n_segments"],
        method=corr["method"],
        show=False, export=False,
        n_used_ch1=corr["n_ch1"], n_used_ch2=corr["n_ch2"],
        T_used=corr["T_used"],
        seg_duration_s=corr.get("seg_duration_s"),
        full_trace_G=False,
        mt_channels=corr["m"], mt_coarsen=corr.get("coarsen"),
        source_name=f"{fcs_data.filepath.name} (bursts excised)",
        std_is_sem=True,
    )
    ax.set_title(
        f"Ch1 × Ch2, bursts excised — {fcs_data.filepath.name}\n"
        f"M = {result.M}, F = {result.F:g}  ·  "
        f"{result.n_coincident:,} bursts removed  ·  "
        f"{corr['n_segments']:,} segments of "
        f"{corr['seg_duration_s']:g} s from "
        f"{corr['n_regions_used']:,} regions, {corr['T_used']:.2f} s used",
        fontsize=11,
    )

    if export:
        meta = dict(_burst_meta(result))
        meta.update({
            "type": "cross",
            "method": corr["method"],
            "method_label": fcs_corr._method_annotation(
                corr["method"], corr["m"], corr.get("coarsen"),
                corr.get("compress")),
            "tau_min_s": f"{corr['tau_min_s']:.6g}",
            "tau_max_s": f"{corr['tau_max_s']:.6g}",
            "symmetrized":
                "yes: G = 0.5*(G_Ch1Ch2 + G_Ch2Ch1), per segment",
            "n_segments": corr["n_segments"],
            "segment_duration_s": f"{corr['seg_duration_s']:.6g}",
            "n_burst_free_regions_used": corr["n_regions_used"],
            "n_regions_dropped_shorter_than_one_segment":
                corr["n_regions_dropped"],
            "burst_free_time_lost_to_fragmentation_s":
                f"{corr['T_dropped_fragments']:.6g}",
            "n_segments_skipped_too_few_photons": corr["n_segments_skipped"],
            "acquisition_time_s": f"{corr['T_used']:.6g}",
            "n_photons_ch1": corr["n_ch1"],
            "n_photons_ch2": corr["n_ch2"],
            "segment_source": (
                "burst-free regions segmented at the fixed segment length; "
                "segments pooled across regions and NOT concatenated"
            ),
            "G_source": "mean of the per-segment G",
            "G_from_full_trace": "no",
            "G_std_definition": (
                "standard error of the mean: "
                "std(G_segments, ddof=1) / sqrt(n_segments)"
            ),
            "mt_channels_per_octave": corr["m"],
            "segment_duration_requested_s":
                f"{corr['seg_duration_requested_s']:.6g}",
            "n_photons_ch1_total": corr["n_ch1_total"],
            "n_photons_ch2_total": corr["n_ch2_total"],
            "n_photons_ch1_after_excision": corr["n_ch1_clean"],
            "n_photons_ch2_after_excision": corr["n_ch2_clean"],
            "tau_grid_note": (
                "multi-tau grid anchored at tau_min (compute_multitau takes "
                "it as base_dt_s), so tau_min sets the lag VALUES, not just "
                "the range"
            ),
        })
        if corr.get("coarsen") is not None:
            meta["mt_coarsen_factor"] = corr["coarsen"]
        if corr.get("compress") is not None:
            meta["mt_compress"] = corr["compress"]
        for _i, _n in enumerate(corr.get("notes", ()), 1):
            meta[f"burst_correlation_note_{_i}"] = _n
        # Same column set as fcs_corr's own correlation export, tau_ms
        # included: these files are read by the other analysis programs, and
        # a column that is present in one correlation CSV and absent from
        # another is a trap for anything reading them in bulk.
        cols = {"tau_s": tau, "tau_ms": tau * 1e3, "G": corr["G"]}
        if bool(np.isfinite(corr["G_std"]).any()):
            cols["G_std"] = corr["G_std"]
        fcs_export.safe_export(
            fcs_data, "burst correlation", cols, meta=meta, suffix="cross",
        )

    if show:
        import fcs_plottools
        fcs_plottools.show_figure(fig, ax)
    return fig, ax


# ── Reopening an export as a live figure (fcs_plotopen) ───────────────────────

#: analysis name (as written into the CSV header) -> what rebuild_plot draws.
#: "burst list" is deliberately absent: it is a table of one row per burst,
#: not a curve, so there is no figure to rebuild and a registered rebuilder
#: would only promise one.
REBUILDABLE = ("burst intensity", "burst lifetime", "burst correlation")


def register(plotopen=None) -> None:
    """Register this module's exports with fcs_plotopen.

    Called once at start-up.  Registration is pushed from outside rather
    than performed on import so that fcs_burst keeps no dependency on
    fcs_plotopen -- the dependency runs the other way, and fcs_plotopen
    imports its rebuilders lazily on dispatch.
    """
    if plotopen is None:
        import fcs_plotopen as plotopen
    for name in REBUILDABLE:
        plotopen.register_rebuilder(name, "fcs_burst", "rebuild_plot")


def _spans_from_flag(time_s: np.ndarray, flag: np.ndarray,
                     bin_width_s: float) -> np.ndarray:
    """Recover burst intervals from the exported per-bin flag column.

    The CSV stores which BINS fell inside a burst rather than the interval
    list, so the reopened shading is bin-resolution rather than the exact
    photon-time extent.  That is a faithful reflection of what the file
    records; it is not refined by guessing.
    """
    f = np.asarray(flag, dtype=np.float64) > 0.5
    if not f.any():
        return np.empty((0, 2), dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    edges = np.diff(f.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1)
    if f[0]:
        starts = np.concatenate(([0], starts))
    if f[-1]:
        ends = np.concatenate((ends, [f.size - 1]))
    return np.column_stack((t[starts], t[ends] + bin_width_s))


def rebuild_plot(meta: dict, columns: dict, show: bool = True, path=None):
    """
    Rebuild a burst-analysis figure from its exported CSV, for fcs_plotopen.

    The CSV is the contract: nothing here opens the .fcs file, so an export
    from months ago reopens as a live figure with the full plot-controls
    panel.  Each case draws through the SAME function that drew the original
    -- the two intensity/lifetime drawers above, and fcs_corr's own
    correlation rebuilder -- so there is no second rendering path to drift.

    Dispatch is on the ``analysis`` header, since one module owns four
    export types here.
    """
    import fcs_plottools

    analysis = str(meta.get("analysis", "")).strip().lower()
    src = meta.get("source file") or meta.get("source") or (
        path.name if path is not None else "")
    detect = ""
    if meta.get("burst_M_photons") is not None:
        detect = (f"   (M = {meta.get('burst_M_photons')}, "
                  f"F = {meta.get('burst_F_multiplier')})")

    if analysis == "burst correlation":
        # Delegated: a burst correlation CSV carries the same columns and
        # header keys as an ordinary one, so fcs_corr's rebuilder draws it
        # correctly.  Only the title needs amending, to say that this curve
        # is of excised data -- which is the one thing a reader must not
        # have to infer from the filename.
        import fcs_corr
        fig, ax = fcs_corr.rebuild_plot(meta, columns, show=False, path=path)
        title = ax.get_title()
        if "excis" not in title.lower():
            ax.set_title(f"{title}\nbursts excised{detect}", fontsize=11)
        if show:
            fcs_plottools.show_figure(fig, ax)
        return fig, ax

    if analysis == "burst intensity":
        if "time_s" not in columns:
            raise ValueError(
                "No 'time_s' column; this CSV does not look like a burst "
                "intensity export.")
        time_s = np.asarray(columns["time_s"], dtype=np.float64)
        bw = _meta_float(meta, "bin_width_s")
        if bw is None:
            bw = float(np.median(np.diff(time_s))) if time_s.size > 1 else 1.0

        def _cps(ch):
            if f"ch{ch}_cps" in columns:
                return np.asarray(columns[f"ch{ch}_cps"], dtype=np.float64)
            if f"ch{ch}_counts" in columns:
                return (np.asarray(columns[f"ch{ch}_counts"],
                                   dtype=np.float64) / bw)
            return None

        spans = (_spans_from_flag(time_s, columns["bin_in_burst"], bw)
                 if "bin_in_burst" in columns
                 else np.empty((0, 2), dtype=np.float64))
        fig, ax = _draw_intensity_bursts(
            time_s, _cps(1), _cps(2), spans, bw,
            title=(f"Intensity with burst detection — {src}\n"
                   f"{spans.shape[0]:,} burst regions{detect}"))
        if show:
            fcs_plottools.show_figure(fig, ax)
        return fig, ax

    if analysis == "burst lifetime":
        if "time_ns" not in columns:
            raise ValueError(
                "No 'time_ns' column; this CSV does not look like a burst "
                "lifetime export.")
        fig, ax = _draw_burst_lifetimes(
            columns,
            title=f"Lifetime decay, burst vs. burst-free — {src}{detect}")
        if show:
            fcs_plottools.show_figure(fig, ax)
        return fig, ax

    raise ValueError(
        f"fcs_burst cannot rebuild analysis {analysis!r}.  It draws "
        f"{', '.join(REBUILDABLE)}; a 'burst list' export is a table of one "
        f"row per burst and has no figure."
    )


def _meta_float(meta: dict, key: str):
    """A float from the header, or None if absent or unparseable."""
    v = meta.get(key)
    if v is None or str(v).strip() == "":
        return None
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None
