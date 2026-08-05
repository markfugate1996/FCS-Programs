"""
fcs_lifetime_fit.py
===================
Fit TCSPC lifetime decays (microtime histograms) to exponential tail models.

This is the lifetime counterpart of fcs_fit.py (which fits correlation curves).
It is launched from the "Model Data" task in the main window, via the Lifetime
button in fcs_fit.run_model_dialog.

Workflow (launched from the main window)
----------------------------------------
    1. Pick the channel, histogram resolution, and fit window
       (peak → end by default; the window can be set visually on the
       histogram, reusing fcs_lifetime.select_gate).      -> _lifetime_data_dialog
    2. Choose a model (single- or two-exponential).         -> _select_lifetime_model_dialog
    3. Set guesses / bounds / fixed flags, then fit.        -> _lifetime_setup_dialog
    4. Fit, plot data + fit + residuals, write a report and curve CSV
       to a 'fits' folder beside the source .fcs file.

Tail fit, not reconvolution
---------------------------
The model is a sum of exponentials plus a constant background.  The data prep
(prepare_decay) drops the first and last microtime bins — these are time-tagger
catch-all artifacts, not fluorescence — and restricts the fit to the chosen
window.  Counts are Poisson-distributed, so the fit is weighted by σ = √counts
and the reduced χ² is meaningful.

The numerical core (prepare_decay, auto_guess_lifetime, fit_lifetime) has no GUI
dependency and can be reused for batch fitting.

Models come from fcs_models.LIFETIME_MODELS.

Dependencies
------------
    pip install numpy scipy matplotlib
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import fcs_plottools
from scipy.optimize import curve_fit

import fcs_models
from fcs_models import FCSModel
from fcs_reader import FCSData
import fcs_lifetime
import fcs_fisher
import fcs_globalfit
import fcs_fitdialogs

import fcs_export

from fcs_fitcommon import (
    fits_dir as _fits_dir,
    new_fit_dir as _new_fit_dir,
    fmt_bound as _fmt,
    parse_bound as _parse_bound,
    slug_name as _slug_name,
    write_params_table as _write_params_table,
)

# ── Multi-component decomposition ─────────────────────────────────────────────

# Multi-exponential models name their parameters A1/tau1, A2/tau2, ...
_COMPONENT_RE = re.compile(r"^(A|tau)(\d+)$")


def component_fractions(values: Dict[str, float],
                        names) -> Optional[dict]:
    """
    Normalised fractional contributions of each component of a multi-
    exponential decay.

    For a decay I(t) = SUM_i A_i exp(-t / tau_i) there are two different and
    equally standard ways to say "how much" component i contributes, and they
    are NOT interchangeable:

    alpha_i = A_i / SUM_j A_j
        Amplitude (pre-exponential) fraction.  Proportional to the NUMBER of
        emitters in that state, because each contributes A_i to the decay at
        t = 0.  This is the one to quote when asking what fraction of the
        molecules are in a given conformation or binding state.

    f_i = A_i tau_i / SUM_j A_j tau_j
        Intensity (photon) fraction.  The fraction of the total emitted
        photons that come from component i, since the integrated area of an
        exponential is A_i tau_i.  This is what a steady-state intensity
        measurement is sensitive to, and it is what a bulk fluorimeter would
        report.

    A dim, long-lived species and a bright, short-lived one can have very
    different alpha and f, so both are reported and labelled rather than
    picking one and calling it "the fraction".

    Also returns both mean lifetimes:

        tau_mean_amp = SUM A_i tau_i / SUM A_i          (amplitude-weighted)
        tau_mean_int = SUM A_i tau_i^2 / SUM A_i tau_i  (intensity-weighted)

    The amplitude-weighted mean is the one proportional to the average time a
    molecule spends excited, and is what the existing ``tau_mean`` field has
    always reported.

    Returns None when the decomposition is not meaningful: fewer than two
    components, a missing A/tau pair, or non-positive amplitudes or lifetimes
    (which a fit can reach if the bounds permit it, and which would make the
    fractions meaningless rather than merely imprecise).
    """
    idx = sorted({int(m.group(2)) for nm in names
                  if (m := _COMPONENT_RE.match(nm))})
    comps = [(i, float(values[f"A{i}"]), float(values[f"tau{i}"]))
             for i in idx
             if f"A{i}" in values and f"tau{i}" in values]
    if len(comps) < 2:
        return None
    if any(a <= 0 or t <= 0 for _, a, t in comps):
        return None

    sum_a  = sum(a for _, a, _ in comps)
    sum_at = sum(a * t for _, a, t in comps)
    if sum_a <= 0 or sum_at <= 0:
        return None

    return {
        "indices":      [i for i, _, _ in comps],
        "alpha":        {i: a / sum_a          for i, a, _ in comps},
        "f":            {i: a * t / sum_at     for i, a, t in comps},
        "tau_mean_amp": sum_at / sum_a,
        "tau_mean_int": sum(a * t * t for _, a, t in comps) / sum_at,
    }


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_decay(
    t_ns: np.ndarray,
    counts: np.ndarray,
    fit_start_ns: float,
    fit_end_ns: float,
    drop_edges: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select the fit window from a TCSPC histogram and return the data to fit.

    The first and last histogram bins are time-tagger catch-all artifacts
    (untimed / clamped records), so by default they are dropped before the
    window is applied.  The window is then [fit_start_ns, fit_end_ns]
    inclusive, in absolute microtime (ns).

    Returns
    -------
    t_ns_win : np.ndarray   absolute bin times within the window (ns)
    counts_win : np.ndarray  counts within the window
    t_rel : np.ndarray       t_ns_win re-zeroed to the window start (ns); this
                             is what the model is evaluated on, so the amplitude
                             refers to the window start.
    """
    t_ns   = np.asarray(t_ns, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if t_ns.shape != counts.shape:
        raise ValueError("t_ns and counts must have the same shape.")
    if fit_end_ns <= fit_start_ns:
        raise ValueError("fit_end_ns must be greater than fit_start_ns.")

    keep = np.ones(len(counts), dtype=bool)
    if drop_edges and len(counts) >= 2:
        keep[0] = False
        keep[-1] = False
    keep &= (t_ns >= fit_start_ns) & (t_ns <= fit_end_ns)

    t_win = t_ns[keep]
    c_win = counts[keep]
    if len(t_win) < 3:
        raise ValueError(
            "Fit window contains fewer than 3 bins after dropping edge bins. "
            "Widen the window or use more histogram bins."
        )
    t_rel = t_win - t_win[0]
    return t_win, c_win, t_rel


def default_window_arrays(t_ns: np.ndarray,
                          counts: np.ndarray) -> Tuple[float, float]:
    """
    Default fit window for an already-built decay: peak to last usable bin.

    Takes arrays rather than an FCSData so the same rule applies whether the
    histogram came from photon records or was read back from an exported CSV.
    :func:`default_window` is the FCSData-shaped wrapper around this.

    The peak is found on the edge-excluded histogram, so the bin-0 spike (the
    TCSPC catch-all for untimed photons) never wins.  The window end is the
    last usable bin's time; the final-bin artifact is dropped separately by
    prepare_decay.
    """
    t_ns = np.asarray(t_ns, dtype=np.float64)
    core = np.asarray(counts, dtype=np.float64).copy()
    if len(core) >= 2:
        core[0] = 0.0
        core[-1] = 0.0
    peak_ns = float(t_ns[int(np.argmax(core))])
    end_ns  = float(t_ns[-2]) if len(t_ns) >= 2 else float(t_ns[-1])
    if end_ns <= peak_ns:
        end_ns = float(t_ns[-1])
    return peak_ns, end_ns


def default_window(d: FCSData, channel: int, n_bins: int) -> Tuple[float, float]:
    """Default fit window for a channel of a photon dataset."""
    t_ns, counts = d.lifetime_histogram(channel=channel, n_bins=n_bins)
    return default_window_arrays(t_ns, counts)


# ── Reading an exported lifetime histogram ────────────────────────────────────

def load_lifetime_csv(path) -> Tuple[np.ndarray, Dict[str, np.ndarray],
                                     Optional[np.ndarray],
                                     Dict[str, np.ndarray], Dict[str, str]]:
    """
    Read a lifetime decay exported by the "Plot lifetime decay" task.

    This is the lifetime counterpart of fcs_fit.load_correlation_csv, and it
    exists for the same reason: once a decay has been exported you should be
    able to fit it without carrying the .fcs photon records around.  A 1024-bin
    histogram is a few tens of kB; the TTTR file it came from is often hundreds
    of MB.

    Two export shapes are accepted, because the plotting task writes both:

    * from photon data (.fcs) --  ``time_ns``, ``ch1_counts``, ``ch2_counts``
      A channel that was never recorded is present but blank, so a column of
      all-NaN means "no data", not "zero counts", and is skipped.
    * from an ISS decay (.ifx) -- ``time_ns``, ``intensity``, and ``irf`` when
      the file carried a measured instrument response.

    Precision
    ---------
    Counts round-trip EXACTLY -- they are integers well inside 10 significant
    figures.  The time axis does not: fcs_export writes 10 sig figs, so a bin
    time comes back agreeing to about 1 part in 1e10 rather than bit-for-bit.
    At a 48 ps bin width that is a discrepancy of order 0.1 fs, ten orders of
    magnitude below the bin itself and far below the Poisson noise on any bin,
    and a fit run from the CSV reproduces one run from the photon records to
    better than 1e-9 ns.  Worth knowing before comparing two fits at full
    printed precision and wondering why the last digits differ.

    Returns
    -------
    t_ns : np.ndarray
        Bin times in ns.
    series : dict of label -> counts
        ``{"Ch1": ..., "Ch2": ...}`` for a photon export (only the channels
        that hold data), or ``{"decay": ...}`` for an .ifx export.
    irf : np.ndarray or None
        Instrument response, when the export carried one.
    columns : dict
        Every column found, by name.
    meta : dict
        Header ``# key : value`` fields, e.g. 'source file', 'n_bins',
        'laser_period_ns', 'channels_recorded'.

    Raises
    ------
    ValueError
        If the file is not a lifetime decay export, with a message that says
        what was found instead.
    """
    path = Path(path)
    meta, columns = fcs_export.read_export(path)

    if "time_ns" not in columns:
        # A fit CURVE export uses t_ns and carries a 'fit' column.  Naming it
        # specifically beats "no time_ns column", because fitting the output of
        # a previous fit is a plausible mistake to make from a file browser.
        if "t_ns" in columns and "fit" in columns:
            raise ValueError(
                f"{path.name} looks like a lifetime FIT curve, not a decay "
                f"histogram.  Fit the exported histogram from 'Plot lifetime "
                f"decay' instead."
            )
        raise ValueError(
            f"{path.name} has no 'time_ns' column - is this a lifetime decay "
            f"export?"
        )

    t_ns = np.asarray(columns["time_ns"], dtype=np.float64)
    if len(t_ns) < 2:
        raise ValueError(f"{path.name} has fewer than 2 time bins.")

    def _has_data(arr) -> bool:
        """A column that exists but is entirely blank carries no data."""
        a = np.asarray(arr, dtype=np.float64)
        return bool(a.size) and not bool(np.all(np.isnan(a)))

    series: Dict[str, np.ndarray] = {}
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

    irf = columns.get("irf")
    if irf is not None and _has_data(irf):
        irf = np.asarray(irf, dtype=np.float64)
    else:
        irf = None

    return t_ns, series, irf, columns, meta


class LoadedDecay:
    """
    A decay read back from an exported CSV, shaped like the piece of FCSData
    that the lifetime plotting and gating code actually uses.

    Why an adapter rather than array-shaped copies of those functions: the
    surface they touch is three attributes wide -- ``filepath``,
    ``laser_period_ns`` and ``lifetime_histogram()``.  Satisfying those means
    :func:`fcs_lifetime.plot_lifetime`, :func:`fcs_lifetime.select_gate` and
    :func:`fcs_lifetime._draw_histogram` all work on a CSV-sourced decay with
    no changes at all, so the interactive window picker behaves identically
    whether the histogram came from photon records or from disk.
    ``fcs_lifetime._load`` already passes any non-path object straight through,
    and the module already duck-types .ifx decays via ``kind``, so this fits
    the existing convention rather than bending it.

    Rebinning
    ---------
    The CSV is already binned.  When a caller asks for FEWER bins by an exact
    integer factor the stored counts are summed in blocks, which reproduces the
    histogram the photon path would have built at that resolution bit-for-bit
    -- summing adjacent bins of a histogram is the same operation as histogram-
    ing into wider bins.  Any other request (more bins than were exported, or a
    non-integer factor) cannot be honoured: the information is not in the file.
    Rather than silently returning a different resolution than was asked for,
    which would make a fit report a bin count it did not use, that raises.
    """

    kind = "loaded_decay"

    def __init__(self, path, t_ns, series, irf=None, meta=None):
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

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def source_name(self) -> str:
        """Name of the .fcs/.ifx the decay was originally measured from."""
        return (self.meta.get("source file")
                or self.meta.get("source")
                or self.filepath.name)

    @property
    def n_bins(self) -> int:
        return len(self.t_ns)

    @property
    def laser_period_ns(self) -> float:
        """
        Laser period in ns.

        Taken from the export header when present.  Otherwise estimated as one
        bin past the last bin's left edge, which is the period the exporter
        divided into these bins in the first place.
        """
        try:
            return float(self.meta["laser_period_ns"])
        except (KeyError, TypeError, ValueError):
            if len(self.t_ns) >= 2:
                width = float(self.t_ns[1] - self.t_ns[0])
                return float(self.t_ns[-1]) + width
            return float("nan")

    @property
    def channel_summary(self) -> str:
        if self._single:
            return "decay"
        if len(self.channels) == 1:
            return f"Ch{self.channels[0]} only"
        return " + ".join(f"Ch{c}" for c in self.channels)

    @property
    def has_ch1(self) -> bool:
        return 1 in self.channels

    @property
    def has_ch2(self) -> bool:
        return 2 in self.channels

    def counts(self, channel=None) -> np.ndarray:
        """Stored counts for *channel* at the exported resolution."""
        if self._single:
            return self.series["decay"]
        key = f"Ch{channel if channel is not None else self.channels[0]}"
        if key not in self.series:
            raise ValueError(
                f"{self.filepath.name} holds {self.channel_summary}; "
                f"there is no {key} decay in this export."
            )
        return self.series[key]

    # ── The FCSData-shaped call ──────────────────────────────────────────────

    @property
    def has_irf(self) -> bool:
        """
        True when the export carries an IRF column.

        fcs_lifetime_recon gates reconvolution on this attribute, so an export
        that saved its IRF stays eligible for a reconvolution fit instead of
        being silently demoted to tail fitting.
        """
        return self.irf is not None and len(np.asarray(self.irf)) > 0

    def decay_curve(self, channel: Optional[int] = None):
        """
        Return ``(t_ns, decay, irf)`` at the stored resolution.

        The third member of the trio fcs_lifetime_recon reads off a decay
        object -- the other two being ``filepath`` and ``has_irf`` -- so a
        loaded decay can be reconvolution-fitted with no change there beyond
        admitting its ``kind``.

        Choosing the channel
        --------------------
        An .ifx decay is one curve, so ``LifetimeData.decay_curve()`` takes no
        argument and the reconvolution fitter never had a channel to pick.  A
        two-channel export is different, and defaulting to the first channel
        would hand back a Ch1 decay to a caller that never said which detector
        it meant -- a fitted lifetime attributed to the wrong channel, with
        nothing in the result to show for it.  So an ambiguous request raises
        and the caller must choose.  A file holding one decay, labelled or not,
        is unambiguous and needs no argument.
        """
        if channel is None:
            if self._single or len(self.channels) == 1:
                channel = None if self._single else self.channels[0]
            else:
                raise ValueError(
                    f"{self.filepath.name} holds {self.channel_summary}; "
                    f"decay_curve() needs a channel to say which decay to "
                    f"return.  Pass channel=1 or channel=2."
                )
        counts = self.counts(channel)
        irf = (np.asarray(self.irf, dtype=np.float64).copy()
               if self.has_irf else None)
        return self.t_ns.copy(), counts.copy(), irf

    def valid_n_bins(self) -> list:
        """
        Bin counts this decay can actually be rebinned to.

        Only exact integer divisors of the stored resolution qualify; see
        :meth:`lifetime_histogram`.  The GUI offers this list rather than the
        full ``fcs_lifetime._VALID_N_BINS`` so a resolution the file cannot
        produce is never selectable, instead of being chosen and then rejected
        with an error two screens later.
        """
        stored = self.n_bins
        opts = [n for n in fcs_lifetime._VALID_N_BINS
                if n <= stored and stored % n == 0]
        if stored not in opts:
            opts.append(stored)
        return sorted(set(opts))

    def gate_n_bins(self, preferred: int = 512) -> int:
        """
        Resolution for the interactive gate picker.

        select_gate() asks for 512 bins by default, which a decay exported at
        256 cannot supply.  This returns the closest valid choice at or below
        *preferred*, so the picker opens rather than raising.
        """
        opts = [n for n in self.valid_n_bins() if n <= preferred]
        return max(opts) if opts else min(self.valid_n_bins())

    def lifetime_histogram(self, channel: int = 1,
                           n_bins: int = None):
        """
        Return ``(bin_times_ns, counts)``, rebinning if asked for fewer bins.

        Signature matches FCSData.lifetime_histogram so the plotting and gating
        code cannot tell the difference.
        """
        counts = self.counts(None if self._single else channel)
        stored = len(counts)
        if n_bins is None or int(n_bins) == stored:
            return self.t_ns.copy(), counts.copy()

        n_bins = int(n_bins)
        if n_bins > stored or stored % n_bins:
            raise ValueError(
                f"{self.filepath.name} was exported with {stored} bins; "
                f"{n_bins} bins cannot be derived from it.  Only an exact "
                f"integer rebin to fewer bins is possible "
                f"({', '.join(str(stored // f) for f in (1, 2, 4, 8, 16) if stored % f == 0)}"
                f", ...).  Re-export the decay at {n_bins} bins to fit at that "
                f"resolution."
            )
        factor = stored // n_bins
        binned = counts.reshape(n_bins, factor).sum(axis=1)
        times  = self.t_ns[::factor][:n_bins]
        return times.copy(), binned


def load_decay_object(path) -> LoadedDecay:
    """Read an exported lifetime decay and wrap it as a :class:`LoadedDecay`."""
    t_ns, series, irf, _cols, meta = load_lifetime_csv(path)
    return LoadedDecay(path, t_ns, series, irf, meta)


def discover_lifetime_csvs(folder) -> list:
    """
    Return the CSVs in *folder* that parse as lifetime decay exports.

    Matches on content rather than filename, exactly as
    fcs_fit._discover_correlation_csvs does, so a file the user renamed is
    still found and a correlation export sitting in the same analysis folder
    is still skipped.
    """
    folder = Path(folder)
    found = []
    if folder.exists():
        for p in sorted(folder.glob("*.csv")):
            try:
                load_lifetime_csv(p)
                found.append(p)
            except Exception:
                continue
    return found


# ── Initial-guess heuristics ──────────────────────────────────────────────────

def auto_guess_lifetime(
    model: FCSModel,
    t_rel: np.ndarray,
    counts: np.ndarray,
) -> Dict[str, float]:
    """
    Sensible starting guesses from the windowed decay.

    Background from the median of the last 10% of the window; amplitude from the
    first bin above background; lifetime from the 1/e crossing.  For the
    two-exponential model the amplitude is split fast/slow and the single-decay
    lifetime estimate is spread around to seed the two components.
    """
    g = model.defaults()
    c = np.asarray(counts, dtype=np.float64)
    t = np.asarray(t_rel, dtype=np.float64)
    if len(c) < 3:
        return g

    n_tail = max(3, len(c) // 10)
    offset = float(np.median(c[-n_tail:]))
    A0 = max(float(c[0]) - offset, 1.0)

    below = np.where((c - offset) < (A0 / np.e))[0]
    tau0 = float(t[below[0]]) if len(below) else float(t[-1] / 2.0)
    tau0 = max(tau0, 1e-2)

    names = set(model.param_names())
    if "offset" in names:
        g["offset"] = max(offset, 0.0)
    if {"A", "tau"} <= names:
        g["A"] = A0
        g["tau"] = tau0
    if {"A1", "tau1", "A2", "tau2"} <= names:
        g["A1"] = 0.6 * A0
        g["tau1"] = max(tau0 * 0.4, 1e-2)
        g["A2"] = 0.4 * A0
        g["tau2"] = tau0 * 1.8
    return g


# ── Fit core ──────────────────────────────────────────────────────────────────

def fit_lifetime(
    model: FCSModel,
    t_ns: np.ndarray,
    counts: np.ndarray,
    fit_start_ns: float,
    fit_end_ns: float,
    guesses: Dict[str, float],
    lowers: Dict[str, float],
    uppers: Dict[str, float],
    fixed: Dict[str, bool],
    weighted: bool = True,
    drop_edges: bool = True,
    channel: Optional[int] = None,
    n_bins: Optional[int] = None,
    maxfev: int = 20000,
) -> dict:
    """
    Weighted least-squares tail fit of ``model`` to a TCSPC decay, honouring
    per-parameter bounds and "fixed" flags.

    The model is evaluated on time re-zeroed to the window start.  When
    ``weighted`` is True (the default) the fit uses Poisson errors
    σ = √max(counts, 1) and parameter errors are absolute, so the reduced χ²
    is meaningful.

    Returns a result dict(values, 1σ errors,
    masked data, fit curve, residuals, R², χ², reduced χ²), plus lifetime
    extras: the absolute and relative time axes, the channel / n_bins / window,
    and — for the two-exponential model — the amplitude-weighted mean lifetime.
    """
    names = model.param_names()

    t_win, c_win, t_rel = prepare_decay(
        t_ns, counts, fit_start_ns, fit_end_ns, drop_edges=drop_edges)

    s = np.sqrt(np.maximum(c_win, 1.0)) if weighted else None

    free = [n for n in names if not fixed.get(n, False)]
    if not free:
        raise ValueError("At least one parameter must be free (not fixed).")
    fixed_vals = {n: guesses[n] for n in names if fixed.get(n, False)}

    def _model_free(tt, *free_vals):
        allv = dict(fixed_vals)
        for n, v in zip(free, free_vals):
            allv[n] = v
        return model.func(tt, **{n: allv[n] for n in names})

    p0 = [guesses[n] for n in free]
    lb = [lowers[n] for n in free]
    ub = [uppers[n] for n in free]
    # Nudge guesses inside the bounds so curve_fit doesn't reject p0.
    p0 = [min(max(p, lo), hi) for p, lo, hi in zip(p0, lb, ub)]

    popt, pcov = curve_fit(
        _model_free, t_rel, c_win, p0=p0, bounds=(lb, ub),
        sigma=s, absolute_sigma=(s is not None), maxfev=maxfev,
    )
    perr = np.sqrt(np.diag(pcov))

    values = dict(fixed_vals)
    errors = {n: 0.0 for n in fixed_vals}        # fixed params have no error
    for n, v, e in zip(free, popt, perr):
        values[n] = float(v)
        errors[n] = float(e)

    yfit  = model.func(t_rel, **{n: values[n] for n in names})
    resid = c_win - yfit

    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((c_win - c_win.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof    = len(c_win) - len(free)
    if s is not None and dof > 0:
        chi2     = float(np.sum((resid / s) ** 2))
        red_chi2 = chi2 / dof
    else:
        chi2 = red_chi2 = float("nan")

    # Derived: amplitude-weighted mean lifetime for the two-exponential model.
    # Amplitude-weighted mean lifetime and the per-component fractions.  This
    # used to test for exactly {A1, tau1, A2, tau2}, so a triple-exponential
    # fit reported no <tau> at all; component_fractions handles any number of
    # components and returns None when the decomposition is not meaningful.
    fractions = component_fractions(values, names)
    tau_mean  = fractions["tau_mean_amp"] if fractions else float("nan")

    return {
        "model": model, "names": names, "free": free,
        "values": values, "errors": errors,
        "t_ns": t_win, "t_rel": t_rel, "counts": c_win,
        "fit": yfit, "resid": resid, "sigma": s,
        "guesses": dict(guesses), "lowers": dict(lowers),
        "uppers": dict(uppers), "fixed": dict(fixed),
        "r2": r2, "chi2": chi2, "red_chi2": red_chi2,
        "ss_res": ss_res, "n_points": len(c_win), "dof": dof,
        "weighted": s is not None,
        "pcov": pcov,
        "channel": channel, "n_bins": n_bins,
        "fit_start_ns": float(fit_start_ns), "fit_end_ns": float(fit_end_ns),
        "drop_edges": drop_edges, "tau_mean": tau_mean,
        "fractions": fractions,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_lifetime_fit(
    result: dict,
    source_name: str,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot the decay (log-y), the fitted curve, and the residuals."""
    model = result["model"]
    t_ns  = result["t_ns"]
    t_rel = result["t_rel"]

    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        layout="constrained",
    )

    # Data + smooth fit curve (log-y)
    ax.semilogy(t_ns, np.maximum(result["counts"], 0.1), linestyle="none",
                marker=".", markersize=3, color="black", label="data")
    t_dense_rel = np.linspace(t_rel.min(), t_rel.max(), 600)
    y_dense = model.func(t_dense_rel, **{n: result["values"][n] for n in result["names"]})
    ax.semilogy(t_dense_rel + t_ns[0], np.maximum(y_dense, 0.1),
                color="tomato", linewidth=1.6, label="fit")
    ax.set_ylabel("Photon counts", fontsize=12)

    # Parameter summary box
    lines = []
    for n in result["names"]:
        val = result["values"][n]
        err = result["errors"][n]
        unit = next((p.unit for p in model.params if p.name == n), "")
        tag = "  (fixed)" if result["fixed"].get(n) else f" ± {err:.3g}"
        lines.append(f"{n} = {val:.4g}{tag} {unit}".rstrip())
    # Normalised component fractions, one compact row per component:
    #   α = amplitude fraction (fraction of MOLECULES)
    #   f = intensity fraction (fraction of PHOTONS)
    # Both on one line so a triple exponential adds three rows, not six.
    fr = result.get("fractions")
    if fr:
        lines.append("")
        lines.append("α = amplitude, f = intensity")
        for i in fr["indices"]:
            lines.append(f"  {i}: α = {fr['alpha'][i]:.3f}   f = {fr['f'][i]:.3f}")
    if np.isfinite(result.get("tau_mean", float("nan"))):
        lines.append(f"⟨τ⟩ = {result['tau_mean']:.4g} ns  (ampl.)")
    if fr:
        lines.append(f"⟨τ⟩ = {fr['tau_mean_int']:.4g} ns  (int.)")
    gof = (f"red. χ² = {result['red_chi2']:.3g}"
           if result["weighted"] else f"R² = {result['r2']:.4f}")
    lines.append(gof)
    ax.text(0.98, 0.95, "\n".join(lines), transform=ax.transAxes,
            ha="right", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))

    ax.legend(loc="lower left", fontsize=10, framealpha=0.85)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.3, alpha=0.3)

    ch = result.get("channel")
    ch_str = f"Ch{ch}  ·  " if ch else ""
    title = f"Lifetime fit — {model.name}"
    subtitle = (f"{source_name}  ·  {ch_str}{result['n_points']} points  ·  "
                f"window {result['fit_start_ns']:.2f}–{result['fit_end_ns']:.2f} ns"
                f"  ·  {model.formula}")
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)

    # Residuals (weighted residuals if a Poisson σ was used)
    res = result["resid"]
    if result["weighted"]:
        res = res / result["sigma"]
        axr.set_ylabel("resid/σ", fontsize=10)
    else:
        axr.set_ylabel("resid", fontsize=10)
    axr.plot(t_ns, res, linestyle="none", marker=".",
             markersize=3, color="steelblue")
    axr.axhline(0, color="grey", linewidth=0.8)
    axr.set_xlabel("Arrival time within laser cycle (ns)", fontsize=12)
    axr.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    axr.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2g}"))

    if show:
        fcs_plottools.show_figure(fig, np.array([ax, axr]))
    return fig, np.array([ax, axr])


# ── Export ────────────────────────────────────────────────────────────────────

def _lifetime_file_stem(name_override: Optional[str] = None,
                        when: Optional[datetime] = None) -> str:
    """
    Shared stem for one lifetime fit's output files.

    Mirrors fcs_fit._fit_file_stem: timestamp, then the user's label.  The
    files used to be named lifetime_fit_report.txt / lifetime_fit_curve.csv,
    identical for every fit and distinguished only by their folder -- which
    Excel cannot cope with, since it refuses to hold two workbooks of the same
    filename open at once regardless of path.  That was harmless while lifetime
    fits wrote no spreadsheet; it stops being harmless now that they do.
    """
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    slug = _slug_name(name_override)
    return f"{stamp}_{slug}" if slug else f"lifetimefit_{stamp}"


def export_lifetime_fit(result: dict, source_path: str | Path,
                        name_override: Optional[str] = None,
                        ) -> Tuple[Path, Path, Path]:
    """
    Write a lifetime fit's report (.txt), fitted curve (.csv) and parameter
    table (.csv + .xlsx) to the 'fits' folder.

    Returns (report_path, curve_path, params_path).

    Both the folder and the filenames carry the timestamp and, when given,
    *name_override*, exactly as the correlation global fit does.
    """
    source_path = Path(source_path)
    model = result["model"]
    out_dir = _new_fit_dir(source_path, name_override)
    stem = _lifetime_file_stem(name_override)
    ch = result.get("channel")
    report_path = out_dir / f"{stem}_report.txt"
    curve_path  = out_dir / f"{stem}_curve.csv"
    params_path = out_dir / f"{stem}_params.csv"

    # ── Report ────────────────────────────────────────────────────────────────
    L: list[str] = []
    L.append("Lifetime fit report")
    L.append("=" * 60)
    L.append(f"source     : {source_path.name}")
    if result.get("origin") and result["origin"] != source_path.name:
        # Fitting a saved decay: name the file the photons were measured in,
        # so the report identifies the measurement and not just the CSV that
        # happened to be on disk.
        L.append(f"measured from : {result['origin']}")
    L.append(f"model      : {model.name}  [{model.key}]")
    L.append(f"formula    : {model.formula}")
    L.append(f"fitted     : {datetime.now().isoformat(timespec='seconds')}")
    if ch is not None:
        L.append(f"channel    : Ch{ch}")
    if result.get("n_bins") is not None:
        L.append(f"histogram  : {result['n_bins']} bins")
    L.append(f"fit window : {result['fit_start_ns']:.4f} – {result['fit_end_ns']:.4f} ns"
             f"  (first/last bins {'dropped' if result['drop_edges'] else 'kept'})")
    L.append(f"points     : {result['n_points']}   "
             f"free params : {len(result['free'])}   dof : {result['dof']}")
    L.append(f"weighted   : {'yes (Poisson σ = √counts)' if result['weighted'] else 'no'}")
    L.append("")
    L.append("Parameters")
    L.append("-" * 60)
    L.append(f"{'name':<8}{'value':>14}{'std err':>14}  {'unit':<7} "
             f"{'bounds':>22}  fixed")
    for n in result["names"]:
        p_unit = next((p.unit for p in model.params if p.name == n), "")
        val = result["values"][n]
        err = result["errors"][n]
        lo  = result["lowers"][n]
        hi  = result["uppers"][n]
        is_fixed = result["fixed"].get(n, False)
        err_str = f"{'—':>14}" if is_fixed else f"{err:>14.6g}"
        bnd = f"[{_fmt(lo)}, {_fmt(hi)}]"
        L.append(f"{n:<8}{val:>14.6g}{err_str}  {p_unit:<7} "
                 f"{bnd:>22}  {'yes' if is_fixed else 'no'}")
    fr = result.get("fractions")
    if fr:
        L.append("")
        L.append("Normalised component fractions")
        L.append("-" * 60)
        L.append("  α = A_i / ΣA        amplitude fraction  (fraction of molecules)")
        L.append("  f = A_i τ_i / ΣAτ   intensity fraction  (fraction of photons)")
        L.append("")
        L.append(f"{'comp':<8}{'alpha':>12}{'f':>12}{'A':>14}{'tau (ns)':>14}")
        for i in fr["indices"]:
            L.append(f"{i:<8}{fr['alpha'][i]:>12.4f}{fr['f'][i]:>12.4f}"
                     f"{result['values'][f'A{i}']:>14.6g}"
                     f"{result['values'][f'tau{i}']:>14.6g}")
    if np.isfinite(result.get("tau_mean", float("nan"))):
        L.append("")
        L.append(f"derived    : amplitude-weighted mean lifetime "
                 f"⟨τ⟩ = {result['tau_mean']:.6g} ns")
        if fr:
            L.append(f"             intensity-weighted mean lifetime "
                     f"⟨τ⟩ = {fr['tau_mean_int']:.6g} ns")
    L.append("")
    L.append("Goodness of fit")
    L.append("-" * 60)
    L.append(f"  SS_res     : {result['ss_res']:.6g}")
    L.append(f"  R^2        : {result['r2']:.6f}")
    if result["weighted"]:
        L.append(f"  chi^2      : {result['chi2']:.6g}")
        L.append(f"  red. chi^2 : {result['red_chi2']:.6g}")
    L.append("")
    
    L.extend(fcs_fisher.format_report_lines(
        fcs_fisher.fisher_from_covariance(
            result.get("pcov"), result["free"], result["weighted"])))
    L.append("")
    
    report_path.write_text("\n".join(L), encoding="utf-8")

    # ── Curve CSV ─────────────────────────────────────────────────────────────
    cols = {
        "t_ns":     result["t_ns"],
        "t_rel_ns": result["t_rel"],
        "counts":   result["counts"],
        "fit":      result["fit"],
        "residual": result["resid"],
    }
    if result["weighted"]:
        cols["sigma"] = result["sigma"]
    names = list(cols.keys())
    with curve_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# Lifetime fit curve — {model.key}\n")
        fh.write(f"# source : {source_path.name}\n")
        if ch is not None:
            fh.write(f"# channel : Ch{ch}\n")
        fh.write(f"# exported : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(",".join(names) + "\n")
        for row in zip(*(cols[n] for n in names)):
            fh.write(",".join(f"{v:.10g}" for v in row) + "\n")

    print(f"[lifetime fit] wrote {report_path}")
    print(f"[lifetime fit] wrote {curve_path}")

    # ── Wide parameter table (one row) for spreadsheets ──────────────────────
    # Same shape as the global fit's *_params.csv: one row per fitted dataset,
    # each parameter a value + error pair with the "_err" suffix used
    # throughout the suite.  A lifetime fit is a single dataset, so there is
    # one row -- which is exactly what makes these files worth concatenating
    # across a series to plot a parameter against a variable.
    p_header = ["dataset", "channel", "model", "n_bins",
                "fit_start_ns", "fit_end_ns"]
    for nm in result["names"]:
        p_header += [nm, f"{nm}_err"]

    fr = result.get("fractions")
    if fr:
        for i in fr["indices"]:
            p_header += [f"alpha{i}", f"f{i}"]
        p_header += ["tau_mean_amp_ns", "tau_mean_int_ns"]

    p_header += ["weighted", "r2", "chi2", "red_chi2", "n_points", "dof"]

    row: list = [
        source_path.name,
        (f"Ch{ch}" if ch is not None else ""),
        model.key,
        (int(result["n_bins"]) if result.get("n_bins") is not None else ""),
        float(result["fit_start_ns"]),
        float(result["fit_end_ns"]),
    ]
    for nm in result["names"]:
        row.append(float(result["values"][nm]))
        # A fixed parameter has no fitted error.  NaN, not 0: openpyxl renders
        # it as an empty cell and read_export maps a blank back to NaN, so it
        # round-trips as "not estimated" rather than "estimated to be zero".
        row.append(float("nan") if result["fixed"].get(nm)
                   else float(result["errors"][nm]))
    if fr:
        for i in fr["indices"]:
            row.append(float(fr["alpha"][i]))
            row.append(float(fr["f"][i]))
        row.append(float(fr["tau_mean_amp"]))
        row.append(float(fr["tau_mean_int"]))
    row += [
        "yes" if result["weighted"] else "no",
        float(result["r2"]),
        float(result["chi2"]) if result["weighted"] else float("nan"),
        float(result["red_chi2"]) if result["weighted"] else float("nan"),
        int(result["n_points"]),
        int(result["dof"]),
    ]

    comments: list[str] = []
    comments.append("Lifetime fit - parameter table")
    comments.append(f"source : {source_path.name}")
    comments.append(f"model : {model.name} [{model.key}]")
    comments.append(f"formula : {model.formula}")
    comments.append(f"exported : {datetime.now().isoformat(timespec='seconds')}")
    comments.append(f"weighted : {'yes (Poisson sigma = sqrt(counts))' if result['weighted'] else 'no'}")
    fixed_names = [nm for nm in result["names"] if result["fixed"].get(nm)]
    comments.append(f"fixed : {', '.join(fixed_names) if fixed_names else '(none)'}")
    comments.append("units : tau in ns, fit window in ns")
    # Distinct keys, not four lines all called "note".  fcs_export.read_export
    # parses "# key : value" into a dict, so repeated keys overwrite each other
    # and only the last note would survive a round trip.
    comments.append("note_fixed : blank *_err means the parameter was held fixed")
    if fr:
        comments.append("note_alpha : alpha_i = A_i / sum(A) - amplitude "
                        "fraction (fraction of molecules)")
        comments.append("note_f : f_i = A_i*tau_i / sum(A*tau) - intensity "
                        "fraction (fraction of photons)")
        comments.append("note_tau_mean : tau_mean_amp = sum(A*tau)/sum(A) ; "
                        "tau_mean_int = sum(A*tau^2)/sum(A*tau)")

    _write_params_table(params_path, comments, p_header, [row],
                        log_tag="lifetime fit")

    return report_path, curve_path, params_path

# ── GUI: entry point and dialogs ──────────────────────────────────────────────

# Remembers the most recently used fit window (ns), so fitting a family of
# datasets over the same range doesn't require re-typing it each time.  Module-
# level → persists across files for the session.
_last_fit_window: dict = {"start": None, "end": None}

def _nearest_gate_line(x: float, lo: float, hi: float,
                       radius: float) -> Optional[str]:
    """
    Which gate line a click at *x* grabs: ``"lo"``, ``"hi"`` or None.

    Same rule as fcs_lifetime.select_gate: the nearer line, but only if the
    click landed within *radius* of it, so clicking in open space pans the plot
    instead of yanking a gate across the axes.
    """
    d_lo, d_hi = abs(x - lo), abs(x - hi)
    if min(d_lo, d_hi) > radius:
        return None
    return "lo" if d_lo <= d_hi else "hi"


def _move_gate_line(which: str, x: float, lo: float, hi: float,
                    t_min: float, t_max: float,
                    min_sep: float = 0.05) -> Tuple[float, float]:
    """
    Move one gate line to *x*, clamped to the axis and kept clear of the other.

    Returns the new ``(lo, hi)``.  The separation floor is what stops a drag
    from inverting the window, which would otherwise produce an empty fit range
    only discovered on the next screen.
    """
    x = float(np.clip(x, t_min, t_max))
    if which == "lo":
        return min(x, hi - min_sep), hi
    return lo, max(x, lo + min_sep)


def _lifetime_window_dialog(parent, datasets: list, on_done):
    """
    Screen — the fit window.

    ONE dataset takes the ordinary single-file route: :func:`_lifetime_data_dialog`,
    which offers the channel, the histogram resolution and the interactive gate
    picker, exactly as fitting a single decay always has.  There is nothing to
    overlay and no per-dataset choice to make, so asking about either is noise.

    SEVERAL datasets get the same interactivity against all of them at once:
    the overlay is drawn in this window with draggable gate lines, and the two
    entry boxes and the lines follow each other.  The alternative -- type
    numbers, press a button, judge a separate figure, close it, try again --
    makes you carry the picture in your head between steps and leaves a trail
    of throwaway figures.

    Calls ``on_done(datasets, windows)``: the datasets may have been re-binned
    or had their channel changed on the way through, so they come back rather
    than being assumed unchanged.
    """
    import tkinter as tk
    from tkinter import messagebox

    D = len(datasets)

    # ── One dataset: the ordinary single-decay flow ──────────────────────────
    if D == 1:
        ds = dict(datasets[0])

        def _got(channel, n_bins, t_ns, counts, fit_start, fit_end):
            ds["t_ns"], ds["counts"] = np.asarray(t_ns), np.asarray(counts)
            ds["channel"] = channel
            if n_bins:
                ds["n_bins"] = int(n_bins)
            # The channel is chosen again here, so the label has to follow it
            # rather than keep whatever the file picker guessed.
            base = Path(ds["path"]).stem
            ds["series"] = "decay" if ds.get("series") == "decay" else f"Ch{channel}"
            ds["name"] = f"{base}[{ds['series']}]"
            on_done([ds], [(float(fit_start), float(fit_end))])

        _lifetime_data_dialog(parent, ds["decay"], _got)
        return

    # ── Several datasets: interactive overlay ────────────────────────────────
    start0, end0 = default_window_for(datasets)
    t_min = min(float(d["t_ns"][0]) for d in datasets)
    t_max = max(float(d["t_ns"][-1]) for d in datasets)
    period = max(float(d.get("laser_period_ns") or t_max) for d in datasets)
    pick_radius = period * 0.015

    win = tk.Toplevel(parent)
    win.title("Lifetime global fit — fit window")
    win.geometry("980x680")
    win.minsize(760, 560)
    win.grab_set()

    tk.Label(win, text="Fit window", font=("Helvetica", 12, "bold"),
             pady=4).pack()
    tk.Label(win, text=f"{D} datasets overlaid.  Drag either dashed line, or "
                       f"type below.  The window starts past the LATEST peak "
                       f"across the datasets, so it begins after the rise of "
                       f"every decay.",
             font=("Helvetica", 9), fg="grey", wraplength=900,
             justify="left").pack(padx=12)

    # Built once and embedded, not popped up per click: the gate lines below
    # are live, so there is nothing to regenerate when the window changes.
    fig, ax = plot_decay_overlay(datasets, window=None, show=False)
    shade = ax.axvspan(start0, end0, color="orange", alpha=0.12, zorder=1)
    line_lo = ax.axvline(start0, color="darkorange", lw=1.6, ls="--", zorder=5)
    line_hi = ax.axvline(end0, color="darkorange", lw=1.6, ls="--", zorder=5)
    gate_text = ax.text(0.5, 1.01, "", transform=ax.transAxes, ha="center",
                        va="bottom", fontsize=9, color="darkorange")

    from matplotlib.backends.backend_tkagg import (
        FigureCanvasTkAgg, NavigationToolbar2Tk)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True, padx=8)
    NavigationToolbar2Tk(canvas, win).update()

    def _gate() -> Tuple[float, float]:
        return (float(line_lo.get_xdata()[0]), float(line_hi.get_xdata()[0]))

    def _redraw():
        nonlocal shade
        lo, hi = _gate()
        # Remove and redraw the span rather than reshaping it: Polygon.set_xy's
        # calling convention has changed across matplotlib versions, and
        # select_gate avoids it for the same reason.
        shade.remove()
        shade = ax.axvspan(lo, hi, color="orange", alpha=0.12, zorder=1)
        gate_text.set_text(f"Fit window: {lo:.2f} – {hi:.2f} ns")
        canvas.draw_idle()

    _drag = {"which": None}

    def _on_press(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        lo, hi = _gate()
        _drag["which"] = _nearest_gate_line(float(event.xdata), lo, hi,
                                            pick_radius)

    def _on_motion(event):
        if _drag["which"] is None or event.inaxes is not ax:
            return
        if event.xdata is None:
            return
        lo, hi = _move_gate_line(_drag["which"], float(event.xdata),
                                 *_gate(), t_min, t_max)
        line_lo.set_xdata([lo, lo])
        line_hi.set_xdata([hi, hi])
        _redraw()
        _sync_entries()

    def _on_release(_event):
        _drag["which"] = None

    canvas.mpl_connect("button_press_event", _on_press)
    canvas.mpl_connect("motion_notify_event", _on_motion)
    canvas.mpl_connect("button_release_event", _on_release)

    ctrl = tk.Frame(win)
    ctrl.pack(fill="x", padx=12, pady=(4, 2))
    tk.Label(ctrl, text="Start (ns):", anchor="e", width=10).pack(side="left")
    s_var = tk.StringVar(value=f"{start0:.2f}")
    s_entry = tk.Entry(ctrl, textvariable=s_var, width=9)
    s_entry.pack(side="left", padx=(2, 16))
    tk.Label(ctrl, text="End (ns):", anchor="e", width=10).pack(side="left")
    e_var = tk.StringVar(value=f"{end0:.2f}")
    e_entry = tk.Entry(ctrl, textvariable=e_var, width=9)
    e_entry.pack(side="left", padx=2)

    def _sync_entries():
        lo, hi = _gate()
        s_var.set(f"{lo:.2f}")
        e_var.set(f"{hi:.2f}")

    def _sync_lines(*_a):
        try:
            lo, hi = float(s_var.get()), float(e_var.get())
        except ValueError:
            return
        lo = float(np.clip(lo, t_min, t_max))
        hi = float(np.clip(hi, t_min, t_max))
        if lo >= hi:
            return
        line_lo.set_xdata([lo, lo])
        line_hi.set_xdata([hi, hi])
        _redraw()

    for widget in (s_entry, e_entry):
        widget.bind("<Return>", _sync_lines)
        widget.bind("<FocusOut>", _sync_lines)

    _redraw()

    # ── Per-dataset override ─────────────────────────────────────────────────
    per_var = tk.BooleanVar(value=False)
    per_frame = tk.LabelFrame(win, text="Per-dataset windows (ns)",
                              padx=10, pady=4)
    per_s: Dict[str, object] = {}
    per_e: Dict[str, object] = {}

    def _pick_one(ds, sv, ev):
        """Pick this dataset's window on its own histogram, interactively."""
        try:
            gate = fcs_lifetime.select_gate(
                ds["decay"], n_bins=ds["decay"].gate_n_bins(512),
                channels=(ds["channel"] or ds["decay"].channels[0],),
                initial_gate=(float(sv.get()), float(ev.get())),
                title=f"Fit window — {ds['name']}")
        except Exception as exc:                # noqa: BLE001 — shown below
            messagebox.showerror("Cannot open the picker", str(exc), parent=win)
            return
        if gate:
            sv.set(f"{gate[0]:.2f}")
            ev.set(f"{gate[1]:.2f}")

    for r, ds in enumerate(datasets):
        tk.Label(per_frame, text=f"{r + 1}  {ds['name']}", anchor="w",
                 font=("Courier", 8), width=36).grid(row=r, column=0, sticky="w")
        sv = tk.StringVar(value=f"{start0:.2f}")
        ev = tk.StringVar(value=f"{end0:.2f}")
        tk.Entry(per_frame, textvariable=sv, width=9).grid(row=r, column=1, padx=3)
        tk.Entry(per_frame, textvariable=ev, width=9).grid(row=r, column=2, padx=3)
        tk.Button(per_frame, text="Pick…", pady=1,
                  command=lambda d=ds, a=sv, b=ev: _pick_one(d, a, b)
                  ).grid(row=r, column=3, padx=4)
        per_s[ds["name"]] = sv
        per_e[ds["name"]] = ev

    def _toggle_per():
        if per_var.get():
            for ds in datasets:
                per_s[ds["name"]].set(s_var.get())
                per_e[ds["name"]].set(e_var.get())
            per_frame.pack(fill="x", padx=12, pady=(0, 4))
        else:
            per_frame.pack_forget()

    tk.Checkbutton(win, text="Use a separate window per dataset  "
                             "(for decays whose histogram phase differs)",
                   variable=per_var, command=_toggle_per,
                   anchor="w").pack(fill="x", padx=14)
    tk.Label(win, text="Amplitudes are measured from the window start, so "
                       "per-dataset windows make A values incomparable between "
                       "datasets — lifetimes are unaffected.",
             font=("Helvetica", 8), fg="grey", wraplength=900,
             justify="left").pack(fill="x", padx=16)

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        try:
            if per_var.get():
                windows = [(float(per_s[d["name"]].get()),
                            float(per_e[d["name"]].get())) for d in datasets]
            else:
                lo, hi = _gate()
                windows = [(lo, hi)] * D
        except ValueError:
            messagebox.showerror("Invalid window",
                                 "Window bounds must be numbers (ns).",
                                 parent=win)
            return
        for ds, (a, b) in zip(datasets, windows):
            if b <= a:
                messagebox.showerror(
                    "Invalid window",
                    f"{ds['name']}: end ({b:g} ns) must be after start "
                    f"({a:g} ns).", parent=win)
                return
            if not (ds["t_ns"][0] <= a < ds["t_ns"][-1]):
                messagebox.showerror(
                    "Window outside the data",
                    f"{ds['name']}: its histogram spans "
                    f"{ds['t_ns'][0]:.3f}–{ds['t_ns'][-1]:.3f} ns, which does "
                    f"not contain a start of {a:g} ns.\n\nThe overlay shows "
                    f"which decay is out of step; tick 'Use a separate window "
                    f"per dataset' to gate it on its own.",
                    parent=win)
                return
        plt.close(fig)
        win.destroy()
        on_done(datasets, windows)

    def _cancel():
        plt.close(fig)
        win.destroy()

    tk.Button(btns, text="Next →", width=12, command=_next,
              pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=_cancel,
              pady=4).pack(side="left", padx=6)
    win.wait_window()


def _lifetime_global_setup_dialog(parent, model: FCSModel, datasets: list,
                                  windows: list):
    """Screen — per-parameter link / guess / bounds / fix / rule, then fit."""
    import tkinter as tk
    from tkinter import messagebox

    D = len(datasets)
    ds_names = [d["name"] for d in datasets]
    out_source = Path(datasets[0]["path"])

    win = tk.Toplevel(parent)
    win.title(f"Lifetime fit — {model.name}")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text=f"Lifetime {'global ' if D > 1 else ''}fit — {model.name}",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text=f"{D} dataset{'s' if D != 1 else ''}  ·  tail fit  ·  "
                       f"{model.formula}",
             font=("Helvetica", 9), fg="grey").pack(pady=(0, 4))

    if D > 1:
        fcs_fitdialogs.dataset_legend(win, ds_names).pack(
            fill="x", padx=12, pady=(0, 6))

    per_dataset_guesses = []
    for ds, (a, b) in zip(datasets, windows):
        t_win, c_win, t_rel = prepare_decay(ds["t_ns"], ds["counts"], a, b)
        per_dataset_guesses.append(auto_guess_lifetime(model, t_rel, c_win))
    guesses0 = fcs_globalfit.combined_guess_from(per_dataset_guesses,
                                                 model.defaults())

    ptable = fcs_fitdialogs.ParamTable(win, model, ds_names, guesses0)
    ptable.pack(fill="x")

    weight_var = tk.BooleanVar(value=True)
    tk.Checkbutton(win, text="Weight by Poisson σ = √counts",
                   variable=weight_var, anchor="w").pack(fill="x", padx=12,
                                                         pady=(6, 0))
    name_row = tk.Frame(win)
    name_row.pack(fill="x", padx=12, pady=(8, 0))
    tk.Label(name_row, text="Save as:", anchor="w",
             font=("Helvetica", 9)).pack(side="left")
    name_var = tk.StringVar(value="")
    tk.Entry(name_row, textvariable=name_var,
             font=("Helvetica", 9)).pack(side="left", fill="x", expand=True,
                                         padx=(4, 0))

    btns = tk.Frame(win)
    btns.pack(pady=10)

    def _do_fit():
        try:
            setup = ptable.read()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=win)
            return
        try:
            result = fit_lifetime_global(
                model, datasets, windows, setup["linked"], setup["guesses"],
                setup["lowers"], setup["uppers"], setup["fixed"],
                weighted=weight_var.get(), plans=setup["plans"])
        except Exception as e:                  # noqa: BLE001 — shown below
            messagebox.showerror("Fit failed", str(e), parent=win)
            return
        label = name_var.get().strip()
        win.destroy()
        report_path, _c, _p = export_lifetime_global(result, out_source,
                                                     name_override=label)
        fig, _axes = plot_lifetime_global_fit(result, show=False)
        fcs_plottools.save_figure(fig, report_path.with_suffix(""))
        shared = _lt_global_value_rows(result, "linked")
        lines = [f"{lab} = {val:.4g} ± {err}" for lab, val, err in shared]
        gof = (f"red. χ² = {result['red_chi2']:.3g}"
               if result["weighted"] else f"{D} dataset{'s' if D != 1 else ''}")
        messagebox.showinfo(
            "Lifetime fit complete",
            f"{model.name}\n\n"
            + ("\n".join(lines) if lines else "(no shared parameters)")
            + f"\n\n{gof}\n\nResults saved to:\n{report_path.parent}",
            parent=parent)
        fcs_plottools.show_figure(fig)

    tk.Button(btns, text="Fit", width=12, command=_do_fit,
              pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(side="left", padx=6)
    win.wait_window()


# ── Batch / global fitting ────────────────────────────────────────────────────

def lifetime_dataset_source():
    """
    Describe saved decays to the shared dataset picker.

    One row per CHANNEL: a two-channel export holds two decays with, in
    general, two different lifetimes, so they are two datasets to a fit.
    """
    def _discover(folder) -> list:
        out = []
        for path in discover_lifetime_csvs(Path(folder)):
            try:
                d = load_decay_object(path)
            except Exception:
                continue
            if d._single:
                out.append(fcs_fitdialogs.DatasetEntry(
                    path=path, series="decay", label=f"{path.name}  [decay]"))
            else:
                for ch in d.channels:
                    out.append(fcs_fitdialogs.DatasetEntry(
                        path=path, series=f"Ch{ch}",
                        label=f"{path.name}  [Ch{ch}]"))
        return out

    def _load(entry) -> dict:
        d = load_decay_object(entry.path)
        channel = (None if entry.series == "decay"
                   else int(str(entry.series)[2:]))
        t_ns, counts = d.lifetime_histogram(
            channel=channel if channel is not None else 1)
        return {
            "name": f"{entry.path.stem}[{entry.series}]",
            "path": entry.path, "series": entry.series, "channel": channel,
            "decay": d, "t_ns": t_ns, "counts": counts,
            "n_bins": int(d.n_bins), "laser_period_ns": float(d.laser_period_ns),
            "origin": d.source_name, "meta": d.meta,
        }

    return fcs_fitdialogs.DatasetSource(
        key="lifetime",
        title="Select lifetime decays",
        noun="lifetime decays",
        discover=_discover,
        load=_load,
        filetypes=(("CSV files", "*.csv"), ("Lifetime export", "*lifetime*.csv"),
                   ("All files", "*.*")),
        empty_hint="These are the CSVs written by the lifetime plotting task.",
    )


def default_window_for(datasets: list) -> Tuple[float, float]:
    """
    A common fit window that suits every decay in *datasets*.

    Taken from the LATEST peak across the datasets, so the window starts past
    the rise of all of them rather than of the first: a window that begins
    before another decay's peak would fit its rising edge as if it were decay,
    and a tail fit has no IRF with which to model a rise.
    """
    starts, ends = [], []
    for ds in datasets:
        t_ns, counts = ds["t_ns"], ds["counts"]
        peak = float(t_ns[int(np.argmax(counts))])
        starts.append(peak + 1.0)
        ends.append(float(t_ns[-1]))
    return max(starts), min(ends)


def plot_decay_overlay(datasets: list, window=None, show: bool = True):
    """
    Overlay every selected decay on one log-y axes.

    This is the screen that answers "can these share one fit window?".  Decays
    recorded with different time-tagger settings sit at different phases in the
    microtime histogram, and a common window that looks right for one of them
    can start halfway up another's rising edge.  Seeing them together makes
    that obvious in a way that a list of numbers does not.
    """
    colours = fcs_plottools.palette(len(datasets))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for ds, colour in zip(datasets, colours):
        ax.plot(ds["t_ns"], np.maximum(ds["counts"], 0.5), color=colour,
                linewidth=1.0, alpha=0.85, label=ds["name"])
    if window is not None:
        for x in window:
            ax.axvline(float(x), color="grey", linestyle="--", linewidth=1.0)
        ax.axvspan(float(window[0]), float(window[1]), color="grey", alpha=0.08)
    ax.set_yscale("log")
    ax.set_xlabel("Microtime (ns)", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(f"Decay overlay — {len(datasets)} dataset"
                 f"{'s' if len(datasets) != 1 else ''}"
                 + ("   ·   shaded = common fit window" if window else ""),
                 fontsize=10)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, framealpha=0.85)
    fig.tight_layout()
    if show:
        fcs_plottools.show_figure(fig)
    return fig, ax


def fit_lifetime_global(model: FCSModel, datasets: list, windows: list,
                        linked, guesses, lowers, uppers, fixed,
                        weighted: bool = True, plans=None,
                        drop_edges: bool = True, maxfev: int = 20000) -> dict:
    """
    Fit one decay model to several TCSPC histograms at once.

    *windows* is one ``(start_ns, end_ns)`` per dataset — the same pair
    repeated for a common window, or different pairs when the histograms sit at
    different phases.

    Each decay is windowed by :func:`prepare_decay` and the model evaluated on
    time re-zeroed to that window's start, exactly as the single fit does.  One
    consequence is worth stating: the amplitudes therefore refer to each
    dataset's OWN window start, so linking A across datasets with different
    windows links values measured from different origins.  Lifetimes are
    immune, which is why they are the parameters linked by default.
    """
    if len(windows) != len(datasets):
        raise ValueError(
            f"{len(windows)} windows for {len(datasets)} datasets.")

    prepared = []
    for ds, (start, end) in zip(datasets, windows):
        t_win, c_win, t_rel = prepare_decay(
            ds["t_ns"], ds["counts"], float(start), float(end),
            drop_edges=drop_edges)
        entry = dict(ds)
        entry.update({
            "x": t_rel, "y": c_win, "t_win": t_win,
            "sigma": np.sqrt(np.maximum(c_win, 1.0)),
            "fit_start": float(start), "fit_end": float(end),
        })
        prepared.append(entry)

    prepped = fcs_globalfit.prepare_datasets(prepared, weighted=weighted)

    def _predict(ds, t_rel, values):
        return model.func(t_rel, **values)

    out = fcs_globalfit.fit_linked(
        model.param_names(), prepped, _predict, linked, guesses, lowers,
        uppers, fixed, weighted=weighted, maxfev=maxfev, plans=plans)

    for entry in out["datasets"]:
        vals = entry["values"]
        extras: Dict[str, float] = {}
        if "A1" in vals and "A2" in vals and "tau1" in vals and "tau2" in vals:
            a1, a2, t1, t2 = vals["A1"], vals["A2"], vals["tau1"], vals["tau2"]
            denom = a1 + a2
            if denom > 0:
                extras["tau_mean_amp"] = (a1 * t1 + a2 * t2) / denom
            w = a1 * t1 + a2 * t2
            if w > 0:
                extras["f1"] = a1 * t1 / w
                extras["f2"] = a2 * t2 / w
        entry["extras"] = extras
    out["model"] = model
    return out


def export_lifetime_global(result: dict, source_path, name_override=None):
    """
    Write a global lifetime fit's report, fitted curves and parameter table.

    Returns (report_path, curves_path, params_path).  The parameter table uses
    a fixed schema, so rows from a bi-exponential fit line up whether or not a
    given dataset produced a usable mean lifetime.
    """
    source_path = Path(source_path)
    model = result["model"]
    out_dir = _new_fit_dir(source_path, name_override)
    stem = "lifetime_globalfit"
    if name_override:
        stem = f"{stem}_{_slug_name(name_override)}"
    report_path = out_dir / f"{stem}_report.txt"
    curves_path = out_dir / f"{stem}_curves.csv"
    params_path = out_dir / f"{stem}_params.csv"

    dsets = result["datasets"]
    names = result["names"]

    L: list = []
    L.append("Lifetime global fit report  (tail fit)")
    L.append("=" * 64)
    L.append(f"generated  : {datetime.now().isoformat(timespec='seconds')}")
    L.append(f"model      : {model.name}  [{model.key}]")
    L.append(f"formula    : {model.formula}")
    L.append(f"datasets   : {result['n_datasets']}")
    L.append(f"weighted   : {'yes (Poisson)' if result['weighted'] else 'no'}")
    L.append("")

    wins = {(round(d["fit_start"], 6), round(d["fit_end"], 6)) for d in dsets}
    L.append("Fit window")
    L.append("-" * 64)
    if len(wins) == 1:
        a, b = next(iter(wins))
        L.append(f"  common to all datasets: {a:.3f} – {b:.3f} ns")
    else:
        # Amplitudes are measured from the window start, so a per-dataset
        # window means they are measured from different origins.  Said here
        # rather than left for the reader to deduce from the numbers.
        L.append("  PER-DATASET — amplitudes refer to each window's own start,")
        L.append("  so A values are not directly comparable between datasets:")
        for d in dsets:
            L.append(f"    {d['name']:<34} {d['fit_start']:.3f} – "
                     f"{d['fit_end']:.3f} ns")
    L.append("")

    L.append("Parameter linking")
    L.append("-" * 64)
    for n in names:
        if result["fixed"].get(n):
            state = "fixed"
        elif result["linked"].get(n):
            state = "linked"
        elif all(not d.get("fixed", {}).get(n) and not d.get("linked", {}).get(n)
                 for d in dsets):
            state = "per-dataset"
        else:
            state = result["plan_summaries"].get(n, "per-dataset")
        L.append(f"  {n:<10} : {state}")
    L.append("")

    for kind, title in (("linked", "Shared values"), ("fixed", "Held values")):
        rows = _lt_global_value_rows(result, kind)
        if not rows:
            continue
        L.append(title)
        L.append("-" * 64)
        for label, val, err in rows:
            L.append(f"  {label:<30} = {val:.6g}"
                     + (f"  ± {err}" if kind == "linked" else "  (held)"))
        L.append("")

    if result["weighted"] and np.isfinite(result["red_chi2"]):
        L.append(f"Goodness of fit:  chi^2 = {result['chi2']:.6g}   "
                 f"red. chi^2 = {result['red_chi2']:.4g}   dof = {result['dof']}")
        L.append("")

    L.append("Per-dataset results")
    L.append("-" * 64)
    for d in dsets:
        L.append(f"  {d['name']}")
        L.append(f"    window {d['fit_start']:.3f} – {d['fit_end']:.3f} ns  ·  "
                 f"{d['n_bins']} bins  ·  {d['n_points']} points")
        for n in names:
            if result["linked"].get(n) or result["fixed"].get(n):
                continue
            tag = ("  (held)" if d.get("fixed", {}).get(n)
                   else "  (shared)" if d.get("linked", {}).get(n) else "")
            err = "—" if d.get("fixed", {}).get(n) else f"{d['errors'][n]:.6g}"
            L.append(f"    {n:<10} = {d['values'][n]:.6g}  ± {err}{tag}")
        for key, val in d["extras"].items():
            L.append(f"    {key:<20} = {val:.6g}")
        L.append(f"    R^2 = {d['r2']:.6f}")
        L.append("")

    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"[lifetime globalfit] wrote {report_path}")

    # ── Fitted curves, on ABSOLUTE microtime so they can be overlaid ─────────
    n_max = max(len(d["t_win"]) for d in dsets)
    cols: Dict[str, np.ndarray] = {}
    for d in dsets:
        for suffix, values in (("t_ns", d["t_win"]), ("counts", d["y"]),
                               ("fit", d["yfit"]), ("resid_w", d["resid"] /
                                (d["sigma"] if d.get("sigma") is not None else 1.0))):
            col = np.full(n_max, np.nan)
            col[:len(values)] = values
            cols[f"{suffix}_{d['name']}"] = col
    with curves_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# FCS analysis export — lifetime global fit curves\n")
        fh.write("# analysis : lifetime global fit\n")
        fh.write(f"# model : {model.name} [{model.key}]\n")
        fh.write(f"# datasets : {result['n_datasets']}\n")
        fh.write(f"# exported : {datetime.now().isoformat(timespec='seconds')}\n")
        fh.write(",".join(cols) + "\n")
        for row in zip(*cols.values()):
            fh.write(",".join(fcs_export.csv_value(v) for v in row) + "\n")
    print(f"[lifetime globalfit] wrote {curves_path}")

    # ── Parameter table (fixed schema) ───────────────────────────────────────
    extra_keys = ["tau_mean_amp", "f1", "f2"]
    present = [k for k in extra_keys if any(k in d["extras"] for d in dsets)]
    p_header = ["dataset", "source", "series", "n_bins",
                "fit_start_ns", "fit_end_ns"]
    for n in names:
        p_header += [n, f"{n}_err", f"{n}_role"]
    p_header += present + ["r2", "n_points"]

    rows = []
    for d in dsets:
        role = lambda n: ("held" if d.get("fixed", {}).get(n)
                          else "shared" if d.get("linked", {}).get(n)
                          else "free")
        row = [d["name"], Path(d["path"]).name, d.get("series") or "",
               d["n_bins"], d["fit_start"], d["fit_end"]]
        for n in names:
            row.append(float(d["values"][n]))
            row.append(float("nan") if role(n) == "held"
                       else float(d["errors"][n]))
            row.append(role(n))
        row += [d["extras"].get(k, float("nan")) for k in present]
        row += [d["r2"], d["n_points"]]
        rows.append(row)

    comments = ["Lifetime global fit (tail fit) — parameter table",
                f"model : {model.name} [{model.key}]",
                f"datasets : {result['n_datasets']}",
                f"weighted : {'yes (Poisson)' if result['weighted'] else 'no'}",
                f"exported : {datetime.now().isoformat(timespec='seconds')}"]
    if len(wins) > 1:
        comments.append("warning_windows : datasets use DIFFERENT fit windows; "
                        "amplitudes are measured from each window's own start "
                        "and are not directly comparable")
    comments.append("note_role : <param>_role is per dataset — free, shared "
                    "or held; a blank *_err means held")
    for n, summary in (result.get("plan_summaries") or {}).items():
        if result["fixed"].get(n) or result["linked"].get(n):
            continue
        if summary and not summary.startswith("free "):
            comments.append(f"rule_{n} : {summary}")

    _write_params_table(params_path, comments, p_header, rows,
                        log_tag="lifetime globalfit",
                        sheet_title="lifetime global fit")
    return report_path, curves_path, params_path


def _lt_global_value_rows(result: dict, kind: str) -> list:
    """Group rows for the report: ``(label, value, err)`` per shared/held set."""
    rows = []
    dsets = result["datasets"]
    for n in result["names"]:
        members = [i for i, d in enumerate(dsets)
                   if d.get(kind, {}).get(n, result[kind].get(n, False))]
        if not members:
            continue
        by_value: Dict[float, list] = {}
        for i in members:
            by_value.setdefault(dsets[i]["values"][n], []).append(i)
        whole = len(members) == len(dsets) and len(by_value) == 1
        for val, group in by_value.items():
            label = n if whole else \
                f"{n}[{'+'.join(dsets[i]['name'] for i in group)}]"
            rows.append((label, float(val),
                         f"{dsets[group[0]]['errors'][n]:.6g}"))
    return rows


def plot_lifetime_global_fit(result: dict, show: bool = True):
    """Overlay every windowed decay with its fit, plus weighted residuals."""
    model = result["model"]
    dsets = result["datasets"]
    colours = fcs_plottools.palette(len(dsets))
    fig, (ax, axr) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 5.5),
        gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    for d, colour in zip(dsets, colours):
        ax.plot(d["x"], np.maximum(d["y"], 0.5), linestyle="none", marker=".",
                markersize=2.5, color=colour, alpha=0.6, label=d["name"])
        ax.plot(d["x"], np.maximum(d["yfit"], 0.5), "-", color=colour,
                linewidth=1.4)
        sig = d["sigma"] if d.get("sigma") is not None else 1.0
        axr.plot(d["x"], d["resid"] / sig, linestyle="none", marker=".",
                 markersize=2, color=colour)
    ax.set_yscale("log")
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(f"Lifetime global fit — {model.name}  ·  "
                 f"{len(dsets)} dataset{'s' if len(dsets) != 1 else ''}",
                 fontsize=10)
    ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8, framealpha=0.85)
    axr.axhline(0.0, color="grey", linewidth=0.8)
    axr.set_xlabel("Time from window start (ns)", fontsize=12)
    axr.set_ylabel("resid/σ", fontsize=10)
    axr.grid(True, linestyle=":", linewidth=0.3, alpha=0.5)
    if show:
        fcs_plottools.show_figure(fig)
    return fig, np.array([ax, axr])


def run_lifetime_fit_dialog(fcs_data: Optional[FCSData] = None, parent=None):
    """
    Full GUI flow for lifetime modelling: choose a source, then channel /
    resolution / window, pick a model, set guesses and bounds, then fit, plot
    and export.

    Two sources are offered.  Photon records give the full choice of channel
    and histogram resolution.  A saved decay CSV is already binned, so the
    resolution choice narrows to the exact integer rebins the file can produce
    -- see :meth:`LoadedDecay.valid_n_bins` -- while the channel choice, the
    window entries and the interactive picker all behave exactly as they do for
    photon data, because :class:`LoadedDecay` satisfies the same three
    attributes fcs_lifetime uses.  With no photon data loaded the chooser is
    bypassed straight to the CSV browser, since there is nothing else to fit.
    """
    def _run(source):
        def _after_data(channel, n_bins, t_ns, counts, fit_start, fit_end):
            def _after_model(model: FCSModel):
                _lifetime_setup_dialog(parent, source, model, channel, n_bins,
                                       t_ns, counts, fit_start, fit_end)
            _select_lifetime_model_dialog(parent, _after_model)
        _lifetime_data_dialog(parent, source, _after_data)

    def _from_csv():
        # Saved decays always go through the multi-dataset path: with one
        # selected it IS the single fit, and adding a second is one click
        # rather than a different menu item.
        def _after_datasets(datasets: list):
            def _after_windows(chosen: list, windows: list):
                def _after_model(model: FCSModel):
                    _lifetime_global_setup_dialog(parent, model, chosen,
                                                  windows)
                _select_lifetime_model_dialog(parent, _after_model)
            _lifetime_window_dialog(parent, datasets, _after_windows)
        fcs_fitdialogs.select_datasets(
            parent, lifetime_dataset_source(),
            _default_decay_dir(fcs_data), _after_datasets)

    if fcs_data is None:
        _from_csv()
        return
    _lifetime_source_dialog(parent, fcs_data,
                            lambda: _run(fcs_data), _from_csv)


def run_lifetime_method_dialog(data=None, parent=None):
    """
    Choose tail fit vs IRF reconvolution, then hand off to that fitter.

    This screen used to live in fcs_main and fired straight off the workspace,
    before the user had said they wanted a lifetime fit at all.  It belongs
    after that choice, and it belongs here, in the module that owns lifetime
    fitting -- fcs_main should route to a workflow, not implement one.

    Both methods stay selectable whatever ``data`` is.  Each fitter has its own
    source screen and can open a saved decay, so greying reconvolution out
    because the file in the workspace has no IRF would hide the CSV browser
    behind a file the user may not have meant to fit at all.  Reconvolution
    without an IRF is a tail fit under another name, and fcs_lifetime_recon
    asks before running one.
    """
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Model lifetime decay")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="Model lifetime decay",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    if data is not None and getattr(data, "filepath", None) is not None:
        tk.Label(win, text=data.filepath.name,
                 font=("Helvetica", 9), fg="grey").pack()

    method_var = tk.StringVar(value="tail")
    frame = tk.LabelFrame(win, text="Method", padx=12, pady=8)
    frame.pack(fill="x", padx=14, pady=8)
    tk.Radiobutton(frame, text="Tail fit  (sum of exponentials, no IRF)",
                   variable=method_var, value="tail", anchor="w").pack(fill="x")
    tk.Radiobutton(frame, text="IRF reconvolution",
                   variable=method_var, value="recon", anchor="w").pack(fill="x")

    if data is not None and hasattr(data, "has_irf"):
        note = ("this decay carries an IRF" if data.has_irf
                else "this decay has no IRF — reconvolution would need a "
                     "saved decay that kept one")
    else:
        note = "reconvolution needs a decay whose export kept its IRF column"
    tk.Label(frame, text=note, font=("Helvetica", 8), fg="grey",
             wraplength=300, justify="left").pack(fill="x", pady=(2, 0))

    def _go():
        method = method_var.get()
        win.destroy()
        if method == "recon":
            # Imported here so the two lifetime fitters stay independent at
            # module scope; either can be used without loading the other.
            import fcs_lifetime_recon
            fcs_lifetime_recon.run_reconv_fit_dialog(data, parent=parent)
        else:
            run_lifetime_fit_dialog(data, parent=parent)

    btns = tk.Frame(win)
    btns.pack(pady=10)
    tk.Button(btns, text="Next →", width=12, command=_go,
              pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(side="left", padx=6)
    win.wait_window()


def _lifetime_source_dialog(parent, fcs_data, on_photons, on_csv):
    """Screen 0 — fit the active file's photon records, or a saved decay."""
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Lifetime fit — source")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="Lifetime fit — choose the data",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    body = tk.Frame(win, padx=16, pady=4)
    body.pack(fill="x")

    def _go(fn):
        win.destroy()
        fn()

    tk.Button(body, text="Active file — photon records", width=32, pady=6,
              command=lambda: _go(on_photons)).pack(pady=3)
    tk.Label(body, text=f"{fcs_data.filepath.name}\n"
                        f"any histogram resolution, binned now",
             font=("Helvetica", 8), fg="grey", justify="center").pack()

    tk.Button(body, text="Saved decay (CSV)…", width=32, pady=6,
              command=lambda: _go(on_csv)).pack(pady=(10, 3))
    tk.Label(body, text="resolution limited to exact rebins of the export",
             font=("Helvetica", 8), fg="grey").pack()

    tk.Button(win, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(pady=10)
    win.wait_window()


# Folder the decay CSV browser last worked in, remembered for the session.
_last_decay_browse_dir: Optional[str] = None


def _default_decay_dir(fcs_data=None) -> Optional[Path]:
    """Folder the decay CSV browser should open on: 'analysis' beside the
    active file, else the folder last browsed this session."""
    if fcs_data is not None and getattr(fcs_data, "filepath", None) is not None:
        start = Path(fcs_data.filepath).parent
        analysis = start / "analysis"
        return analysis if analysis.exists() else start
    if _last_decay_browse_dir:
        prev = Path(_last_decay_browse_dir)
        if prev.exists():
            return prev
    return None


def _lifetime_csv_dialog(parent, fcs_data, on_done):
    """Screen 0b — pick a saved decay CSV and hand back a LoadedDecay."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    global _last_decay_browse_dir

    init_dir = _default_decay_dir(fcs_data)
    paths = discover_lifetime_csvs(init_dir) if init_dir else []

    win = tk.Toplevel(parent)
    win.title("Lifetime fit — saved decay")
    win.geometry("620x430")
    win.minsize(520, 360)
    win.grab_set()

    tk.Label(win, text="Select a saved lifetime decay",
             font=("Helvetica", 12, "bold"), pady=6).pack()
    tk.Label(win, text="Decays exported by the 'Plot lifetime decay' task. "
                       "Files that do not parse as a decay are not listed.",
             font=("Helvetica", 9), fg="grey", wraplength=580,
             justify="left").pack()

    folder_var = tk.StringVar()

    def _update_folder():
        folder_var.set(f"Folder: {init_dir}" if init_dir
                       else "Folder: none chosen yet")
    _update_folder()
    tk.Label(win, textvariable=folder_var, font=("Courier", 8), fg="grey",
             wraplength=580, justify="left").pack()

    lb_frame = tk.Frame(win)
    lb_frame.pack(fill="both", expand=True, padx=12, pady=6)
    scroll = tk.Scrollbar(lb_frame, orient="vertical")
    listbox = tk.Listbox(lb_frame, yscrollcommand=scroll.set,
                         activestyle="none", font=("Courier", 9),
                         exportselection=False)
    scroll.config(command=listbox.yview)
    scroll.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)

    info = tk.StringVar(value="")
    tk.Label(win, textvariable=info, font=("Courier", 8), fg="grey",
             wraplength=580, justify="left").pack(padx=12, anchor="w")

    cache: dict = {}

    def _current():
        sel = listbox.curselection()
        if not sel:
            return None
        p = paths[sel[0]]
        if p not in cache:
            try:
                cache[p] = load_decay_object(p)
            except Exception as e:
                cache[p] = e
        got = cache[p]
        return None if isinstance(got, Exception) else got

    def _refresh(*_):
        d = _current()
        if d is None:
            info.set("")
            return
        info.set(f"{d.channel_summary}  ·  {d.n_bins} bins  ·  "
                 f"period {d.laser_period_ns:.3g} ns  ·  "
                 f"IRF {'yes' if d.has_irf else 'no'}  ·  "
                 f"from {d.source_name}")

    listbox.bind("<<ListboxSelect>>", _refresh)

    def _populate(select=0):
        listbox.delete(0, tk.END)
        for p in paths:
            listbox.insert(tk.END, p.name)
        if paths:
            listbox.selection_set(min(select, len(paths) - 1))
        _refresh()

    def _browse_folder():
        nonlocal init_dir, paths
        chosen = filedialog.askdirectory(
            title="Choose a folder of decay exports",
            initialdir=str(init_dir) if init_dir else "",
            mustexist=True, parent=win)
        if not chosen:
            return
        folder = Path(chosen)
        found = discover_lifetime_csvs(folder)
        if not found:
            messagebox.showinfo(
                "No decay exports",
                f"Nothing in '{folder.name}' parsed as a lifetime decay.",
                parent=win)
            return
        init_dir, paths = folder, found
        _update_folder()
        _populate()

    def _add_files():
        nonlocal init_dir, paths
        new = filedialog.askopenfilenames(
            title="Add decay export files",
            initialdir=str(init_dir) if init_dir else "",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            parent=win)
        if not new:
            return
        errors, added = [], 0
        for n in new:
            q = Path(n)
            try:
                load_lifetime_csv(q)
            except Exception as e:
                errors.append(f"{q.name}: {e}")
                continue
            if q not in paths:
                paths.append(q)
                added += 1
        init_dir = Path(new[0]).parent
        _update_folder()
        _populate(select=len(paths) - 1 if added else 0)
        if errors:
            messagebox.showerror("Some files could not be read",
                                 "\n\n".join(errors), parent=win)

    bar = tk.Frame(win)
    bar.pack(fill="x", padx=12)
    tk.Button(bar, text="Add files…", command=_add_files, pady=3).pack(side="left")
    tk.Button(bar, text="Browse folder…", command=_browse_folder,
              pady=3).pack(side="left", padx=6)

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        d = _current()
        if d is None:
            sel = listbox.curselection()
            if sel and isinstance(cache.get(paths[sel[0]]), Exception):
                messagebox.showerror("Cannot read this file",
                                     str(cache[paths[sel[0]]]), parent=win)
            else:
                messagebox.showinfo("Nothing selected",
                                    "Select a decay export first.", parent=win)
            return
        if init_dir:
            _last_decay_browse_dir = str(init_dir)
        win.destroy()
        on_done(d)

    tk.Button(btns, text="Next →", width=12, command=_next,
              pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy,
              pady=4).pack(side="left", padx=6)

    _populate()
    win.wait_window()

def _lifetime_data_dialog(parent, fcs_data, on_done):
    """Screen 1 — channel, histogram resolution, and fit window.

    Accepts photon data (FCSData) or an already-binned lifetime decay
    (fcs_ifx.LifetimeData).  For a decay there is a single curve and a fixed
    native resolution, so the channel and bin controls are hidden.
    """
    import tkinter as tk
    from tkinter import messagebox

    kind = getattr(fcs_data, "kind", None)
    is_lt     = kind == "lifetime_decay"      # .ifx: one curve, fixed grid
    is_loaded = kind == "loaded_decay"        # CSV: binned, but rebinnable

    # .ifx is already binned at a resolution it cannot change → fit natively
    # (n_bins=None).  A loaded CSV is also already binned, but CAN be summed
    # down to any exact divisor, so it keeps a resolution control restricted to
    # those divisors.  Photon data uses the full choice.
    def _nbins():
        return None if is_lt else bin_var.get()

    win = tk.Toplevel(parent)
    win.title("Lifetime fit — data")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text="Lifetime fit — select data",
             font=("Helvetica", 12, "bold"), pady=8).pack()
    _origin = getattr(fcs_data, "source_name", None)
    _hdr = f"File: {fcs_data.filepath.name}"
    if _origin and _origin != fcs_data.filepath.name:
        _hdr += f"   (measured from {_origin})"
    tk.Label(win, text=_hdr, font=("Helvetica", 9), fg="grey").pack()

    body = tk.Frame(win, padx=14, pady=8)
    body.pack(fill="x")

    # Channel — photon data has Ch1/Ch2; an .ifx decay is a single curve.
    # For photon data the choice is limited to the channels the file actually
    # recorded: a single-channel acquisition has no decay to fit on the other
    # detector, and offering it would produce a fit to an empty histogram.
    available_ch = tuple(getattr(fcs_data, "channels", (1, 2)))
    ch_var = tk.IntVar(value=(available_ch[0] if available_ch else 1))
    tk.Label(body, text="Channel:", anchor="w").grid(row=0, column=0, sticky="w", pady=3)
    if is_lt:
        tk.Label(body, text="decay (.ifx)", anchor="w").grid(row=0, column=1, sticky="w")
    elif is_loaded and getattr(fcs_data, "_single", False):
        # An unlabelled single-curve export: there is nothing to choose, and
        # offering Ch1/Ch2 would imply a detector identity the file never had.
        tk.Label(body, text="decay (single curve)", anchor="w").grid(
            row=0, column=1, sticky="w")
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

    # Bins — an .ifx decay is fixed; a loaded CSV can only be summed down to an
    # exact divisor of what was exported, so only those are offered.  Listing
    # the full set would let the user pick a resolution the file cannot produce
    # and meet an error on the next screen instead of a greyed-out option here.
    if is_loaded:
        bin_choices = fcs_data.valid_n_bins()
        bin_default = fcs_data.n_bins            # native, i.e. no rebinning
    else:
        bin_choices = list(fcs_lifetime._VALID_N_BINS)
        bin_default = 4096
    bin_var = tk.IntVar(value=bin_default)
    if not is_lt:
        tk.Label(body, text="Histogram bins:", anchor="w").grid(row=1, column=0, sticky="w", pady=3)
        tk.OptionMenu(body, bin_var, *bin_choices).grid(row=1, column=1, sticky="w")
        if is_loaded:
            tk.Label(body,
                     text=f"(exported at {fcs_data.n_bins}; "
                          f"exact rebins only)",
                     font=("Helvetica", 8), fg="grey").grid(
                         row=1, column=2, sticky="w", padx=(6, 0))

    # Window entries
    tk.Label(body, text="Fit start (ns):", anchor="w").grid(row=2, column=0, sticky="w", pady=3)
    start_var = tk.StringVar()
    tk.Entry(body, textvariable=start_var, width=12).grid(row=2, column=1, sticky="w")
    tk.Label(body, text="Fit end (ns):", anchor="w").grid(row=3, column=0, sticky="w", pady=3)
    end_var = tk.StringVar()
    tk.Entry(body, textvariable=end_var, width=12).grid(row=3, column=1, sticky="w")

    def _set_default_window(*_):
        try:
            lo, hi = default_window(fcs_data, ch_var.get(), _nbins())
            start_var.set(f"{lo:.2f}")
            end_var.set(f"{hi:.2f}")
        except Exception:
            pass

    # Pre-fill with the last-used window if we have one; otherwise compute the
    # per-file default (peak → end).
    if _last_fit_window["start"] is not None and _last_fit_window["end"] is not None:
        start_var.set(f"{_last_fit_window['start']:.2f}")
        end_var.set(f"{_last_fit_window['end']:.2f}")
    else:
        _set_default_window()
    ch_var.trace_add("write", _set_default_window)
    bin_var.trace_add("write", _set_default_window)

    def _current_window():
        try:
            return (float(start_var.get()), float(end_var.get()))
        except ValueError:
            return None

    def _pick_on_hist():
        # select_gate defaults to 512 bins, which a decay exported at 256
        # cannot supply; ask the source what it can render.
        gate_bins = (fcs_data.gate_n_bins(512) if is_loaded else 512)
        gate = fcs_lifetime.select_gate(
            fcs_data, n_bins=gate_bins, channels=(ch_var.get(),),
            initial_gate=_current_window(),
            title="Set lifetime fit window",
            gate_label="Fit window",
            confirm_text="Use this window",
        )
        if gate is not None:
            start_var.set(f"{gate[0]:.2f}")
            end_var.set(f"{gate[1]:.2f}")

    tk.Button(win, text="Pick window on histogram…", command=_pick_on_hist,
              pady=3).pack(pady=(0, 4))

    tk.Label(win, text="The first and last bins (edge artifacts)\n"
                       "are always excluded from the fit.",
             font=("Helvetica", 9), fg="grey", justify="center").pack()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        win_range = _current_window()
        if win_range is None:
            messagebox.showerror("Invalid window",
                                 "Fit start and end must be numbers.", parent=win)
            return
        lo, hi = win_range
        if hi <= lo:
            messagebox.showerror("Invalid window",
                                 "Fit end must be greater than fit start.", parent=win)
            return
        # Remember this window so the next dataset opens pre-filled with it.
        _last_fit_window["start"], _last_fit_window["end"] = lo, hi
        channel = ch_var.get()

        # Robustness net.  The radio for a channel this file lacks is disabled
        # above, so this should be unreachable from the GUI; it is kept for the
        # paths that bypass the greying-out (a programmatic caller, or a future
        # edit that adds a channel control without consulting available_ch).
        # Fitting an absent channel would otherwise fit an empty histogram and
        # report a lifetime with no data behind it.
        if not is_lt and channel not in available_ch:
            messagebox.showerror(
                "Channel not available",
                f"{fcs_data.filepath.name} recorded "
                f"{getattr(fcs_data, 'channel_summary', 'one channel')}; "
                f"there is no Ch{channel} decay to fit.",
                parent=win)
            return

        n_bins  = _nbins()
        t_ns, counts = fcs_data.lifetime_histogram(channel=channel, n_bins=n_bins)
        win.destroy()
        on_done(channel, n_bins, t_ns, counts, lo, hi)

    tk.Button(btns, text="Next →", width=12, command=_next, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


def _select_lifetime_model_dialog(parent, on_choose):
    """Screen 2 — choose a lifetime model from the registry."""
    import tkinter as tk

    win = tk.Toplevel(parent)
    win.title("Lifetime fit — select model")
    win.geometry("525x380")
    win.minsize(440, 340)
    win.resizable(True, True)
    win.grab_set()

    tk.Label(win, text="Select a lifetime model",
             font=("Helvetica", 12, "bold"), pady=8).pack()

    models = fcs_models.list_lifetime_models()
    key_var = tk.StringVar(value=models[0].key)

    list_frame = tk.LabelFrame(win, text="Models", padx=10, pady=6)
    list_frame.pack(fill="x", padx=12, pady=(0, 6))
    for m in models:
        tk.Radiobutton(list_frame, text=m.name, variable=key_var,
                       value=m.key, anchor="w").pack(fill="x")

    desc = tk.Text(win, height=8, wrap="word", font=("Courier", 9),
                   bg="#f7f7f7", relief="flat", padx=8, pady=6)
    desc.pack(fill="both", expand=True, padx=12, pady=(0, 6))

    def _refresh_desc(*_):
        m = fcs_models.get_lifetime_model(key_var.get())
        desc.config(state="normal")
        desc.delete("1.0", tk.END)
        desc.insert(tk.END, m.description)
        desc.config(state="disabled")

    key_var.trace_add("write", _refresh_desc)
    _refresh_desc()

    btns = tk.Frame(win)
    btns.pack(pady=8)

    def _next():
        m = fcs_models.get_lifetime_model(key_var.get())
        win.destroy()
        on_choose(m)

    tk.Button(btns, text="Next →", width=12, command=_next, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


def _lifetime_setup_dialog(parent, fcs_data, model, channel, n_bins,
                           t_ns, counts, fit_start, fit_end):
    """Screen 3 — initial guesses, bounds and fixed flags, then fit."""
    import tkinter as tk
    from tkinter import messagebox

    win = tk.Toplevel(parent)
    win.title(f"Lifetime fit — {model.name}")
    win.resizable(False, False)
    win.grab_set()

    tk.Label(win, text=model.name, font=("Helvetica", 12, "bold"), pady=6).pack()
    _origin = getattr(fcs_data, "source_name", None)
    _name = fcs_data.filepath.name
    if _origin and _origin != _name:
        _name = f"{_name}  (from {_origin})"
    tk.Label(win, text=f"{_name}  ·  Ch{channel}  ·  "
                       f"window {fit_start:.2f}–{fit_end:.2f} ns",
             font=("Helvetica", 9), fg="grey").pack()
    tk.Label(win, text=model.formula, font=("Courier", 9), fg="#444").pack(pady=(0, 6))

    # Auto guesses from the windowed data
    try:
        _t, _c, _trel = prepare_decay(t_ns, counts, fit_start, fit_end)
        guesses0 = auto_guess_lifetime(model, _trel, _c)
    except Exception:
        guesses0 = model.defaults()

    table = tk.Frame(win, padx=12, pady=4)
    table.pack(fill="x")

    headers = ["Parameter", "Guess", "Lower", "Upper", "Fix"]
    for c, h in enumerate(headers):
        tk.Label(table, text=h, font=("Helvetica", 10, "bold")).grid(
            row=0, column=c, padx=4, pady=(0, 4))

    guess_vars: Dict[str, tk.StringVar] = {}
    lower_vars: Dict[str, tk.StringVar] = {}
    upper_vars: Dict[str, tk.StringVar] = {}
    fixed_vars: Dict[str, tk.BooleanVar] = {}

    for r, p in enumerate(model.params, start=1):
        label = f"{p.name}" + (f" ({p.unit})" if p.unit else "")
        tk.Label(table, text=label, anchor="w", width=12).grid(
            row=r, column=0, sticky="w", padx=4, pady=2)

        gv = tk.StringVar(value=f"{guesses0.get(p.name, p.default):.6g}")
        lv = tk.StringVar(value=_fmt(p.lower))
        uv = tk.StringVar(value=_fmt(p.upper))
        fv = tk.BooleanVar(value=p.fixed)

        tk.Entry(table, textvariable=gv, width=12).grid(row=r, column=1, padx=4)
        tk.Entry(table, textvariable=lv, width=10).grid(row=r, column=2, padx=4)
        tk.Entry(table, textvariable=uv, width=10).grid(row=r, column=3, padx=4)
        tk.Checkbutton(table, variable=fv).grid(row=r, column=4, padx=4)

        guess_vars[p.name] = gv
        lower_vars[p.name] = lv
        upper_vars[p.name] = uv
        fixed_vars[p.name] = fv

    # Poisson weighting toggle (on by default for counts)
    weight_var = tk.BooleanVar(value=True)
    tk.Checkbutton(
        win, text="Weight by Poisson σ = √counts  (recommended)",
        variable=weight_var, anchor="w",
    ).pack(fill="x", padx=12, pady=(6, 0))

    # Optional label folded into the fit folder and the output filenames, so a
    # set of fits can be told apart in a file listing (and in Excel's window
    # list) without opening them.  Same control, wording and behaviour as the
    # correlation fit dialog.
    name_row = tk.Frame(win)
    name_row.pack(fill="x", padx=12, pady=(8, 0))
    tk.Label(name_row, text="Save as:", anchor="w",
             font=("Helvetica", 9)).pack(side="left")
    name_var = tk.StringVar(value="")
    tk.Entry(name_row, textvariable=name_var,
             font=("Helvetica", 9)).pack(side="left", fill="x", expand=True,
                                         padx=(4, 0))
    tk.Label(
        win,
        text=("      optional label; added after the timestamp, in both the\n"
              "      folder and the filenames, e.g.\n"
              "      2026-07-23_14-25-30_«myLabel»/"
              "20260723_142530_«myLabel»_params.csv"),
        font=("Helvetica", 8), fg="grey", anchor="w", justify="left",
    ).pack(fill="x", padx=12)

    btns = tk.Frame(win)
    btns.pack(pady=10)

    def _do_fit():
        try:
            guesses = {n: float(guess_vars[n].get()) for n in guess_vars}
            lowers  = {n: _parse_bound(lower_vars[n].get(), -np.inf) for n in lower_vars}
            uppers  = {n: _parse_bound(upper_vars[n].get(),  np.inf) for n in upper_vars}
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "Guesses and bounds must be numbers "
                                 "(use 'inf' / '-inf' for open bounds).",
                                 parent=win)
            return
        fixed = {n: fixed_vars[n].get() for n in fixed_vars}

        for n in guesses:
            if lowers[n] >= uppers[n]:
                messagebox.showerror("Invalid bounds",
                                     f"For '{n}', lower must be < upper.",
                                     parent=win)
                return

        try:
            result = fit_lifetime(
                model, t_ns, counts, fit_start, fit_end,
                guesses, lowers, uppers, fixed,
                weighted=weight_var.get(),
                channel=channel, n_bins=n_bins,
            )
        except Exception as e:
            messagebox.showerror("Fit failed", str(e), parent=win)
            return

        # Read the label BEFORE destroying the window: the StringVar outlives
        # the widget, but reading it first keeps the dependency obvious.
        fit_label = name_var.get()
        win.destroy()

        result["origin"] = getattr(fcs_data, "source_name", None)

        report_path, _curve, _params = export_lifetime_fit(
            result, fcs_data.filepath, fit_label)
        fig, _axes = plot_lifetime_fit(result, fcs_data.filepath.name, show=False)
        try:
            fig.savefig(report_path.with_suffix(".png"), dpi=150)
        except Exception as e:
            print(f"[lifetime fit] could not save figure: {e}")

        summary = "\n".join(
            f"{n} = {result['values'][n]:.4g}"
            + ("" if result['fixed'].get(n) else f" ± {result['errors'][n]:.2g}")
            for n in result["names"]
        )
        if np.isfinite(result.get("tau_mean", float("nan"))):
            summary += f"\n⟨τ⟩ = {result['tau_mean']:.4g} ns (amplitude-weighted)"
        _fr = result.get("fractions")
        if _fr:
            summary += "\n" + "   ".join(
                f"α{i} = {_fr['alpha'][i]:.3f} / f{i} = {_fr['f'][i]:.3f}"
                for i in _fr["indices"])
        gof = (f"red. χ² = {result['red_chi2']:.3g}"
               if result["weighted"] else f"R² = {result['r2']:.4f}")
        messagebox.showinfo(
            "Lifetime fit complete",
            f"{model.name}  (Ch{channel})\n\n{summary}\n\n{gof}\n\n"
            f"Results saved to:\n{report_path.parent}",
            parent=parent,
        )
        fcs_plottools.show_figure(fig, _axes)

    tk.Button(btns, text="Fit", width=12, command=_do_fit, pady=4).pack(side="left", padx=6)
    tk.Button(btns, text="Cancel", width=10, command=win.destroy, pady=4).pack(side="left", padx=6)

    win.wait_window()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from fcs_reader import read_fcs

    if len(sys.argv) < 2:
        print("Usage: python fcs_lifetime_fit.py <file.fcs> [model_key] [channel]")
        print("Available lifetime models:")
        for m in fcs_models.list_lifetime_models():
            print(f"  {m.key:<16} {m.name}")
        sys.exit(1)

    path = Path(sys.argv[1])
    key  = sys.argv[2] if len(sys.argv) > 2 else fcs_models.list_lifetime_models()[0].key
    ch   = int(sys.argv[3]) if len(sys.argv) > 3 else None
    model = fcs_models.get_lifetime_model(key)

    d = read_fcs(path)
    # Default to the first channel the file recorded rather than a hard-coded
    # Ch1, so a Ch2-only file works without having to name the channel.
    if ch is None:
        ch = d.channels[0]
    elif ch not in d.channels:
        sys.exit(f"{path.name} recorded {d.channel_summary}; "
                 f"there is no Ch{ch} decay to fit.")
    t_ns, counts = d.lifetime_histogram(channel=ch, n_bins=4096)
    lo, hi = default_window(d, ch, 4096)
    t_, c_, trel = prepare_decay(t_ns, counts, lo, hi)
    guesses = auto_guess_lifetime(model, trel, c_)
    lowers = {p.name: p.lower for p in model.params}
    uppers = {p.name: p.upper for p in model.params}
    fixed  = {p.name: p.fixed for p in model.params}

    result = fit_lifetime(model, t_ns, counts, lo, hi,
                          guesses, lowers, uppers, fixed,
                          channel=ch, n_bins=4096)
    export_lifetime_fit(result, path)
    plot_lifetime_fit(result, path.name)
