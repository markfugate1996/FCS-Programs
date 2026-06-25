"""
fcs_lifetime_models.py
======================
Library of time-domain fluorescence-decay models for lifetime fitting,
mirroring the structure of :mod:`fcs_models` (the correlation models) so the
fit GUI can build its parameter table the same way.

What a lifetime model is
------------------------
Each model describes the *ideal* (un-broadened) fluorescence decay as a sum of
exponentials::

    f(t) = Σ_i  a_i · exp(−t / τ_i)        for t ≥ 0,   else 0

The measured TCSPC histogram is this ideal decay **convolved with the
instrument response function (IRF)**, plus a constant background::

    model(t) = [ IRF(t) ⊗ f(t) ]  +  bg

The convolution (iterative reconvolution) and the optional IRF time ``shift``
are handled by the fitter in :mod:`fcs_lifetime_fit`, so a model here only has
to supply ``ideal(t_ns, **params)``.  Every model therefore shares two extra
parameters beyond its amplitudes/lifetimes:

    bg     constant background / dark-count offset (counts)
    shift  IRF colour shift relative to the decay (ns; usually small)

Adding a model
--------------
Append a :class:`LifetimeModel` to :data:`MODELS`.  No other file needs to
change — the selection screen and parameter table are built from whatever is
registered here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

# Reuse the same Param descriptor the correlation models use, so the fit
# dialog can render lifetime and correlation parameters identically.
from fcs_models import Param


# ── Model descriptor ──────────────────────────────────────────────────────────

@dataclass
class LifetimeModel:
    """
    A named multi-exponential decay model.

    Attributes
    ----------
    key, name, description, formula : str
    params : list[Param]
        Amplitudes and lifetimes, followed by the shared ``bg`` and ``shift``.
    n_exp : int
        Number of exponential components.
    ideal : callable
        ``ideal(t_ns, **params) -> ndarray`` giving the un-convolved decay.
    """
    key: str
    name: str
    description: str
    params: List[Param]
    n_exp: int
    ideal: Callable[..., np.ndarray]
    formula: str = ""
    amp_names:  List[str] = field(default_factory=list)
    tau_names:  List[str] = field(default_factory=list)

    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    def defaults(self) -> Dict[str, float]:
        return {p.name: p.default for p in self.params}


# ── Ideal-decay evaluation ────────────────────────────────────────────────────

def _multiexp(t_ns: np.ndarray, amps: List[float], taus: List[float]) -> np.ndarray:
    """
    Σ_i a_i · exp(−t/τ_i) for t ≥ 0 (0 for t < 0).  τ_i are clamped away from
    zero to keep the exponential well-defined during fitting.
    """
    t = np.asarray(t_ns, dtype=np.float64)
    out = np.zeros_like(t)
    pos = t >= 0
    tp = t[pos]
    acc = np.zeros_like(tp)
    for a, tau in zip(amps, taus):
        tau = max(float(tau), 1e-6)
        acc = acc + a * np.exp(-tp / tau)
    out[pos] = acc
    return out


def _make_ideal(n_exp: int) -> Callable[..., np.ndarray]:
    """Build an ``ideal(t_ns, **params)`` closure for an n-exponential model."""
    amp_names = [f"a{i}" for i in range(1, n_exp + 1)]
    tau_names = [f"tau{i}" for i in range(1, n_exp + 1)]

    def ideal(t_ns, **p):
        amps = [p[a] for a in amp_names]
        taus = [p[t] for t in tau_names]
        return _multiexp(t_ns, amps, taus)

    return ideal


def _build_model(key, name, n_exp, description, formula) -> LifetimeModel:
    """Assemble a LifetimeModel with standard amplitude/lifetime/bg/shift params."""
    params: List[Param] = []
    amp_names, tau_names = [], []

    # Spread default lifetimes across a plausible 0.5–8 ns range so multi-
    # component starts are not degenerate.
    default_taus = [1.0, 3.0, 6.0, 9.0][:n_exp]
    for i in range(1, n_exp + 1):
        an, tn = f"a{i}", f"tau{i}"
        amp_names.append(an)
        tau_names.append(tn)
        params.append(Param(an, 1.0e4, 0.0, np.inf, "cnt",
                             f"amplitude of component {i}"))
        params.append(Param(tn, default_taus[i - 1], 1e-3, 1e3, "ns",
                             f"lifetime of component {i}"))

    params.append(Param("bg", 0.0, 0.0, np.inf, "cnt",
                         "constant background / dark counts"))
    params.append(Param("shift", 0.0, -2.0, 2.0, "ns",
                        "IRF colour shift (decay vs IRF timing)", fixed=False))

    return LifetimeModel(
        key=key, name=name, description=description, formula=formula,
        params=params, n_exp=n_exp, ideal=_make_ideal(n_exp),
        amp_names=amp_names, tau_names=tau_names,
    )


# ── Registry ──────────────────────────────────────────────────────────────────

_EXP1 = _build_model(
    key="lifetime_exp1",
    name="Mono-exponential decay",
    n_exp=1,
    formula="model(t) = IRF ⊗ [a1·e^(−t/τ1)] + bg",
    description=(
        "Single-exponential fluorescence decay, fitted to the data by "
        "iterative reconvolution with the measured IRF:\n\n"
        "    model(t) = [ IRF(t) ⊗ a1·exp(−t/τ1) ] + bg\n\n"
        "    a1     amplitude of the decay (counts)\n"
        "    τ1     fluorescence lifetime (ns)  ← the quantity of interest\n"
        "    bg     constant background / dark-count offset (counts)\n"
        "    shift  IRF colour shift (ns); small timing offset between the\n"
        "           IRF and the sample decay.  Free by default, since a\n"
        "           sub-bin offset between the IRF and decay is common; fix\n"
        "           it at 0 if the recovered shift is unstable.\n\n"
        "Use for a single, well-behaved emitter.  If the weighted residuals "
        "are structured (not flat and random about 0), try the bi-exponential "
        "model."
    ),
)

_EXP2 = _build_model(
    key="lifetime_exp2",
    name="Bi-exponential decay",
    n_exp=2,
    formula="model(t) = IRF ⊗ [a1·e^(−t/τ1) + a2·e^(−t/τ2)] + bg",
    description=(
        "Two-component exponential decay (iterative reconvolution):\n\n"
        "    model(t) = [ IRF(t) ⊗ (a1·exp(−t/τ1) + a2·exp(−t/τ2)) ] + bg\n\n"
        "    a1, a2   component amplitudes (counts)\n"
        "    τ1, τ2   the two lifetimes (ns)\n"
        "    bg       constant background (counts)\n"
        "    shift    IRF colour shift (ns), free by default\n\n"
        "Amplitude-weighted mean lifetime  ⟨τ⟩ = Σ aᵢτᵢ / Σ aᵢ  and the "
        "fractional contributions  fᵢ = aᵢτᵢ / Σ aⱼτⱼ  are reported with the "
        "fit.  Two lifetimes are usually only separable when τ2/τ1 ≳ 2.\n\n"
        "Typical of a dye in two environments, FRET (donor with/without "
        "acceptor), or a free + bound population."
    ),
)

_EXP3 = _build_model(
    key="lifetime_exp3",
    name="Tri-exponential decay",
    n_exp=3,
    formula="model(t) = IRF ⊗ [Σ aᵢ·e^(−t/τᵢ)] + bg",
    description=(
        "Three-component exponential decay (iterative reconvolution):\n\n"
        "    model(t) = [ IRF(t) ⊗ Σᵢ aᵢ·exp(−t/τᵢ) ] + bg,   i = 1..3\n\n"
        "    aᵢ, τᵢ   amplitudes (counts) and lifetimes (ns)\n"
        "    bg       constant background (counts)\n"
        "    shift    IRF colour shift (ns), free by default\n\n"
        "Three components are easy to over-fit: only use this when the "
        "bi-exponential residuals are clearly structured and the recovered "
        "lifetimes stay well separated and stable across repeats.  Fixing one "
        "lifetime (e.g. a known free-dye value) often stabilises the fit."
    ),
)


MODELS: Dict[str, LifetimeModel] = {
    _EXP1.key: _EXP1,
    _EXP2.key: _EXP2,
    _EXP3.key: _EXP3,
}


# ── Accessors (same surface as fcs_models) ────────────────────────────────────

def list_models() -> List[LifetimeModel]:
    """All registered lifetime models, in registry order."""
    return list(MODELS.values())


def get_model(key: str) -> LifetimeModel:
    """Look up a lifetime model by key."""
    return MODELS[key]
