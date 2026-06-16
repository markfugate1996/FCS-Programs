"""
fcs_models.py
=============
Library of FCS correlation models available for fitting.

Design
------
Each model is an :class:`FCSModel` carrying

  * a list of :class:`Param` descriptors (name, default guess, bounds,
    unit, human description, and a suggested "fixed" state), and
  * a vectorised ``func(tau_s, **params)`` that evaluates G(tau).

``tau`` is always in **seconds** and the baseline convention matches the
rest of the suite: G(tau) -> 0 as tau -> infinity (the correlator already
subtracts 1), so the model's ``offset`` parameter is ~0.

Adding a model
--------------
Append a new :class:`FCSModel` to the :data:`MODELS` registry below.  No
other file needs to change — the fit GUI builds its parameter table and the
model-selection screen directly from whatever is registered here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


# ── Descriptors ───────────────────────────────────────────────────────────────

@dataclass
class Param:
    """One fit parameter and its default guess / bounds."""
    name: str
    default: float
    lower: float = -np.inf
    upper: float = np.inf
    unit: str = ""
    description: str = ""
    fixed: bool = False          # suggested initial "hold fixed" state
    link_default: bool = False   # suggested initial "link across datasets" state


@dataclass
class FCSModel:
    """A named correlation model: parameters + an evaluable function."""
    key: str
    name: str
    description: str
    params: List[Param]
    func: Callable[..., np.ndarray]
    formula: str = ""

    def param_names(self) -> List[str]:
        return [p.name for p in self.params]

    def defaults(self) -> Dict[str, float]:
        return {p.name: p.default for p in self.params}


# ── Model functions ───────────────────────────────────────────────────────────

def _g_diffusion_3d_1c(tau_s, G0, tau_D, kappa, offset):
    """
    Single-component free 3D diffusion through a Gaussian confocal volume.

        G(tau) = G0 · (1 + tau/tauD)^-1 · (1 + tau/(kappa^2 · tauD))^-1/2 + offset
    """
    tau_s = np.asarray(tau_s, dtype=np.float64)
    x       = tau_s / tau_D
    lateral = 1.0 / (1.0 + x)
    axial   = 1.0 / np.sqrt(1.0 + x / (kappa * kappa))
    return G0 * lateral * axial + offset


def _g_diffusion_3d_2c(tau_s, G0, f1, tau_D1, tau_D2, kappa, offset):
    """
    Two-component free 3D diffusion through a Gaussian confocal volume.

        G(tau) = G0 · [f1·D1(tau) + (1−f1)·D2(tau)] + offset

    where Di(tau) = (1 + tau/tau_Di)^-1 · (1 + tau/(kappa^2 · tau_Di))^-1/2
    is the single-species diffusion term for species i.
    """
    tau_s = np.asarray(tau_s, dtype=np.float64)
    kk    = kappa * kappa
    x1    = tau_s / tau_D1
    x2    = tau_s / tau_D2
    D1    = (1.0 / (1.0 + x1)) / np.sqrt(1.0 + x1 / kk)
    D2    = (1.0 / (1.0 + x2)) / np.sqrt(1.0 + x2 / kk)
    return G0 * (f1 * D1 + (1.0 - f1) * D2) + offset


def _g_diffusion_3d_1c_triplet(tau_s, G0, tau_D, T, tau_T, kappa, offset):
    """
    Single-component 3D diffusion with triplet-state (dark-state) blinking.

        G(tau) = G0 · [1 + T/(1−T) · exp(−tau/tau_T)] · D(tau) + offset

    The triplet pre-factor multiplies the diffusion term, so:
        G(0) = G0 / (1 − T)  ≠  G0.
    Setting T = 0 recovers the single-component model.
    """
    tau_s = np.asarray(tau_s, dtype=np.float64)
    kk    = kappa * kappa
    x     = tau_s / tau_D
    D     = (1.0 / (1.0 + x)) / np.sqrt(1.0 + x / kk)
    trip  = 1.0 + (T / (1.0 - T)) * np.exp(-tau_s / tau_T)
    return G0 * trip * D + offset


def _g_diffusion_3d_2c_triplet(tau_s, G0, f1, tau_D1, tau_D2, T, tau_T, kappa, offset):
    """
    Two-component 3D diffusion with a shared triplet-state correction.

        G(tau) = G0 · [1 + T/(1−T) · exp(−tau/tau_T)]
                     · [f1·D1(tau) + (1−f1)·D2(tau)] + offset

    Both species are assumed to carry the same fluorophore with the same
    triplet kinetics (e.g. free dye vs. dye-labelled complex).
    Setting T = 0 recovers the two-component model.
    """
    tau_s = np.asarray(tau_s, dtype=np.float64)
    kk    = kappa * kappa
    x1    = tau_s / tau_D1
    x2    = tau_s / tau_D2
    D1    = (1.0 / (1.0 + x1)) / np.sqrt(1.0 + x1 / kk)
    D2    = (1.0 / (1.0 + x2)) / np.sqrt(1.0 + x2 / kk)
    trip  = 1.0 + (T / (1.0 - T)) * np.exp(-tau_s / tau_T)
    return G0 * trip * (f1 * D1 + (1.0 - f1) * D2) + offset


def _g_diffusion_3d_2c_reaction(tau_s, G0, f1, tau_D1, tau_D2, tau_R, A_R, kappa, offset):
    """
    Two-component 3D diffusion with first-order chemical interconversion
    between the two states (product-form approximation).

        G(tau) = G0 · [1 + A_R · exp(−tau/tau_R)]
                     · [f1·D1(tau) + (1−f1)·D2(tau)] + offset

    tau_R = 1/(k_12 + k_21) is the chemical relaxation time.
    A_R   is the kinetic amplitude set by the brightness contrast between
          states; A_R = 0 when both states are equally bright.
    Setting A_R = 0 recovers the two-component model.
    """
    tau_s = np.asarray(tau_s, dtype=np.float64)
    kk    = kappa * kappa
    x1    = tau_s / tau_D1
    x2    = tau_s / tau_D2
    D1    = (1.0 / (1.0 + x1)) / np.sqrt(1.0 + x1 / kk)
    D2    = (1.0 / (1.0 + x2)) / np.sqrt(1.0 + x2 / kk)
    kin   = 1.0 + A_R * np.exp(-tau_s / tau_R)
    return G0 * kin * (f1 * D1 + (1.0 - f1) * D2) + offset


# ── Registry ──────────────────────────────────────────────────────────────────

_DIFFUSION_3D_1C = FCSModel(
    key="diffusion_3d_1comp",
    name="3D Diffusion — single component",
    description=(
        "Free three-dimensional diffusion of a single species through a "
        "Gaussian confocal observation volume.\n\n"
        "    G(τ) = G0 · (1 + τ/τD)⁻¹ · (1 + τ/(κ²·τD))⁻¹ᐟ² + offset\n\n"
        "    G0      amplitude ≈ 1/N  (N = mean molecules in the volume)\n"
        "    τD      diffusion time (seconds)\n"
        "    κ       structure parameter, axial/radial radius ratio z0/w0\n"
        "    offset  baseline (≈ 0 for this suite's normalisation)"
    ),
    formula="G0·(1+τ/τD)⁻¹·(1+τ/(κ²τD))⁻¹ᐟ² + offset",
    params=[
        Param("G0",     1.0,    0.0,     np.inf, "",  "amplitude ≈ 1/N"),
        Param("tau_D",  1e-3,   1e-9,    1e2,    "s", "diffusion time"),
        Param("kappa",  5.0,    1.0,     50.0,   "",  "structure parameter z0/w0",
              link_default=True),
        Param("offset", 0.0,   -np.inf,  np.inf, "",  "baseline offset",
              link_default=True),
    ],
    func=_g_diffusion_3d_1c,
)


MODELS: Dict[str, FCSModel] = {
    _DIFFUSION_3D_1C.key: _DIFFUSION_3D_1C,
}


# ── Additional models ─────────────────────────────────────────────────────────

_DIFFUSION_3D_2C = FCSModel(
    key="diffusion_3d_2comp",
    name="3D Diffusion — two components",
    description=(
        "Free 3D diffusion of two independently diffusing, equally bright "
        "species through a Gaussian confocal volume.  The correlation is a "
        "brightness-weighted sum of the two single-species terms:\n\n"
        "    G(τ) = G0·[f1·D1(τ) + (1−f1)·D2(τ)] + offset\n\n"
        "    Di(τ) = (1+τ/τDi)⁻¹·(1+τ/(κ²·τDi))⁻¹ᐟ²\n\n"
        "    G0      amplitude ≈ 1/N  (N = total mean molecules)\n"
        "    f1      number fraction of species 1  (f2 = 1 − f1)\n"
        "    τD1     diffusion time of species 1 (typically the faster)\n"
        "    τD2     diffusion time of species 2 (typically the slower)\n"
        "    κ       structure parameter z0/w0\n"
        "    offset  baseline (≈ 0 for this suite's normalisation)\n\n"
        "Note: the two components are only resolvable when τD2/τD1 ≳ 1.6 "
        "(Meseth et al. 1999 Biophys J 76:1619).  At equal brightness the "
        "model assumes equal molecular brightnesses; unequal brightnesses "
        "require brightness-weighted fractions instead of number fractions."
    ),
    formula="G0·[f1·D1(τ)+(1−f1)·D2(τ)] + offset",
    params=[
        Param("G0",    1.0,  0.0,    np.inf, "",  "amplitude ≈ 1/N"),
        Param("f1",    0.5,  0.0,    1.0,    "",  "number fraction of species 1  (f2 = 1−f1)"),
        Param("tau_D1", 3e-4, 1e-9,  1e2,   "s", "diffusion time of species 1  (fast)"),
        Param("tau_D2", 5e-3, 1e-9,  1e2,   "s", "diffusion time of species 2  (slow)"),
        Param("kappa",  5.0,  1.0,   50.0,   "",  "structure parameter z0/w0",
              link_default=True),
        Param("offset", 0.0, -np.inf, np.inf, "", "baseline offset",
              link_default=True),
    ],
    func=_g_diffusion_3d_2c,
)

_DIFFUSION_3D_1C_TRIPLET = FCSModel(
    key="diffusion_3d_1comp_triplet",
    name="3D Diffusion — single component + triplet",
    description=(
        "Free 3D diffusion of a single species with a fast triplet-state "
        "(or any dark-state) blinking correction.  The triplet term "
        "multiplies the diffusion factor:\n\n"
        "    G(τ) = G0·[1 + T/(1−T)·exp(−τ/τT)]·D(τ) + offset\n\n"
        "    D(τ) = (1+τ/τD)⁻¹·(1+τ/(κ²·τD))⁻¹ᐟ²\n\n"
        "    G0     diffusion amplitude ≈ 1/N  (G(0) = G0/(1−T))\n"
        "    τD     diffusion time\n"
        "    T      triplet (dark-state) fraction  (0 ≤ T < 1)\n"
        "    τT     triplet lifetime\n"
        "    κ      structure parameter z0/w0\n"
        "    offset baseline (≈ 0)\n\n"
        "Because a fraction T of molecules are non-emissive at any instant, "
        "the zero-lag amplitude is G0/(1−T) rather than G0.  The product "
        "form (diffusion × triplet) is exact when τT ≪ τD, which holds for "
        "all common visible-range fluorophores (τT ~ µs, τD ~ ms).\n\n"
        "Reference: Widengren J, Mets U, Rigler R (1995) J Phys Chem "
        "99:13368–13379."
    ),
    formula="G0·(1+T/(1−T)·e^(−τ/τT))·D(τ) + offset",
    params=[
        Param("G0",    1.0,   0.0,    np.inf, "",  "diffusion amplitude ≈ 1/N; G(0) = G0/(1−T)"),
        Param("tau_D", 1e-3,  1e-9,   1e2,    "s", "diffusion time"),
        Param("T",     0.10,  0.0,    0.9999, "",  "triplet (dark-state) fraction"),
        Param("tau_T", 2e-6,  1e-9,   1e-3,   "s", "triplet lifetime",
              link_default=True),
        Param("kappa", 5.0,   1.0,    50.0,   "",  "structure parameter z0/w0",
              link_default=True),
        Param("offset", 0.0, -np.inf, np.inf, "",  "baseline offset",
              link_default=True),
    ],
    func=_g_diffusion_3d_1c_triplet,
)

_DIFFUSION_3D_2C_TRIPLET = FCSModel(
    key="diffusion_3d_2comp_triplet",
    name="3D Diffusion — two components + triplet",
    description=(
        "Two-component 3D diffusion with a shared triplet-state correction.  "
        "Both species are assumed to carry the same fluorophore so they share "
        "the same triplet fraction and lifetime (e.g. free dye vs. "
        "dye-labelled protein of different size):\n\n"
        "    G(τ) = G0·[1 + T/(1−T)·exp(−τ/τT)]\n"
        "              ·[f1·D1(τ) + (1−f1)·D2(τ)] + offset\n\n"
        "    Di(τ) = (1+τ/τDi)⁻¹·(1+τ/(κ²·τDi))⁻¹ᐟ²\n\n"
        "    G0     diffusion amplitude ≈ 1/N;  G(0) = G0/(1−T)\n"
        "    f1     number fraction of species 1  (f2 = 1 − f1)\n"
        "    τD1    diffusion time of species 1  (fast)\n"
        "    τD2    diffusion time of species 2  (slow)\n"
        "    T      shared triplet fraction\n"
        "    τT     shared triplet lifetime\n"
        "    κ      structure parameter z0/w0\n"
        "    offset baseline (≈ 0)\n\n"
        "If the two species carry different labels with distinct triplet "
        "kinetics, apply per-species triplet factors before summing instead "
        "of using this shared-triplet form.\n\n"
        "Setting T = 0 recovers the two-component model.  "
        "Setting f1 = 1 recovers the single-component + triplet model."
    ),
    formula="G0·(1+T/(1−T)·e^(−τ/τT))·[f1·D1(τ)+(1−f1)·D2(τ)] + offset",
    params=[
        Param("G0",    1.0,   0.0,    np.inf, "",  "diffusion amplitude ≈ 1/N;  G(0) = G0/(1−T)"),
        Param("f1",    0.5,   0.0,    1.0,    "",  "number fraction of species 1  (f2 = 1−f1)"),
        Param("tau_D1", 3e-4, 1e-9,  1e2,    "s", "diffusion time of species 1  (fast)"),
        Param("tau_D2", 5e-3, 1e-9,  1e2,    "s", "diffusion time of species 2  (slow)"),
        Param("T",     0.10,  0.0,   0.9999,  "",  "shared triplet fraction"),
        Param("tau_T", 2e-6,  1e-9,  1e-3,   "s", "shared triplet lifetime",
              link_default=True),
        Param("kappa", 5.0,   1.0,   50.0,    "",  "structure parameter z0/w0",
              link_default=True),
        Param("offset", 0.0, -np.inf, np.inf, "",  "baseline offset",
              link_default=True),
    ],
    func=_g_diffusion_3d_2c_triplet,
)

_DIFFUSION_3D_2C_REACTION = FCSModel(
    key="diffusion_3d_2comp_reaction",
    name="3D Diffusion — two components + reaction",
    description=(
        "Two-component 3D diffusion with first-order interconversion between "
        "the two states (conformational dynamics, binding equilibrium, or any "
        "first-order chemical reaction A ⇌ B).  Uses the product-form "
        "approximation:\n\n"
        "    G(τ) = G0·[1 + A_R·exp(−τ/τR)]\n"
        "              ·[f1·D1(τ) + (1−f1)·D2(τ)] + offset\n\n"
        "    Di(τ) = (1+τ/τDi)⁻¹·(1+τ/(κ²·τDi))⁻¹ᐟ²\n\n"
        "    G0     diffusion amplitude ≈ 1/N;  G(0) = G0·(1+A_R)\n"
        "    f1     equilibrium fraction in state 1  (f2 = 1 − f1)\n"
        "    τD1    diffusion time of state 1  (typically the faster)\n"
        "    τD2    diffusion time of state 2  (typically the slower)\n"
        "    τR     chemical relaxation time  =  1/(k₁₂ + k₂₁)\n"
        "    A_R    kinetic amplitude (brightness-contrast term):\n"
        "           A_R = (η1−η2)²·f1·f2 / (η1·f1 + η2·f2)²\n"
        "           Special cases:\n"
        "             η2 = 0  (dark state)    →  A_R = f2/f1\n"
        "             η1 = η2 (equal brightness) →  A_R = 0\n"
        "    κ      structure parameter z0/w0\n"
        "    offset baseline (≈ 0)\n\n"
        "The product form is exact when τD1 = τD2 (same size, different "
        "brightness) and is a good approximation in the fast-exchange "
        "(τR ≪ τD) or slow-exchange (τR ≫ τD) limits.  For intermediate "
        "exchange with substantially different diffusion times, the full "
        "coupled diffusion-reaction equations should be used instead "
        "(Meseth et al. 1999).\n\n"
        "Setting A_R = 0 recovers the two-component diffusion model.  "
        "Setting τD1 = τD2 and A_R = T/(1−T) recovers the single-component "
        "triplet model.\n\n"
        "References:\n"
        "  Elson EL, Magde D (1974) Biopolymers 13:1–27.\n"
        "  Meseth U et al. (1999) Biophys J 76:1619–1631.\n"
        "  Haustein E, Schwille P (2004) Curr Opin Struct Biol 14:531–540."
    ),
    formula="G0·(1+A_R·e^(−τ/τR))·[f1·D1(τ)+(1−f1)·D2(τ)] + offset",
    params=[
        Param("G0",    1.0,   0.0,    np.inf, "",  "diffusion amplitude ≈ 1/N;  G(0) = G0·(1+A_R)"),
        Param("f1",    0.5,   0.0,    1.0,    "",  "equilibrium fraction in state 1  (f2 = 1−f1)"),
        Param("tau_D1", 3e-4, 1e-9,  1e2,    "s", "diffusion time of state 1  (fast)"),
        Param("tau_D2", 5e-3, 1e-9,  1e2,    "s", "diffusion time of state 2  (slow)"),
        Param("tau_R", 1e-4,  1e-9,  1e2,    "s", "chemical relaxation time  1/(k₁₂+k₂₁)"),
        Param("A_R",   0.25,  0.0,   np.inf, "",  "kinetic amplitude  (0 if equal brightness)"),
        Param("kappa", 5.0,   1.0,   50.0,   "",  "structure parameter z0/w0",
              link_default=True),
        Param("offset", 0.0, -np.inf, np.inf, "",  "baseline offset",
              link_default=True),
    ],
    func=_g_diffusion_3d_2c_reaction,
)


MODELS: Dict[str, FCSModel] = {
    _DIFFUSION_3D_1C.key:          _DIFFUSION_3D_1C,
    _DIFFUSION_3D_2C.key:          _DIFFUSION_3D_2C,
    _DIFFUSION_3D_1C_TRIPLET.key:  _DIFFUSION_3D_1C_TRIPLET,
    _DIFFUSION_3D_2C_TRIPLET.key:  _DIFFUSION_3D_2C_TRIPLET,
    _DIFFUSION_3D_2C_REACTION.key: _DIFFUSION_3D_2C_REACTION,
}


# ── Lifetime (TCSPC tail-fit) models ──────────────────────────────────────────
#
# These reuse the same FCSModel/Param machinery as the correlation models, but
# the independent variable is the photon arrival time WITHIN the laser period
# (microtime), measured in nanoseconds from the start of the fit window, and the
# dependent variable is photon counts per bin.  They are tail fits: a sum of
# exponentials plus a constant background, with NO instrument-response
# reconvolution.  Fit only the decaying tail (from at/after the peak), and
# exclude the first and last microtime bins, which are time-tagger catch-all
# artifacts rather than fluorescence (see fcs_lifetime_fit for the data prep).
# They live in a SEPARATE registry (LIFETIME_MODELS) so they never appear in the
# correlation model chooser.

def _decay_1exp(t_ns, A, tau, offset):
    """
    Single-exponential decay (tail fit).

        I(t) = A · exp(−t/τ) + offset

    t is the arrival time in ns measured from the start of the fit window, so
    A is the (background-subtracted) count level at the window start and τ is
    the fluorescence lifetime.
    """
    t_ns = np.asarray(t_ns, dtype=np.float64)
    return A * np.exp(-t_ns / tau) + offset


def _decay_2exp(t_ns, A1, tau1, A2, tau2, offset):
    """
    Two-exponential decay (tail fit).

        I(t) = A1·exp(−t/τ1) + A2·exp(−t/τ2) + offset

    By convention component 1 is the faster (shorter τ).  The amplitude-weighted
    mean lifetime ⟨τ⟩ = (A1·τ1 + A2·τ2)/(A1 + A2) is reported as a derived
    quantity by the fitter.
    """
    t_ns = np.asarray(t_ns, dtype=np.float64)
    return A1 * np.exp(-t_ns / tau1) + A2 * np.exp(-t_ns / tau2) + offset


_LIFETIME_1EXP = FCSModel(
    key="lifetime_1exp",
    name="Lifetime — single exponential (tail)",
    description=(
        "Single-exponential fluorescence decay, fitted to the tail of the "
        "TCSPC histogram (no IRF reconvolution).\n\n"
        "    I(t) = A · exp(−t/τ) + offset\n\n"
        "    A       count level at the start of the fit window\n"
        "    τ       fluorescence lifetime (ns)\n"
        "    offset  constant background (dark counts, scatter)\n\n"
        "t is measured in nanoseconds from the start of the fit window.  Choose "
        "the window to start at or just after the decay peak and to stop before "
        "the final bin; the fitter additionally drops the first and last "
        "microtime bins, which are time-tagger edge artifacts.  Counts are "
        "Poisson-distributed, so the fit is weighted by σ = √counts and the "
        "reduced χ² is meaningful (≈ 1 for a good fit).\n\n"
        "A tail fit is accurate when τ is large compared with the instrument "
        "response width; for τ comparable to the IRF, use a reconvolution fit "
        "instead (not implemented here)."
    ),
    formula="A·exp(−t/τ) + offset",
    params=[
        Param("A",      1000.0, 0.0,  np.inf, "counts", "amplitude at fit-window start"),
        Param("tau",    3.0,    1e-3, 1e3,    "ns",     "fluorescence lifetime"),
        Param("offset", 0.0,    0.0,  np.inf, "counts", "constant background"),
    ],
    func=_decay_1exp,
)

_LIFETIME_2EXP = FCSModel(
    key="lifetime_2exp",
    name="Lifetime — two exponentials (tail)",
    description=(
        "Two-component (bi-exponential) fluorescence decay, fitted to the tail "
        "of the TCSPC histogram (no IRF reconvolution).\n\n"
        "    I(t) = A1·exp(−t/τ1) + A2·exp(−t/τ2) + offset\n\n"
        "    A1, τ1  amplitude and lifetime of component 1 (faster)\n"
        "    A2, τ2  amplitude and lifetime of component 2 (slower)\n"
        "    offset  constant background\n\n"
        "The amplitude-weighted mean lifetime ⟨τ⟩ = (A1·τ1 + A2·τ2)/(A1 + A2) "
        "is reported alongside the parameters.  Two lifetimes are only "
        "resolvable when they differ by roughly a factor of two or more and the "
        "decay has enough counts; if the data are truly single-exponential the "
        "fit will tend to drive τ1 → τ2 and the reduced χ² will not improve over "
        "the single-exponential model.  As with the single-exponential model, "
        "fit the tail only and let the fitter drop the edge bins; the weighting "
        "is Poisson (σ = √counts)."
    ),
    formula="A1·exp(−t/τ1) + A2·exp(−t/τ2) + offset",
    params=[
        Param("A1",     700.0, 0.0,  np.inf, "counts", "amplitude of component 1 (fast)"),
        Param("tau1",   1.5,   1e-3, 1e3,    "ns",     "lifetime of component 1 (fast)"),
        Param("A2",     300.0, 0.0,  np.inf, "counts", "amplitude of component 2 (slow)"),
        Param("tau2",   5.0,   1e-3, 1e3,    "ns",     "lifetime of component 2 (slow)"),
        Param("offset", 0.0,   0.0,  np.inf, "counts", "constant background"),
    ],
    func=_decay_2exp,
)

LIFETIME_MODELS: Dict[str, FCSModel] = {
    _LIFETIME_1EXP.key: _LIFETIME_1EXP,
    _LIFETIME_2EXP.key: _LIFETIME_2EXP,
}


# ── PCH (photon counting histogram) models ────────────────────────────────────
#
# The single-species photon counting histogram for a 3D Gaussian PSF (Chen,
# Müller, So & Gratton, Biophys J 1999, 77:553).  Two parameters per species:
#
#     N        mean number of molecules in the observation volume
#     epsilon  molecular brightness = detected counts per molecule per bin
#
# Moment relations (used for the initial guess): <k> = N·epsilon and the Mandel
# parameter Q = Var(k)/<k> − 1 = gamma2·epsilon, with gamma2 = 2^(−3/2) for the
# 3D Gaussian.  Multiple species combine by convolution (their generating-
# function exponents add), so the two-species PCH is built from the same
# single-species kernel.  This is the ideal single-bin-time PCH: it does NOT
# include detector dead-time or the diffusion blur that matters when the bin
# time is not small compared with the diffusion time.
#
# Unlike the correlation and lifetime model functions, evaluating a PCH requires
# a numerical spatial integral plus an FFT, so the kernel is implemented here
# rather than as a one-line closed form.  ``func(k, **params)`` returns the
# probability Pi(k) at the requested integer counts; the fitter scales by the
# number of sampled bins to compare with the measured histogram.

from scipy.stats import poisson as _poisson           # noqa: E402
from numpy.fft import fft as _fft, ifft as _ifft       # noqa: E402

_PCH_GAMMA2 = 2.0 ** -1.5    # 2nd-order gamma factor for a 3D Gaussian PSF


def _pch_single_molecule_bk(epsilon: float, K: int, n_s: int = 2500) -> np.ndarray:
    """
    Reduced single-molecule PCH coefficients b_k for k = 1..K (3D Gaussian PSF).

        b_k = (2/√π) ∫_0^∞ Poisson(k; ε·e^{-s}) · √s ds

    These satisfy Σ_k k·b_k = ε and Σ_k k(k−1)·b_k = ε²·gamma2, so that an
    occupation N gives <k> = N·ε and Q = gamma2·ε.
    """
    eps = max(float(epsilon), 1e-12)
    s_max = max(25.0, np.log(eps) + 30.0)
    s = np.linspace(0.0, s_max, n_s)
    mu = eps * np.exp(-s)
    sq = np.sqrt(s)
    ks = np.arange(1, K + 1)[:, None]
    pmf = _poisson.pmf(ks, mu[None, :])               # (K, n_s)
    integ = np.trapezoid(pmf * sq[None, :], s, axis=1)
    return (2.0 / np.sqrt(np.pi)) * integ


def _pch_pmf(K: int, species) -> np.ndarray:
    """
    PCH probability vector Pi(0..K) for a list of (N, epsilon) species.

    The compound-Poisson generating function G(ξ) = exp(Σ_k a_k (ξ^k − 1)) with
    a_k = Σ_species N·b_k(ε) is inverted by FFT.  Multiple species simply add
    their a_k (equivalent to convolving their PCHs).
    """
    a = np.zeros(K + 1)
    for (N, eps) in species:
        a[1:] += float(N) * _pch_single_molecule_bk(eps, K)
    L = 1
    while L < 4 * (K + 1):
        L *= 2
    a_pad = np.zeros(L)
    a_pad[:K + 1] = a
    Pi = np.real(_ifft(np.exp(_fft(a_pad)))) * np.exp(-a.sum())
    return np.clip(Pi[:K + 1], 0.0, None)


def _pch_eval(k, species) -> np.ndarray:
    """Evaluate the PCH probability at the integer counts ``k`` (array)."""
    k = np.asarray(k)
    kmax = int(np.max(k)) if k.size else 0
    Pi = _pch_pmf(kmax + 16, species)        # build with headroom to avoid FFT aliasing
    idx = np.clip(k.astype(int), 0, len(Pi) - 1)
    return Pi[idx]


def _pch_1species(k, N, epsilon):
    """Single-species 3D-Gaussian PCH probability Pi(k)."""
    return _pch_eval(k, [(N, epsilon)])


def _pch_2species(k, N1, epsilon1, N2, epsilon2):
    """Two-species 3D-Gaussian PCH probability Pi(k) = conv(species1, species2)."""
    return _pch_eval(k, [(N1, epsilon1), (N2, epsilon2)])


_PCH_1SPECIES = FCSModel(
    key="pch_1species",
    name="PCH — single species (3D Gaussian)",
    description=(
        "Photon counting histogram for one diffusing species in a 3D Gaussian "
        "observation volume (Chen et al. 1999).\n\n"
        "    Π(k) = single-species 3DG PCH(N, ε)\n\n"
        "    N        mean number of molecules in the observation volume\n"
        "    ε        molecular brightness = counts per molecule per bin\n\n"
        "Moments: ⟨k⟩ = N·ε and the Mandel parameter Q = Var(k)/⟨k⟩ − 1 = "
        "γ₂·ε, with γ₂ = 2^(−3/2) ≈ 0.354 for the 3D Gaussian.  N and ε are "
        "separated by the super-Poissonian shape: ε controls the excess width "
        "(Q), N then sets the mean.  When the data are essentially Poisson "
        "(Q ≈ 0, e.g. very dim molecules), ε and N become poorly separable.\n\n"
        "This is the ideal single-bin-time PCH (no detector dead-time and no "
        "diffusion-during-the-bin correction); choose a bin time small compared "
        "with the diffusion time.  The fit is weighted by the Poisson error of "
        "each histogram bin (σ = √counts)."
    ),
    formula="Π(k) = 3DG-PCH(N, ε)",
    params=[
        Param("N",       1.0, 1e-6, np.inf, "",            "mean molecules in observation volume"),
        Param("epsilon", 1.0, 1e-4, np.inf, "cnts/mol/bin", "molecular brightness"),
    ],
    func=_pch_1species,
)

_PCH_2SPECIES = FCSModel(
    key="pch_2species",
    name="PCH — two species (3D Gaussian)",
    description=(
        "Photon counting histogram for two independent species in a 3D Gaussian "
        "observation volume.  The two-species PCH is the convolution of the two "
        "single-species histograms (their generating-function exponents add):\n\n"
        "    Π(k) = conv( 3DG-PCH(N1, ε1), 3DG-PCH(N2, ε2) )\n\n"
        "    N1, ε1   occupation and brightness of species 1\n"
        "    N2, ε2   occupation and brightness of species 2\n\n"
        "The total mean is ⟨k⟩ = N1·ε1 + N2·ε2.  Two species are only "
        "resolvable when their brightnesses ε differ appreciably (a brightness "
        "ratio of roughly two or more) and the histogram has enough counts in "
        "its tail; PCH separates species by brightness, not by diffusion time. "
        "If the data are really single-species the fit tends to drive the two "
        "brightnesses together and the reduced χ² will not improve over the "
        "single-species model.  Reported derived quantities include the number "
        "fraction f1 = N1/(N1 + N2).  Same ideal single-bin-time assumptions and "
        "Poisson weighting as the single-species model."
    ),
    formula="Π(k) = conv(PCH(N1,ε1), PCH(N2,ε2))",
    params=[
        Param("N1",       1.0, 1e-6, np.inf, "",            "occupation of species 1"),
        Param("epsilon1", 1.5, 1e-4, np.inf, "cnts/mol/bin", "brightness of species 1 (brighter)"),
        Param("N2",       1.0, 1e-6, np.inf, "",            "occupation of species 2"),
        Param("epsilon2", 0.5, 1e-4, np.inf, "cnts/mol/bin", "brightness of species 2 (dimmer)"),
    ],
    func=_pch_2species,
)

PCH_MODELS: Dict[str, FCSModel] = {
    _PCH_1SPECIES.key: _PCH_1SPECIES,
    _PCH_2SPECIES.key: _PCH_2SPECIES,
}


# ── Accessors ─────────────────────────────────────────────────────────────────

def list_models() -> List[FCSModel]:
    """All registered correlation models, in registry order."""
    return list(MODELS.values())


def get_model(key: str) -> FCSModel:
    """Look up a correlation model by key."""
    return MODELS[key]


def list_lifetime_models() -> List[FCSModel]:
    """All registered lifetime (TCSPC tail-fit) models, in registry order."""
    return list(LIFETIME_MODELS.values())


def get_lifetime_model(key: str) -> FCSModel:
    """Look up a lifetime model by key."""
    return LIFETIME_MODELS[key]


def list_pch_models() -> List[FCSModel]:
    """All registered PCH models, in registry order."""
    return list(PCH_MODELS.values())


def get_pch_model(key: str) -> FCSModel:
    """Look up a PCH model by key."""
    return PCH_MODELS[key]
