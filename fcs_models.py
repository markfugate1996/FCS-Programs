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


# ── Accessors ─────────────────────────────────────────────────────────────────

def list_models() -> List[FCSModel]:
    """All registered models, in registry order."""
    return list(MODELS.values())


def get_model(key: str) -> FCSModel:
    """Look up a model by key."""
    return MODELS[key]
