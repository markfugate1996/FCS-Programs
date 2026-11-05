#!/usr/bin/env python3
"""
add_augmented_models.py
=======================
Insert two models into fcs_models.py:

    "Augmented 3D diffusion - single component"
    "Augmented 3D diffusion - two components"

Both are the existing triplet models reparameterised so that the triplet
lifetime is tau_T = alpha * tau_D, with alpha a dimensionless fit/fix
parameter.  Nothing existing is modified; three insertions are made:

    1. two model functions, after _g_diffusion_3d_2c_reaction
    2. two FCSModel objects, after _DIFFUSION_3D_2C_REACTION
    3. two entries in the MODELS registry

Usage:
    python add_augmented_models.py [path/to/fcs_models.py]
    python add_augmented_models.py --check [path]     # dry run, no write

Line endings of the target file are detected and preserved.
"""

import sys
from pathlib import Path

# ── Insertion 1: model functions ──────────────────────────────────────────────

ANCHOR_FUNCS = (
    "    kin   = 1.0 + A_R * np.exp(-tau_s / tau_R)\n"
    "    return G0 * kin * (f1 * D1 + (1.0 - f1) * D2) + offset\n"
)

BLOCK_FUNCS = '''

def _g_diffusion_3d_1c_triplet_aug(tau_s, G0, tau_D, kappa, alpha, T, offset):
    """
    Single-component 3D diffusion with a triplet term whose lifetime is tied
    to the diffusion time:  tau_T = alpha * tau_D.

        G(tau) = G0 · [1 + T/(1−T) · exp(−tau/(alpha·tau_D))] · D(tau) + offset

    Mathematically identical to _g_diffusion_3d_1c_triplet under the
    substitution tau_T -> alpha·tau_D; with both alpha and tau_D free the two
    are exact reparameterisations of each other and give identical fits.  The
    point of this form is that alpha can be HELD, which pins the triplet
    lifetime to a fixed fraction of the diffusion time instead of to an
    absolute value.
    """
    tau_s = np.asarray(tau_s, dtype=np.float64)
    kk    = kappa * kappa
    x     = tau_s / tau_D
    D     = (1.0 / (1.0 + x)) / np.sqrt(1.0 + x / kk)
    trip  = 1.0 + (T / (1.0 - T)) * np.exp(-tau_s / (alpha * tau_D))
    return G0 * trip * D + offset


def _g_diffusion_3d_2c_triplet_aug(tau_s, G0, f1, tau_D1, tau_D2, kappa,
                                   alpha1, T1, alpha2, T2, offset):
    """
    Weighted sum of two augmented single-component curves, each carrying its
    OWN triplet fraction and its own alpha:

        G(tau) = G0 · [ f1·X1(tau)·D1(tau) + (1−f1)·X2(tau)·D2(tau) ] + offset

        Xi(tau) = 1 + Ti/(1−Ti) · exp(−tau/(alpha_i·tau_Di))
        Di(tau) = (1 + tau/tau_Di)^-1 · (1 + tau/(kappa^2·tau_Di))^-1/2

    This differs from _g_diffusion_3d_2c_triplet, where a single triplet
    factor multiplies the whole sum.  Here each species blinks on its own
    timescale, so the dark-state term is inside the sum.  Consequently

        G(0) = G0 · [ f1/(1−T1) + (1−f1)/(1−T2) ]

    rather than G0/(1−T).  Setting T1 = T2 and alpha1·tau_D1 = alpha2·tau_D2
    recovers the shared-triplet two-component model; setting f1 = 1 recovers
    the augmented single-component model.
    """
    tau_s = np.asarray(tau_s, dtype=np.float64)
    kk    = kappa * kappa
    x1    = tau_s / tau_D1
    x2    = tau_s / tau_D2
    D1    = (1.0 / (1.0 + x1)) / np.sqrt(1.0 + x1 / kk)
    D2    = (1.0 / (1.0 + x2)) / np.sqrt(1.0 + x2 / kk)
    trip1 = 1.0 + (T1 / (1.0 - T1)) * np.exp(-tau_s / (alpha1 * tau_D1))
    trip2 = 1.0 + (T2 / (1.0 - T2)) * np.exp(-tau_s / (alpha2 * tau_D2))
    return G0 * (f1 * trip1 * D1 + (1.0 - f1) * trip2 * D2) + offset
'''

# ── Insertion 2: FCSModel objects ─────────────────────────────────────────────

ANCHOR_MODELS = (
    "    func=_g_diffusion_3d_2c_reaction,\n"
    ")\n"
)

BLOCK_MODELS = '''

_DIFFUSION_3D_1C_AUG = FCSModel(
    key="diffusion_3d_1comp_aug",
    name="Augmented 3D diffusion \u2014 single component",
    description=(
        "Single-component 3D diffusion with a triplet (dark-state) term whose "
        "lifetime is expressed as a fixed multiple of the diffusion time, "
        "\u03c4T = \u03b1\u00b7\u03c4D:\\n\\n"
        "    G(\u03c4) = G0\u00b7[1 + T/(1\u2212T)\u00b7exp(\u2212\u03c4/(\u03b1\u00b7\u03c4D))]\u00b7D(\u03c4) + offset\\n\\n"
        "    D(\u03c4) = (1+\u03c4/\u03c4D)\u207b\u00b9\u00b7(1+\u03c4/(\u03ba\u00b2\u00b7\u03c4D))\u207b\u00b9\u141f\u00b2\\n\\n"
        "    G0     diffusion amplitude \u2248 1/N;  G(0) = G0/(1\u2212T)\\n"
        "    \u03c4D     diffusion time\\n"
        "    \u03ba      structure parameter z0/w0\\n"
        "    \u03b1      dimensionless triplet/diffusion time ratio \u03c4T/\u03c4D\\n"
        "    T      triplet (dark-state) fraction  (0 \u2264 T < 1)\\n"
        "    offset baseline (\u2248 0)\\n\\n"
        "This is the standard single-component + triplet model under the "
        "substitution \u03c4T \u2192 \u03b1\u00b7\u03c4D.  With \u03b1 free the two forms are exact "
        "reparameterisations and fit identically; the useful case is \u03b1 HELD, "
        "which ties the blinking timescale to the diffusion timescale instead "
        "of to an absolute value \u2014 e.g. across a dilution or viscosity series "
        "where \u03c4D moves but the underlying ratio is taken as constant.\\n\\n"
        "The product form (diffusion \u00d7 triplet) assumes \u03c4T \u226a \u03c4D, i.e. \u03b1 \u226a 1; "
        "\u03b1 \u2248 2e-3 corresponds to the usual \u03c4T \u223c \u00b5s against \u03c4D \u223c ms.\\n\\n"
        "Reference: Widengren J, Mets U, Rigler R (1995) J Phys Chem "
        "99:13368\u201313379."
    ),
    formula="G0\u00b7(1+T/(1\u2212T)\u00b7e^(\u2212\u03c4/(\u03b1\u03c4D)))\u00b7D(\u03c4) + offset",
    params=[
        Param("G0",     1.0,   0.0,    np.inf, "",  "diffusion amplitude \u2248 1/N; G(0) = G0/(1\u2212T)"),
        Param("tau_D",  1e-3,  1e-9,   1e2,    "s", "diffusion time"),
        Param("kappa",  5.0,   1.0,    50.0,   "",  "structure parameter z0/w0",
              link_default=True),
        Param("alpha",  2e-3,  1e-9,   1.0,    "",  "triplet/diffusion time ratio  \u03c4T = \u03b1\u00b7\u03c4D",
              link_default=True),
        Param("T",      0.10,  0.0,    0.9999, "",  "triplet (dark-state) fraction"),
        Param("offset", 0.0,  -np.inf, np.inf, "",  "baseline offset",
              link_default=True),
    ],
    func=_g_diffusion_3d_1c_triplet_aug,
)

_DIFFUSION_3D_2C_AUG = FCSModel(
    key="diffusion_3d_2comp_aug",
    name="Augmented 3D diffusion \u2014 two components",
    description=(
        "Weighted sum of two augmented single-component curves.  Each species "
        "carries its own triplet fraction and its own \u03b1, so each dark-state "
        "term sits INSIDE the sum rather than multiplying it:\\n\\n"
        "    G(\u03c4) = G0\u00b7[f1\u00b7X1(\u03c4)\u00b7D1(\u03c4) + (1\u2212f1)\u00b7X2(\u03c4)\u00b7D2(\u03c4)] + offset\\n\\n"
        "    Xi(\u03c4) = 1 + Ti/(1\u2212Ti)\u00b7exp(\u2212\u03c4/(\u03b1i\u00b7\u03c4Di))\\n"
        "    Di(\u03c4) = (1+\u03c4/\u03c4Di)\u207b\u00b9\u00b7(1+\u03c4/(\u03ba\u00b2\u00b7\u03c4Di))\u207b\u00b9\u141f\u00b2\\n\\n"
        "    G0     diffusion amplitude \u2248 1/N  (N = total mean molecules);\\n"
        "           G(0) = G0\u00b7[f1/(1\u2212T1) + (1\u2212f1)/(1\u2212T2)]\\n"
        "    f1     fraction of species 1  (f2 = 1 \u2212 f1)\\n"
        "    \u03c4D1    diffusion time of species 1\\n"
        "    \u03c4D2    diffusion time of species 2\\n"
        "    \u03ba      structure parameter z0/w0  (shared \u2014 one observation volume)\\n"
        "    \u03b11, T1 triplet ratio and fraction of species 1\\n"
        "    \u03b12, T2 triplet ratio and fraction of species 2\\n"
        "    offset baseline (\u2248 0)\\n\\n"
        "Unlike the shared-triplet two-component model, this form allows the "
        "two species to blink on different timescales \u2014 two different labels, "
        "or one fluorophore whose dark-state kinetics change on binding.  "
        "Setting T1 = T2 and \u03b11\u00b7\u03c4D1 = \u03b12\u00b7\u03c4D2 recovers the shared-triplet "
        "model; setting f1 = 1 recovers the augmented single-component model.\\n\\n"
        "As in the other triplet models, G0 is the DIFFUSION amplitude: above "
        "both blinking timescales X1, X2 \u2192 1 and G \u2192 G0\u00b7[f1\u00b7D1 + (1\u2212f1)\u00b7D2], so "
        "N = 1/G0 keeps its usual meaning.  The zero-lag value is larger, by "
        "the f1-weighted dark-state factor written above."
    ),
    formula=("G0\u00b7[f1\u00b7(1+T1/(1\u2212T1)e^(\u2212\u03c4/(\u03b11\u03c4D1)))\u00b7D1(\u03c4)"
             " + (1\u2212f1)\u00b7(1+T2/(1\u2212T2)e^(\u2212\u03c4/(\u03b12\u03c4D2)))\u00b7D2(\u03c4)] + offset"),
    params=[
        Param("G0",     1.0,   0.0,    np.inf, "",  "diffusion amplitude \u2248 1/N (total); see description for G(0)"),
        Param("f1",     0.5,   0.0,    1.0,    "",  "fraction of species 1  (f2 = 1\u2212f1)"),
        Param("tau_D1", 3e-4,  1e-9,   1e2,    "s", "diffusion time of species 1  (fast)"),
        Param("tau_D2", 5e-3,  1e-9,   1e2,    "s", "diffusion time of species 2  (slow)"),
        Param("kappa",  5.0,   1.0,    50.0,   "",  "structure parameter z0/w0",
              link_default=True),
        Param("alpha1", 2e-3,  1e-9,   1.0,    "",  "triplet/diffusion time ratio of species 1  \u03c4T1 = \u03b11\u00b7\u03c4D1",
              link_default=True),
        Param("T1",     0.10,  0.0,    0.9999, "",  "triplet fraction of species 1"),
        Param("alpha2", 2e-3,  1e-9,   1.0,    "",  "triplet/diffusion time ratio of species 2  \u03c4T2 = \u03b12\u00b7\u03c4D2",
              link_default=True),
        Param("T2",     0.10,  0.0,    0.9999, "",  "triplet fraction of species 2"),
        Param("offset", 0.0,  -np.inf, np.inf, "",  "baseline offset",
              link_default=True),
    ],
    func=_g_diffusion_3d_2c_triplet_aug,
)
'''

# ── Insertion 3: registry entries ─────────────────────────────────────────────

ANCHOR_REGISTRY = "    _DIFFUSION_3D_2C_REACTION.key: _DIFFUSION_3D_2C_REACTION,\n"

BLOCK_REGISTRY = (
    "    _DIFFUSION_3D_1C_AUG.key:      _DIFFUSION_3D_1C_AUG,\n"
    "    _DIFFUSION_3D_2C_AUG.key:      _DIFFUSION_3D_2C_AUG,\n"
)


def main(argv):
    check_only = "--check" in argv
    args = [a for a in argv[1:] if not a.startswith("--")]
    path = Path(args[0]) if args else Path("fcs_models.py")

    if not path.is_file():
        sys.exit(f"ERROR: {path} not found")

    with open(path, "r", encoding="utf-8", newline="") as fh:
        src = fh.read()

    nl = "\r\n" if "\r\n" in src else "\n"
    print(f"target      : {path}")
    print(f"line endings: {'CRLF' if nl == chr(13) + chr(10) else 'LF'}")
    print(f"size        : {len(src)} chars")

    if "diffusion_3d_1comp_aug" in src or "_DIFFUSION_3D_1C_AUG" in src:
        sys.exit("ERROR: the augmented models already appear in this file; "
                 "nothing done.")

    def conv(s):
        return s.replace("\n", nl) if nl != "\n" else s

    edits = [
        ("model functions",  ANCHOR_FUNCS,    BLOCK_FUNCS),
        ("FCSModel objects", ANCHOR_MODELS,   BLOCK_MODELS),
        ("registry entries", ANCHOR_REGISTRY, BLOCK_REGISTRY),
    ]

    out = src
    for label, anchor, block in edits:
        a = conv(anchor)
        n = out.count(a)
        if n != 1:
            sys.exit(f"ERROR: anchor for {label} matched {n} times "
                     f"(expected exactly 1); nothing written.\n"
                     f"anchor was:\n{anchor}")
        out = out.replace(a, a + conv(block), 1)
        print(f"  ok  {label}: anchor unique, {len(block)} chars inserted")

    if check_only:
        print("\n--check: anchors verified, file NOT modified.")
        return

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)
    print(f"\nwrote {path}  ({len(src)} -> {len(out)} chars)")
    print("now run:  python -m py_compile", path)


if __name__ == "__main__":
    main(sys.argv)
