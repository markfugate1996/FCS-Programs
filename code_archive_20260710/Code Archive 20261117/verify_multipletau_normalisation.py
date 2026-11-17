"""
verify_multipletau_normalisation.py
===================================
Run this ONCE on your machine, in the env where the FCS suite runs:

    pip install multipletau
    python verify_multipletau_normalisation.py

It answers the one question I could not answer from the documentation, and
could not test myself (my sandbox has no network access, so I could neither
install multipletau nor read its source):

  With normalize=True, does multipletau return
      (a)  G(tau) -> 0  at large lag   [it subtracts the 1 for us], or
      (b)  G(tau) -> 1  at large lag   [we must subtract 1 ourselves]?

The docstrings say "the convention of the curve decaying to zero", but that
phrase is also attached to the normalize=False path, so it is ambiguous about
whether the -1 is applied. Guessing wrong puts a constant offset of exactly 1
into every G(tau) from the new backend -- which a fit would absorb into the
amplitude and quietly corrupt every N and every brightness downstream.

Paste the output back to me and I will finalise the adapter.
"""
import numpy as np
import multipletau

print("multipletau version:", multipletau.__version__)

rng = np.random.default_rng(0)

# Uncorrelated Poisson intensity trace: G(tau) has NO correlation at all, so
# the large-lag baseline IS the whole answer -- 0 or 1, nothing in between.
N = 2 ** 18
mean_counts = 5.0
trace = rng.poisson(mean_counts, N).astype(np.float64)

g = multipletau.autocorrelate(trace, m=16, deltat=1e-6,
                              normalize=True, compress="average")
lag, G = g[:, 0], g[:, 1]

tail = G[len(G) // 2:]          # large-lag half
baseline = np.nanmean(tail)

print(f"\ntrace mean            = {trace.mean():.4f}")
print(f"n lag points          = {len(G)}")
print(f"G at first few lags   = {np.array2string(G[:4], precision=5)}")
print(f"large-lag baseline    = {baseline:.6f}")

print("\n>>> VERDICT:")
if abs(baseline) < 0.05:
    print("    normalize=True ALREADY subtracts 1  (G -> 0).")
    print("    Adapter must NOT subtract 1.")
elif abs(baseline - 1.0) < 0.05:
    print("    normalize=True does NOT subtract 1  (G -> 1).")
    print("    Adapter MUST subtract 1 to match the suite's baseline-0 convention.")
else:
    print(f"    UNEXPECTED baseline {baseline:.6f} -- neither 0 nor 1.")
    print("    Do not proceed; send me this output.")

# Secondary check: is m honoured as points-per-level, and is the grid base-2?
print("\nfirst 20 lag values (s):")
print(np.array2string(lag[:20], precision=9))
ratios = [lag[i+1] / lag[i] for i in range(1, min(len(lag) - 1, 40))
          if lag[i] > 0]
print(f"\nmax successive lag ratio = {max(ratios):.3f}  (2.0 => base-2 octaves)")
