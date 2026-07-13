#!/usr/bin/env python3
"""
Power analysis for Rayleigh test on ETNO clustering.
Uses scipy.stats.vonmises for proper sampling.
"""
import math, numpy as np
from scipy.stats import vonmises

def rayleigh_p(angles):
    n = len(angles)
    if n < 3:
        return 1.0
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    R = np.sqrt(sin_sum**2 + cos_sum**2) / n
    return math.exp(-n * R**2)

rng = np.random.default_rng(42)
kappa_values = [0.2, 0.5, 1.0, 1.17, 1.5, 2.0]
N_values = [19, 30, 50, 100]
n_trials = 5000
alpha = 0.05

print(f"=== Rayleigh Test Power Analysis ({n_trials} trials per cell, scipy vonmises) ===")
print(f"{'kappa':>6} {'N=19':>8} {'N=30':>8} {'N=50':>8} {'N=100':>8}")
print("-" * 42)

for kappa in kappa_values:
    row = f"{kappa:6.2f}"
    for N in N_values:
        # Generate samples from scipy vonmises (mean=0, kappa=kappa)
        detections = 0
        for trial in range(n_trials):
            angles = vonmises.rvs(kappa, loc=0, size=N, random_state=rng)
            p = rayleigh_p(angles)
            if p < alpha:
                detections += 1
        power = detections / n_trials * 100
        row += f" {power:7.1f}%"
    print(row)

print("\nDone.")
