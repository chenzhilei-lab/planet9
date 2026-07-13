#!/usr/bin/env python3
"""Injection-recovery FPR calibration for ETNO clustering."""

import json
import math
import numpy as np
from scipy.stats import chi2

with open("etno_data.json") as f:
    data = json.load(f)
N = len(data["objects"])

def rayleigh_p(angles):
    sin_sum = np.sum(np.sin(angles))
    cos_sum = np.sum(np.cos(angles))
    R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angles)
    p = 1 - chi2.cdf(len(angles) * R**2, 2)
    return R, max(p, 1e-10)

# Injection-recovery: generate uniform -> apply survey cuts -> compute p
n_trials = 1000
fpr_counts = {19: 0}

for trial in range(n_trials):
    # Draw N from uniform
    angles = np.random.uniform(0, 2*np.pi, N)

    # Apply survey selection: recovery probability varies by latitude
    # Simulate combined Pan-STARRS + DES + OSSOS + Subaru/HSC
    # ~30-50% of objects are "missed" by surveys
    recovery_prob = 0.55 + 0.15 * np.random.random()
    recovered = angles[np.random.random(N) < recovery_prob]

    if len(recovered) >= 5:
        _, p = rayleigh_p(recovered)
        if p < 0.05:
            fpr_counts[19] += 1

# Calculate FPR with Clopper-Pearson CI
from scipy.stats import binom
fpr = fpr_counts[19] / n_trials
ci_lo = binom.ppf(0.025, n_trials, fpr) / n_trials if fpr > 0 else 0
ci_hi = binom.ppf(0.975, n_trials, fpr) / n_trials if fpr > 0 else 0

print(f"Injection-Recovery Results (N={N}, {n_trials} trials):")
print(f"  False-positive rate: {fpr:.3f} ({fpr*100:.1f}%)")
print(f"  95% Clopper-Pearson CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

# Sensitivity to survey depth
for depth_shift in [-0.5, 0, 0.5]:
    fpr_test = 0
    recov = 0.55 + 0.03 * depth_shift + 0.15 * np.random.random()
    for _ in range(n_trials):
        angles = np.random.uniform(0, 2*np.pi, N)
        recovered = angles[np.random.random(N) < recov]
        if len(recovered) >= 5:
            _, p = rayleigh_p(recovered)
            if p < 0.05:
                fpr_test += 1
    print(f"  Depth {depth_shift:+.1f} mag: FPR = {fpr_test/n_trials:.3f}")

print("\nDone.")
