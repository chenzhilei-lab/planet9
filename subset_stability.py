#!/usr/bin/env python3
"""
Subset stability analysis for ETNO clustering.
For each k in {12,14,16,18}, draws 5,000 random subsets of size k from the
19-object ETNO sample and computes the uncorrected Rayleigh p-value.
Outputs median p, IQR, and fraction exceeding a=0.05 for each k.
"""
import json, math, random, sys
from collections import Counter

# Load ETNO data
with open('/mnt/d/Papers/A_ApJ_ETNO/code/etno_data.json') as f:
    data = json.load(f)
varpis = [math.radians(obj['varpi']) for obj in data['objects']]
N = len(varpis)
print(f"Loaded {N} ETNOs.")

def rayleigh_p(angles):
    n = len(angles)
    if n < 3:
        return 1.0
    sin_sum = sum(math.sin(a) for a in angles)
    cos_sum = sum(math.cos(a) for a in angles)
    R = math.sqrt(sin_sum**2 + cos_sum**2) / n
    # Exact Rayleigh p = exp(-n * R^2)
    p = math.exp(-n * R**2)
    return p

# Subset stability analysis
random.seed(42)
k_values = [12, 14, 16, 18]
n_subsets = 5000

print(f"\n=== Subset Stability Analysis ({n_subsets} subsets per k) ===")
print(f"{'k':>4} {'Median p':>10} {'IQR':>20} {'% p<0.05':>12}")
print("-" * 50)

for k in k_values:
    p_vals = []
    for _ in range(n_subsets):
        subset = random.sample(varpis, k)
        p = rayleigh_p(subset)
        p_vals.append(p)
    p_vals.sort()
    median = p_vals[n_subsets // 2]
    iqr_low = p_vals[n_subsets // 4]
    iqr_high = p_vals[3 * n_subsets // 4]
    frac_sig = sum(1 for p in p_vals if p < 0.05) / n_subsets * 100
    print(f"{k:4d} {median:10.4f} [{iqr_low:.4f}, {iqr_high:.4f}] {frac_sig:11.1f}%")

# Full sample reference
p_full = rayleigh_p(varpis)
print(f"\nFull sample (N=19): p = {p_full:.4f}")
print(f"  Bootstrap SE: TBD (need clustering_audit.py output)")
print("\nDone.")
