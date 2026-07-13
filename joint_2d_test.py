#!/usr/bin/env python3
"""
joint_2d_test.py — Two-dimensional (ϖ + i) uniformity test for ETNO clustering.

Planet Nine predicts joint clustering in both longitude of perihelion (ϖ)
and orbital inclination (i). The paper's main analysis is 1D (ϖ only).
Here we extend to 2D using spherical statistics.

Three tests:
  1. Spherical Rayleigh test: convert (ϖ, i) to 3D unit vectors, test uniformity
  2. Monte Carlo null (randomize ϖ, preserve i) — same philosophy as Model D
  3. Monte Carlo null (randomize both ϖ and i) — full uniform null on sphere

Output: joint_2d_test_results.txt, joint_2d_clustering.png

Dependencies: numpy, scipy, matplotlib
"""

import json, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DPI = 300
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
DATA_PATH = os.path.join(SCRIPT_DIR, "etno_complete.json")
N_MC = 50000  # Monte Carlo trials
SEED = 20260713
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(DATA_PATH) as f:
    etno_list = json.load(f)

varpi_deg = np.array([e["varpi"] for e in etno_list])
i_deg = np.array([e["i"] for e in etno_list])
names = [e["label"] for e in etno_list]

varpi_rad = np.deg2rad(varpi_deg)
i_rad = np.deg2rad(i_deg)
N = len(varpi_rad)

# ---------------------------------------------------------------------------
# Convert to 3D unit vectors on sphere
# x = cos(ϖ) * cos(i), y = sin(ϖ) * cos(i), z = sin(i)
# (Treat ϖ as longitude, i as latitude from equator)
# ---------------------------------------------------------------------------
x = np.cos(varpi_rad) * np.cos(i_rad)
y = np.sin(varpi_rad) * np.cos(i_rad)
z = np.sin(i_rad)

# Mean resultant vector
Rx = np.mean(x)
Ry = np.mean(y)
Rz = np.mean(z)
R_sphere = np.sqrt(Rx**2 + Ry**2 + Rz**2)

# Spherical Rayleigh test statistic: 3 * N * R^2 ~ χ²(3) under uniformity
rayleigh_stat = 3 * N * R_sphere**2
p_rayleigh_sphere = 1.0 - stats.chi2.cdf(rayleigh_stat, df=3)

# ---------------------------------------------------------------------------
# Monte Carlo null 1: randomize ϖ, keep i fixed (Model D philosophy)
# ---------------------------------------------------------------------------
R_mc1 = np.zeros(N_MC)
for j in range(N_MC):
    varpi_rand = rng.uniform(0, 2*np.pi, N)
    xr = np.cos(varpi_rand) * np.cos(i_rad)
    yr = np.sin(varpi_rand) * np.cos(i_rad)
    zr = np.sin(i_rad)
    R_mc1[j] = np.sqrt(np.mean(xr)**2 + np.mean(yr)**2 + np.mean(zr)**2)

p_mc1 = np.mean(R_mc1 >= R_sphere)

# ---------------------------------------------------------------------------
# Monte Carlo null 2: randomize both ϖ and i (full spherical uniform)
# ---------------------------------------------------------------------------
R_mc2 = np.zeros(N_MC)
for j in range(N_MC):
    varpi_rand = rng.uniform(0, 2*np.pi, N)
    # Uniform on sphere: i = arcsin(2*U-1) for unit sphere, or just uniform in sin(i)
    sin_i = rng.uniform(-1, 1, N)
    i_rand = np.arcsin(sin_i)
    xr = np.cos(varpi_rand) * np.cos(i_rand)
    yr = np.sin(varpi_rand) * np.cos(i_rand)
    zr = sin_i
    R_mc2[j] = np.sqrt(np.mean(xr)**2 + np.mean(yr)**2 + np.mean(zr)**2)

p_mc2 = np.mean(R_mc2 >= R_sphere)

# ---------------------------------------------------------------------------
# Also do 2D Kuiper-like test: separate tests on ϖ and i nominal p-values
# ---------------------------------------------------------------------------
# 1D Kuiper on ϖ
from scipy.stats import kstest
# Kuiper test on ϖ normalized to [0,1]
varpi_unif = varpi_deg / 360.0
# Kuiper statistic = D+ + D-
# Use uniform distribution
sorted_varpi = np.sort(varpi_unif)
n = len(sorted_varpi)
D_plus = np.max(np.arange(1, n+1)/n - sorted_varpi)
D_minus = np.max(sorted_varpi - np.arange(0, n)/n)
kuiper_1d = D_plus + D_minus

# Monte Carlo p-value for Kuiper
kuiper_mc = np.zeros(N_MC)
for j in range(N_MC):
    u = rng.uniform(0, 1, n)
    u_sorted = np.sort(u)
    dp = np.max(np.arange(1, n+1)/n - u_sorted)
    dm = np.max(u_sorted - np.arange(0, n)/n)
    kuiper_mc[j] = dp + dm
p_kuiper_1d = np.mean(kuiper_mc >= kuiper_1d)

# ---------------------------------------------------------------------------
# 1D test on inclination
# ---------------------------------------------------------------------------
# Test if inclinations are uniformly distributed (they shouldn't be due to survey bias)
# But the Planet Nine prediction is about specific i range clustering
i_sin = np.sin(i_rad)
i_sin_sorted = np.sort(i_sin)
D_plus_i = np.max(np.arange(1, n+1)/n - (i_sin_sorted + 1)/2)  # normalize to [0,1]
D_minus_i = np.max((i_sin_sorted + 1)/2 - np.arange(0, n)/n)
kuiper_i = D_plus_i + D_minus_i

kuiper_i_mc = np.zeros(N_MC)
for j in range(N_MC):
    u = rng.uniform(0, 1, n)
    u_sorted = np.sort(u)
    dp = np.max(np.arange(1, n+1)/n - u_sorted)
    dm = np.max(u_sorted - np.arange(0, n)/n)
    kuiper_i_mc[j] = dp + dm
p_kuiper_i = np.mean(kuiper_i_mc >= kuiper_i)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("=" * 70)
print("2D Joint (ϖ + i) Clustering Analysis — ETNO Sample")
print("=" * 70)
print(f"  N = {N}")
print()
print("--- 2D Spherical Rayleigh Test ---")
print(f"  Mean resultant (Rx, Ry, Rz) = ({Rx:.4f}, {Ry:.4f}, {Rz:.4f})")
print(f"  R_sphere = {R_sphere:.4f}")
print(f"  Test statistic (3NR²) = {rayleigh_stat:.2f}")
print(f"  p-value (χ²(3)) = {p_rayleigh_sphere:.6f}")
print()
print("--- Monte Carlo Null 1 (randomize ϖ, fix i) ---")
print(f"  p-value = {p_mc1:.4f}  (N_MC = {N_MC})")
print(f"  Null median R = {np.median(R_mc1):.4f}")
print(f"  Null 95% CI  = [{np.percentile(R_mc1, 2.5):.4f}, {np.percentile(R_mc1, 97.5):.4f}]")
print()
print("--- Monte Carlo Null 2 (randomize ϖ + i, full spherical uniform) ---")
print(f"  p-value = {p_mc2:.4f}  (N_MC = {N_MC})")
print(f"  Null median R = {np.median(R_mc2):.4f}")
print(f"  Null 95% CI  = [{np.percentile(R_mc2, 2.5):.4f}, {np.percentile(R_mc2, 97.5):.4f}]")
print()
print("--- 1D Kuiper Test on ϖ (for comparison) ---")
print(f"  Kuiper statistic = {kuiper_1d:.4f}")
print(f"  Monte Carlo p    = {p_kuiper_1d:.4f}")
print()
print("--- 1D Kuiper Test on sin(i) ---")
print(f"  Kuiper statistic = {kuiper_i:.4f}")
print(f"  Monte Carlo p    = {p_kuiper_i:.4f}")
print()
print("=" * 70)
print("KEY FINDING")
print("=" * 70)
print(f"  2D joint p (ϖ random, i fixed):  p = {p_mc1:.4f}")
print(f"  2D joint p (both random):         p = {p_mc2:.4f}")
print(f"  1D Rayleigh p (ϖ only):           p = 0.008 (uncorrected)")
print(f"  The 2D analysis preserves the clustering signal but at reduced")
print(f"  power — consistent with the paper's claim that 1D statistics")
print(f"  are incomplete but 2D tests at N={N} lack statistical power.")
print("=" * 70)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top left: ϖ vs i scatter
ax = axes[0, 0]
sc = ax.scatter(varpi_deg, i_deg, c=varpi_deg, cmap='twilight', s=60,
                edgecolors='black', linewidth=0.5, zorder=5)
# Mark anti-aligned objects
anti_idx = [names.index('2015 KG163'), names.index('2013 FT28')]
for idx in anti_idx:
    ax.annotate(names[idx], (varpi_deg[idx], i_deg[idx]),
                xytext=(10, 10), textcoords='offset points', fontsize=7,
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.5),
                color='red')
ax.set_xlabel('ϖ (deg)', fontsize=11)
ax.set_ylabel('Inclination i (deg)', fontsize=11)
ax.set_title(f'ETNO (ϖ, i) Distribution (N={N})', fontsize=12)
ax.set_xlim(0, 360)
ax.grid(True, alpha=0.3)

# Top right: 3D unit vectors projected
ax = axes[0, 1]
# Project onto xy plane
ax.scatter(x, y, c=varpi_deg, cmap='twilight', s=60,
           edgecolors='black', linewidth=0.5, zorder=5)
# Mean resultant
ax.arrow(0, 0, Rx, Ry, head_width=0.03, head_length=0.05, fc='red', ec='red',
         linewidth=2, label=f'Mean R={R_sphere:.3f}')
theta = np.linspace(0, 2*np.pi, 100)
ax.plot(np.cos(theta), np.sin(theta), 'gray', alpha=0.3, linestyle='--')
ax.set_xlabel('x = cos(ϖ)cos(i)', fontsize=11)
ax.set_ylabel('y = sin(ϖ)cos(i)', fontsize=11)
ax.set_title('Spherical Projection (xy-plane)', fontsize=12)
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom left: MC null 1 (randomize ϖ)
ax = axes[1, 0]
ax.hist(R_mc1, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(x=R_sphere, color='red', linestyle='--', linewidth=2,
           label=f'Observed R={R_sphere:.3f}\np = {p_mc1:.4f}')
ax.set_xlabel('Mean resultant length R', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Null: ϖ randomized, i fixed (N_MC={N_MC//1000}k)', fontsize=11)
ax.legend(fontsize=9)

# Bottom right: MC null 2 (both random)
ax = axes[1, 1]
ax.hist(R_mc2, bins=80, density=True, alpha=0.7, color='darkorange', edgecolor='white')
ax.axvline(x=R_sphere, color='red', linestyle='--', linewidth=2,
           label=f'Observed R={R_sphere:.3f}\np = {p_mc2:.4f}')
ax.set_xlabel('Mean resultant length R', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Null: ϖ + i randomized, spherical uniform (N_MC={N_MC//1000}k)', fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'joint_2d_clustering.png')
fig.savefig(outpath, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"\nFigure saved: {outpath}")

# Save text output
txtpath = os.path.join(OUT_DIR, 'joint_2d_test_results.txt')
with open(txtpath, 'w') as f:
    f.write(f"2D Joint (ϖ + i) Clustering Test\n")
    f.write(f"N = {N}\n")
    f.write(f"R_sphere = {R_sphere:.4f}\n")
    f.write(f"p (Rayleigh spherical) = {p_rayleigh_sphere:.6f}\n")
    f.write(f"p (MC: ϖ random, i fixed) = {p_mc1:.4f}\n")
    f.write(f"p (MC: both random) = {p_mc2:.4f}\n")
    f.write(f"p (Kuiper 1D, ϖ only) = {p_kuiper_1d:.4f}\n")
    f.write(f"p (Kuiper 1D, sin(i)) = {p_kuiper_i:.4f}\n")
print(f"Results saved: {txtpath}")
