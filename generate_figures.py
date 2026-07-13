#!/usr/bin/env python3
"""
Generate fig_sensitivity.png — leave-one-out sensitivity analysis.
Uses JPL-verified ϖ values for 19 ETNOs.
Computes circular mean shifts independently — does not copy paper's numbers.
"""
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ============================================================
# DATA: 19 ETNO ϖ values (JPL-verified, 2026-06-17)
# ============================================================
objects = [
    ("90377 Sedna",       96.0),
    ("2012 VP113",        24.9),
    ("2015 TG387",        59.0),
    ("2013 FT28",        258.5),
    ("2014 SR349",        16.9),
    ("2013 RF98",         19.7),
    ("2014 FE72",        111.0),
    ("2015 RX245",        73.7),
    ("2010 GB174",       118.0),
    ("2007 TG422",        39.0),
    ("2010 VZ98",         71.0),
    ("2015 KG163",       250.9),
    ("2013 RA109",         8.0),
    ("2015 BP519",       123.0),
    ("2013 UH15",        100.0),
    ("2013 SY99",         61.8),
    ("2014 WB556",       351.0),
    ("2015 RY245",       337.0),
    ("2021 RR205",       317.0),
]
N = len(objects)
names = [o[0] for o in objects]
varpi_vals = np.array([o[1] for o in objects])
varpi_rad = np.deg2rad(varpi_vals)

# ============================================================
# COMPUTE: circular mean and leave-one-out shifts
# ============================================================
def circular_mean(angles_rad):
    sx = np.sum(np.cos(angles_rad))
    sy = np.sum(np.sin(angles_rad))
    return np.arctan2(sy, sx) % (2 * np.pi)

cm_all = circular_mean(varpi_rad)
cm_all_deg = np.rad2deg(cm_all)

deltas = []
R_changes = []
for i in range(N):
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    sub = varpi_rad[mask]

    # circular mean of subset
    cm_i = circular_mean(sub)
    cm_i_deg = np.rad2deg(cm_i)

    # circular distance (Δϖ)
    d = abs(cm_i_deg - cm_all_deg)
    if d > 180:
        d = 360 - d
    deltas.append(d)

    # Rayleigh R change
    sx_sub = np.sum(np.cos(sub))
    sy_sub = np.sum(np.sin(sub))
    R_sub = np.sqrt(sx_sub**2 + sy_sub**2) / (N - 1)
    sx_all = np.sum(np.cos(varpi_rad))
    sy_all = np.sum(np.sin(varpi_rad))
    R_all = np.sqrt(sx_all**2 + sy_all**2) / N
    dR_pct = abs(R_sub - R_all) / R_all * 100
    R_changes.append(dR_pct)

deltas = np.array(deltas)
R_changes = np.array(R_changes)

# Print verification
print(f"N = {N}")
print(f"ϖ̄_all = {cm_all_deg:.1f}°")
print(f"R_all  = {np.sqrt(np.sum(np.cos(varpi_rad))**2 + np.sum(np.sin(varpi_rad))**2) / N:.4f}")
print(f"\nObjects with Δϖ > 5°:")
for i in np.argsort(-deltas):
    if deltas[i] > 5:
        print(f"  {names[i]:15s}  ϖ={varpi_vals[i]:6.1f}°  Δϖ={deltas[i]:4.1f}°  ΔR={R_changes[i]:4.1f}%")
    else:
        break

# ============================================================
# PLOT: fig_sensitivity.png
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

# Sort by ϖ value for display (not by Δ)
sort_idx = np.argsort(varpi_vals)
sorted_names = [names[i] for i in sort_idx]
sorted_deltas = deltas[sort_idx]
sorted_varpi = varpi_vals[sort_idx]

# Colors: red for Δ>5°, grey otherwise
colors = ['#d62728' if d > 5 else '#7f7f7f' for d in sorted_deltas]
edgecolors = ['#a51d1d' if d > 5 else '#5a5a5a' for d in sorted_deltas]

bars = ax.bar(range(N), sorted_deltas, color=colors, edgecolor=edgecolors,
              linewidth=0.5, width=0.7)

# Threshold line at 5°
ax.axhline(y=5, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(N - 0.5, 5.3, r'$\Delta\varpi = 5^\circ$', ha='right', va='bottom',
        fontsize=9, color='#d62728', alpha=0.9)

# Labels
ax.set_xticks(range(N))
ax.set_xticklabels(sorted_names, rotation=55, ha='right', fontsize=7.5)
ax.set_ylabel(r'$\Delta\bar{\varpi}$ (degrees)', fontsize=12)
ax.set_xlabel(r'Object (sorted by $\varpi$)', fontsize=12)

# Annotate bars with ϖ values for high-leverage objects
for i, (d, v) in enumerate(zip(sorted_deltas, sorted_varpi)):
    if d > 5:
        ax.annotate(f'{v:.0f}°', xy=(i, d), xytext=(i, d + 0.4),
                    ha='center', va='bottom', fontsize=7, color='#d62728',
                    fontweight='bold')

ax.set_ylim(0, max(deltas) * 1.25)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', edgecolor='#a51d1d', label=r'$\Delta\bar{\varpi} > 5^\circ$'),
    Patch(facecolor='#7f7f7f', edgecolor='#5a5a5a', label=r'$\Delta\bar{\varpi} \leq 5^\circ$'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)

# Title with key numbers
ax.set_title(
    f'Leave-One-Out Sensitivity: {sum(deltas > 5)} of {N} objects shift $\\bar{{\\varpi}}$ by $>5^\\circ$ '
    f'(full-sample $\\bar{{\\varpi}} = {cm_all_deg:.1f}^\\circ$)',
    fontsize=11, pad=10
)

plt.tight_layout()

# Save
outpath = '/mnt/d/第九行星/ray76/cometh/fig_sensitivity.png'
plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {outpath}")

# Also generate a data table for verification
print(f"\n--- Full leave-one-out table ---")
print(f"{'Object':16s} {'ϖ':>6s} {'Δϖ':>6s} {'ΔR%':>6s} {'Flag'}")
print("-" * 45)
for i in np.argsort(-deltas):
    flag = "← >5°" if deltas[i] > 5 else ""
    print(f"{names[i]:16s} {varpi_vals[i]:5.1f}° {deltas[i]:5.1f}° {R_changes[i]:5.1f}% {flag}")
