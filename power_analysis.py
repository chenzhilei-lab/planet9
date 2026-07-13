#!/usr/bin/env python3
"""
Power Analysis v2 — Required N for 80% detection power
======================================================
Uses analytic power (non-central chi2) for idealized case,
and effective-sample-size model for survey-biased case.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.special import i0, i1

def analytic_power(kappa, N, alpha=0.05):
    """Analytic power for Rayleigh test via non-central chi2."""
    mu_R = i1(kappa) / i0(kappa)
    ncp = 2 * N * mu_R**2
    crit = chi2.ppf(1 - alpha, 2)
    return chi2.sf(crit, 2, ncp) * 100

def find_N_for_power(kappa, target_power=80, N_max=300, survey_eff=1.0):
    """Find smallest N achieving target_power, given survey efficiency."""
    for N in range(3, N_max + 1):
        Neff = max(3, int(N * survey_eff))
        if analytic_power(kappa, Neff) >= target_power:
            return N
    return None

print("=" * 60)
print("Power Analysis v2")
print("=" * 60)

kappa_grid = [0.2, 0.3, 0.5, 0.8, 1.17, 1.5, 2.0, 3.0]
results = []

for kv in kappa_grid:
    # Idealized (survey_eff = 1.0)
    N_ideal = find_N_for_power(kv, survey_eff=1.0)
    
    # Survey-biased (Neff ≈ 14.5/19 ≈ 76% efficiency)
    N_survey = find_N_for_power(kv, survey_eff=0.76)
    
    results.append((kv, N_ideal, N_survey))
    print(f"  κ={kv:.2f}:  N_ideal={N_ideal},  N_survey={N_survey}")

# Generate figure
fig = plt.figure(figsize=(10, 5))

# Panel A: Full power curves
ax1 = fig.add_subplot(121)
N_range = np.arange(5, 301, 1)
for kv, color, ls, label in [
    (0.5, '#377eb8', '-', r'$\kappa=0.5$ (weak)'),
    (1.17, '#e41a1c', '-', r'$\kappa=1.17$ (observed)'),
    (2.0, '#4daf4a', '-', r'$\kappa=2.0$ (strong)'),
]:
    # Idealized
    pwr = [analytic_power(kv, N) for N in N_range]
    ax1.plot(N_range, pwr, color=color, ls='-', lw=1.5,
             label=label + ' (idealized)')
    # Survey-biased
    pwr_s = [analytic_power(kv, max(3, int(N * 0.76))) for N in N_range]
    ax1.plot(N_range, pwr_s, color=color, ls='--', lw=1.2, alpha=0.6)

ax1.axhline(80, color='gray', ls='--', lw=1)
ax1.text(290, 81, '80% power', fontsize=9, color='gray', ha='right')
ax1.axvline(19, color='red', ls=':', lw=1.5)
ax1.text(19, 2, 'N=19', fontsize=9, color='red', ha='center')
ax1.set_xlim(0, 300)
ax1.set_ylim(0, 102)
ax1.set_xlabel('Sample size N', fontsize=11)
ax1.set_ylabel('Detection power (%)', fontsize=11)
ax1.set_title('Rayleigh test power\n(solid = idealized, dashed = survey-biased)', fontsize=10)
ax1.legend(fontsize=7, loc='lower right')
ax1.grid(alpha=0.3)

# Panel B: Required N bar chart
ax2 = fig.add_subplot(122)
x = np.arange(len(kappa_grid))
w = 0.35

N_ideals = [r[1] if r[1] else 300 for r in results]
N_surveys = [r[2] if r[2] else 300 for r in results]

bars1 = ax2.bar(x - w/2, N_ideals, w, color='steelblue', alpha=0.85,
                label='Idealized')
bars2 = ax2.bar(x + w/2, N_surveys, w, color='#e74c3c', alpha=0.85,
                label='Survey-biased (N$_{\\rm eff}$≈0.76N)')

# Annotate N=19 threshold
ax2.axhline(19, color='red', ls=':', lw=1.5)
ax2.text(7.2, 20, 'Current N=19', fontsize=8, color='red')

# Add N labels on bars
for i, (iv, sv) in enumerate(zip(N_ideals, N_surveys)):
    if iv < 300:
        ax2.text(i - w/2, iv + 3, str(iv), fontsize=7, ha='center', color='steelblue')
    if sv < 300:
        ax2.text(i + w/2, sv + 3, str(sv), fontsize=7, ha='center', color='#e74c3c')

ax2.set_xticks(x)
ax2.set_xticklabels([f'{k:.2f}' for k in kappa_grid], fontsize=8)
ax2.set_xlabel('Clustering strength κ', fontsize=11)
ax2.set_ylabel('N needed for 80% power', fontsize=11)
ax2.set_title('Required sample size for 80% power', fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/d/Papers/MNRAS/fig_required_N.png', dpi=200, bbox_inches='tight')
print(f"\nFigure saved: fig_required_N.png")
print("\nDone.")
