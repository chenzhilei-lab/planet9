#!/usr/bin/env python3
"""
model_c_selection_likelihood.py — Proper Model C likelihood profile.

Model C (Napier et al.) simultaneously fits the intrinsic ϖ distribution
AND the survey selection function. This creates a fundamental degeneracy:
at small N, you cannot distinguish intrinsic clustering from selection bias.

Implementation:
  Intrinsic distribution:  P_int(ϖ) ∝ exp(κ cos(ϖ - μ))    [von Mises]
  Selection function:      S(ϖ)   ∝ 1 + ε cos(ϖ - ϕ)       [1st Fourier mode]
  Observed distribution:   P_obs(ϖ) ∝ P_int(ϖ) × S(ϖ)

The selection function parameter ε captures latitude-dependent detection bias.
At N=19, fitting 4 parameters (μ, κ, ε, ϕ) is underdetermined. We compute the
profile likelihood: fix κ, optimize (μ, ε, ϕ), report NLL(κ).

This directly addresses the reviewer's question about Model C's likelihood
flatness with the selection function properly included.

Output: model_c_selection_likelihood.png, model_c_selection_likelihood.txt

Dependencies: numpy, scipy, matplotlib
"""

import json, os, sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0, i1
from scipy.integrate import quad
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
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(DATA_PATH) as f:
    etno_list = json.load(f)

varpi_deg = np.array([e["varpi"] for e in etno_list])
varpi_rad = np.deg2rad(varpi_deg)
N = len(varpi_rad)

# ---------------------------------------------------------------------------
# Normalization integral: Z(μ, κ, ε, ϕ) = ∫₀^{2π} P_int × S dϖ
# ---------------------------------------------------------------------------
# P_int(ϖ) ∝ exp(κ cos(ϖ - μ))
# S(ϖ)    ∝ 1 + ε cos(ϖ - ϕ)   with |ε| < 1
#
# Z(μ, κ, ε, ϕ) = ∫₀^{2π} exp(κ cos(ϖ - μ)) × (1 + ε cos(ϖ - ϕ)) dϖ
#                = 2π I₀(κ) + ε ∫₀^{2π} exp(κ cos(ϖ - μ)) cos(ϖ - ϕ) dϖ
#
# Use identity: ∫₀^{2π} exp(κ cos θ) cos(θ - α) dθ = 2π I₁(κ) cos(α)
# where θ = ϖ - μ, α = μ - ϕ
#
# So: ∫₀^{2π} exp(κ cos(ϖ - μ)) cos(ϖ - ϕ) dϖ = 2π I₁(κ) cos(μ - ϕ)
#
# Therefore: Z(μ, κ, ε, ϕ) = 2π [I₀(κ) + ε I₁(κ) cos(μ - ϕ)]

def log_normalization(mu, kappa, eps, phi):
    """Log of normalization constant Z(μ, κ, ε, ϕ)."""
    # Z = 2π [I₀(κ) + ε I₁(κ) cos(μ - ϕ)]
    # Require Z > 0: |ε I₁(κ)/I₀(κ)| < 1/cos(μ-ϕ)... simpler: require ε small enough
    z_inner = i0(kappa) + eps * i1(kappa) * np.cos(mu - phi)
    if z_inner <= 0:
        return -1e10  # invalid parameter region
    return np.log(2 * np.pi) + np.log(max(z_inner, 1e-300))

def negative_log_likelihood(params, kappa_fixed=None):
    """NLL for Model C. If kappa_fixed is set, κ is fixed and not optimized."""
    if kappa_fixed is not None:
        mu, eps, phi = params
        kappa_val = kappa_fixed
    else:
        mu, log_kappa, eps, phi = params
        kappa_val = np.exp(log_kappa)
    
    # Bounds check
    if abs(eps) >= 1.0:
        return 1e10
    if abs(mu) > 2 * np.pi:
        return 1e10
    
    log_Z = log_normalization(mu, kappa_val, eps, phi)
    if log_Z < -1e9:
        return 1e10
    
    # log-likelihood = Σ [κ cos(ϖᵢ - μ) + log(1 + ε cos(ϖᵢ - ϕ))] - N log Z
    ll = np.sum(kappa_val * np.cos(varpi_rad - mu) + 
                np.log(np.maximum(1.0 + eps * np.cos(varpi_rad - phi), 1e-10)))
    nll = -(ll - N * log_Z)
    return nll

# ---------------------------------------------------------------------------
# Profile likelihood: fix κ, optimize (μ, ε, ϕ)
# ---------------------------------------------------------------------------
def profile_nll_at_kappa(kappa_val, n_restarts=10):
    """Profile NLL at fixed κ, optimizing μ, ε, ϕ from multiple starts."""
    rng = np.random.default_rng(42)
    best_nll = np.inf
    
    for _ in range(n_restarts):
        # Random initial guess
        mu0 = rng.uniform(0, 2*np.pi)
        eps0 = rng.uniform(-0.5, 0.5)
        phi0 = rng.uniform(0, 2*np.pi)
        
        try:
            result = minimize(
                lambda p: negative_log_likelihood(p, kappa_fixed=kappa_val),
                x0=[mu0, eps0, phi0],
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue
    
    return best_nll, best_params

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------
print("=" * 70)
print("Model C — Joint Likelihood (Intrinsic + Selection Function)")
print("=" * 70)
print(f"  N = {N}")
print(f"  Model: P_obs(ϖ) ∝ exp(κ cos(ϖ - μ)) × (1 + ε cos(ϖ - ϕ))")
print(f"  Parameters: μ (mean), κ (concentration), ε (selection strength), ϕ (selection phase)")
print()

# Pure von Mises (no selection) — for comparison
nll_vm_fixed = {}
for k in [0.001, 0.5, 1.0, 1.17, 2.0, 2.5, 3.0]:
    nll_vm_fixed[k] = profile_nll_at_kappa(k, n_restarts=3)[0]  # quick, few restarts

# With selection function — proper Model C
kappa_grid = np.linspace(0.001, 3.5, 30)
nll_model_c = np.zeros(len(kappa_grid))
eps_model_c = np.zeros(len(kappa_grid))

print("Computing profile likelihood (this takes 2-3 minutes)...")
print()
print(f"{'κ':>6s}  {'NLL':>10s}  {'ε':>8s}  {'ΔNLL vs κ=0':>14s}")
print("-" * 50)

for i, kappa_val in enumerate(kappa_grid):
    nll_val, params = profile_nll_at_kappa(kappa_val, n_restarts=15)
    nll_model_c[i] = nll_val
    eps_model_c[i] = params[1]  # ε
    
    delta_vs_zero = nll_val - nll_model_c[0]  # vs κ≈0
    print(f" {kappa_val:5.2f}  {nll_val:10.2f}  {params[1]:+8.4f}  {delta_vs_zero:+14.4f}")

# ---------------------------------------------------------------------------
# Also compute pure von Mises profile (no selection) on fine grid for comparison
# ---------------------------------------------------------------------------
kappa_fine = np.linspace(0.001, 3.5, 100)
from scipy.special import i0e
nll_vm_fine = np.zeros(len(kappa_fine))
for i, k in enumerate(kappa_fine):
    # Pure von Mises NLL = N log(2π I₀(κ)) - κ N R_obs
    log_i0 = k + np.log(i0e(k) + 1e-300)
    R_obs = np.abs(np.mean(np.exp(1j * varpi_rad)))
    nll_vm_fine[i] = N * (np.log(2*np.pi) + log_i0) - k * R_obs * N

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
nll_best = np.min(nll_model_c)
idx_best = np.argmin(nll_model_c)
kappa_best = kappa_grid[idx_best]

nll_at_0 = nll_model_c[0]  # κ ≈ 0
nll_at_25_idx = np.argmin(np.abs(kappa_grid - 2.5))
nll_at_25 = nll_model_c[nll_at_25_idx]
delta_0_25 = abs(nll_at_0 - nll_at_25)

print()
print("=" * 70)
print("RESULTS: Model C Likelihood Flatness")
print("=" * 70)
print(f"  Pure von Mises:         ΔNLL(κ=0 → 2.5) = {abs(nll_vm_fine[0] - nll_vm_fine[-1]):.2f}")
print(f"  Model C (with sel.fn.): ΔNLL(κ=0 → 2.5) = {delta_0_25:.2f}")
print(f"  Best-fit κ              = {kappa_best:.2f}")
print(f"  Best-fit ε              = {eps_model_c[idx_best]:.4f}")
print(f"  κ range, ΔNLL < 0.5:    ", end="")
mask_flat = (nll_model_c - nll_best) < 0.5
if np.any(mask_flat):
    k_flat = kappa_grid[mask_flat]
    print(f"[{k_flat[0]:.2f}, {k_flat[-1]:.2f}]")
else:
    print("none")
print(f"  κ range, ΔNLL < 2.0:    ", end="")
mask_ci = (nll_model_c - nll_best) < 2.0
if np.any(mask_ci):
    k_ci = kappa_grid[mask_ci]
    print(f"[{k_ci[0]:.2f}, {k_ci[-1]:.2f}]")
else:
    print("none")
print()
print(f"  KEY: With selection function, ΔNLL(0→2.5) = {delta_0_25:.2f}")
if delta_0_25 < 0.5:
    print(f"  This is < 0.5, confirming the paper's claim at L302.")
else:
    print(f"  This is {'< 2.0' if delta_0_25 < 2.0 else '≥ 2.0'}.")
    print(f"  The paper's claim of ΔNLL < 0.5 requires more careful selection modeling.")
print("=" * 70)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: NLL comparison
# Shift both to NLL=0 at their respective minima
nll_vm_shifted = nll_vm_fine - np.min(nll_vm_fine)
nll_mc_shifted = nll_model_c - nll_best

ax1.plot(kappa_fine, nll_vm_shifted, 'gray', linewidth=1.5, alpha=0.6,
         label='Pure von Mises (no selection)')
ax1.plot(kappa_grid, nll_mc_shifted, 'o-', color='steelblue', linewidth=2, 
         markersize=5, label='Model C (with selection function)')
ax1.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='ΔNLL = 0.5')
ax1.axhline(y=2.0, color='red', linestyle='-.', alpha=0.7, label='ΔNLL = 2.0 (≈95% CI)')

# Shade flat region for Model C
if np.any(mask_flat):
    ax1.fill_between(kappa_grid, 0, nll_mc_shifted, where=mask_flat,
                     color='steelblue', alpha=0.12)

# Annotate
delta_str = f"ΔNLL(0→2.5)\n= {delta_0_25:.2f}"
ann_x = 2.5
ann_y = nll_mc_shifted[nll_at_25_idx]
ax1.annotate(delta_str, xy=(ann_x, ann_y),
             xytext=(ann_x + 0.5, ann_y + 2.0),
             fontsize=9, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='gray'))

ax1.set_xlabel('κ (von Mises concentration)', fontsize=11)
ax1.set_ylabel('ΔNLL from best fit', fontsize=11)
ax1.set_title(f'Model C Profile Likelihood — With vs Without Selection (N={N})', fontsize=12)
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 8)

# Right: selection function parameter ε vs κ
ax2.plot(kappa_grid, eps_model_c, 'o-', color='darkorange', linewidth=2, markersize=5)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axvline(x=kappa_best, color='red', linestyle=':', alpha=0.5,
            label=f'Best-fit κ = {kappa_best:.2f}')
ax2.set_xlabel('κ (von Mises concentration)', fontsize=11)
ax2.set_ylabel('Selection strength ε', fontsize=11)
ax2.set_title(f'Selection Function Parameter ε vs κ', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Annotate: when ε ≠ 0, selection is absorbing clustering signal
ax2.annotate('ε > 0: selection function\nabsorbs clustering signal',
             xy=(kappa_grid[5], eps_model_c[5]),
             xytext=(kappa_grid[5] + 0.5, eps_model_c[5] + 0.1),
             fontsize=8, ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
             arrowprops=dict(arrowstyle='->', color='gray'))

plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'model_c_selection_likelihood.png')
fig.savefig(outpath, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"\nFigure saved: {outpath}")

# Save results
txtpath = os.path.join(OUT_DIR, 'model_c_selection_likelihood.txt')
with open(txtpath, 'w') as f:
    f.write(f"Model C Joint Likelihood (Intrinsic + Selection Function)\n")
    f.write(f"N = {N}\n")
    f.write(f"Pure von Mises: ΔNLL(0→2.5) = {abs(nll_vm_fine[0] - nll_vm_fine[-1]):.2f}\n")
    f.write(f"Model C: ΔNLL(0→2.5) = {delta_0_25:.2f}\n")
    f.write(f"Best-fit κ = {kappa_best:.2f}, ε = {eps_model_c[idx_best]:.4f}\n")
    for i, k in enumerate(kappa_grid):
        f.write(f"  κ={k:.2f}  NLL={nll_model_c[i]:.2f}  ε={eps_model_c[i]:+.4f}\n")
print(f"Results saved: {txtpath}")
