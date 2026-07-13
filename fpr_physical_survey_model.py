#!/usr/bin/env python3
"""
fpr_physical_survey_model.py — FPR from ecliptic coverage + orbital mechanics.

Each survey covers a rectangle in ecliptic coordinates (λ, b).
For each synthetic ETNO (with observed orbital elements but uniform ϖ),
we compute the fraction of its orbit inside each survey's footprint.
Detection prob = max over surveys of that fraction.

The mechanism: different surveys cover different λ ranges. Objects with
different Ω values have different longitude distributions. This creates
ϖ-dependent detection probability → false clustering under uniform null.

Output: fpr_physical_model.txt, fpr_physical_model.png

Dependencies: numpy, scipy, matplotlib
"""

import json, os, sys
import numpy as np
from scipy.stats import binom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(os.path.join(SCRIPT_DIR, "etno_complete.json")) as f:
    etno_list = json.load(f)

varpi_deg_obs = np.array([e["varpi"] for e in etno_list])
varpi_rad_obs = np.deg2rad(varpi_deg_obs)
# Need Ω and ω individually for orbit integration
Omega_deg_obs = np.array([e.get("Omega", 0.0) for e in etno_list])
omega_deg_obs = np.array([e.get("omega", 0.0) for e in etno_list])
# If Ω, ω not in JSON, derive from ϖ (approximate)
# ϖ = Ω + ω, but we don't know the split. Default: ω = ϖ/2, Ω = ϖ/2
mask_no_Omega = np.abs(Omega_deg_obs) < 1e-10
Omega_deg_obs[mask_no_Omega] = varpi_deg_obs[mask_no_Omega] * 0.5
omega_deg_obs[mask_no_Omega] = varpi_deg_obs[mask_no_Omega] * 0.5

i_deg_obs = np.array([e["i"] for e in etno_list])
e_obs = np.array([e["e"] for e in etno_list])
N = len(varpi_deg_obs)
R_obs = np.abs(np.mean(np.exp(1j * varpi_rad_obs)))

# ---------------------------------------------------------------------------
# Survey coverage rectangles in ecliptic (λ, b) coordinates
# λ = ecliptic longitude (0-360°), b = ecliptic latitude
# Coverage from published survey papers (approximate)
# ---------------------------------------------------------------------------
SURVEYS = {
    "Pan-STARRS": {"lam_min": 0.0,   "lam_max": 360.0, "b_min": -30.0, "b_max": 30.0},
    "DES":        {"lam_min": 0.0,   "lam_max": 180.0, "b_min": -35.0, "b_max": 35.0},
    "Subaru/HSC": {"lam_min": 330.0, "lam_max": 60.0,  "b_min": -5.0,  "b_max": 5.0},
    "OSSOS":      {"lam_min": 0.0,   "lam_max": 360.0, "b_min": -2.0,  "b_max": 2.0},
    "Other":      {"lam_min": 0.0,   "lam_max": 360.0, "b_min": -20.0, "b_max": 20.0},
}

# Wrap-around for Subaru/HSC: lam_min=330, lam_max=60 means coverage across 0°
def longitude_in_range(lam, lam_min, lam_max):
    """Check if longitude λ is in [lam_min, lam_max] with wrap-around."""
    if lam_max >= lam_min:
        return (lam >= lam_min) & (lam <= lam_max)
    else:
        return (lam >= lam_min) | (lam <= lam_max)

# ---------------------------------------------------------------------------
# Orbital sampling: ecliptic coords at N_mean_anomaly points
# ---------------------------------------------------------------------------
N_ORBIT_PTS = 50  # sample points per orbit

def orbit_to_ecliptic(varpi_deg, Omega_deg, i_deg, e_val, n_pts=N_ORBIT_PTS):
    """
    Sample N_ORBIT_PTS points along the orbit, return (λ, b) in degrees.
    
    For low-e orbits, we approximate true anomaly ≈ mean anomaly.
    λ = Ω + arctan(cos(i) * tan(ω + M))   [approx for low e]
    sin(b) = sin(i) * sin(ω + M)
    
    ω = varpi - Ω
    """
    omega_deg = (varpi_deg - Omega_deg) % 360.0
    Omega_rad = np.deg2rad(Omega_deg)
    omega_rad = np.deg2rad(omega_deg)
    i_rad = np.deg2rad(i_deg)
    
    M_vals = np.linspace(0, 2*np.pi, n_pts)
    omega_plus_M = omega_rad + M_vals
    
    # Ecliptic latitude
    sin_b = np.sin(i_rad) * np.sin(omega_plus_M)
    b = np.arcsin(np.clip(sin_b, -1, 1))
    
    # Ecliptic longitude
    cos_i = np.cos(i_rad)
    lam = Omega_rad + np.arctan2(cos_i * np.sin(omega_plus_M), 
                                   np.cos(omega_plus_M))
    lam = np.mod(lam, 2*np.pi)
    
    return np.rad2deg(lam), np.rad2deg(b)

def detection_probability(varpi_deg, Omega_deg, i_deg, e_val):
    """
    Probability of detection = max over surveys of (fraction of orbit in coverage).
    """
    lam_pts, b_pts = orbit_to_ecliptic(varpi_deg, Omega_deg, i_deg, e_val)
    
    best_frac = 0.0
    for sname, cov in SURVEYS.items():
        in_lam = longitude_in_range(lam_pts, cov["lam_min"], cov["lam_max"])
        in_b = (b_pts >= cov["b_min"]) & (b_pts <= cov["b_max"])
        frac = np.mean(in_lam & in_b)
        if frac > best_frac:
            best_frac = frac
    
    return best_frac

# ---------------------------------------------------------------------------
# Test: detection prob for observed sample
# ---------------------------------------------------------------------------
print("=" * 70)
print("FPR from Physical Survey Model (Ecliptic Coverage + Orbits)")
print("=" * 70)
print(f"  N = {N}")
print(f"  N_orbit_pts = {N_ORBIT_PTS}")
print()

p_detect_obs = np.array([
    detection_probability(varpi_deg_obs[j], Omega_deg_obs[j], i_deg_obs[j], e_obs[j])
    for j in range(N)
])
print(f"  Observed detection probs: mean={np.mean(p_detect_obs):.3f}, "
      f"min={np.min(p_detect_obs):.3f}, max={np.max(p_detect_obs):.3f}")

# Does detection prob correlate with ϖ?
corr_pdet_varpi = np.corrcoef(p_detect_obs, varpi_deg_obs)[0, 1]
print(f"  Corr(p_detect, ϖ) = {corr_pdet_varpi:.4f}")
if abs(corr_pdet_varpi) > 0.1:
    print(f"  → Detection IS ϖ-dependent → FPR > 5% expected")
else:
    print(f"  → Detection nearly ϖ-independent → FPR ≈ 5% expected")
print()

# ---------------------------------------------------------------------------
# Injection-recovery
# ---------------------------------------------------------------------------
N_TRIALS = 5000
SEED = 20260713

def run_fpr_physical(n_trials, seed):
    rng = np.random.default_rng(seed)
    fpr_count = 0
    p_list = []
    
    for j in range(n_trials):
        # Uniform ϖ
        varpi_syn = rng.uniform(0, 360.0, N)
        
        # Bootstrap observed (Ω, i, e) — these are NOT randomized
        idx_boot = rng.integers(0, N, N)
        Omega_boot = Omega_deg_obs[idx_boot]
        i_boot = i_deg_obs[idx_boot]
        e_boot = e_obs[idx_boot]
        
        # Compute detection probs
        p_detect = np.array([
            detection_probability(varpi_syn[k], Omega_boot[k], i_boot[k], e_boot[k])
            for k in range(N)
        ])
        
        # Detect
        detected = rng.random(N) < p_detect
        n_det = np.sum(detected)
        
        if n_det >= 5:
            varpi_det_rad = np.deg2rad(varpi_syn[detected])
            R = np.abs(np.mean(np.exp(1j * varpi_det_rad)))
            Z = n_det * R**2
            p_val = np.exp(-Z) * (1.0 + Z)
            p_val = np.clip(p_val, 1e-15, 1.0)
            p_list.append(p_val)
            if p_val < 0.05:
                fpr_count += 1
        else:
            p_list.append(np.nan)
    
    fpr = fpr_count / n_trials
    p_arr = np.array(p_list)
    valid = ~np.isnan(p_arr)
    
    if fpr > 0:
        ci_lo = binom.ppf(0.025, n_trials, fpr) / n_trials
        ci_hi = binom.ppf(0.975, n_trials, fpr) / n_trials
    else:
        ci_lo, ci_hi = 0.0, 0.0
    
    return {'fpr': fpr, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'valid': np.sum(valid), 'median_p': np.median(p_arr[valid])}

print(f"Running {N_TRIALS} injection-recovery trials...")
print(f"Each trial: N objects × {N_ORBIT_PTS} orbit points × 5 surveys")
res = run_fpr_physical(N_TRIALS, SEED)

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print(f"  FPR = {res['fpr']*100:.1f}%")
print(f"  95% CI = [{res['ci_lo']*100:.1f}%, {res['ci_hi']*100:.1f}%]")
print(f"  Valid trials = {res['valid']}/{N_TRIALS}")
print(f"  Median p = {res['median_p']:.4f}")
print()
print(f"  The physical survey model (ecliptic coverage + orbit integration)")
print(f"  gives FPR = {res['fpr']*100:.1f}%. This replaces the previous 34% value.")
print("=" * 70)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
# Null R distribution
rng = np.random.default_rng(SEED)
R_null = np.zeros(2000)
for j in range(2000):
    varpi_syn = rng.uniform(0, 360.0, N)
    idx_boot = rng.integers(0, N, N)
    Omega_boot = Omega_deg_obs[idx_boot]
    i_boot = i_deg_obs[idx_boot]
    e_boot = e_obs[idx_boot]
    p_detect = np.array([
        detection_probability(varpi_syn[k], Omega_boot[k], i_boot[k], e_boot[k])
        for k in range(N)
    ])
    detected = rng.random(N) < p_detect
    if np.sum(detected) >= 5:
        R_null[j] = np.abs(np.mean(np.exp(1j * np.deg2rad(varpi_syn[detected]))))
    else:
        R_null[j] = np.nan
R_null = R_null[~np.isnan(R_null)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: detection prob vs ϖ
varpi_grid = np.linspace(0, 360, 72)
p_mean = np.zeros(72)
for i, v in enumerate(varpi_grid):
    probs = [detection_probability(v, Omega_deg_obs[k], i_deg_obs[k], e_obs[k])
             for k in range(N)]
    p_mean[i] = np.mean(probs)

ax1.plot(varpi_grid, p_mean, 'steelblue', linewidth=2)
ax1.axhline(y=np.mean(p_mean), color='gray', linestyle='--', alpha=0.5,
            label=f'Mean = {np.mean(p_mean):.3f}')
ax1.set_xlabel('ϖ (deg)', fontsize=11)
ax1.set_ylabel('Mean detection probability', fontsize=11)
ax1.set_title(f'Detection Prob vs ϖ (Physical Model)', fontsize=12)
ax1.legend(fontsize=9)
ax1.set_xlim(0, 360)
ax1.grid(True, alpha=0.3)

# Right: null R distribution
ax2.hist(R_null, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax2.axvline(x=R_obs, color='red', linestyle='--', linewidth=2,
            label=f'Observed R = {R_obs:.3f}')
ax2.axvline(x=np.percentile(R_null, 95), color='gray', linestyle=':',
            label=f'Null 95% = {np.percentile(R_null, 95):.3f}')
ax2.set_xlabel('Rayleigh R', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title(f'Null R Distribution (FPR={res["fpr"]*100:.1f}%)', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'fpr_physical_model.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nFigure saved: {outpath}")

txtpath = os.path.join(OUT_DIR, 'fpr_physical_model.txt')
with open(txtpath, 'w') as f:
    f.write(f"FPR from Physical Survey Model\n")
    f.write(f"N={N}, N_trials={N_TRIALS}, N_orbit_pts={N_ORBIT_PTS}\n")
    f.write(f"FPR = {res['fpr']*100:.1f}% [{res['ci_lo']*100:.1f}%, {res['ci_hi']*100:.1f}%]\n")
    f.write(f"Median p = {res['median_p']:.4f}\n")
print(f"Results saved: {txtpath}")
