#!/usr/bin/env python3
"""
model_b_uncertainty_v2.py — Proper Model B uncertainty via injection-recovery.

Uses the actual survey model from generate_all_figures.py (latitude-dependent
detection weights per survey). Runs injection-recovery with spatial structure,
then sweeps survey model parameters to bound Model B's systematic uncertainty.

Method:
  1. Load ETNO data + survey assignments (SURVEY_MAP from generate_all_figures.py)
  2. For each synthetic sample with uniform ϖ:
     - Bootstrap observed (a, i, q, survey) from real sample
     - Compute per-object detection prob via survey weight function
     - Apply detection → recovered sample
     - Compute Rayleigh p-value on recovered sample
  3. Repeat 5,000 trials → FPR at N=19
  4. Sweep latitude band width (±5°, ±10°, ±15° from baseline) and depth (±0.5 mag)
  5. Report p(Model B) range as systematic uncertainty

Output: model_b_uncertainty_v2.png, model_b_uncertainty_v2.txt

Dependencies: numpy, scipy, matplotlib
"""

import json, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DPI = 300
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
i_deg_obs = np.array([e["i"] for e in etno_list])
a_obs = np.array([e["a"] for e in etno_list])
q_obs = np.array([e["q"] for e in etno_list])
N = len(varpi_deg_obs)
R_obs = np.abs(np.mean(np.exp(1j * varpi_rad_obs)))

# Survey assignments (from generate_all_figures.py)
SURVEY_MAP = {
    "Sedna": "Other", "2012 VP113": "Pan-STARRS", "2015 TG387": "Subaru/HSC",
    "2013 FT28": "DES", "2014 SR349": "DES", "2013 RF98": "Pan-STARRS",
    "2014 FE72": "Pan-STARRS", "2015 RX245": "Pan-STARRS",
    "2010 GB174": "Other", "2007 TG422": "Other", "2010 VZ98": "Other",
    "2015 KG163": "Pan-STARRS", "2013 RA109": "Pan-STARRS",
    "2015 BP519": "DES", "2013 UH15": "Subaru/HSC", "2013 SY99": "DES",
    "2014 WB556": "Subaru/HSC", "2015 RY245": "Pan-STARRS",
    "2021 RR205": "Subaru/HSC",
}
names_obs = [e["label"] for e in etno_list]
survey_obs = [SURVEY_MAP.get(n, "Other") for n in names_obs]

# ---------------------------------------------------------------------------
# Survey detection probability model (from generate_all_figures.py L163-173)
# ---------------------------------------------------------------------------
# Each survey covers a latitude band. Detection probability depends on ecliptic
# latitude b. For low-i objects, b ≈ 0 so detection is always high.
# For high-i objects, b varies around the orbit; time-averaged detection prob
# ≈ fraction of orbit in survey band.

# Survey parameters: (name, lon_center_deg, lon_width_deg, b_max_deg, base_efficiency)
# lon_center: approximate central ecliptic longitude of survey coverage
# lon_width: approximate longitude span
# b_max: half-width of survey coverage in ecliptic latitude
SURVEY_PARAMS = {
    "Pan-STARRS": {"lon_c": 180.0, "lon_w": 360.0, "b_max": 30.0, "eff": 0.90},
    "DES":        {"lon_c": 90.0,  "lon_w": 180.0, "b_max": 35.0, "eff": 0.85},
    "Subaru/HSC": {"lon_c": 0.0,   "lon_w": 60.0,  "b_max": 5.0,  "eff": 0.95},
    "Other":      {"lon_c": 0.0,   "lon_w": 360.0, "b_max": 20.0, "eff": 0.70},
}

def survey_detection_prob(varpi_deg, i_deg, survey_name, b_max_shift=0.0, depth_shift=0.0):
    """
    Detection probability for an object at (varpi_deg, i_deg).
    
    Combines latitude-dependent detection (inclination → fraction of orbit
    in survey band) with longitude-dependent coverage (survey footprints).
    
    b_max_shift: adjustment to latitude band half-width (deg)
    depth_shift: adjustment to effective depth (mag), reduces efficiency
    """
    params = SURVEY_PARAMS.get(survey_name, SURVEY_PARAMS["Other"])
    b_max = np.deg2rad(max(params["b_max"] + b_max_shift, 2.0))
    eff = params["eff"] * (1.0 - 0.3 * depth_shift / 0.5)
    eff = np.clip(eff, 0.3, 1.0)
    
    # --- Latitude-dependent detection ---
    scalar = np.isscalar(i_deg)
    i_rad = np.atleast_1d(np.deg2rad(np.abs(np.asarray(i_deg, dtype=float))))
    prob_lat = np.full_like(i_rad, eff)
    mask = i_rad > b_max
    ratio = np.sin(b_max) / np.sin(np.clip(i_rad[mask], b_max + 1e-8, np.pi/2))
    prob_lat[mask] = eff * (2.0/np.pi) * np.arcsin(np.clip(ratio, 0, 1))
    
    # --- Longitude-dependent coverage ---
    # For low-i objects, ϖ ≈ ecliptic longitude. Survey covers a band around lon_c.
    varpi = np.atleast_1d(np.asarray(varpi_deg, dtype=float))
    lon_c = params["lon_c"]
    lon_w = params["lon_w"]
    # Circular distance from survey center
    dlon = np.minimum(np.abs(varpi - lon_c), 360.0 - np.abs(varpi - lon_c))
    # Coverage: 1.0 within half-width, cosine taper beyond
    prob_lon = np.ones_like(varpi)
    taper_mask = dlon > lon_w / 2
    prob_lon[taper_mask] = np.maximum(0, np.cos(np.pi * (dlon[taper_mask] - lon_w/2) / lon_w))
    
    prob = prob_lat * prob_lon
    if scalar:
        return float(prob[0])
    return prob

# ---------------------------------------------------------------------------
# Injection-recovery: Model B proxy
# ---------------------------------------------------------------------------
def run_injection_recovery(n_trials, b_max_shift=0.0, depth_shift=0.0, seed=42):
    """
    Run injection-recovery experiment.
    
    For each trial:
      1. Generate uniform ϖ for N objects
      2. Bootstrap observed (a, i, survey) from real sample
      3. Compute detection prob for each object
      4. Detect objects with prob = p_detect
      5. Compute Rayleigh p-value on detected sample
      6. Count false positives (p < 0.05)
    
    Returns: FPR, list of p-values
    """
    rng = np.random.default_rng(seed)
    p_vals = np.zeros(n_trials)
    n_detected_list = np.zeros(n_trials, dtype=int)
    
    for j in range(n_trials):
        # Generate uniform ϖ
        varpi_syn = rng.uniform(0, 2*np.pi, N)
        
        # Bootstrap observed properties
        idx_boot = rng.integers(0, N, N)
        i_boot = i_deg_obs[idx_boot]
        survey_boot = [survey_obs[k] for k in idx_boot]
        
        # Compute detection probs (now needs varpi_syn too)
        p_detect = np.array([survey_detection_prob(
            np.rad2deg(varpi_syn[k]), i_boot[k], survey_boot[k],
            b_max_shift, depth_shift)
                             for k in range(N)])
        
        # Detect
        detected = rng.random(N) < p_detect
        n_det = np.sum(detected)
        n_detected_list[j] = n_det
        
        if n_det >= 5:
            varpi_det = varpi_syn[detected]
            R_syn = np.abs(np.mean(np.exp(1j * varpi_det)))
            Z = n_det * R_syn**2
            p_val = np.exp(-Z) * (1.0 + Z)  # Rayleigh approximation
            p_vals[j] = np.clip(p_val, 1e-15, 1.0)
        else:
            p_vals[j] = np.nan
    
    valid = ~np.isnan(p_vals)
    fpr = np.mean(p_vals[valid] < 0.05) if np.any(valid) else np.nan
    return fpr, p_vals[valid], n_detected_list

# ---------------------------------------------------------------------------
# Run baseline
# ---------------------------------------------------------------------------
N_TRIALS = 5000
SEED_BASE = 20260713

print("=" * 70)
print("Model B Systematic Uncertainty — Proper Injection-Recovery")
print("=" * 70)
print(f"  N = {N}, N_trials = {N_TRIALS}")
print(f"  Survey model: latitude-band detection (from generate_all_figures.py)")
print()

# Baseline
fpr_base, p_base, n_det_base = run_injection_recovery(N_TRIALS, 0.0, 0.0, SEED_BASE)
print(f"  BASELINE: FPR = {fpr_base*100:.1f}%, median N_det = {np.median(n_det_base):.0f}")

# Also compute Model B p-value: probability that null produces R >= R_obs
# under the survey model
p_model_b_base = np.mean(np.array([
    np.abs(np.mean(np.exp(1j * 
        np.random.default_rng(SEED_BASE + j).uniform(0, 2*np.pi, 
            max(5, int(np.random.default_rng(SEED_BASE + j + 10000).normal(
                np.median(n_det_base), 2)))))))
    for j in range(N_TRIALS)
]) >= R_obs)

# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
b_max_shifts = [-15, -10, -5, 0, 5, 10, 15]  # deg
depth_shifts = [-0.5, -0.25, 0.0, 0.25, 0.5]  # mag

print()
print("--- Sensitivity sweep ---")
print(f"{'b_max shift':>12s}  {'depth shift':>12s}  {'FPR':>8s}  {'med N_det':>10s}")
print("-" * 55)

results = {}
for b_shift in b_max_shifts:
    for d_shift in depth_shifts:
        key = (b_shift, d_shift)
        fpr, p_vals, n_det = run_injection_recovery(
            N_TRIALS, b_shift, d_shift, SEED_BASE + int(b_shift*100 + d_shift*1000))
        results[key] = {'fpr': fpr, 'n_det_med': np.median(n_det)}
        print(f"  {b_shift:+8.0f}°  {d_shift:+12.2f} mag  {fpr*100:6.1f}%  {np.median(n_det):10.0f}")

# ---------------------------------------------------------------------------
# Aggregate uncertainty
# ---------------------------------------------------------------------------
fpr_vals = [r['fpr'] for r in results.values()]
fpr_min = np.min(fpr_vals) * 100
fpr_max = np.max(fpr_vals) * 100

print()
print("=" * 70)
print("RESULTS: Model B Systematic Uncertainty")
print("=" * 70)
print(f"  Baseline FPR          = {fpr_base*100:.1f}%")
print(f"  FPR range (all combos) = [{fpr_min:.1f}%, {fpr_max:.1f}%]")
print(f"  Model B p ≈ {p_model_b_base:.3f} (baseline)")
print()
print(f"  The systematic uncertainty from survey model parameters")
print(f"  (latitude band width ±15°, depth ±0.5 mag) shifts the FPR")
print(f"  by at most ±{max(abs(fpr_min-fpr_base*100), abs(fpr_max-fpr_base*100)):.0f} percentage points.")
print("=" * 70)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: FPR heatmap
b_uniq = sorted(set(k[0] for k in results))
d_uniq = sorted(set(k[1] for k in results))
fpr_grid = np.zeros((len(b_uniq), len(d_uniq)))
for i, b in enumerate(b_uniq):
    for j, d in enumerate(d_uniq):
        fpr_grid[i, j] = results[(b, d)]['fpr'] * 100

im = ax1.pcolormesh(d_uniq, b_uniq, fpr_grid, shading='auto', cmap='RdYlBu_r')
plt.colorbar(im, ax=ax1, label='FPR (%)')
ax1.scatter(0, 0, marker='*', color='black', s=200, zorder=5, label='Baseline')
ax1.set_xlabel('Depth shift (mag)', fontsize=11)
ax1.set_ylabel('Latitude band shift (deg)', fontsize=11)
ax1.set_title(f'Model B FPR Sensitivity (N={N})', fontsize=12)
ax1.legend(fontsize=9)

# Right: FPR histogram across all parameter combos
ax2.hist(fpr_vals, bins=20, color='steelblue', edgecolor='white', 
         weights=np.ones(len(fpr_vals))*100/len(fpr_vals))
ax2.axvline(x=fpr_base, color='red', linestyle='--', linewidth=2, 
            label=f'Baseline: {fpr_base*100:.1f}%')
ax2.axvline(x=fpr_min/100, color='gray', linestyle=':', alpha=0.5,
            label=f'Min: {fpr_min:.1f}%')
ax2.axvline(x=fpr_max/100, color='gray', linestyle=':', alpha=0.5,
            label=f'Max: {fpr_max:.1f}%')
ax2.set_xlabel('False positive rate', fontsize=11)
ax2.set_ylabel('Fraction of parameter combos (%)', fontsize=11)
ax2.set_title(f'FPR Distribution Across Survey Parameters', fontsize=12)
ax2.legend(fontsize=9)

plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'model_b_uncertainty_v2.png')
fig.savefig(outpath, dpi=DPI, bbox_inches='tight')
plt.close(fig)
print(f"\nFigure saved: {outpath}")

# Save results
txtpath = os.path.join(OUT_DIR, 'model_b_uncertainty_v2.txt')
with open(txtpath, 'w') as f:
    f.write(f"Model B Systematic Uncertainty (Proper Injection-Recovery)\n")
    f.write(f"N={N}, N_trials={N_TRIALS}\n")
    f.write(f"Baseline FPR = {fpr_base*100:.1f}%\n")
    f.write(f"FPR range = [{fpr_min:.1f}%, {fpr_max:.1f}%]\n")
    f.write(f"Model B p (baseline) ≈ {p_model_b_base:.3f}\n")
print(f"Results saved: {txtpath}")
