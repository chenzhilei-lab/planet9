#!/usr/bin/env python3
"""
model_b_fpr_from_paper_model.py — Use the paper's OWN survey model to compute FPR.

Extracts the survey_weight() function from generate_all_figures.py (L163-173),
which is the paper's actual per-survey, ϖ-dependent detection model. Runs
injection-recovery to verify FPR, then sweeps model parameters for uncertainty.

Method:
  1. Load ETNO data + survey assignments
  2. Use survey_weight(varpi, survey) as detection probability
  3. For each trial: uniform ϖ → bootstrap survey + properties → detect → p-value
  4. Compute FPR = fraction of trials with p < 0.05
  5. Sweep survey model parameters (scale weights, shift coverage)

Output: model_b_fpr_paper_model.txt, model_b_fpr_paper_model.png

Dependencies: numpy, scipy, matplotlib
"""

import json, os, sys
import numpy as np

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
N = len(varpi_deg_obs)
R_obs = np.abs(np.mean(np.exp(1j * varpi_rad_obs)))

# Survey assignments (from generate_all_figures.py SURVEY_MAP)
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

# Survey distribution in observed sample
survey_counts = {}
for s in survey_obs:
    survey_counts[s] = survey_counts.get(s, 0) + 1
survey_probs = {k: v/N for k, v in survey_counts.items()}
survey_names_list = list(survey_probs.keys())
survey_probs_list = [survey_probs[s] for s in survey_names_list]

# ---------------------------------------------------------------------------
# Paper's survey_weight function (from generate_all_figures.py L163-173)
# ---------------------------------------------------------------------------
def survey_weight(varpi_deg, survey_name):
    """Ecliptic-latitude weight per survey (from paper's generate_all_figures.py)."""
    b_model = varpi_deg / 360.0 * 60.0 - 30.0  # crude b proxy from varpi
    if survey_name == "Pan-STARRS":
        return 0.5 + 0.5 * np.sin(np.radians(b_model * 0.5))**2
    elif survey_name == "DES":
        return 0.3 + 0.7 * np.maximum(0, np.cos(np.radians(b_model - 180)))**2
    elif survey_name == "Subaru/HSC":
        return 0.4 + 0.3 * np.sin(np.radians(b_model * 0.3))**2
    else:
        return 0.5 + 0.3 * np.cos(np.radians(b_model * 0.7))**2

# ---------------------------------------------------------------------------
# Injection-recovery using paper's survey model
# ---------------------------------------------------------------------------
def run_injection_recovery_paper(n_trials, scale_factor=1.0, seed=42):
    """
    Injection-recovery using paper's survey_weight as detection probability.
    
    For each trial:
      1. Generate N uniform ϖ values
      2. Assign each synthetic object to a random survey (matching observed distribution)
      3. Compute detection prob = survey_weight(ϖ, survey) * scale_factor
      4. Detect each object with that probability
      5. Compute Rayleigh p on recovered sample
    
    scale_factor: multiplicative factor on detection weights (1.0 = baseline)
    """
    rng = np.random.default_rng(seed)
    p_vals = np.zeros(n_trials)
    n_detected = np.zeros(n_trials, dtype=int)
    R_syn_vals = np.zeros(n_trials)
    
    for j in range(n_trials):
        # Generate uniform ϖ
        varpi_syn_deg = rng.uniform(0, 360.0, N)
        
        # Assign random surveys matching observed distribution
        surv_assigned = rng.choice(survey_names_list, size=N, p=survey_probs_list)
        
        # Compute detection probs using paper's survey_weight
        p_detect = np.array([
            min(survey_weight(v, s) * scale_factor, 1.0)
            for v, s in zip(varpi_syn_deg, surv_assigned)
        ])
        
        # Detect
        detected = rng.random(N) < p_detect
        n_det = np.sum(detected)
        n_detected[j] = n_det
        
        if n_det >= 5:
            varpi_det = np.deg2rad(varpi_syn_deg[detected])
            R_syn = np.abs(np.mean(np.exp(1j * varpi_det)))
            R_syn_vals[j] = R_syn
            Z = n_det * R_syn**2
            p_val = np.exp(-Z) * (1.0 + Z)
            p_vals[j] = np.clip(p_val, 1e-15, 1.0)
        else:
            p_vals[j] = np.nan
            R_syn_vals[j] = np.nan
    
    valid = ~np.isnan(p_vals)
    fpr = np.mean(p_vals[valid] < 0.05) if np.sum(valid) > 0 else np.nan
    
    return {
        'fpr': fpr,
        'p_vals': p_vals[valid],
        'R_vals': R_syn_vals[valid],
        'n_det': n_detected,
        'n_valid': np.sum(valid),
    }

# ---------------------------------------------------------------------------
# Run baseline
# ---------------------------------------------------------------------------
N_TRIALS = 10000
SEED_BASE = 20260713

print("=" * 70)
print("FPR from Paper's Own Survey Model")
print("=" * 70)
print(f"  N = {N}, N_trials = {N_TRIALS}")
print(f"  Using survey_weight() from generate_all_figures.py L163-173")
print()

# Also compute: what does the survey model predict for detection probability
# as a function of ϖ? Average over all survey assignments.
varpi_grid = np.linspace(0, 360, 72)
mean_p_detect = np.zeros(72)
for i, v in enumerate(varpi_grid):
    probs = [survey_weight(v, s) * survey_probs[s] for s in survey_names_list]
    mean_p_detect[i] = np.sum(probs)

print(f"  Mean detection prob over ϖ: {np.mean(mean_p_detect):.4f}")
print(f"  Min/Max detection prob:     {np.min(mean_p_detect):.4f} / {np.max(mean_p_detect):.4f}")
print(f"  Ratio max/min:              {np.max(mean_p_detect)/np.min(mean_p_detect):.2f}")
print(f"  (Ratio > 1 means detection IS ϖ-dependent → FPR > 5% expected)")
print()

# Baseline
res_base = run_injection_recovery_paper(N_TRIALS, scale_factor=1.0, seed=SEED_BASE)
print(f"  BASELINE (scale=1.0):")
print(f"    FPR     = {res_base['fpr']*100:.1f}%")
print(f"    med N_det = {np.median(res_base['n_det']):.0f}")
print(f"    valid trials = {res_base['n_valid']}/{N_TRIALS}")
print()

# ---------------------------------------------------------------------------
# Sweep scale factor
# ---------------------------------------------------------------------------
scale_factors = np.arange(0.6, 1.5, 0.1)
print("--- Scale factor sweep ---")
print(f"{'scale':>8s}  {'FPR':>8s}  {'med N_det':>10s}")
print("-" * 35)

sweep_results = {}
for sf in scale_factors:
    res = run_injection_recovery_paper(N_TRIALS, scale_factor=sf, seed=SEED_BASE + int(sf*1000))
    sweep_results[sf] = res
    print(f"  {sf:5.1f}   {res['fpr']*100:6.1f}%  {np.median(res['n_det']):10.0f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
fpr_vals = [r['fpr'] for r in sweep_results.values()]
fpr_min = np.min(fpr_vals) * 100
fpr_max = np.max(fpr_vals) * 100
fpr_baseline = res_base['fpr'] * 100

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Baseline FPR   = {fpr_baseline:.1f}%")
print(f"  FPR range      = [{fpr_min:.1f}%, {fpr_max:.1f}%] (scale 0.6-1.4)")
print(f"  Paper's FPR    = 34% (claimed)")
print()

if abs(fpr_baseline - 34.0) < 10:
    print(f"  REPRODUCED: baseline FPR ({fpr_baseline:.1f}%) is within ~10pp of paper's 34%.")
    print(f"  The survey_weight model DOES produce elevated false positive rates.")
else:
    print(f"  NOTE: baseline FPR ({fpr_baseline:.1f}%) differs from paper's 34%.")
    print(f"  The paper's 34% may come from a different survey model or parameters.")

print("=" * 70)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Detection prob vs ϖ
ax1.plot(varpi_grid, mean_p_detect, 'steelblue', linewidth=2)
ax1.axhline(y=np.mean(mean_p_detect), color='gray', linestyle='--', alpha=0.5,
            label=f'Mean = {np.mean(mean_p_detect):.3f}')
ax1.set_xlabel('ϖ (deg)', fontsize=11)
ax1.set_ylabel('Mean detection probability', fontsize=11)
ax1.set_title('Survey Detection Probability vs ϖ', fontsize=12)
ax1.legend(fontsize=9)
ax1.set_xlim(0, 360)
ax1.grid(True, alpha=0.3)

# Right: FPR vs scale factor
sf_vals = sorted(sweep_results.keys())
fpr_plot = [sweep_results[sf]['fpr']*100 for sf in sf_vals]
ax2.plot(sf_vals, fpr_plot, 'o-', color='darkorange', linewidth=2, markersize=8)
ax2.axhline(y=34, color='red', linestyle='--', alpha=0.5, label="Paper's FPR = 34%")
ax2.axhline(y=5, color='gray', linestyle=':', alpha=0.5, label='Nominal α = 5%')
ax2.axvline(x=1.0, color='blue', linestyle=':', alpha=0.5, label='Baseline scale=1.0')
ax2.set_xlabel('Detection weight scale factor', fontsize=11)
ax2.set_ylabel('False positive rate (%)', fontsize=11)
ax2.set_title(f'FPR vs Survey Model Scale (N={N})', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'model_b_fpr_paper_model.png')
fig.savefig(outpath, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"\nFigure saved: {outpath}")

# Save
txtpath = os.path.join(OUT_DIR, 'model_b_fpr_paper_model.txt')
with open(txtpath, 'w') as f:
    f.write(f"FPR from Paper's Survey Model\n")
    f.write(f"N={N}, N_trials={N_TRIALS}\n")
    f.write(f"Baseline FPR = {fpr_baseline:.1f}%\n")
    f.write(f"FPR range (scale 0.6-1.4) = [{fpr_min:.1f}%, {fpr_max:.1f}%]\n")
    f.write(f"Paper's claimed FPR = 34%\n")
print(f"Results saved: {txtpath}")
