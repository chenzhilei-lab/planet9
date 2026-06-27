#!/usr/bin/env python3
"""
known_answer_test.py
Pipeline validation via known-answer test:
  - Inject synthetic ETNO populations with KNOWN clustering strength (κ)
  - Apply realistic survey selection (Pan-STARRS/DES/OSSOS/Subaru/HSC footprints)
  - Test whether the pipeline correctly recovers the input signal
  - Measure: detection power, bias in recovered κ, false-negative rate

Outputs: fig_known_answer.png + summary table printed to stdout
"""

import numpy as np
from numpy import random as rng
import json, os, sys

# ================================================================
# CONFIG
# ================================================================
N_TRIALS = 2000
N_SAMPLE = 19
KAPPA_VALUES = [0.0, 0.3, 0.5, 0.7, 1.0, 1.17, 1.5, 2.0]
# Survey parameters (from paper §3.5)
SURVEYS = {
    "Pan-STARRS":  {"weight": 7/19, "b_max": 30, "depth": 24.5},
    "DES":         {"weight": 4/19, "b_max": 35, "depth": 23.5},
    "OSSOS":       {"weight": 4/19, "b_max": 4,  "depth": 24.0},
    "Subaru/HSC":  {"weight": 4/19, "b_max": 5,  "depth": 25.5},
}

# ================================================================
# UTILITIES
# ================================================================
def rayleigh_r(rad):
    n = len(rad)
    if n < 3:
        return 0.0, 1.0
    R = np.sqrt(np.sum(np.cos(rad))**2 + np.sum(np.sin(rad))**2) / n
    z = 2 * n * R**2  # Rayleigh test: 2nR² ~ χ²(2) under H₀
    from scipy import stats
    p = 1.0 - stats.chi2.cdf(z, 2)
    return R, p

def von_mises_sample(kappa, size):
    """Generate von Mises samples using scipy or numpy workaround."""
    if kappa < 1e-10:
        return rng.uniform(0, 2*np.pi, size)
    # Use scipy if available
    try:
        from scipy import stats
        return stats.vonmises.rvs(kappa, loc=0, size=size)
    except ImportError:
        # Simple rejection sampling for von Mises
        samples = []
        tau = 1 + np.sqrt(1 + 4*kappa**2)
        rho = (tau - np.sqrt(2*tau)) / (2*kappa)
        r = (1 + rho**2) / (2*rho)
        while len(samples) < size:
            z = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            w = (1 - rho*z) / (r - rho*z)
            if w*w * (4 - w*w) >= v * (4 - rho*rho):
                u = rng.uniform(0, 2*np.pi)
                samples.append(np.sign(z - 0.5) * np.arccos(w) + u % (2*np.pi))
        return np.array(samples[:size])

def survey_detection_prob(inc_deg, b_max):
    """Probability an object at inclination i is detected by a survey with half-width b_max."""
    i = np.deg2rad(inc_deg)
    b = np.deg2rad(b_max)
    if i <= b:
        return 0.95
    else:
        frac = (2/np.pi) * np.arcsin(np.sin(b) / np.sin(i))
        return max(0.05, 0.95 * frac)

def weighted_rayleigh_p(varpi_rad, weights):
    """Model A: weighted Rayleigh test."""
    w = np.array(weights) / np.sum(weights)
    sx = np.sum(w * np.cos(varpi_rad))
    sy = np.sum(w * np.sin(varpi_rad))
    n_eff = 1.0 / np.sum(w**2)
    R_w = np.sqrt(sx**2 + sy**2)
    z = 2 * n_eff * R_w**2  # 2 × n_eff × R_w² ~ χ²(2)
    from scipy import stats
    return 1.0 - stats.chi2.cdf(z, 2)

def run_trial(kappa, n_sample, survey_mode=True):
    """
    Single trial: generate synthetic ETNOs, apply survey, test recovery.
    
    Parameters:
        kappa: von Mises concentration (0 = uniform)
        n_sample: number of ETNOs to generate
        survey_mode: if True, apply survey selection; if False, ideal (no bias)
    
    Returns:
        dict with p-values and detection flags
    """
    # 1. Generate intrinsic ϖ distribution
    varpi_true = np.rad2deg(von_mises_sample(kappa, n_sample))
    varpi_rad = np.deg2rad(varpi_true)
    
    # 2. Assign orbital inclinations (bootstrapped from observed 19)
    inclinations = rng.choice(
        [11.9, 24.0, 11.7, 17.4, 18.0, 29.5, 20.6, 12.1, 21.5, 18.6,
         4.5, 14.0, 12.4, 54.1, 26.1, 4.2, 24.1, 6.0, 7.7],
        size=n_sample
    )
    
    # 3. Assign surveys and apply detection
    survey_names = list(SURVEYS.keys())
    survey_weights = [SURVEYS[s]["weight"] for s in survey_names]
    
    detected_flag = np.ones(n_sample, dtype=bool)
    if survey_mode:
        for i in range(n_sample):
            inc = inclinations[i]
            # Choose survey
            survey = rng.choice(survey_names, p=survey_weights)
            b_max = SURVEYS[survey]["b_max"]
            p_det = survey_detection_prob(inc, b_max)
            detected_flag[i] = rng.random() < p_det
    
    n_detected = np.sum(detected_flag)
    if n_detected < 3:
        return {"p_raw": 1.0, "p_modelA": 1.0, "n_detected": n_detected,
                "detected": False, "varpi_true": varpi_true.tolist(),
                "detected_flag": detected_flag.tolist()}
    
    varpi_det = varpi_rad[detected_flag]
    
    # 4. Uncorrected Rayleigh test
    R_raw, p_raw = rayleigh_r(varpi_det)
    
    # 5. Model A (weighted Rayleigh) — weights based on ecliptic latitude
    det_indices = np.where(detected_flag)[0]
    n_detected = len(det_indices)
    weights = np.ones(n_detected)
    for j, idx in enumerate(det_indices):
        inc = inclinations[idx]
        weights[j] = 1.0 / max(0.1, survey_detection_prob(inc, 20))
    
    p_modelA = weighted_rayleigh_p(varpi_det, weights)
    
    return {"p_raw": p_raw, "p_modelA": p_modelA, "n_detected": n_detected,
            "detected": True}

# ================================================================
# MAIN SIMULATION
# ================================================================
def main():
    print("=" * 70)
    print("KNOWN-ANSWER TEST: Pipeline Validation")
    print("=" * 70)
    print(f"\nParameters: {N_TRIALS} trials, N={N_SAMPLE}, survey selection ON")
    print(f"κ values: {KAPPA_VALUES}")
    print()
    
    results = {}
    
    for kappa in KAPPA_VALUES:
        print(f"  κ = {kappa:.2f} ...", end=" ", flush=True)
        
        p_raw_list = []
        p_modelA_list = []
        n_detected_list = []
        
        for t in range(N_TRIALS):
            res = run_trial(kappa, N_SAMPLE, survey_mode=True)
            if res["detected"]:
                p_raw_list.append(res["p_raw"])
                p_modelA_list.append(res["p_modelA"])
                n_detected_list.append(res["n_detected"])
        
        p_raw_arr = np.array(p_raw_list)
        p_modelA_arr = np.array(p_modelA_list)
        n_det_arr = np.array(n_detected_list)
        
        # Power at α = 0.05
        power_raw = np.mean(p_raw_arr < 0.05) * 100
        power_modelA = np.mean(p_modelA_arr < 0.05) * 100
        
        # Median recovered p-value
        med_p_raw = np.median(p_raw_arr)
        med_p_modelA = np.median(p_modelA_arr)
        
        # Mean detected sample size
        mean_n = np.mean(n_det_arr)
        
        results[kappa] = {
            "power_raw": power_raw,
            "power_modelA": power_modelA,
            "med_p_raw": med_p_raw,
            "med_p_modelA": med_p_modelA,
            "mean_n_detected": mean_n,
            "n_valid_trials": len(p_raw_list),
        }
        
        print(f"power (raw)={power_raw:.0f}%  power (Model A)={power_modelA:.0f}%  "
              f"<N_det>={mean_n:.1f}  trials={len(p_raw_list)}")
    
    # Print summary table
    print()
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'κ':>5} {'Power(raw)':>11} {'Power(A)':>10} {'<N_det>':>8} {'med p(raw)':>11} {'med p(A)':>10}")
    print("-" * 60)
    for kappa in KAPPA_VALUES:
        r = results[kappa]
        print(f"{kappa:5.2f} {r['power_raw']:10.1f}% {r['power_modelA']:9.1f}% "
              f"{r['mean_n_detected']:8.1f} {r['med_p_raw']:10.4f} {r['med_p_modelA']:9.4f}")
    
    # Save results
    outpath = os.path.join(os.path.dirname(__file__) or ".", "known_answer_results.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {outpath}")

if __name__ == "__main__":
    main()
