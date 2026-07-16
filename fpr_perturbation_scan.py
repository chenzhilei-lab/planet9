#!/usr/bin/env python3
"""
fpr_perturbation_scan.py — FPR parameter perturbation scan
===========================================================

Verifies claim: "Varying survey parameters within published uncertainties
(depth +/-0.5 mag, coverage +/-10%) shifts FPR by +/-2 percentage points."

Monkey-patches etno_simulator module-level SURVEY_EFF/SURVEY_BMAX dicts
BEFORE creating each SurveySimulator AND keeps them patched throughout
compute_fpr() — detection_probability() reads from these globals at call time.

Usage:
    python fpr_perturbation_scan.py

Output:
    fpr_perturbation_results.json
"""

import json, sys, os, copy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(SCRIPT_DIR, "reproducible_package")
sys.path.insert(0, PKG_DIR)

import etno_simulator

DATA_FILE = os.path.join(PKG_DIR, "etno_complete.json")
OUT_FILE  = os.path.join(SCRIPT_DIR, "fpr_perturbation_results.json")

ORIG_EFF  = copy.deepcopy(etno_simulator.SURVEY_EFF)
ORIG_BMAX = copy.deepcopy(etno_simulator.SURVEY_BMAX)

N_TRIALS = 5000
N_BOOT   = 1000
SEED     = 20260713


def apply_params(eff_delta, bmax_scale):
    """Apply perturbations to module-level dicts. Call BEFORE SurveySimulator()."""
    etno_simulator.SURVEY_EFF = {
        k: float(max(0.0, min(1.0, v + eff_delta)))
        for k, v in ORIG_EFF.items()
    }
    etno_simulator.SURVEY_BMAX = {
        k: float(v * bmax_scale) for k, v in ORIG_BMAX.items()
    }


def restore_params():
    """Restore module-level dicts to originals."""
    etno_simulator.SURVEY_EFF = copy.deepcopy(ORIG_EFF)
    etno_simulator.SURVEY_BMAX = copy.deepcopy(ORIG_BMAX)


def run_fpr(eff_delta=0.0, bmax_scale=1.0, label=""):
    """Run FPR with perturbed parameters. Keeps patch alive during compute_fpr."""
    apply_params(eff_delta, bmax_scale)
    sim = etno_simulator.SurveySimulator(DATA_FILE, seed=SEED)
    fpr, cnt, _ = sim.compute_fpr(n_trials=N_TRIALS, n_boot_fpr=N_BOOT)
    restore_params()
    return fpr, cnt


# ===================================================================
print("=" * 65)
print("FPR PARAMETER PERTURBATION SCAN")
print("=" * 65)

# Baseline
sim0 = etno_simulator.SurveySimulator(DATA_FILE, seed=SEED)
print(f"  Objects: {sim0.N}  |  Trials: {N_TRIALS}  |  Seed: {SEED}")
print()

print("Computing baseline FPR ...", end=" ", flush=True)
fpr0, cnt0 = run_fpr(0.0, 1.0, "baseline")
print(f"  FPR = {fpr0:.1f}%  ({cnt0}/{N_TRIALS})")
print()

# Perturbations
cases = [
    ("depth_+0.5mag",      +0.05, 1.00),
    ("depth_-0.5mag",      -0.05, 1.00),
    ("coverage_+10%",       0.00, 1.10),
    ("coverage_-10%",       0.00, 0.90),
    ("deep_+wide",         +0.05, 1.10),
    ("shallow_+narrow",    -0.05, 0.90),
]

results = {
    "baseline": {"fpr_pct": round(fpr0, 2), "count": cnt0, "trials": N_TRIALS},
    "perturbations": {},
}

all_shifts = []

for label, eff_d, bmax_s in cases:
    print(f"  {label:<22s} eff{eff_d:+.2f} bmax x{bmax_s:.2f} ...",
          end=" ", flush=True)
    fpr, cnt = run_fpr(eff_d, bmax_s)
    shift = fpr - fpr0
    results["perturbations"][label] = {
        "fpr_pct": round(fpr, 2),
        "count": cnt,
        "shift_pp": round(shift, 2),
        "eff_delta": eff_d,
        "bmax_scale": bmax_s,
    }
    all_shifts.append(shift)
    print(f"FPR={fpr:.1f}%  shift={shift:+.1f} pp")

print()

# Summary
max_shift = max(abs(s) for s in all_shifts)
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Baseline FPR:            {fpr0:.1f}%")
print(f"  Shift range:             [{min(all_shifts):+.1f}, {max(all_shifts):+.1f}] pp")
print(f"  Max |shift|:             {max_shift:.1f} pp")

if max_shift <= 2.5:
    verdict = "CONSISTENT — all shifts within 2.5 pp, paper claim '±2 pp' is supported"
elif max_shift <= 3.5:
    verdict = "MARGINAL — max shift slightly exceeds ±2 pp; consider '±3 pp'"
else:
    verdict = f"DISCREPANCY — max shift {max_shift:.1f} pp exceeds ±2 pp claim"

print(f"  Paper claim: '±2 pp'     {verdict}")
print()

results["_meta"] = {
    "script":      "fpr_perturbation_scan.py",
    "date":        "2026-07-16",
    "data_file":   DATA_FILE,
    "n_objects":   sim0.N,
    "n_trials":    N_TRIALS,
    "n_boot_fpr":  N_BOOT,
    "seed":        SEED,
    "baseline_fpr": round(fpr0, 2),
    "shift_range":  [round(min(all_shifts), 2), round(max(all_shifts), 2)],
    "max_abs_shift": round(max_shift, 2),
    "verdict":      verdict,
}

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved -> {OUT_FILE}")
