#!/usr/bin/env python3
"""
test_known_answer.py — Known-answer tests for the ETNO survey simulator.

Run with:
    python test_known_answer.py

Each test generates synthetic data with known clustering properties
and verifies that the simulator produces the expected statistical output.
"""

import sys, json, tempfile, os
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from etno_simulator import SurveySimulator

PASS = 0
FAIL = 0


def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {label}")
    else:
        FAIL += 1
        print(f"  FAIL: {label} — {detail}")


def make_test_data(n=19, concentration=0.0, seed=42):
    """Generate synthetic ETNO data with known clustering."""
    rng = np.random.default_rng(seed)

    varpi_deg = np.rad2deg(
        rng.vonmises(0.0, concentration, n) if concentration > 0
        else rng.uniform(0, 2 * np.pi, n)
    )
    obj_list = []
    for j in range(n):
        obj_list.append({
            "label": f"TEST-{j:02d}",
            "varpi": float(varpi_deg[j]),
            "i": float(rng.uniform(5, 30)),
            "Omega": float(rng.uniform(0, 360)),
            "a": float(rng.uniform(150, 500)),
            "e": float(rng.uniform(0.5, 0.95)),
            "q": float(rng.uniform(30, 50)),
        })
    return obj_list


# ---------------------------------------------------------------------------
print("=" * 60)
print("KNOWN-ANSWER TESTS")
print("=" * 60)

# ---- Test 1: Uniform input → p ≈ 0.5 ----
print("\n--- Test 1: Uniform varpi → p ~ 0.5 ---")
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(make_test_data(n=100, concentration=0.0), f)
    tmp_path = f.name

sim = SurveySimulator(tmp_path, seed=42)
p_unc = sim.uncorrected_rayleigh(n_boot=5000)
check("Uncorrected p on uniform data (N=100)", 0.2 < p_unc < 0.8,
      f"p = {p_unc:.3f} (expected ~0.5)")
os.unlink(tmp_path)

# ---- Test 2: Strongly clustered input → p << 0.05 ----
print("\n--- Test 2: Clustered varpi (kappa=5) → p << 0.05 ---")
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(make_test_data(n=19, concentration=5.0), f)
    tmp_path = f.name

sim = SurveySimulator(tmp_path, seed=42)
p_clust = sim.uncorrected_rayleigh(n_boot=5000)
check("Uncorrected p on clustered data (kappa=5)", p_clust < 0.01,
      f"p = {p_clust:.4f} (expected < 0.01)")
os.unlink(tmp_path)

# ---- Test 3: R_obs reproduces known value ----
print("\n--- Test 3: Known R on real data ---")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "etno_complete.json")
if os.path.exists(data_path):
    sim = SurveySimulator(data_path, seed=20260713)
    check("R_obs matches expected", abs(sim.R_obs - 0.5052) < 0.001,
          f"R_obs = {sim.R_obs:.4f} (expected 0.5052)")
    check("N matches expected", sim.N == 19,
          f"N = {sim.N} (expected 19)")
    check("Circ mean matches expected",
          abs(sim.circ_mean_deg - 46.3) < 0.1,
          f"circ_mean = {sim.circ_mean_deg:.1f} (expected 46.3)")
else:
    check("Real data file exists", False, f"{data_path} not found")

# ---- Test 4: Model D = Uncorrected (structural identity) ----
print("\n--- Test 4: Model D ≡ Uncorrected (same test statistic) ---")
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(make_test_data(n=19, concentration=1.0), f)
    tmp_path = f.name

sim = SurveySimulator(tmp_path, seed=12345)
p_u = sim.uncorrected_rayleigh(n_boot=5000)
p_d = sim.model_d_bootstrap_null(n_boot=5000)
check("Model D p == Uncorrected p", abs(p_u - p_d) < 0.005,  # MC noise
      f"uncorrected={p_u:.4f}, model_d={p_d:.4f}")
os.unlink(tmp_path)

# ---- Test 5: N=14 comparison sample size ----
print("\n--- Test 5: N=14 sample consistency ---")
if os.path.exists(data_path):
    sim = SurveySimulator(data_path, seed=20260713)
    n14 = sim.n14_comparison(n_trials_b=1000, n_boot=5000)
    check("N14 has 14 objects", n14["N"] == 14,
          f"N = {n14['N']}")
    check("N14 R_obs > 0", n14["R_obs"] > 0,
          f"R_obs = {n14['R_obs']:.4f}")

# ---- Summary ----
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
print("=" * 60)
sys.exit(0 if FAIL == 0 else 1)
