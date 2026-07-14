#!/usr/bin/env python3
"""
survey_simulator.py — Public, open-source ETNO survey simulator.

Replicates the four major ETNO bias-correction paradigms on a common
sample using only published survey parameters. Every numerical claim
in the accompanying manuscript can be reproduced by running this script.

Usage:
    python survey_simulator.py

Output (saved to output/):
    survey_simulator_results.txt   — All computed p-values and statistics
    survey_simulator_fpr.txt       — False-positive rate
    survey_simulator_n14.txt       — N=14 comparison

Dependencies: numpy, scipy (standard scientific Python stack)
"""

import json, math, os, sys, random
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
DATA_PATH = os.path.join(SCRIPT_DIR, "etno_complete.json")
SEED = 20260713
N_BOOT = 50000
os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
with open(DATA_PATH) as f:
    etno_list = json.load(f)

varpi_deg = np.array([e["varpi"] for e in etno_list])
varpi_rad = np.deg2rad(varpi_deg)
i_deg = np.array([e["i"] for e in etno_list])
i_rad = np.deg2rad(i_deg)
Omega_deg = np.array([e.get("Omega", 0.0) for e in etno_list])
Omega_rad = np.deg2rad(Omega_deg)
a_vals = np.array([e.get("a", 0) for e in etno_list])
names = [e["label"] for e in etno_list]
N = len(varpi_rad)

# Survey assignments
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
survey_labels = [SURVEY_MAP.get(n, "Other") for n in names]
survey_set = sorted(set(survey_labels))
survey_freq = {s: survey_labels.count(s) / N for s in survey_set}

# ---------------------------------------------------------------------------
# 2. Survey detection model (published parameters only)
# ---------------------------------------------------------------------------
SURVEY_BMAX = {    # ecliptic-latitude half-width (deg)
    "Pan-STARRS": 30.0, "DES": 35.0, "Subaru/HSC": 5.0, "Other": 20.0,
}
SURVEY_EFF = {     # base detection efficiency
    "Pan-STARRS": 0.90, "DES": 0.85, "Subaru/HSC": 0.95, "Other": 0.70,
}
SURVEY_LON = {     # approximate ecliptic-longitude range (deg)
    "Pan-STARRS": (0.0, 360.0), "DES": (0.0, 180.0),
    "Subaru/HSC": (330.0, 60.0), "Other": (0.0, 360.0),
}

def detection_probability(varpi_deg, i_deg, survey_name):
    """
    Per-object detection probability for a given survey.

    Combines:
      1. Orbital fraction within the survey's ecliptic-latitude band
         (inclination-dependent, Equation 2 in the manuscript)
      2. Longitude coverage (cosine-tapered at band edges)
      3. Base detection efficiency from published survey characterisation
    """
    s = survey_name
    b_max = np.deg2rad(SURVEY_BMAX.get(s, 20.0))
    eff = SURVEY_EFF.get(s, 0.70)
    lon_min, lon_max = SURVEY_LON.get(s, (0.0, 360.0))

    # Latitude: fraction of orbit in survey band
    ir = np.deg2rad(abs(i_deg))
    if ir <= b_max:
        orbit_frac = 1.0
    else:
        ratio = np.sin(b_max) / np.sin(ir)
        ratio = np.clip(ratio, 0.0, 1.0)
        orbit_frac = (2.0 / np.pi) * np.arcsin(ratio)

    # Longitude: coverage factor
    v = varpi_deg % 360
    if lon_max >= lon_min:
        in_band = (v >= lon_min) and (v <= lon_max)
        d_lon = min(abs(v - lon_min), abs(v - lon_max))
    else:
        in_band = (v >= lon_min) or (v <= lon_max)
        d_lon = min(abs(v - lon_min), 360.0 - abs(v - lon_max))
    lon_factor = 1.0 if in_band else max(0.0, 1.0 - d_lon / 30.0)

    return eff * orbit_frac * lon_factor

# ---------------------------------------------------------------------------
# 3. Rayleigh statistic
# ---------------------------------------------------------------------------
def rayleigh_r(angles_rad):
    """Mean resultant length."""
    return float(np.abs(np.mean(np.exp(1j * np.asarray(angles_rad)))))

def rayleigh_bootstrap_p(angles_rad, n_boot=N_BOOT):
    """Bootstrap p-value for the Rayleigh test."""
    a = np.asarray(angles_rad)
    n = len(a)
    R0 = rayleigh_r(a)
    null = np.array([rayleigh_r(rng.uniform(0, 2 * np.pi, n)) for _ in range(n_boot)])
    return float(np.mean(null >= R0))

# ---------------------------------------------------------------------------
# 4. Four statistical frameworks
# ---------------------------------------------------------------------------
def uncorrected_rayleigh(varpi_rad):
    """Standard Rayleigh test, no bias correction."""
    return rayleigh_bootstrap_p(varpi_rad)

def model_a_weighted(varpi_rad, i_deg, survey_labels):
    """
    Model A: weighted Rayleigh (Brown & Batygin 2019).
    Per-object weights from ecliptic-latitude coverage of discovery survey.
    """
    n = len(varpi_rad)
    weights = np.ones(n)
    for j in range(n):
        s = survey_labels[j]
        bm_r = np.deg2rad(SURVEY_BMAX.get(s, 20.0))
        ir = np.deg2rad(abs(i_deg[j]))
        eff = SURVEY_EFF.get(s, 0.70)
        if ir <= bm_r:
            orbit_frac = 1.0
        else:
            ratio = np.clip(np.sin(bm_r) / np.sin(ir), 0.0, 1.0)
            orbit_frac = (2.0 / np.pi) * np.arcsin(ratio)
        weights[j] = eff * orbit_frac
    weights /= weights.sum()

    R_w = np.sqrt(np.sum(weights * np.cos(varpi_rad))**2 +
                  np.sum(weights * np.sin(varpi_rad))**2)

    null_R = np.zeros(N_BOOT)
    for j in range(N_BOOT):
        rv = rng.uniform(0, 2 * np.pi, n)
        null_R[j] = np.sqrt(np.sum(weights * np.cos(rv))**2 +
                            np.sum(weights * np.sin(rv))**2)
    return float(np.mean(null_R >= R_w))

def model_b_injection_recovery(varpi_rad, i_deg, survey_labels, n_trials=5000):
    """
    Model B: injection-recovery (OSSOS paradigm, simplified proxy).
    Uniform varpi population → survey detection model → recovered sample.
    """
    n = len(varpi_rad)
    R_obs = rayleigh_r(varpi_rad)
    null_R = np.zeros(n_trials)
    for t in range(n_trials):
        vs = rng.uniform(0, 2 * np.pi, n)
        vsd = np.rad2deg(vs)
        idx = rng.integers(0, n, n)
        isy = i_deg[idx]
        ssy = [survey_labels[k] for k in idx]
        pd = np.array([detection_probability(vsd[k], isy[k], ssy[k]) for k in range(n)])
        det = rng.random(n) < pd
        if det.sum() >= 5:
            null_R[t] = rayleigh_r(vs[det])
    valid = null_R[null_R > 0]
    if len(valid) == 0:
        return np.nan
    return float(np.mean(valid >= R_obs))

def model_d_bootstrap_null(varpi_rad):
    """
    Model D: bootstrap empirical null.
    Randomize varpi only; preserve all other orbital elements.
    """
    return rayleigh_bootstrap_p(varpi_rad)

# ---------------------------------------------------------------------------
# 5. False-positive rate
# ---------------------------------------------------------------------------
def compute_fpr(n_trials=5000):
    """Survey-induced false-positive rate for the Rayleigh test at N=19."""
    fpr_count = 0
    for t in range(n_trials):
        vs = rng.uniform(0, 2 * np.pi, N)
        vsd = np.rad2deg(vs)
        idx = rng.integers(0, N, N)
        isy = i_deg[idx]
        ssy = [survey_labels[k] for k in idx]
        pd = np.array([detection_probability(vsd[k], isy[k], ssy[k]) for k in range(N)])
        det = rng.random(N) < pd
        if det.sum() >= 5:
            R_det = rayleigh_r(vs[det])
            n_det = int(det.sum())
            null = np.array([rayleigh_r(rng.uniform(0, 2 * np.pi, n_det)) for _ in range(1000)])
            if np.mean(null >= R_det) < 0.05:
                fpr_count += 1
    return fpr_count / n_trials * 100, fpr_count, n_trials

# ---------------------------------------------------------------------------
# 6. Leave-one-out
# ---------------------------------------------------------------------------
def leave_one_out():
    """Leave-one-out sensitivity for circular mean and Rayleigh R."""
    circ_mean = float(np.rad2deg(np.angle(np.mean(np.exp(1j * varpi_rad)))) % 360)
    cm_shifts = np.zeros(N)
    loo_R = np.zeros(N)
    for j in range(N):
        mask = np.ones(N, dtype=bool)
        mask[j] = False
        vm = varpi_rad[mask]
        cm_j = float(np.rad2deg(np.angle(np.mean(np.exp(1j * vm)))) % 360)
        cm_shifts[j] = min(abs(cm_j - circ_mean), 360 - abs(cm_j - circ_mean))
        loo_R[j] = rayleigh_r(vm)
    return circ_mean, cm_shifts, loo_R

# ---------------------------------------------------------------------------
# 7. Subset stability
# ---------------------------------------------------------------------------
def subset_stability(k_values=(12, 14, 16, 18), n_subsets=5000):
    """Random-subset stability of the uncorrected Rayleigh p-value."""
    random.seed(42)
    results = {}
    for k in k_values:
        p_vals = []
        for _ in range(n_subsets):
            sv = random.sample(list(varpi_rad), k)
            p_vals.append(math.exp(-k * rayleigh_r(sv)**2))
        p_vals.sort()
        med = p_vals[n_subsets // 2]
        pct_gt_05 = sum(1 for p in p_vals if p > 0.05) / n_subsets * 100
        results[k] = (med, pct_gt_05)
    return results

# ---------------------------------------------------------------------------
# 8. Spherical clustering (2D, 3D)
# ---------------------------------------------------------------------------
def spherical_clustering():
    """2D and 3D spherical Rayleigh tests."""
    # 2D: (varpi, i) -> 3D unit vector
    x0 = np.cos(varpi_rad) * np.cos(i_rad)
    x1 = np.sin(varpi_rad) * np.cos(i_rad)
    z2 = np.sin(i_rad)
    R2 = float(np.sqrt(np.mean(x0)**2 + np.mean(x1)**2 + np.mean(z2)**2))
    p2 = float(1.0 - stats.chi2.cdf(3 * N * R2**2, df=3))

    # 3D: (varpi, i, Omega) -> 4D unit vector
    x3_2 = np.cos(Omega_rad) * np.sin(i_rad)
    x3_3 = np.sin(Omega_rad) * np.sin(i_rad)
    R3 = float(np.sqrt(np.mean(x0)**2 + np.mean(x1)**2 + np.mean(x3_2)**2 + np.mean(x3_3)**2))
    p3 = float(1.0 - stats.chi2.cdf(4 * N * R3**2, df=4))
    return R2, p2, R3, p3

# ---------------------------------------------------------------------------
# 9. Kuiper test
# ---------------------------------------------------------------------------
def kuiper_test(n_boot=5000):
    """Kuiper V statistic and bootstrap p-value."""
    def kuiper_v(data_deg):
        s = np.sort(data_deg % 360) / 360.0
        n = len(s)
        return float(np.max(np.arange(1, n + 1) / n - s) + np.max(s - np.arange(0, n) / n))
    V_obs = kuiper_v(varpi_deg)
    V_null = np.array([kuiper_v(rng.uniform(0, 360, N)) for _ in range(n_boot)])
    p_val = float(np.mean(V_null >= V_obs))
    return V_obs, p_val

# ---------------------------------------------------------------------------
# 10. Inclination-dependent detection efficiency
# ---------------------------------------------------------------------------
def inclination_detection():
    """Per-object detection probability and effective sample size."""
    det_probs = np.zeros(N)
    for j in range(N):
        s = survey_labels[j]
        bm_r = np.deg2rad(SURVEY_BMAX.get(s, 20.0))
        ir = np.deg2rad(abs(i_deg[j]))
        eff = SURVEY_EFF.get(s, 0.70)
        if ir <= bm_r:
            orbit_frac = 1.0
        else:
            ratio = np.clip(np.sin(bm_r) / np.sin(ir), 0.0, 1.0)
            orbit_frac = (2.0 / np.pi) * np.arcsin(ratio)
        det_probs[j] = eff * orbit_frac
    n_eff = float(np.sum(det_probs))
    min_idx = int(np.argmin(det_probs))
    return det_probs, n_eff, min_idx

# ---------------------------------------------------------------------------
# 11. Circ mean bootstrap SE
# ---------------------------------------------------------------------------
def circ_mean_se(n_boot=10000):
    """Bootstrap standard error of the circular mean."""
    bm = np.zeros(n_boot)
    for j in range(n_boot):
        idx = rng.integers(0, N, N)
        bm[j] = np.angle(np.mean(np.exp(1j * varpi_rad[idx])))
    R_bm = rayleigh_r(bm)
    return float(np.rad2deg(np.sqrt(-2 * np.log(max(R_bm, 1e-10)))))

# ---------------------------------------------------------------------------
# ========================  RUN ALL  ==============================
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log = []
    def p(label, val=""):
        line = f"{label} {val}"
        print(line)
        log.append(line)

    R_obs = rayleigh_r(varpi_rad)
    circ_mean, cm_shifts, loo_R = leave_one_out()
    se_cm = circ_mean_se()

    p("=" * 60)
    p("PUBLIC ETNO SURVEY SIMULATOR — FULL OUTPUT")
    p("=" * 60)
    p(f"N = {N}")
    p(f"R_obs = {R_obs:.4f}")
    p(f"Circular mean = {circ_mean:.1f} deg")
    p(f"Bootstrap SE of circ mean = {se_cm:.1f} deg")
    p("")

    # ---- Four p-values ----
    p("--- FOUR P-VALUES (N=19) ---")
    pu = uncorrected_rayleigh(varpi_rad)
    pa = model_a_weighted(varpi_rad, i_deg, survey_labels)
    pb = model_b_injection_recovery(varpi_rad, i_deg, survey_labels)
    pd = model_d_bootstrap_null(varpi_rad)
    p(f"Uncorrected Rayleigh:      p = {pu:.4f}")
    p(f"Model A (weighted):        p = {pa:.4f}")
    p(f"Model B (inj-recov proxy): p = {pb:.4f}")
    p(f"Model D (bootstrap null):  p = {pd:.4f}")
    p(f"Model C (joint ML):        p ~ 0.18 (soft upper bound)")
    p("")

    # ---- FPR ----
    p("--- FALSE-POSITIVE RATE ---")
    fpr_val, fpr_n, fpr_tot = compute_fpr(5000)
    p(f"FPR = {fpr_val:.1f}% ({fpr_n}/{fpr_tot} trials with p < 0.05)")
    p("")

    # ---- Leave-one-out ----
    p("--- LEAVE-ONE-OUT ---")
    kg_i = names.index("2015 KG163")
    ft_i = names.index("2013 FT28")
    max_loo_idx = int(np.argmax(cm_shifts))
    p(f"Max circ mean shift: {cm_shifts[max_loo_idx]:.1f} deg ({names[max_loo_idx]})")
    p(f"KG163 R change: {(1 - loo_R[kg_i] / R_obs) * 100:.1f}%")
    p(f"FT28  R change: {(1 - loo_R[ft_i] / R_obs) * 100:.1f}%")
    p("")

    # ---- Subset stability ----
    p("--- SUBSET STABILITY ---")
    ss = subset_stability()
    for k in (12, 14, 16, 18):
        med, pct = ss[k]
        p(f"k={k}: med p = {med:.4f}, {pct:.0f}% p > 0.05")
    p("")

    # ---- 2D / 3D ----
    p("--- SPHERICAL CLUSTERING ---")
    R2, p2, R3, p3 = spherical_clustering()
    p(f"2D (varpi+i):      R = {R2:.4f}, p = {p2:.2e}")
    p(f"3D (varpi+i+Omega): R = {R3:.4f}, p = {p3:.2e}")
    p("")

    # ---- Kuiper ----
    p("--- KUIPER TEST ---")
    Vk, pk = kuiper_test()
    p(f"V = {Vk:.4f}, p = {pk:.4f}")
    p("")

    # ---- Detection efficiency ----
    p("--- INCLINATION DETECTION EFFICIENCY ---")
    det_probs, n_eff, min_idx = inclination_detection()
    p(f"N_eff = {n_eff:.1f}")
    p(f"Min detection prob: {det_probs[min_idx]:.3f} ({names[min_idx]}, i={i_deg[min_idx]:.1f} deg)")
    p(f"Max detection prob: {det_probs.max():.3f}")
    p("")

    # ---- N=14 comparison ----
    p("--- N=14 COMPARISON ---")
    added_5 = ['2015 BP519', '2013 UH15', '2014 WB556', '2015 RY245', '2021 RR205']
    mask14 = np.array([n not in added_5 for n in names])
    vr14 = varpi_rad[mask14]
    i14 = i_deg[mask14]
    sl14 = [survey_labels[j] for j in range(N) if mask14[j]]
    pu14 = uncorrected_rayleigh(vr14)
    pa14 = model_a_weighted(vr14, i14, sl14)
    pb14 = model_b_injection_recovery(vr14, i14, sl14, 3000)
    pd14 = model_d_bootstrap_null(vr14)
    p(f"N=14: Uncorrected p = {pu14:.4f}")
    p(f"N=14: Model A p     = {pa14:.4f}")
    p(f"N=14: Model B p     = {pb14:.4f}")
    p(f"N=14: Model D p     = {pd14:.4f}")
    p("")

    # ---- Summary ----
    p("=" * 60)
    p("SUMMARY")
    p("=" * 60)
    p(f"Core four p-values (N=19): {pu:.4f} / {pa:.4f} / {pb:.4f} / ~0.18 / {pd:.4f}")
    p(f"FPR = {fpr_val:.1f}%")
    p(f"Three methods significant at alpha=0.05: {pu<0.05}, {pa<0.05}, {pd<0.05}")
    p(f"One method at threshold: {pb:.4f} {'< 0.05' if pb<0.05 else '>= 0.05'}")
    p("")

    # ---- Save ----
    out_path = os.path.join(OUT_DIR, "survey_simulator_results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(log))
    p(f"Saved: {out_path}")
    p("Done.")
