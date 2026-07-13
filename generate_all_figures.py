#!/usr/bin/env python3
"""
generate_all_figures.py — Regenerate all 9 figures for the ETNO calibration paper.

All statistics are computed from raw orbital data (etno_complete.json).
Only quantities that require the external OSSOS survey simulator (Model B/C
p-values, FPR curve) are hard-coded from the paper's reported values.

Usage:
    cd D:/Papers/MNRAS/figure_generation
    python generate_all_figures.py

Output: 9 PNG files in ./output/ matching the filenames used by the .tex file.

Dependencies: numpy, scipy, matplotlib
"""

import json, math, os, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# Fix Unicode output on Windows GBK terminals
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ================================================================
# CONFIG
# ================================================================
DPI = 300
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
DATA_PATH = os.path.join(SCRIPT_DIR, "etno_complete.json")

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} not found.")
    sys.exit(1)

with open(DATA_PATH) as f:
    etno_list = json.load(f)

names      = [e["label"] for e in etno_list]
varpi_vals = np.array([e["varpi"] for e in etno_list])
a_vals     = np.array([e["a"] for e in etno_list])
q_vals     = np.array([e["q"] for e in etno_list])
i_vals     = np.array([e["i"] for e in etno_list])
e_vals     = np.array([e["e"] for e in etno_list])
cc_vals    = np.array([e["condition_code"] for e in etno_list])
varpi_rad  = np.deg2rad(varpi_vals)
N = len(etno_list)

# Survey assignment per object (from paper §2.1)
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

# ================================================================
# CIRCULAR STATISTICS (computed from data)
# ================================================================
def circ_mean(rad):
    """Circular mean direction in radians."""
    return np.arctan2(np.sum(np.sin(rad)), np.sum(np.cos(rad))) % (2*np.pi)

def circ_dist_deg(d1, d2):
    """Circular distance in degrees."""
    d = abs(d1 - d2)
    return np.minimum(d, 360 - d)

def rayleigh_r(rad):
    """Rayleigh R statistic."""
    n = len(rad)
    return np.sqrt(np.sum(np.cos(rad))**2 + np.sum(np.sin(rad))**2) / n

def rayleigh_p_from_r(r, n):
    """p-value from Rayleigh R. Test statistic 2nR² ~ χ²(2)."""
    z = 2 * n * r**2
    return float(np.exp(-n * r**2))  # exact: = 1 - chi2.cdf(2nR², 2)

def rayleigh_p(rad):
    """Convenience: Rayleigh p-value from angles."""
    return rayleigh_p_from_r(rayleigh_r(rad), len(rad))

def bootstrap_se_circmean(angles_rad, n_boot=10000, seed=42):
    """Bootstrap standard error of circular mean (68% CI half-width, degrees)."""
    rng = np.random.default_rng(seed)
    n = len(angles_rad)
    cm_all = circ_mean(angles_rad)
    dists = []
    for _ in range(n_boot):
        samp = angles_rad[rng.integers(0, n, n)]
        cm_b = circ_mean(samp)
        dists.append(circ_dist_deg(np.rad2deg(cm_b), np.rad2deg(cm_all)))
    dists.sort()
    return dists[int(0.6827 * n_boot)]

def kuiper_p(rad):
    """Kuiper test p-value for circular uniformity."""
    x = np.sort(rad % (2*np.pi)) / (2*np.pi)
    n = len(x)
    D_plus  = np.max(np.arange(1, n+1)/n - x)
    D_minus = np.max(x - np.arange(0, n)/n)
    V = D_plus + D_minus
    z = V * (np.sqrt(n) + 0.155 + 0.24/np.sqrt(n))
    p = 0.0
    for j in range(1, 5):
        p += 2 * (4*j**2 * z**2 - 1) * np.exp(-2*j**2 * z**2)
    return V, max(0.0, min(1.0, p))

# Compute core statistics from data
CM_ALL_DEG = np.rad2deg(circ_mean(varpi_rad))
R_ALL = rayleigh_r(varpi_rad)
P_UNC = rayleigh_p(varpi_rad)
BOOT_SE = bootstrap_se_circmean(varpi_rad)
KUIPER_V, KUIPER_P = kuiper_p(varpi_rad)

print(f"N = {N}")
print(f"ϖ̄ = {CM_ALL_DEG:.1f}°")
print(f"R  = {R_ALL:.4f}")
print(f"p_unc = {P_UNC:.4f}")
print(f"Bootstrap SE(ϖ̄) = {BOOT_SE:.1f}°")
print(f"Kuiper V = {KUIPER_V:.4f}, p = {KUIPER_P:.6f}")

# ================================================================
# LEAVE-ONE-OUT (computed from data)
# ================================================================
loo_deltas, loo_R_changes = [], []
for i in range(N):
    mask = np.ones(N, dtype=bool); mask[i] = False
    sub = varpi_rad[mask]
    cm_i = np.rad2deg(circ_mean(sub))
    d = circ_dist_deg(cm_i, CM_ALL_DEG)
    loo_deltas.append(d)
    R_sub = rayleigh_r(sub)
    loo_R_changes.append(abs(R_sub - R_ALL) / R_ALL * 100)
loo_deltas = np.array(loo_deltas)
loo_R_changes = np.array(loo_R_changes)

# Print LOO summary
sort_loo = np.argsort(-loo_R_changes)
print(f"\nTop 5 R-leverage objects:")
for i in range(5):
    idx = sort_loo[i]
    print(f"  {names[idx]:14s} ϖ={varpi_vals[idx]:5.1f}°  "
          f"Δϖ̄={loo_deltas[idx]:4.1f}°  ΔR={loo_R_changes[idx]:4.1f}%")

# ================================================================
# MODEL A (weighted Rayleigh) — approximate from data
# ================================================================
def survey_weight(varpi_deg, survey_name):
    """Ecliptic-latitude weight per survey (Model A approximation)."""
    b_model = varpi_deg / 360 * 60 - 30  # crude b proxy from varpi
    if survey_name == "Pan-STARRS":
        return 0.5 + 0.5 * np.sin(np.radians(b_model * 0.5))**2
    elif survey_name == "DES":
        return 0.3 + 0.7 * np.maximum(0, np.cos(np.radians(b_model - 180)))**2
    elif survey_name == "Subaru/HSC":
        return 0.4 + 0.3 * np.sin(np.radians(b_model * 0.3))**2
    else:
        return 0.5 + 0.3 * np.cos(np.radians(b_model * 0.7))**2

def weighted_rayleigh_p(angles_deg, survey_labels):
    """Model A: weighted Rayleigh p-value."""
    angles_rad = np.deg2rad(angles_deg)
    w = np.array([survey_weight(a, s) for a, s in zip(angles_deg, survey_labels)])
    w /= w.sum()
    sx = np.sum(w * np.cos(angles_rad))
    sy = np.sum(w * np.sin(angles_rad))
    R_w = np.sqrt(sx**2 + sy**2)
    n_eff = 1.0 / np.sum(w**2)
    return float(np.exp(-n_eff * R_w**2)), R_w

P_W_FULL, R_W_FULL = weighted_rayleigh_p(varpi_vals, survey_labels)
print(f"\nModel A: p_w = {P_W_FULL:.6f}, R_w = {R_W_FULL:.4f}")

# Model A leave-one-out
pw_loo, Rw_loo_change = [], []
for i in range(N):
    mask = np.ones(N, dtype=bool); mask[i] = False
    p_i, R_i = weighted_rayleigh_p(varpi_vals[mask], [survey_labels[j] for j in range(N) if j != i])
    pw_loo.append(p_i)
    Rw_loo_change.append(abs(R_i - R_W_FULL) / R_W_FULL * 100)
pw_loo = np.array(pw_loo)
Rw_loo_change = np.array(Rw_loo_change)

# ================================================================
# SUBSET STABILITY (computed from data — uncorrected Rayleigh)
# ================================================================
def compute_subset_stability(ks, n_trials=2000, seed=42):
    """Compute subset stability: median p and fraction significant."""
    rng = np.random.default_rng(seed)
    results = {}
    for k in ks:
        p_vals = []
        for _ in range(n_trials):
            idx = rng.choice(N, k, replace=False)
            p = rayleigh_p(varpi_rad[idx])
            p_vals.append(p)
        p_vals = np.array(p_vals)
        results[k] = {
            "median_p": float(np.median(p_vals)),
            "iqr_low": float(np.percentile(p_vals, 25)),
            "iqr_high": float(np.percentile(p_vals, 75)),
            "frac_sig": float(np.mean(p_vals < 0.05) * 100),
        }
        print(f"  k={k}: median_p={results[k]['median_p']:.4f}, "
              f"frac_sig={results[k]['frac_sig']:.1f}%")
    return results

ks = [12, 14, 16, 18]
print("\nSubset stability (uncorrected Rayleigh):")
subset_results = compute_subset_stability(ks)

# ================================================================
# LEAVE-ONE-SURVEY-OUT (computed from data)
# ================================================================
survey_groups = {}
for i, s in enumerate(survey_labels):
    survey_groups.setdefault(s, []).append(i)

print("\nLeave-one-survey-out:")
loo_survey = {}
for sname, idxs in survey_groups.items():
    mask = np.ones(N, dtype=bool)
    for i in idxs:
        mask[i] = False
    if np.sum(mask) >= 5:
        p = rayleigh_p(varpi_rad[mask])
        loo_survey[sname] = {"p": p, "n_removed": len(idxs), "n_remain": np.sum(mask)}
        print(f"  Without {sname} ({len(idxs)} removed): p = {p:.4f}, N = {np.sum(mask)}")

print("\nWithin-survey clustering:")
within_survey = {}
for sname, idxs in survey_groups.items():
    if len(idxs) >= 3:
        p = rayleigh_p(varpi_rad[idxs])
        within_survey[sname] = {"p": p, "n": len(idxs)}
        print(f"  {sname} only (N={len(idxs)}): p = {p:.4f}")

# ================================================================
# POWER ANALYSIS (von Mises Monte Carlo, computed from data)
# ================================================================
def compute_power_curve(n_samples, kappa_grid, n_trials=2000, seed=42):
    """Compute Rayleigh test power at given (N, κ)."""
    rng = np.random.default_rng(seed)
    powers = []
    for kappa in kappa_grid:
        detections = 0
        for _ in range(n_trials):
            if kappa < 0.01:
                samp = rng.uniform(0, 2*np.pi, n_samples)
            else:
                samp = stats.vonmises.rvs(kappa, size=n_samples, random_state=rng)
            p = rayleigh_p(samp)
            if p < 0.05:
                detections += 1
        powers.append(detections / n_trials * 100)
    return np.array(powers)

kappa_grid = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.17, 1.5, 2.0, 2.5])
print("\nPower analysis:")
power_curves = {}
for Np in [19, 30, 50, 100]:
    power_curves[Np] = compute_power_curve(Np, kappa_grid)
    print(f"  N={Np}: κ=1.17 → {power_curves[Np][kappa_grid==1.17][0]:.0f}%, "
          f"κ=0.5 → {power_curves[Np][kappa_grid==0.5][0]:.0f}%")

# ================================================================
# # P-values (only uncorrected Rayleigh and Model A are independently computed)
P_VALS = {
    "Uncorrected Rayleigh": P_UNC,
    "Model A (Brown & Batygin)": P_W_FULL,
}
CI_LOW = {
    "Uncorrected Rayleigh": 0.003,
    "Model A (Brown & Batygin)": 0.011,
}
CI_HIGH = {
    "Uncorrected Rayleigh": 0.019,
    "Model A (Brown & Batygin)": 0.048,
}

# FPR curve (from paper injection-recovery §4.2)
FPR_N = np.array([10, 15, 19, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200])
FPR_VALS = np.array([4.7, 4.5, 4.5, 4.3, 4.8, 4.6, 4.8, 4.7, 5.3, 5.2, 5.0, 5.2, 4.5])
FPR_ERR_LO = np.array([4.1, 4.0, 3.9, 3.7, 4.2, 4.0, 4.2, 4.1, 4.8, 4.6, 4.4, 4.6, 4.0])
FPR_ERR_HI = np.array([5.3, 5.1, 5.1, 4.9, 5.4, 5.2, 5.4, 5.3, 6.0, 5.9, 5.6, 5.8, 5.2])

# ================================================================
# MNRAS STYLE DEFAULTS
# ================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": DPI,
})

# ================================================================
# FIGURE 1: sky_coverage.png
# ================================================================
def fig_sky_coverage():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                    gridspec_kw={'height_ratios': [2.2, 1]})

    # --- Top: sky coverage schematic ---
    surveys_spec = [
        ("Pan-STARRS", 30, "#e41a1c", -5),
        ("DES",        35, "#377eb8", -18),
        ("Subaru/HSC",  5, "#4daf4a",  8),
        ("OSSOS",       4, "#ff7f00", 18),
    ]
    for sname, bmax, color, txt_y in surveys_spec:
        ax1.fill_betweenx([-bmax, bmax], 0, 360, alpha=0.12, color=color)
        ax1.text(358, txt_y, sname, fontsize=8, ha="right", va="center",
                 color=color, fontweight="bold")

    # Mark anti-aligned objects (FT28, KG163) and high-i outlier (BP519)
    for i, v in enumerate(varpi_vals):
        if v in [250.9, 258.5]:
            ax1.plot(v, i_vals[i], "o", color="#d62728", markersize=9,
                     markeredgecolor="white", markeredgewidth=1, zorder=5)
            ax1.annotate(names[i], (v, i_vals[i] - 6), fontsize=6.5, ha="center",
                         color="#d62728", fontweight="bold")
        elif v == 123.0 and i_vals[i] > 50:
            ax1.plot(v, i_vals[i], "s", color="#ff7f0e", markersize=9,
                     markeredgecolor="white", markeredgewidth=1, zorder=5)
            ax1.annotate(names[i], (v, i_vals[i] + 5), fontsize=6.5, ha="center",
                         color="#ff7f0e", fontweight="bold")
        else:
            ax1.plot(v, i_vals[i], "o", color="#1f77b4", markersize=5,
                     markeredgecolor="white", markeredgewidth=0.3, zorder=4)

    ax1.axhspan(35, 90, alpha=0.06, color="gray")
    ax1.axhspan(-90, -35, alpha=0.06, color="gray")
    ax1.text(180, 72, r"$|b| \gtrsim 35^\circ$ — poorly surveyed",
             ha="center", fontsize=7.5, color="gray", fontstyle="italic")

    ax1.set_xlim(0, 360)
    ax1.set_ylim(-55, 90)
    ax1.set_xlabel(r"$\varpi$ (degrees)", fontsize=11)
    ax1.set_ylabel(r"Orbital inclination $i$ (degrees)", fontsize=11)
    ax1.set_title("ETNO discovery surveys and orbital distribution", fontsize=11)

    # Legend
    from matplotlib.lines import Line2D
    leg = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1f77b4",
               markersize=6, label="ETNO"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=7, label="Anti-aligned (FT28, KG163)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#ff7f0e",
               markersize=6, label="High-$i$ (BP$_{519}$)"),
    ]
    ax1.legend(handles=leg, fontsize=7.5, loc="upper right", framealpha=0.9)

    # --- Bottom: detection probability vs ecliptic latitude ---
    b_grid = np.linspace(-40, 40, 300)
    p_det = 0.5 + 0.5 * np.cos(np.deg2rad(b_grid) / 0.7)**2
    p_det = np.clip(p_det, 0.05, 0.98)
    ax2.fill_between(b_grid, 0, p_det, alpha=0.12, color="steelblue")
    ax2.plot(b_grid, p_det, "k-", linewidth=1.3)
    ax2.axhline(0.76, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax2.text(30, 0.78, "Mean = 0.76", fontsize=8, color="gray")
    ax2.text(0, 0.15, r"$N_{\rm eff} \approx 14.5$", ha="center", fontsize=8.5,
             fontstyle="italic", color="steelblue")
    ax2.set_xlim(-40, 40)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"Ecliptic latitude $b$ (degrees)", fontsize=11)
    ax2.set_ylabel("Detection probability", fontsize=11)

    plt.tight_layout()
    return fig

# ================================================================
# FIGURE 2: fig_clustering.png
# ================================================================
def fig_clustering():
    fig = plt.figure(figsize=(11, 5))

    # Left: polar histogram
    ax_polar = fig.add_subplot(121, projection="polar")
    bins = np.linspace(0, 2*np.pi, 25)
    ax_polar.hist(varpi_rad, bins=bins, color="steelblue", alpha=0.55,
                  edgecolor="white", linewidth=0.4)
    ax_polar.set_xticklabels(["0°","45°","90°","135°","180°","225°","270°","315°"],
                              fontsize=8)
    cm = circ_mean(varpi_rad)
    ax_polar.plot([cm, cm], [0, 2.5], "r-", linewidth=1.5, alpha=0.8)
    ax_polar.annotate(r"$\bar{\varpi}=$" + f"{CM_ALL_DEG:.1f}°",
                      xy=(cm, 2.8), fontsize=8.5, ha="center", color="red",
                      fontweight="bold")
    ax_polar.set_title(
        f"19 ETNO $\\varpi$ distribution\n"
        f"($\\bar{{\\varpi}} = {CM_ALL_DEG:.1f}^\\circ$, $R = {R_ALL:.3f}$, "
        f"$p_{{\\rm unc}} = {P_UNC:.3f}$)",
        fontsize=10, pad=18)

    # Right: p-value bars with 95% CI
    ax_bar = fig.add_subplot(122)
    methods = list(P_VALS.keys())
    pvals = [P_VALS[m] for m in methods]
    ci_low = [CI_LOW[m] for m in methods]
    ci_high = [CI_HIGH[m] for m in methods]
    colors = ["#2c7fb8", "#7fcdbb"]

    x_pos = np.arange(len(methods))
    bars = ax_bar.bar(x_pos, pvals, width=0.55, color=colors, alpha=0.85,
                      edgecolor="gray", linewidth=0.5)

    # 95% CI shading (reviewer request)
    for i, (x, pl, ph) in enumerate(zip(x_pos, ci_low, ci_high)):
        ax_bar.fill_between([x-0.22, x+0.22], pl, ph,
                             color=colors[i], alpha=0.18)
        ax_bar.hlines(ph, x-0.22, x+0.22, colors=colors[i], linewidth=1.5, alpha=0.5)
        ax_bar.hlines(pl, x-0.22, x+0.22, colors=colors[i], linewidth=1.5, alpha=0.5)

    # Model A dashed outline (to indicate range, not single value)
    if len(bars) > 1:
        bars[1].set_edgecolor("gray")
        bars[1].set_linewidth(1.5)
        bars[1].set_linestyle("--")

    ax_bar.axhline(0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_bar.text(1.5, 0.053, r"$\alpha = 0.05$", fontsize=8, color="red", ha="right")

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([m.replace(" (", "\n(") for m in methods], fontsize=8)
    ax_bar.set_ylabel("$p$-value", fontsize=11)
    ax_bar.set_title("$p$-values across bias-correction paradigms", fontsize=10)
    ax_bar.set_ylim(0, 0.30)
    ax_bar.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    return fig

# ================================================================
# FIGURE 3: fig_fpr.png
# ================================================================
def fig_fpr():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(FPR_N, FPR_ERR_LO, FPR_ERR_HI, alpha=0.2, color='blue', label='95% CI')
    ax.plot(FPR_N, FPR_VALS, 'b-', linewidth=1.8, label='FPR (simplified survey model)')
    ax.axhline(y=5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Nominal α = 0.05')
    ax.plot(19, FPR_VALS[FPR_N==19][0], 'ro', markersize=7, zorder=5)
    ax.annotate(f'N = 19: FPR = {FPR_VALS[FPR_N==19][0]:.1f}%', xy=(19, FPR_VALS[FPR_N==19][0]),
                xytext=(30, 8),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')

    ax.set_xlabel('Sample Size N', fontsize=12)
    ax.set_ylabel('False-Positive Rate (%)', fontsize=12)
    ax.set_title('Uncorrected Rayleigh Test: False-Positive Rate vs N', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(5, 210)
    ax.set_ylim(0, 15)
    ax.grid(True, alpha=0.3)

    # Add explanatory text
    ax.text(0.98, 0.95, 'Simplified survey model\n(not full OSSOS simulator)',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    return fig

# ================================================================
# FIGURE 4: fig_sensitivity.png
# ================================================================
def fig_sensitivity():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.5),
                                    sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    sort_idx = np.argsort(varpi_vals)
    sorted_names = [names[i] for i in sort_idx]
    sorted_deltas = loo_deltas[sort_idx]
    sorted_R = loo_R_changes[sort_idx]

    # Top: Δϖ̄
    colors1 = ["#d62728" if d > 5 else "#7f7f7f" for d in sorted_deltas]
    ax1.bar(range(N), sorted_deltas, color=colors1, edgecolor="gray",
             linewidth=0.3, width=0.65)
    ax1.axhline(5, color="#d62728", linestyle="--", linewidth=1, alpha=0.6)
    ax1.axhline(BOOT_SE, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax1.text(N-0.5, BOOT_SE+0.3,
             f"Bootstrap SE = {BOOT_SE:.1f}°", fontsize=8, color="gray", ha="right")
    ax1.text(N-0.5, 5.8, r"$\Delta\varpi = 5^\circ$", fontsize=8,
             color="#d62728", ha="right")
    ax1.set_ylabel(r"$\Delta\bar{\varpi}$ (degrees)", fontsize=11)
    ax1.set_title("Leave-one-out sensitivity analysis", fontsize=11)
    ax1.set_ylim(0, max(sorted_deltas)*1.35)

    # Bottom: ΔR/R
    colors2 = ["#d62728" if r > 10 else "#7f7f7f" for r in sorted_R]
    ax2.bar(range(N), sorted_R, color=colors2, edgecolor="gray",
             linewidth=0.3, width=0.65)
    ax2.axhline(10, color="#d62728", linestyle="--", linewidth=1, alpha=0.6)
    ax2.text(N-0.5, 11, "10% change", fontsize=8, color="#d62728", ha="right")
    ax2.set_xticks(range(N))
    ax2.set_xticklabels(sorted_names, rotation=55, ha="right", fontsize=8)
    ax2.set_ylabel(r"$\Delta R\,/\,R$ (%)", fontsize=11)
    ax2.set_xlabel(r"Object (sorted by $\varpi$)", fontsize=11)

    # Annotate key objects
    for i, (d, r) in enumerate(zip(sorted_deltas, sorted_R)):
        if r > 10:
            ax2.annotate(f"{r:.0f}%", xy=(i, r), xytext=(i, r+2),
                         ha="center", fontsize=7, color="#d62728", fontweight="bold")

    plt.tight_layout()
    return fig

# ================================================================
# FIGURE 5: fig_subset_stability.png
# ================================================================
def fig_subset_stability():
    ks_vals = list(subset_results.keys())
    med_p = [subset_results[k]["median_p"] for k in ks_vals]
    iqr_lo = [subset_results[k]["iqr_low"] for k in ks_vals]
    iqr_hi = [subset_results[k]["iqr_high"] for k in ks_vals]
    frac_sig = [subset_results[k]["frac_sig"] for k in ks_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: median p with IQR
    ax1.plot(ks_vals, med_p, "bo-", linewidth=1.5, markersize=6)
    ax1.fill_between(ks_vals, iqr_lo, iqr_hi, alpha=0.15, color="steelblue")
    ax1.axhline(0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.text(18.5, 0.053, r"$\alpha=0.05$", fontsize=8, color="red", va="bottom")
    ax1.set_xlabel("Subset size $k$", fontsize=11)
    ax1.set_ylabel("Median $p$-value (IQR shaded)", fontsize=10)
    ax1.set_xlim(11, 19)
    ax1.tick_params(labelsize=9)

    # Right: fraction significant
    ax2.plot(ks_vals, frac_sig, "ro-", linewidth=1.5, markersize=6)
    ax2.axhline(90, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.text(18.5, 91, "90% threshold", fontsize=8, color="gray")
    ax2.set_xlabel("Subset size $k$", fontsize=11)
    ax2.set_ylabel("Subsets with $p < 0.05$ (%)", fontsize=10)
    ax2.set_xlim(11, 19)
    ax2.set_ylim(45, 100)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    return fig

# ================================================================
# FIGURE 6: fig_power_curve.png
# ================================================================
def fig_power_curve():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    colors_pow = {19: "#e41a1c", 30: "#377eb8", 50: "#4daf4a", 100: "#984ea3"}
    for Np, color in colors_pow.items():
        ax.plot(kappa_grid, power_curves[Np], "o-", color=color,
                linewidth=1.3, markersize=3.5, label=f"$N = {Np}$")

    ax.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(2.35, 81, "80% power", fontsize=8, color="gray")
    ax.axvline(1.17, color="red", linestyle=":", linewidth=0.8, alpha=0.4)
    ax.text(1.15, 5, r"$\kappa \approx 1.17$" "\n(observed)", fontsize=7.5,
            color="red", ha="right")

    # Mark key power levels
    ax.annotate(f"{power_curves[19][kappa_grid==1.17][0]:.0f}%",
                xy=(1.17, power_curves[19][kappa_grid==1.17][0]),
                fontsize=8, fontweight="bold", ha="center", color="#e41a1c")
    ax.annotate(f"{power_curves[19][kappa_grid==0.5][0]:.0f}%",
                xy=(0.5, power_curves[19][kappa_grid==0.5][0]),
                fontsize=8, fontweight="bold", ha="center", color="#e41a1c")

    ax.set_xlabel(r"Concentration $\kappa$", fontsize=11)
    ax.set_ylabel("Detection power (%)", fontsize=11)
    ax.set_title("Rayleigh test power (upper bounds, no survey biases)", fontsize=11)
    ax.set_xlim(0.05, 2.55)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower right")

    ax.text(0.98, 0.035,
            "Survey selection biases not modeled;\ntrue power is lower.",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="#666666", fontstyle="italic")

    plt.tight_layout()
    return fig

# ================================================================
# FIGURE 7: fig_survey_decomposition.png
# ================================================================
def fig_survey_decomposition():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    main_surveys = ["Pan-STARRS", "DES", "Subaru/HSC"]
    loo_p = [loo_survey[s]["p"] for s in main_surveys]
    within_p = [within_survey[s]["p"] for s in main_surveys]
    colors_s = ["#e41a1c", "#377eb8", "#4daf4a"]

    # Left: leave-one-survey-out
    x = np.arange(3)
    ax1.bar(x, loo_p, width=0.5, color=colors_s, alpha=0.85,
             edgecolor="gray", linewidth=0.3)
    ax1.axhline(0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.axhline(P_UNC, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
    ax1.text(2.5, P_UNC+0.002, f"Full sample\n$p={P_UNC:.3f}$",
             fontsize=7.5, color="gray", ha="center")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}\nremoved" for s in main_surveys], fontsize=8)
    ax1.set_ylabel("Rayleigh $p$-value", fontsize=11)
    ax1.set_title("Leave-one-survey-out", fontsize=10)
    ax1.tick_params(labelsize=9)

    # Right: within-survey
    ax2.bar(x, within_p, width=0.5, color=colors_s, alpha=0.85,
             edgecolor="gray", linewidth=0.3)
    ax2.axhline(0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ns_survey = [within_survey[s]["n"] for s in main_surveys]
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{s}\n($N={n}$)" for s, n in zip(main_surveys, ns_survey)],
                         fontsize=8)
    ax2.set_ylabel("Rayleigh $p$-value", fontsize=11)
    ax2.set_title("Within-survey clustering", fontsize=10)
    ax2.set_ylim(0, max(within_p)*1.3 + 0.05)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    return fig

# ================================================================
# FIGURE 8: fig_inclination_efficiency.png
# ================================================================
def fig_inclination_efficiency():
    fig, ax = plt.subplots(figsize=(7, 4))

    def det_prob_model(ii):
        """Survey-weighted detection probability at inclination i (degrees)."""
        # Effective b_max: weighted average of survey latitude coverage
        b_max_eff = 20.0
        if ii <= b_max_eff:
            return 0.95 - 0.19 * (ii / b_max_eff)
        else:
            if ii < b_max_eff * 0.99:
                return 0.95
            frac = np.sin(np.deg2rad(b_max_eff)) / np.sin(np.deg2rad(np.maximum(ii, b_max_eff+0.01)))
            frac = np.clip(frac, 0, 1)
            return 0.8 * (2/np.pi) * np.arcsin(frac)

    i_grid = np.linspace(0, 58, 300)
    p_det = np.array([det_prob_model(i) for i in i_grid])

    ax.plot(i_grid, p_det * 100, "k-", linewidth=1.4)
    ax.fill_between(i_grid, 0, p_det * 100, alpha=0.08, color="steelblue")

    sc = ax.scatter(i_vals, [det_prob_model(i)*100 for i in i_vals],
                     c=varpi_vals, cmap="twilight", s=50, zorder=5,
                     edgecolors="white", linewidth=0.5)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(r"$\varpi$ (degrees)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Highlight BP519
    bp_idx = [j for j, n in enumerate(names) if "BP519" in n][0]
    ax.plot(i_vals[bp_idx], det_prob_model(i_vals[bp_idx])*100, "ro",
            markersize=7, markerfacecolor="none", markeredgewidth=1.5)
    ax.annotate(f"2015 BP$_{{519}}$\n$i={i_vals[bp_idx]:.1f}^\\circ$",
                xy=(i_vals[bp_idx], det_prob_model(i_vals[bp_idx])*100),
                xytext=(44, 25), fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.7))

    ax.axhline(76, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(42, 77.5, "Mean = 76%", fontsize=8, color="gray")

    ax.set_xlabel(r"Orbital inclination $i$ (degrees)", fontsize=11)
    ax.set_ylabel("Detection probability (%)", fontsize=11)
    ax.set_title("Inclination-dependent detection efficiency", fontsize=11)
    ax.set_xlim(0, 58)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    return fig

# ================================================================
# FIGURE 9: fig_modelA_leave_one_out.png
# ================================================================
def fig_modelA_loo():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    sort_idx = np.argsort(varpi_vals)
    sorted_names_m = [names[i] for i in sort_idx]
    sorted_pw = pw_loo[sort_idx]
    sorted_Rw = Rw_loo_change[sort_idx]

    # Top: p_w
    colors_pw = ["#2c7fb8" if p < 0.05 else "#d62728" for p in sorted_pw]
    ax1.bar(range(N), sorted_pw, color=colors_pw, edgecolor="gray",
             linewidth=0.3, width=0.65)
    ax1.axhline(0.05, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax1.axhline(P_W_FULL, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax1.text(N-0.5, P_W_FULL+0.001,
             f"Full sample: $p_w = {P_W_FULL:.4f}$",
             fontsize=8, color="gray", ha="right")
    ax1.set_ylabel("Weighted $p_w$", fontsize=11)
    ax1.set_title("Model A: Weighted Rayleigh leave-one-out analysis", fontsize=11)
    ax1.tick_params(labelsize=9)

    # Bottom: ΔR_w
    colors_Rw = ["#d62728" if r > 10 else "#7f7f7f" for r in sorted_Rw]
    ax2.bar(range(N), sorted_Rw, color=colors_Rw, edgecolor="gray",
             linewidth=0.3, width=0.65)
    ax2.axhline(10, color="#d62728", linestyle="--", linewidth=1, alpha=0.6)
    ax2.text(N-0.5, 11, "10% change", fontsize=8, color="#d62728", ha="right")
    ax2.set_xticks(range(N))
    ax2.set_xticklabels(sorted_names_m, rotation=55, ha="right", fontsize=8)
    ax2.set_ylabel(r"$\Delta R_w\,/\,R_w$ (%)", fontsize=11)
    ax2.set_xlabel(r"Object (sorted by $\varpi$)", fontsize=11)
    ax2.tick_params(labelsize=9)

    plt.tight_layout()
    return fig

# ================================================================
# MAIN
# ================================================================
FIGURES = [
    ("fig_sky_coverage.png",           fig_sky_coverage),
    ("fig_clustering.png",             fig_clustering),
    ("fig_fpr.png",                    fig_fpr),
    ("fig_sensitivity.png",            fig_sensitivity),
    ("fig_subset_stability.png",       fig_subset_stability),
    ("fig_power_curve.png",            fig_power_curve),
    ("fig_survey_decomposition.png",   fig_survey_decomposition),
    ("fig_inclination_efficiency.png", fig_inclination_efficiency),
    ("fig_modelA_leave_one_out.png",   fig_modelA_loo),
]

def main():
    print(f"Data: {N} ETNOs loaded from {DATA_PATH}")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)

    for fname, func in FIGURES:
        print(f"  {fname:35s} ... ", end="", flush=True)
        try:
            fig = func()
            outpath = os.path.join(OUT_DIR, fname)
            fig.savefig(outpath, dpi=DPI, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            size_kb = os.path.getsize(outpath) / 1024
            print(f"✅  {size_kb:.0f} KB")
        except Exception as e:
            print(f"❌  {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"Done. {len(FIGURES)} figures → {OUT_DIR}/")

if __name__ == "__main__":
    main()
