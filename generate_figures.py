#!/usr/bin/env python3
"""
generate_figures.py — Reproduce the two figures in the paper.

Requires: etno_complete.json (same directory)
Output: fig_polar.png, fig_pvalues.png
"""
import json, os, sys
import numpy as np
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "etno_complete.json")

with open(data_path) as f:
    etno_list = json.load(f)

varpi_deg = np.array([e["varpi"] for e in etno_list])
varpi_rad = np.deg2rad(varpi_deg)
i_deg = np.array([e["i"] for e in etno_list])
N = len(varpi_rad)

def rayleigh_r(angles_rad):
    return float(np.abs(np.mean(np.exp(1j * np.asarray(angles_rad)))))

R_obs = rayleigh_r(varpi_rad)
circ_mean = float(np.rad2deg(np.angle(np.mean(np.exp(1j * varpi_rad)))) % 360)

# ============================================================
# Figure 1: Polar histogram
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})

# Two anti-aligned for labeling
kg163_deg = 250.9
ft28_deg = 258.5

theta_rad = np.deg2rad(varpi_deg)
ax.hist(theta_rad, bins=18, color="steelblue", edgecolor="white", alpha=0.85)
ax.axvline(np.deg2rad(circ_mean), color="darkorange", ls="--", lw=2,
           label=fr"Circular mean: ${circ_mean:.1f}^\circ$")

# Mark anti-aligned objects
ax.annotate("2015 KG$_{163}$", xy=(np.deg2rad(kg163_deg), 3.0),
            fontsize=8, color="darkred", ha="center")
ax.annotate("2013 FT$_{28}$", xy=(np.deg2rad(ft28_deg), 3.0),
            fontsize=8, color="darkred", ha="center")

ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.legend(loc="lower right", fontsize=9)
ax.set_title(f"ETNO $\\varpi$ (N={N}, $R={R_obs:.3f}$)", pad=20)

fig.tight_layout()
fig.savefig("fig_polar.png", dpi=150)
plt.close()
print(f"Saved: fig_polar.png  (R={R_obs:.3f}, circ_mean={circ_mean:.1f} deg)")

# ============================================================
# Figure 2: Four p-values bar chart
# ============================================================
methods = ["Uncorrected\nRayleigh", "Model A\n(Weighted)", "Model B\n(Inj.-Recov.)", "Model D\n(Bootstrap)"]
p_vals = [0.0068, 0.0096, 0.0692, 0.0065]
colors = ["#2c7bb6", "#abd9e9", "#fdae61", "#2c7bb6"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(methods, p_vals, color=colors, edgecolor="gray", linewidth=0.8)
ax.axhline(0.05, color="black", ls="--", lw=1.2, label=r"$\alpha = 0.05$")

for bar, p in zip(bars, p_vals):
    y_pos = bar.get_height() + 0.003
    ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"$p={p:.4f}$",
            ha="center", va="bottom", fontsize=10)

ax.set_ylabel("$p$-value")
ax.set_title("Four $p$-values on the Identical 19-Object ETNO Sample")
ax.legend(fontsize=10)
ax.set_ylim(0, max(p_vals) * 1.25)
fig.tight_layout()
fig.savefig("fig_pvalues.png", dpi=150)
plt.close()
print("Saved: fig_pvalues.png")
print("Done.")
