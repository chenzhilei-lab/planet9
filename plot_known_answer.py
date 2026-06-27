"""Generate fig_known_answer.png from the known-answer test results."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load results
with open(os.path.join(os.path.dirname(__file__) or ".", "known_answer_results.json")) as f:
    results = json.load(f)

kappa_vals = sorted([float(k) for k in results.keys()])
power_raw = [results[str(k)]["power_raw"] for k in kappa_vals]
power_modelA = [results[str(k)]["power_modelA"] for k in kappa_vals]
med_p_raw = [results[str(k)]["med_p_raw"] for k in kappa_vals]
med_p_modelA = [results[str(k)]["med_p_modelA"] for k in kappa_vals]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

# Left: Power vs κ
ax1.plot(kappa_vals, power_raw, "bo-", linewidth=1.5, markersize=5, label="Uncorrected Rayleigh")
ax1.plot(kappa_vals, power_modelA, "rs--", linewidth=1.5, markersize=5, label="Model A (weighted)")
ax1.axhline(5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
ax1.text(1.9, 6, "FPR ≈ 5% (κ=0)", fontsize=7, color="gray")
ax1.axhline(80, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax1.text(1.9, 81, "80% power", fontsize=7, color="gray")
ax1.axvline(1.17, color="red", linestyle=":", linewidth=0.8, alpha=0.4)
ax1.text(1.17, 3, "Observed\nκ≈1.17", fontsize=7, color="red", ha="center")
ax1.annotate("60%", xy=(1.17, 60), fontsize=9, fontweight="bold", ha="center",
             color="blue")
ax1.set_xlabel("Concentration κ", fontsize=10)
ax1.set_ylabel("Detection power at α=0.05 (%)", fontsize=10)
ax1.set_title("Known-answer test: survey-biased power", fontsize=10)
ax1.set_xlim(0, 2.1)
ax1.set_ylim(0, 100)
ax1.tick_params(labelsize=8)
ax1.legend(fontsize=7, loc="lower right")
ax1.text(0.02, 0.98, "Survey selection reduces\neffective N from 19 to ≈12",
         transform=ax1.transAxes, fontsize=7, ha="left", va="top",
         color="gray", fontstyle="italic")

# Right: Median p-value vs κ
ax2.semilogy(kappa_vals, med_p_raw, "bo-", linewidth=1.5, markersize=5, label="Uncorrected Rayleigh")
ax2.semilogy(kappa_vals, med_p_modelA, "rs--", linewidth=1.5, markersize=5, label="Model A (weighted)")
ax2.axhline(0.05, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.text(1.9, 0.052, "α=0.05", fontsize=7, color="red")
ax2.set_xlabel("Concentration κ", fontsize=10)
ax2.set_ylabel("Median recovered p-value", fontsize=10)
ax2.set_title("Recovered p-value vs input clustering", fontsize=10)
ax2.set_xlim(0, 2.1)
ax2.tick_params(labelsize=8)
ax2.legend(fontsize=7, loc="lower right")

# Comparison ideal vs survey-biased at κ=1.17
ax2.annotate("Ideal (no biases):\np=0.008, power=88%",
             xy=(1.17, 0.028), xytext=(1.4, 0.15),
             fontsize=7, ha="center",
             arrowprops=dict(arrowstyle="->", color="gray"))
ax2.annotate("Survey-biased:\np=0.028, power=60%",
             xy=(1.17, 0.028), fontsize=7, ha="center",
             xytext=(0.6, 0.15),
             arrowprops=dict(arrowstyle="->", color="gray"))

plt.tight_layout()
outpath = os.path.join(os.path.dirname(__file__) or ".", "fig_known_answer.png")
plt.savefig(outpath, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {outpath}")
