#!/usr/bin/env python3
"""
multiple_testing_correction.py — Bonferroni and Benjamini-Hochberg corrections
for the four ETNO clustering p-values.

The four p-values come from incommensurable statistical frameworks (different
null hypotheses, test statistics, and likelihood functions). Standard multiple-
comparison corrections assume exchangeable tests under a global null, which
does not strictly apply here. We provide the corrected values for completeness
while noting this caveat.

Output: multiple_testing_correction.txt (LaTeX table rows)

Dependencies: numpy, scipy
"""

import os, sys
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data: p-values from G038 paper
# ---------------------------------------------------------------------------
methods = [
    "Uncorrected Rayleigh",
    "Model~A (Brown \\& Batygin)",
    "Model~B (OSSOS, simplified)",
    "Model~C (Napier et al., soft upper bound)",
    "Model~D (bootstrap empirical null)",
]
p_values = np.array([0.008, 0.023, 0.089, 0.178, 0.377])
n_tests = len(p_values)

# ---------------------------------------------------------------------------
# Bonferroni correction
# ---------------------------------------------------------------------------
p_bonf = np.minimum(p_values * n_tests, 1.0)

# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR
# ---------------------------------------------------------------------------
order = np.argsort(p_values)
p_sorted = p_values[order]
# BH critical values: (i / m) * α for α=0.05
bh_critical = np.arange(1, n_tests + 1) / n_tests * 0.05
# Find largest i where p_(i) <= (i/m)*α
bh_reject = p_sorted <= bh_critical
# BH adjusted p-values (simplified)
p_bh = np.minimum(p_sorted * n_tests / np.arange(1, n_tests + 1), 1.0)
# Reorder back
p_bh_unsorted = np.zeros(n_tests)
p_bh_unsorted[order] = p_bh

# ---------------------------------------------------------------------------
# Holm-Bonferroni (step-down)
# ---------------------------------------------------------------------------
holm_critical = 0.05 / (n_tests - np.arange(n_tests))
holm_reject = p_sorted <= holm_critical
# Holm adjusted p-values
p_holm_unsorted = np.zeros(n_tests)
for i in range(n_tests):
    idx = order[i]
    p_holm_unsorted[idx] = min(p_values[idx] * (n_tests - i), 1.0)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("=" * 70)
print("Multiple Testing Correction for ETNO Clustering p-values")
print("=" * 70)
print(f"  Number of tests (k) = {n_tests}")
print(f"  Nominal α           = 0.05")
print()
print(f"{'Method':<40} {'p_raw':>8} {'Bonferroni':>12} {'BH-FDR':>10} {'Holm':>10}")
print("-" * 70)
for i in range(n_tests):
    print(f"{methods[i]:<40} {p_values[i]:>8.4f} {p_bonf[i]:>12.4f} {p_bh_unsorted[i]:>10.4f} {p_holm_unsorted[i]:>10.4f}")

print()
print("Significance at α=0.05 after correction:")
print("-" * 70)
for i in range(n_tests):
    sig_raw = "✓" if p_values[i] < 0.05 else "—"
    sig_bonf = "✓" if p_bonf[i] < 0.05 else "—"
    sig_bh = "✓" if p_bh_unsorted[i] < 0.05 else "—"
    sig_holm = "✓" if p_holm_unsorted[i] < 0.05 else "—"
    print(f"  {methods[i]:<40} raw={sig_raw}  Bonf={sig_bonf}  BH={sig_bh}  Holm={sig_holm}")

# ---------------------------------------------------------------------------
# Summary for paper
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("KEY FINDING")
print("=" * 70)
print("After Bonferroni correction (×5), only the uncorrected Rayleigh test")
print(f"remains significant (p_bonf = {p_bonf[0]:.3f}). All bias-corrected models")
print(f"cross the α=0.05 threshold after correction:")
print(f"  Model A: p_bonf = {p_bonf[1]:.3f}")
print(f"  Model B: p_bonf = {p_bonf[2]:.3f}")
print(f"  Model C: p_bonf = {p_bonf[3]:.3f}")
print(f"  Model D: p_bonf = {p_bonf[4]:.3f}")
print()
print("CAVEAT: The four models test incommensurable null hypotheses and are")
print("not exchangeable replicates. The Bonferroni correction is conservative")
print("and provided for completeness only; the paper's central argument does")
print("not rest on any single p-value crossing α=0.05, but on the range of")
print("p-values across defensible methods.")

# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------
latex_table = r"""\begin{table}
\centering
\caption{Multiple-testing corrections for the four ETNO clustering tests.
All four methods test distinct null hypotheses and are not exchangeable;
standard corrections are provided for completeness and may be over-conservative.}
\label{tab:multiple_testing}
\footnotesize
\begin{tabular}{lcccc}
\hline\noalign{\smallskip}
\textbf{Method} & \textbf{Raw $p$} & \textbf{Bonferroni} & \textbf{BH-FDR} & \textbf{Holm} \\
\hline\noalign{\smallskip}
"""

for i in range(n_tests):
    latex_table += f"{methods[i]} & {p_values[i]:.3f} & {p_bonf[i]:.3f} & {p_bh_unsorted[i]:.3f} & {p_holm_unsorted[i]:.3f} \\\\\n"

latex_table += r"""\noalign{\smallskip}\hline
\multicolumn{5}{l}{\footnotesize All corrections use $k=5$ tests. Bonferroni: $p_{\rm adj} = \min(k \cdot p, 1.0)$. BH: Benjamini--Hochberg (1995). Holm: Holm (1979) step-down.}
\end{tabular}
\end{table}
"""

outpath = os.path.join(OUT_DIR, "multiple_testing_correction.tex")
with open(outpath, 'w') as f:
    f.write(latex_table)
print(f"\nLaTeX table saved: {outpath}")

# Also save as plain text for the paper
outpath_txt = os.path.join(OUT_DIR, "multiple_testing_correction.txt")
with open(outpath_txt, 'w') as f:
    f.write(f"Multiple Testing Correction\n")
    f.write(f"{'Method':<40} {'Raw':>8} {'Bonf':>8} {'BH':>8} {'Holm':>8}\n")
    f.write("-" * 70 + "\n")
    for i in range(n_tests):
        f.write(f"{methods[i]:<40} {p_values[i]:>8.4f} {p_bonf[i]:>8.4f} {p_bh_unsorted[i]:>8.4f} {p_holm_unsorted[i]:>8.4f}\n")
print(f"Plain text saved: {outpath_txt}")
