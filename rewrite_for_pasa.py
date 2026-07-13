"""Rewrite A manuscript for PASA: remove fabricated data, keep verified data."""
import shutil

src = "/mnt/d/Papers/A_ApJ_ETNO/current/A_PASA_v01.tex"
with open(src, 'r') as f:
    lines = f.readlines()

# First pass: identify line ranges
print(f"Total lines: {len(lines)}")

# Replace title
for i, line in enumerate(lines):
    if "Calibrating the Planet Nine Evidence" in line:
        lines[i] = lines[i].replace(
            "Calibrating the Planet Nine Evidence: A Multi-Model Benchmark for ETNO Clustering",
            "Small-Sample Limitations in ETNO Orbital Clustering Diagnostics: A Reproducible Framework at N=19"
        )
        print(f"Title replaced at line {i+1}")
        break

# Also check for alternative title in abstract
for i, line in enumerate(lines):
    if "Calibrating ETNO clustering statistics" in line:
        lines[i] = lines[i].replace(
            "Calibrating ETNO clustering statistics: An open-source pre-LSST benchmark at N = 19",
            "Small-Sample Limitations in ETNO Orbital Clustering Diagnostics: A Reproducible Framework at N=19"
        )
        print(f"Short title replaced at line {i+1}")
        break

# Update abstract - remove references to FPR and Models B/C
for i, line in enumerate(lines):
    if "false-positive rate" in line.lower() and "34" in line:
        lines[i] = ""
        print(f"FPR mention removed at line {i+1}")

# Remove injection-recovery FPR section (lines 160-179 range)
in_fpr = False
fpr_start = None
for i, line in enumerate(lines):
    if "subsection{Injection--Recovery: False-Positive Rate Calibration}" in line:
        in_fpr = True
        fpr_start = i
    if in_fpr and "\\\\subsection" in line and i > fpr_start + 5:
        in_fpr = False
        break

if fpr_start:
    # Find end of FPR section
    end = fpr_start + 1
    while end < len(lines) and not lines[end].startswith('\\\\subsection') and not lines[end].startswith('\\\\subsection'):
        # Check for next major section
        if '\\\\subsection{' in lines[end] or '\\\\section{' in lines[end]:
            break
        end += 1
    
    # Instead of removing, replace with a brief note
    fpr_note = [
        "\\\\subsection{False-Positive Rate Considerations}\n",
        "A rigorous false-positive rate calibration via injection--recovery simulation requires the OSSOS survey simulator code,\n",
        "which is not publicly available and therefore not reproduced here. We refer readers to the original OSSOS characterisation \\\\citep{shankman2017}\n",
        "for a detailed injection--recovery analysis of survey selection biases.\n"
    ]
    lines[fpr_start:end] = fpr_note
    print(f"FPR section replaced at lines {fpr_start+1}-{end}")

# Replace Model B/C detailed p-values with Model A only  
for i, line in enumerate(lines):
    if "Model~B (OSSOS)" in line and "p = 0.089" in line:
        lines[i] = line.replace("Model~B (OSSOS) gives $p = 0.089$ (CI: 0.052--0.143). Model~C (Napier et al.) gives $p = 0.178$ (CI: 0.124--0.248).", "")
        print(f"Model B/C p-values removed at line {i+1}")
    if "Models~B and~C" in line and "re-implementation" in line.lower():
        lines[i] = ""  # Remove reference to B/C re-implementation
        print(f"Model B/C re-implementation ref removed at line {i+1}")

# Fix the abstract to remove FPR claim
for i, line in enumerate(lines):
    if "false-positive" in line.lower() and "high" in line.lower() and "baseline" in line.lower():
        lines[i] = ""  # Remove FPR mentions
    if "34" in line and "FPR" in line:
        lines[i] = ""
    if "28--41" in line:
        lines[i] = ""
    if "N > 120" in line and "5" in line:
        lines[i] = ""

# Update conclusions to remove FPR/implications
for i, line in enumerate(lines):
    if "34\\\\%" in line or "28--41" in line or "34\\%" in line:
        lines[i] = ""

# Replace power analysis values with real ones (at kappa=1.17)
for i, line in enumerate(lines):
    if "88\\\\%" in line and "power" in line.lower():
        lines[i] = line.replace("88\\\\%", "83\\\\%")
        print(f"Power value corrected at line {i+1}")
    if "45\\\\%" in line and "power" in line.lower():
        lines[i] = line.replace("45\\\\%", "25\\\\%")
        print(f"Power value corrected at line {i+1}")

# Remove "The N > 120 threshold" paragraph
for i, line in enumerate(lines):
    if "The $N > 120$ threshold is read directly" in line:
        lines[i] = ""
        # Remove subsequent line too
        if i+1 < len(lines):
            lines[i+1] = ""

# Clean up empty lines (remove consecutive empty lines)
cleaned = []
prev_empty = False
for line in lines:
    is_empty = line.strip() == ''
    if is_empty and prev_empty:
        continue
    cleaned.append(line)
    prev_empty = is_empty

with open(src, 'w') as f:
    f.writelines(cleaned)

print(f"\nDone. Lines after cleanup: {len(cleaned)}")
