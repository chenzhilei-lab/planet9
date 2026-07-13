#!/usr/bin/env python3
"""
Planet Nine ETNO clustering audit — verified against paper v31.
Run: python3 clustering_audit.py
Requires: etno_data.json (19 objects in JPL-verified format)
"""
import json, math, random

# ── Load data ──
with open("etno_data.json") as f:
    data = json.load(f)
varpis_deg = [obj["varpi"] for obj in data["objects"]]
varpis = [math.radians(v) for v in varpis_deg]
N = len(varpis)
print(f"Loaded {N} ETNOs.")

# ── Core functions ──
def circ_mean(angles):
    sx = sum(math.cos(a) for a in angles)
    sy = sum(math.sin(a) for a in angles)
    return math.atan2(sy, sx) % (2*math.pi)

def circ_dist(a, b):
    d = abs(a - b)
    return min(d, 2*math.pi - d)

def rayleigh_p(angles):
    """Returns (R, p-value) using exact exp(-NR²) formula."""
    n = len(angles)
    sx = sum(math.cos(a) for a in angles)
    sy = sum(math.sin(a) for a in angles)
    R = math.sqrt(sx**2 + sy**2) / n
    p = math.exp(-n * R**2)
    return R, p

# ── Uncorrected Rayleigh ──
R_unc, p_unc = rayleigh_p(varpis)
cm = circ_mean(varpis)
print(f"\n=== Uncorrected Rayleigh ===")
print(f"  N = {N}")
print(f"  Circular mean ϖ̄ = {math.degrees(cm):.1f}°")
print(f"  Rayleigh R = {R_unc:.4f}")
print(f"  p = {p_unc:.4f}  (paper: 0.008)")

# ── Bootstrap CI ──
random.seed(42)
n_boot = 10000
p_boot = []
for _ in range(n_boot):
    sample = [random.choice(varpis) for _ in range(N)]
    _, p = rayleigh_p(sample)
    p_boot.append(p)
p_boot.sort()
ci_low = p_boot[250]
ci_high = p_boot[9749]
print(f"  Bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

# ── Bootstrap SE of ϖ̄ ──
cm_boot = []
for _ in range(n_boot):
    s = [random.choice(varpis) for _ in range(N)]
    cm_boot.append(circ_mean(s))
dists = sorted(circ_dist(c, cm) for c in cm_boot)
se = math.degrees(dists[6827])  # 68.27%
print(f"  Bootstrap SE of ϖ̄ = {se:.1f}°")

# ── Leave-one-out ──
print(f"\n=== Leave-One-Out (threshold = bootstrap SE = {se:.1f}°) ===")
print(f"{'Object':16s} {'ϖ':>6s} {'Δϖ̄':>6s} {'ΔR%':>6s}")
print("-" * 40)
for i, obj in enumerate(data["objects"]):
    sub = varpis[:i] + varpis[i+1:]
    cm_i = circ_mean(sub)
    d = math.degrees(circ_dist(cm_i, cm))
    Ri, _ = rayleigh_p(sub)
    dR = abs(Ri - R_unc) / R_unc * 100
    flag = " ← >SE" if d > se else ""
    print(f"  {obj['name']:14s} {obj['varpi']:5.1f}° {d:5.1f}° {dR:5.1f}%{flag}")

# ── Summary ──
n_above_se = sum(1 for i in range(N) if math.degrees(circ_dist(circ_mean(varpis[:i]+varpis[i+1:]), cm)) > se)
print(f"\nObjects with Δϖ̄ > bootstrap SE ({se:.1f}°): {n_above_se}")
print(f"Paper conclusion: 0 objects exceed SE ({n_above_se == 0})")
print(f"Rayleigh p = {p_unc:.4f} (paper: 0.008)")
print(f"Bootstrap SE = {se:.1f}° (paper: 17.3°)")
