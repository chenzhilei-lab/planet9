#!/usr/bin/env python3
"""Leave-one-out sensitivity for ETNO ϖ clustering."""
import json, numpy as np

with open("etno_complete.json") as f:
    data = json.load(f)

varpi_rad = np.deg2rad([d["varpi"] for d in data])
names = [d["label"] for d in data]
N = len(varpi_rad)

def R(a): return float(np.abs(np.mean(np.exp(1j*np.asarray(a)))))

R_full = R(varpi_rad)
cm_full = float(np.rad2deg(np.angle(np.mean(np.exp(1j*varpi_rad)))) % 360)

print(f"N={N}, R_obs={R_full:.4f}, circ_mean={cm_full:.1f} deg\n")
print(f"{'Object':20s}  {'ϖ':>6s}  {'Shift':>6s}  {'ΔR':>7s}")
print("-"*48)

for i in range(N):
    mask = np.ones(N, bool); mask[i] = False
    vm = varpi_rad[mask]
    cm_j = float(np.rad2deg(np.angle(np.mean(np.exp(1j*vm)))) % 360)
    shift = min(abs(cm_j-cm_full), 360-abs(cm_j-cm_full))
    dR = (1 - R(vm)/R_full)*100
    flag = " ***" if shift > 5 else ""
    print(f"  {names[i]:18s}  {data[i]['varpi']:6.1f}  {shift:6.1f}  {dR:+7.1f}%{flag}")

# Identify KG163 and FT28
for n in names:
    if "KG163" in n:
        idx_k = names.index(n)
    if "FT28" in n:
        idx_f = names.index(n)

mask_k = np.ones(N, bool); mask_k[idx_k] = False
mask_f = np.ones(N, bool); mask_f[idx_f] = False
dR_k = (1 - R(varpi_rad[mask_k])/R_full)*100
dR_f = (1 - R(varpi_rad[mask_f])/R_full)*100
print(f"\nKG163 R change: {dR_k:+.1f}%")
print(f"FT28  R change: {dR_f:+.1f}%")
