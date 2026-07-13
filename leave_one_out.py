#!/usr/bin/env python3
"""Leave-one-out sensitivity analysis for ETNO clustering."""

import json
import math
import numpy as np

with open("etno_data.json") as f:
    data = json.load(f)

N = len(data["objects"])
varpis_rad = np.radians([obj["varpi"] for obj in data["objects"]])

# Full-sample circular mean
sin_all = np.sum(np.sin(varpis_rad))
cos_all = np.sum(np.cos(varpis_rad))
vmean_all = math.degrees(math.atan2(sin_all, cos_all)) % 360

print(f"Full sample: N={N}, circular mean varpi = {vmean_all:.1f} deg\n")
print(f"{'Object':16s} {'varpi':>6s} {'Shift':>7s} {'Leverage':>10s}")
print("-" * 45)

shifts = []
for i, obj in enumerate(data["objects"]):
    mask = np.ones(N, dtype=bool)
    mask[i] = False
    loo = varpis_rad[mask]
    sin_loo = np.sum(np.sin(loo))
    cos_loo = np.sum(np.cos(loo))
    vmean_loo = math.degrees(math.atan2(sin_loo, cos_loo)) % 360
    shift = abs(vmean_loo - vmean_all)
    if shift > 180:
        shift = 360 - shift
    shifts.append((obj["name"], obj["varpi"], shift))
    lever = ">>> HIGH" if shift > 5 else ""
    print(f"  {obj['name']:14s} {obj['varpi']:6.1f}  {shift:6.1f} deg  {lever}")

shifts.sort(key=lambda x: x[2], reverse=True)
print(f"\nTop 3 high-leverage objects:")
for name, v, s in shifts[:3]:
    print(f"  {name:14s}  varpi={v:.0f} deg  shift={s:.1f} deg")
