# ETNO Survey Simulator

Public, open-source survey simulator for ETNO perihelion clustering statistics. Replicates the four major published bias-correction paradigms on a common sample using only published survey parameters. Accompanies Chen (2026), MNRAS.

## Quick start

```bash
# 1. Run all analyses
python run_etno_simulator.py --n-boot 50000 --n-fpr 5000

# 2. Verify correctness
python test_known_answer.py

# 3. Use as a Python library
python -c "
from etno_simulator import SurveySimulator
sim = SurveySimulator('etno_complete.json')
results = sim.run_all()
print(sim.summary())
"
```

## Requirements

- Python 3.8+
- numpy
- scipy

```bash
pip install numpy scipy matplotlib
```

## File overview

| File | Purpose |
|------|---------|
| `etno_simulator.py` | Core module: `SurveySimulator` class with all statistical frameworks |
| `run_etno_simulator.py` | CLI entry point |
| `test_known_answer.py` | Known-answer tests for correctness verification |
| `etno_complete.json` | JPL SBDB orbital elements for the 19-object ETNO sample (2026-06-13) |

## What it computes

All analyses run on the identical 19-object sample (a > 150 AU, q > 30 AU):

- **Four statistical frameworks:** Uncorrected Rayleigh, Weighted Rayleigh (Brown & Batygin 2019), Injection-recovery proxy (OSSOS paradigm), Bootstrap empirical null (Model D)
- **False-positive rate:** Survey-induced FPR via 5,000 injection-recovery trials
- **Leave-one-out sensitivity:** Per-object impact on circular mean and R
- **Random-subset stability:** p-value distribution at k = 12, 14, 16, 18 (5,000 subsets each)
- **Spherical clustering:** 2D (ϖ, i) and 3D (ϖ, i, Ω) Rayleigh tests
- **Kuiper test:** Sensitive to non-unimodal departures from uniformity
- **Inclination-dependent detection efficiency:** Effective sample size N_eff
- **N=14 comparison:** All four frameworks on the pre-2021 14-object sample

## Survey parameters

All detection model parameters are from published survey descriptions (Shankman et al. 2017; Brown & Batygin 2019; Napier et al. 2021):

| Survey | Ecliptic latitude | Longitude range | Base efficiency |
|--------|:---:|:---:|:---:|
| Pan-STARRS | ±30° | 0°–360° | 0.90 |
| DES | ±35° | 0°–180° | 0.85 |
| Subaru/HSC | ±5° | 330°–60° | 0.95 |
| Other | ±20° | 0°–360° | 0.70 |

## Known-answer tests

Run `python test_known_answer.py` to verify:

1. Uniform ϖ (N=100) → p ≈ 0.5 (simulator does not create spurious clustering)
2. Clustered ϖ (κ=5) → p ≪ 0.05 (simulator detects real clustering)
3. Real data: R_obs = 0.505, N = 19, circular mean = 46.3°
4. Model D ≡ Uncorrected (structural identity verified)

## Reproducing paper numbers

Every numerical claim in the accompanying manuscript can be reproduced:

```bash
python run_etno_simulator.py --seed 20260713 --n-boot 50000 --n-fpr 5000 --output results.txt
```

The output file `results.txt` contains all p-values and statistics reported in the paper.

**Expected output (values may fluctuate within MC noise):**

```
Core four p-values (N=19): ~0.007 / ~0.010 / ~0.07 / ~0.18 / ~0.007
FPR ≈ 6%
```

## Limitations

- **Model B is a simplified proxy.** Omits weather losses, chip gaps, per-night pointing variation, and moving-object trailing. The true OSSOS simulator depends on proprietary survey-operations data not publicly available.
- **Model C is numerically unreliable at N=19.** The joint maximum-likelihood fit has a nearly flat likelihood surface. p ~ 0.18 is a soft upper bound.
- **No N-body validation.** Synthetic ETNO populations from published simulations are not publicly available in machine-readable format.

## Citation

If you use this simulator in your research, please cite:

Chen, Z. (2026). "Statistical Framework Choice—Not Selection Bias—Drives the Decade-Long ETNO Perihelion Clustering Controversy." MNRAS, submitted.

```bibtex
@article{chen2026etno,
  author = {Chen, Zhilei},
  title = {Statistical Framework Choice—Not Selection Bias—Drives the Decade-Long ETNO Perihelion Clustering Controversy},
  journal = {MNRAS},
  year = {2026},
  note = {submitted}
}
```

## License

MIT License. See the repository for details.
