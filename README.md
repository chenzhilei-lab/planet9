# Orbital-Anomaly Predictions: Monte Carlo Power Analysis

Supporting code for the manuscript:

> **"Orbital-Anomaly Predictions: A Historical Diagnostic Checklist for the LSST Era"**
> Zhilei Chen
> *Submitted to Nature Astronomy*

## Contents

- `etno_power_analysis_compact.py` — Monte Carlo simulation that generates Table 3 (small-N convergence power analysis). Simulates synthetic ETNO orbital distributions at varying sample sizes and clustering strengths, analysed with six published clustering statistics.
- `etno_power_analysis.py` — Extended version of the simulation with additional diagnostic output.
- `results/power_analysis.json` — Pre-computed output data used to produce Table 3.

## Requirements

- Python 3.8+
- NumPy
- SciPy

## Usage

```bash
# Reproduce Table 3
python3 etno_power_analysis_compact.py

# Expected output: tabular summary of p-value spreads and convergence fractions
# for κ = 0.0, 0.3, 0.5, 0.7 at N = 20, 50, 100, 150, 200
```

The simulation draws synthetic ETNO samples from isotropic (κ = 0.0) and clustered (κ = 0.3, 0.5, 0.7) orbital distributions, then applies:
- Rayleigh test (uniformity of longitudes)
- Survey-weighted Rayleigh tests (Pan-STARRS, DES, OSSOS footprints)
- Kuiper test
- Monte Carlo reference

Convergence is defined as all six methods simultaneously yielding p < 0.05.

## License

MIT
