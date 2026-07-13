# planet9 — ETNO Clustering Statistical Calibration Pipeline

An open-source pipeline for auditing the statistical evidence for Planet Nine from extreme trans-Neptunian object (ETNO) perihelion clustering.

**Manuscript**: "How much does the null hypothesis matter? A model-free bootstrap exposes the assumption-dependence of ETNO clustering significance" (under review at RAA / SCPMA, 2026).

## Repository Contents

### Core pipeline
- `generate_all_figures.py` — Main pipeline: all four bias-correction models (A–D), leave-one-out, subset stability, power analysis
- `clustering_audit.py` — Rayleigh test + bootstrap + CI estimation
- `subset_stability.py` — Random-subset stability for uncorrected Rayleigh and Model A
- `leave_one_out.py` — Leave-one-out sensitivity for circular mean and Rayleigh R

### Supplementary analyses
- `model_c_selection_likelihood.py` — Model C joint likelihood profile (intrinsic VM + selection function)
- `model_c_likelihood_profile.py` — Pure von Mises profile likelihood (no selection function)
- `bayesian_prior_sensitivity.py` — Bayes factor prior sensitivity (4 families × 19 combinations)
- `joint_2d_test.py` — 2D spherical Rayleigh test (ϖ + i) with MC null distributions
- `multiple_testing_correction.py` — Bonferroni, Holm, Benjamini-Hochberg corrections
- `model_b_fpr_paper_model.py` — Injection–recovery using paper's survey_weight model
- `model_b_uncertainty_v2.py` — Model B systematic uncertainty via injection–recovery
- `fpr_physical_survey_model.py` — FPR from ecliptic coverage + orbital mechanics

### Data
- `etno_complete.json` — Orbital elements for 19 ETNOs (JPL SBDB, 2026 June)
- `etno_data.json` — Alternative data format
- `etno_fetch.py` — JPL SBDB API retrieval script

### Output
- `output/` — All script outputs (.txt, .log, .png, .tex)

### Experimental / pending
- `nbody_scattering.py` — REBOUND N-body scattering simulation (requires REBOUND + CUDA; not yet validated)
- `run_nbody_grid.bat` — Windows batch file for 6-GPU grid runs

## Dependencies
- numpy, scipy, matplotlib
- (optional) rebound, reboundx for N-body module

## License
MIT License.

## Citation
If you use this code, please cite the accompanying manuscript and the Zenodo archive:
[Zenodo DOI to be added]

## Author
Zhilei Chen, Guangdong Peizheng College. 2604513@peizheng.edu.cn
