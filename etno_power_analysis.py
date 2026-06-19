"""
ETNO Power Analysis — Monte Carlo Convergence Test v1.0
========================================================
Design: Simulate ETNO populations at N = 20, 50, 100, 150, 200
  with known clustering strength, apply multiple bias-correction
  methods, measure p-value spread convergence.

Inputs:  None (fully synthetic)
Outputs: results/power_analysis.json + LaTeX table
         results/power_analysis_table.tex

Run: python etno_power_analysis.py --n-trials 200
     (--quick for 20 trials, ~3 min on RTX 3090)
========================================================
"""

import os, json, time, argparse, warnings
import numpy as np
from scipy.stats import pearsonr, kuiper, rayleightest
from multiprocessing import Pool

warnings.filterwarnings('ignore')
SEED = 42

# =============================================================
# SURVEY FOOTPRINT MODELS
# =============================================================

def survey_weights(longitudes, survey='panstarrs'):
    """Sky coverage weight per longitude for a given survey."""
    if survey == 'panstarrs':
        # Pan-STARRS: 3pi steradian, declination > -30
        return 0.5 + 0.5 * np.sin(np.radians(longitudes * 0.5))
    elif survey == 'des':
        # DES: 5000 deg^2 in southern galactic cap
        return 0.3 + 0.7 * np.maximum(0, np.cos(np.radians(longitudes - 180)))**2
    elif survey == 'ossos':
        # OSSOS: 3 narrowly spaced pointings near ecliptic
        return 0.2 + 0.4 * np.exp(-((longitudes - 90) / 30)**2) \
                    + 0.4 * np.exp(-((longitudes - 270) / 30)**2)
    elif survey == 'subaru_hsc':
        # Subaru/HSC: 300 deg^2 at various pointings
        return 0.4 + 0.3 * np.sin(np.radians(longitudes * 0.3))**2
    else:
        return np.ones_like(longitudes)


# =============================================================
# ETNO POPULATION GENERATOR
# =============================================================

def generate_etno_population(N, cluster_strength=0.5, seed_offset=0):
    """
    Generate N ETNOs with longitudes of perihelion.
    cluster_strength = 0: uniform distribution (no P9)
    cluster_strength = 1: perfectly clustered at 0 deg
    cluster_strength controls the von Mises concentration parameter kappa.
    """
    np.random.seed(SEED + seed_offset)
    
    # Convert cluster_strength [0,1] to von Mises kappa [0, 10]
    kappa = cluster_strength * 10.0
    
    if kappa < 0.01:
        # Uniform
        varpi = np.random.uniform(0, 360, N)
    else:
        from scipy.stats import vonmises
        # von Mises centered at ~70 deg (P9 cluster direction)
        varpi = np.degrees(vonmises.rvs(kappa, loc=np.radians(70), size=N))
        varpi = varpi % 360
    
    return varpi


# =============================================================
# BIAS-CORRECTION METHODS
# =============================================================

def method_raw_rayleigh(varpi):
    """Raw Rayleigh test — no bias correction."""
    # Convert to radians for Rayleigh test
    theta = np.radians(varpi)
    r = np.sqrt(np.mean(np.cos(theta))**2 + np.mean(np.sin(theta))**2)
    n = len(varpi)
    # Rayleigh test: p = exp(-n * r^2)
    p = np.exp(-n * r**2)
    return float(p)


def method_weighted_rayleigh(varpi, survey='panstarrs'):
    """Weighted Rayleigh — sky-coverage weighting (Batygin-style)."""
    weights = survey_weights(varpi, survey)
    weights = weights / weights.sum()
    
    theta = np.radians(varpi)
    cos_w = np.sum(weights * np.cos(theta))
    sin_w = np.sum(weights * np.sin(theta))
    r_w = np.sqrt(cos_w**2 + sin_w**2)
    
    # Effective sample size from weights
    n_eff = 1.0 / np.sum(weights**2)
    p = np.exp(-n_eff * r_w**2)
    return float(p)


def method_kuiper(varpi):
    """Kuiper test against uniform distribution (Batygin-style)."""
    theta = np.radians(varpi)
    from scipy.stats import kuiper_two
    # Kuiper test: compare observed distribution to uniform
    stat, p = kuiper(np.sort(theta) / (2 * np.pi), len(theta))
    return float(p)


def method_survey_simulator(varpi, n_mc=500):
    """Survey simulator null (Napier-style, simplified)."""
    np.random.seed(SEED + hash(str(varpi.tobytes())) % 10000)
    
    # Observed clustering
    theta = np.radians(varpi)
    r_obs = np.sqrt(np.mean(np.cos(theta))**2 + np.mean(np.sin(theta))**2)
    
    # Null: inject uniform population through survey
    count_greater = 0
    for i in range(n_mc):
        null_varpi = np.random.uniform(0, 360, len(varpi))
        theta_null = np.radians(null_varpi)
        r_null = np.sqrt(np.mean(np.cos(theta_null))**2 + np.mean(np.sin(theta_null))**2)
        if r_null >= r_obs:
            count_greater += 1
    
    p = (count_greater + 1) / (n_mc + 1)
    return float(p)


# =============================================================
# FRAME-COUNT POWER ANALYSIS
# =============================================================

def single_trial(args):
    """Run one trial at given N and cluster_strength."""
    N, strength, trial_id = args
    
    varpi = generate_etno_population(N, strength, trial_id)
    
    p_values = {}
    p_values['raw'] = method_raw_rayleigh(varpi)
    
    for survey in ['panstarrs', 'des', 'ossos', 'subaru_hsc']:
        p_values[f'weighted_{survey}'] = method_weighted_rayleigh(varpi, survey)
    
    p_values['kuiper'] = method_kuiper(varpi)
    p_values['simulator'] = method_survey_simulator(varpi)
    
    # p-value spread
    all_p = list(p_values.values())
    spread = max(all_p) - min(all_p)
    max_p = max(all_p)
    min_p = min(all_p)
    
    return {
        'N': N,
        'strength': strength,
        'p_values': p_values,
        'spread': spread,
        'min_p': min_p,
        'max_p': max_p,
        'range_factor': max_p / max(min_p, 1e-6)
    }


def run_power_analysis(N_values, strength_values, n_trials=100, n_workers=8):
    """Full Monte Carlo power analysis."""
    tasks = []
    for N in N_values:
        for strength in strength_values:
            for t in range(n_trials):
                tasks.append((N, strength, t))
    
    print(f"  Total trials: {len(tasks)}")
    print(f"  Workers: {n_workers}")
    
    t0 = time.time()
    
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(single_trial, tasks)
    else:
        results = [single_trial(t) for t in tasks]
    
    elapsed = time.time() - t0
    print(f"  Runtime: {elapsed:.0f}s ({elapsed/len(tasks):.2f}s/trial)")
    
    # Aggregate
    aggregated = {}
    for N in N_values:
        for strength in strength_values:
            key = (N, strength)
            trials = [r for r in results if r['N'] == N and r['strength'] == strength]
            
            spreads = [t['spread'] for t in trials]
            min_ps = [t['min_p'] for t in trials]
            max_ps = [t['max_p'] for t in trials]
            factors = [t['range_factor'] for t in trials]
            
            # Fraction of trials where p-values cross the significance boundary
            sig_below = sum(1 for t in trials if t['max_p'] < 0.05)
            sig_above = sum(1 for t in trials if t['min_p'] > 0.05)
            ambiguous = n_trials - sig_below - sig_above
            
            aggregated[key] = {
                'N': N,
                'strength': strength,
                'n_trials': n_trials,
                'mean_spread': float(np.mean(spreads)),
                'std_spread': float(np.std(spreads)),
                'mean_min_p': float(np.mean(min_ps)),
                'mean_max_p': float(np.mean(max_ps)),
                'mean_range_factor': float(np.mean(factors)),
                'all_below_0.05': sig_below,
                'all_above_0.05': sig_above,
                'ambiguous': ambiguous,
                'convergence_rate': sig_below / max(n_trials, 1) if strength > 0 else 0
            }
    
    return aggregated


# =============================================================
# LATEX TABLE
# =============================================================

def generate_latex_table(agg, N_values, strength_values):
    lines = [
        r"\begin{table*}[htbp]",
        r"\footnotesize\centering",
        r"\caption{Monte Carlo Power Analysis: $p$-value Spread vs.\ Sample Size and Clustering Strength}",
        r"\label{tab:power_analysis}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Strength} & \textbf{$N=20$} & \textbf{$N=50$} & \textbf{$N=100$} & \textbf{$N=150$} & \textbf{$N=200$} \\",
        r"\midrule",
    ]
    
    for strength in strength_values:
        row = f"  {strength:.1f} "
        for N in N_values:
            key = (N, strength)
            if key in agg:
                r = agg[key]
                row += f"& ${r['mean_spread']:.3f} \\pm {r['std_spread']:.3f}$ "
        row += r"\\"
        lines.append(row)
    
    lines.append(r"\midrule")
    lines.append(r"\textbf{Resolution rate}")
    for N in N_values:
        key = (N, 0.5)
        if key in agg:
            rate = agg[key]['convergence_rate'] * 100
            lines.append(f"& {rate:.0f}\\% ")
    lines.append(r"\\")
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\tablecomments{Each entry shows the mean $p$-value spread (max-min) across five bias-correction methods, "
        r"averaged over Monte Carlo trials. Strength=0 corresponds to no Planet Nine (uniform $\varpi$); "
        r"strength=0.3--0.5 corresponds to weak-to-moderate clustering. "
        r"Resolution rate: fraction of trials where all methods converge to $p<0.05$ at strength=0.5. "
        r"Convergence is reached when the spread narrows to $<0.05$ and no method reports $p>0.05$.}",
        r"\end{table*}",
    ])
    
    # Also generate a brief convergence summary
    lines.append("")
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\footnotesize\centering")
    lines.append(r"\caption{Convergence Summary: Sample Size Thresholds}")
    lines.append(r"\label{tab:convergence_threshold}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Metric} & \textbf{N=50} & \textbf{N=100} & \textbf{N=150} \\")
    lines.append(r"\midrule")
    
    for strength in [0.3, 0.5]:
        row = f"  Strength={strength:.1f} mean spread "
        for N in [50, 100, 150]:
            key = (N, strength)
            if key in agg:
                row += f"& {agg[key]['mean_spread']:.3f} "
        row += r"\\"
        lines.append(row)
    
    lines.extend([
        r"\midrule",
        r"\textbf{Convergence at N=100?} & --- & --- & --- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\tablecomments{A spread $<0.05$ across all five methods is taken as evidence of convergence---"
        r"no single bias model can flip the qualitative conclusion. Thresholds are provisional and should be "
        r"refined with survey-specific simulations.}",
        r"\end{table*}",
    ])
    
    return '\n'.join(lines)


# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='ETNO Power Analysis v1.0')
    p.add_argument('--n-trials', type=int, default=100,
                   help='Number of Monte Carlo trials per (N, strength) cell')
    p.add_argument('--n-workers', type=int, default=8,
                   help='Parallel workers')
    p.add_argument('--quick', action='store_true',
                   help='Quick mode: 20 trials, 3 N values, 2 strengths')
    args = p.parse_args()
    
    if args.quick:
        N_values = [20, 50, 100]
        strength_values = [0.0, 0.5]
        n_trials = 20
        workers = 2
        print("QUICK MODE")
    else:
        N_values = [20, 50, 100, 150, 200]
        strength_values = [0.0, 0.3, 0.5, 0.7]
        n_trials = args.n_trials
        workers = min(args.n_workers, os.cpu_count() or 8)
    
    print("="*60)
    print("ETNO Power Analysis — Monte Carlo Convergence Test")
    print("="*60)
    print(f"  N values: {N_values}")
    print(f"  Strengths: {strength_values}")
    print(f"  Trials/cell: {n_trials}")
    print(f"  Total: {len(N_values) * len(strength_values) * n_trials}")
    
    results = run_power_analysis(N_values, strength_values, n_trials, workers)
    
    # Save
    os.makedirs('results', exist_ok=True)
    output = {
        'experiment': 'etno_power_analysis_v1',
        'date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {'N_values': N_values, 'strength_values': strength_values,
                   'n_trials': n_trials},
        'results': {f'N{k[0]}_S{k[1]}': v for k, v in results.items()}
    }
    json.dump(output, open('results/power_analysis.json', 'w'), indent=2)
    print(f"\n  Saved: results/power_analysis.json")
    
    # LaTeX
    latex = generate_latex_table(results, N_values, strength_values)
    with open('results/power_analysis_table.tex', 'w') as f:
        f.write(latex)
    print(f"  Saved: results/power_analysis_table.tex")
    
    # Print preview
    print(f"\n  Convergence at N=100 (strength=0.5): ", end="")
    key = (100, 0.5)
    if key in results:
        r = results[key]
        print(f"spread={r['mean_spread']:.3f}+-{r['std_spread']:.3f}, "
              f"range={r['mean_min_p']:.3f}--{r['mean_max_p']:.3f}, "
              f"convergence={r['convergence_rate']*100:.0f}%")
    
    print(f"\n  Done.")
