#!/usr/bin/env python3
"""ETNO Power Analysis -- Monte Carlo Convergence Test v1.0"""
import os, json, time, argparse, warnings
import numpy as np
from scipy.stats import pearsonr

def _kuiper_one_sample(x, n=None):
    """One-sample Kuiper test (circular KS)."""
    x = np.sort(np.asarray(x))
    n = len(x)
    D_plus = np.max(np.arange(1, n + 1) / n - x)
    D_minus = np.max(x - np.arange(0, n) / n)
    V = D_plus + D_minus
    z = V * (np.sqrt(n) + 0.155 + 0.24 / np.sqrt(n))
    p = 0.0
    for j in range(1, 5):
        p += 2 * (4 * j**2 * z**2 - 1) * np.exp(-2 * j**2 * z**2)
    return V, max(0.0, min(1.0, p))
from multiprocessing import Pool
warnings.filterwarnings('ignore')
SEED = 42

def survey_weights(lons, s='panstarrs'):
    if s == 'panstarrs': return 0.5 + 0.5 * np.sin(np.radians(lons * 0.5))
    elif s == 'des': return 0.3 + 0.7 * np.maximum(0, np.cos(np.radians(lons - 180)))**2
    elif s == 'ossos': return 0.2 + 0.4 * np.exp(-((lons - 90)/30)**2) + 0.4 * np.exp(-((lons - 270)/30)**2)
    elif s == 'subaru': return 0.4 + 0.3 * np.sin(np.radians(lons * 0.3))**2
    else: return np.ones_like(lons)

def gen_etno(N, strength=0.5, off=0):
    np.random.seed(SEED + off)
    kappa = strength * 10.0
    if kappa < 0.01: return np.random.uniform(0, 360, N)
    from scipy.stats import vonmises
    return np.degrees(vonmises.rvs(kappa, loc=np.radians(70), size=N)) % 360

def p_raw(v):
    t = np.radians(v); n = len(v)
    r = np.sqrt(np.mean(np.cos(t))**2 + np.mean(np.sin(t))**2)
    return float(np.exp(-n * r**2))

def p_weighted(v, s):
    w = survey_weights(v, s); w /= w.sum()
    t = np.radians(v)
    r = np.sqrt(np.sum(w * np.cos(t))**2 + np.sum(w * np.sin(t))**2)
    neff = 1.0 / np.sum(w**2)
    return float(np.exp(-neff * r**2))

def p_kuiper(v):
    t = np.sort(np.radians(v)) / (2 * np.pi)
    _, pv = _kuiper_one_sample(t)
    return float(pv)

def p_sim(v, mc=200):
    np.random.seed(SEED + hash(str(v.tobytes())) % 10000)
    t = np.radians(v)
    r_obs = np.sqrt(np.mean(np.cos(t))**2 + np.mean(np.sin(t))**2)
    cnt = sum(1 for _ in range(mc) if np.sqrt(np.mean(np.cos(np.radians(np.random.uniform(0,360,len(v)))))**2 + np.mean(np.sin(np.radians(np.random.uniform(0,360,len(v)))))**2) >= r_obs)
    return (cnt + 1) / (mc + 1)

def trial(args):
    N, s, tid = args
    v = gen_etno(N, s, tid)
    ps = [p_raw(v)]
    for surv in ['panstarrs','des','ossos','subaru']: ps.append(p_weighted(v, surv))
    ps.append(p_kuiper(v))
    ps.append(p_sim(v))
    sp = max(ps) - min(ps)
    return {'N':N,'str':s,'spread':sp,'min_p':min(ps),'max_p':max(ps),'range_f':max(ps)/max(min(ps),1e-6)}

def run(Ns, Ss, nt=100):
    tasks = [(N,s,t) for N in Ns for s in Ss for t in range(nt)]
    print(f"  Trials: {len(tasks)}")
    t0 = time.time()
    with Pool(min(16, os.cpu_count() or 8)) as p:
        res = p.map(trial, tasks)
    print(f"  Runtime: {time.time()-t0:.0f}s")
    agg = {}
    for N in Ns:
        for s in Ss:
            tr = [r for r in res if r['N']==N and r['str']==s]
            agg[(N,s)] = {'N':N,'str':s,'nt':nt,
                'mean_spread':float(np.mean([t['spread'] for t in tr])),
                'std_spread':float(np.std([t['spread'] for t in tr])),
                'mean_min':float(np.mean([t['min_p'] for t in tr])),
                'mean_max':float(np.mean([t['max_p'] for t in tr])),
                'converged_pct':float(sum(1 for t in tr if t['max_p']<0.05)/nt*100)}
    return agg

def latex(agg, Ns, Ss):
    L = [r"\begin{table*}[htbp]", r"\footnotesize\centering",
         r"\caption{Power Analysis: p-value Spread vs Sample Size}",
         r"\label{tab:power_analysis}",
         r"\begin{tabular}{l"+"c"*len(Ns)+"}", r"\toprule",
         r"\textbf{Strength} & " + " & ".join(f"$N$={N}" for N in Ns) + r" \\", r"\midrule"]
    for s in Ss:
        row = f"  {s:.1f} "
        for N in Ns:
            if (N,s) in agg: row += f"& ${agg[(N,s)]['mean_spread']:.3f}\\pm{agg[(N,s)]['std_spread']:.3f}$ "
        L.append(row + r"\\")
    L += [r"\midrule", r"\textbf{Conv.@N=100}", r"& --- & ---"]
    if (100,0.5) in agg: L[-1] = f"& --- & --- & {agg[(100,0.5)]['converged_pct']:.0f}\\%"
    if (100,0.3) in agg: L.append(f"& {agg[(100,0.3)]['converged_pct']:.0f}\\%")
    L += [r"\\", r"\bottomrule", r"\end{tabular}",
          r"\tablecomments{Spread = max(p)-min(p) across 7 bias methods. Convergence = all methods report p<0.05.}",
          r"\end{table*}"]
    return '\n'.join(L)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nt', type=int, default=100)
    p.add_argument('--quick', action='store_true')
    a = p.parse_args()
    Ns = [20,50,100,150,200] if not a.quick else [20,50,100]
    Ss = [0.0,0.3,0.5,0.7] if not a.quick else [0.0,0.5]
    nt = 20 if a.quick else a.nt
    print(f"N={Ns} Strength={Ss} Trials={nt}")
    agg = run(Ns, Ss, nt)
    os.makedirs('results', exist_ok=True)
    json.dump({f'N{k[0]}_S{k[1]}':v for k,v in agg.items()}, open('results/power_analysis.json','w'), indent=2)
    open('results/power_analysis_table.tex','w').write(latex(agg, Ns, Ss))
    print("Saved results/")
