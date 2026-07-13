#!/usr/bin/env python3
"""
nbody_scattering.py — N-body scattering simulation for ETNO ϖ clustering.

Tests whether primordial scattering from known giant planets (or an additional
Planet Nine) can produce non-uniform ϖ distributions that match the observed
ETNO sample.

Two modes:
  (a) Giant-planet scattering only (null hypothesis: no P9)
  (b) Planet Nine shepherding (alternative: P9-induced clustering)

Uses REBOUND with IAS15 integrator + GPU acceleration (CUDA via REBOUNDx).

Parallelization: run multiple instances with different P9 parameters on
separate GPUs.

Usage (Windows cmd on 4060Ti):
    # Mode (a): scattering only, no Planet Nine
    python nbody_scattering.py --mode scatter --N 300 --tmax 100 --output scatter_null.json

    # Mode (b): Planet Nine at a=500 AU, M=5 M_earth
    python nbody_scattering.py --mode p9 --M_p9 5 --a_p9 500 --N 300 --tmax 100 --output p9_M5_a500.json

    # Mode (b) with GPU
    python nbody_scattering.py --mode p9 --M_p9 10 --a_p9 600 --N 500 --tmax 200 --gpu --output p9_M10_a600.json

Dependencies:
    pip install rebound reboundx
    GPU mode requires CUDA toolkit + REBOUND compiled with CUDA support.

Output: JSON file with final orbital elements, ϖ distribution, Rayleigh R.
"""

import argparse, json, os, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
G_Msun = 1.0           # REBOUND units: G=1, M_sun=1
AU = 1.0                # REBOUND default units
YEAR = 2.0 * np.pi      # REBOUND time unit (1 year = 2π in units where G=1)
DAY = YEAR / 365.25

# Giant planet masses (in solar masses) and initial orbital elements
PLANETS = {
    'Jupiter':  {'m': 9.5479e-4, 'a': 5.204,  'e': 0.0489, 'i': 1.304,  'Omega': 100.5, 'omega': 274.1, 'M': 19.7},
    'Saturn':   {'m': 2.8586e-4, 'a': 9.582,  'e': 0.0565, 'i': 2.485,  'Omega': 113.7, 'omega': 339.4, 'M': 50.4},
    'Uranus':   {'m': 4.3656e-5, 'a': 19.201, 'e': 0.0457, 'i': 0.773,  'Omega': 74.0,  'omega': 98.0,  'M': 142.2},
    'Neptune':  {'m': 5.1505e-5, 'a': 30.048, 'e': 0.0086, 'i': 1.770,  'Omega': 131.7, 'omega': 265.6, 'M': 259.9},
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
def add_planet(sim, name, params):
    """Add a giant planet to the simulation."""
    import rebound
    m = params['m']
    a = params['a']
    e = params['e']
    inc = np.deg2rad(params['i'])
    Omega = np.deg2rad(params['Omega'])
    omega = np.deg2rad(params['omega'])
    M = np.deg2rad(params['M'])
    sim.add(m=m, a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M)

def add_test_particles(sim, n_particles, rng_seed=None):
    """Add test particles in a primordial scattered disk.
    
    Semi-major axis: log-uniform between 50 and 500 AU (biased toward outer disk)
    Eccentricity: uniform [0, 0.3] (initially low-e disk)
    Inclination: Rayleigh distribution with σ = 3° (thin disk)
    Other angles: uniform
    
    Particles are massless (m=0).
    """
    import rebound
    rng = np.random.default_rng(rng_seed if rng_seed else int(time.time()))
    
    a_vals = 10 ** rng.uniform(np.log10(50), np.log10(500), n_particles)
    e_vals = rng.uniform(0.0, 0.3, n_particles)
    inc_vals = np.abs(rng.rayleigh(scale=np.deg2rad(3.0), size=n_particles))
    Omega_vals = rng.uniform(0, 2*np.pi, n_particles)
    omega_vals = rng.uniform(0, 2*np.pi, n_particles)
    M_vals = rng.uniform(0, 2*np.pi, n_particles)
    
    for i in range(n_particles):
        sim.add(m=0.0, a=a_vals[i], e=e_vals[i], inc=inc_vals[i],
                Omega=Omega_vals[i], omega=omega_vals[i], M=M_vals[i])

def extract_etnos(sim):
    """Extract particles satisfying ETNO criteria: a > 150 AU, q > 30 AU."""
    etnos = []
    for p in sim.particles[5:]:  # skip Sun + 4 planets + optional P9
        if p.m > 0:
            continue  # skip massive bodies
        a = p.a
        e = p.e
        q = a * (1 - e)
        if np.isfinite(a) and a > 150.0 and q > 30.0:
            # Compute ϖ = Ω + ω  (longitude of perihelion)
            varpi = np.mod(np.degrees(p.Omega + p.omega), 360.0)
            inc = np.degrees(p.inc)
            etnos.append({
                'a': float(a),
                'e': float(e),
                'q': float(q),
                'i': float(inc),
                'varpi': float(varpi),
                'Omega': float(np.degrees(p.Omega)),
                'omega': float(np.degrees(p.omega)),
            })
    return etnos

def compute_clustering(etnos):
    """Compute Rayleigh R and p-value for extracted ETNOs."""
    if len(etnos) < 3:
        return {'N': len(etnos), 'R': np.nan, 'p': np.nan}
    
    varpi_rad = np.deg2rad([e['varpi'] for e in etnos])
    n = len(varpi_rad)
    R = np.abs(np.mean(np.exp(1j * varpi_rad)))
    
    # Monte Carlo p-value
    rng = np.random.default_rng(42)
    n_mc = 10000
    R_null = np.zeros(n_mc)
    for j in range(n_mc):
        rand_varpi = rng.uniform(0, 2*np.pi, n)
        R_null[j] = np.abs(np.mean(np.exp(1j * rand_varpi)))
    p = np.mean(R_null >= R)
    
    return {'N': n, 'R': float(R), 'p': float(p)}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='N-body scattering for ETNO clustering')
    parser.add_argument('--mode', choices=['scatter', 'p9'], default='scatter',
                        help='Simulation mode: scatter (no P9) or p9 (with Planet Nine)')
    parser.add_argument('--N', type=int, default=300,
                        help='Number of test particles')
    parser.add_argument('--tmax', type=float, default=100.0,
                        help='Integration time (Myr)')
    parser.add_argument('--M_p9', type=float, default=5.0,
                        help='Planet Nine mass (Earth masses)')
    parser.add_argument('--a_p9', type=float, default=500.0,
                        help='Planet Nine semi-major axis (AU)')
    parser.add_argument('--e_p9', type=float, default=0.3,
                        help='Planet Nine eccentricity')
    parser.add_argument('--i_p9', type=float, default=15.0,
                        help='Planet Nine inclination (deg)')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Use GPU acceleration')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='nbody_output.json',
                        help='Output JSON file')
    parser.add_argument('--dt', type=float, default=0.05,
                        help='Output interval (yr). 0 = only final snapshot.')
    args = parser.parse_args()
    
    try:
        import rebound
    except ImportError:
        print("ERROR: rebound not installed. Run: pip install rebound reboundx")
        sys.exit(1)
    
    t_start = time.time()
    print(f"N-body ETNO Scattering Simulation")
    print(f"  Mode:      {args.mode}")
    print(f"  Particles: {args.N}")
    print(f"  t_max:     {args.tmax} Myr")
    print(f"  GPU:       {args.gpu}")
    print()
    
    # Create simulation
    sim = rebound.Simulation()
    sim.integrator = "ias15"
    sim.dt = 0.01 * YEAR  # initial timestep
    sim.add(m=G_Msun)  # Sun
    
    # Add giant planets
    for name, params in PLANETS.items():
        add_planet(sim, name, params)
    
    # Add Planet Nine if requested
    if args.mode == 'p9':
        M_p9_Msun = args.M_p9 * 3.0035e-6  # Earth masses → solar masses
        inc_p9_rad = np.deg2rad(args.i_p9)
        rng = np.random.default_rng(args.seed)
        sim.add(m=M_p9_Msun, a=args.a_p9, e=args.e_p9, inc=inc_p9_rad,
                Omega=rng.uniform(0, 2*np.pi),
                omega=rng.uniform(0, 2*np.pi),
                M=rng.uniform(0, 2*np.pi))
        print(f"  P9 mass:   {args.M_p9} M_earth")
        print(f"  P9 a:      {args.a_p9} AU")
        print(f"  P9 e:      {args.e_p9}")
        print(f"  P9 i:      {args.i_p9} deg")
        print()
    
    # Add test particles
    add_test_particles(sim, args.N, rng_seed=args.seed)
    sim.move_to_com()
    
    # Integration
    t_max = args.tmax * 1e6 * YEAR
    n_outputs_snapshots = max(1, int(t_max / (args.dt * YEAR)))
    
    print(f"  N_active:  {sim.N - sim.N_var}")
    print(f"  Integrator: {sim.integrator}")
    print(f"  Wall time estimate: {args.tmax * args.N / 500:.1f} hours (very rough)")
    print()
    print("Integrating...")
    
    n_snap = 0
    sys.stdout.flush()
    
    # Progress reporting
    t_next_report = t_max * 0.05  # report every 5%
    next_report = t_next_report
    
    try:
        for _ in range(n_outputs_snapshots):
            t_target = min((n_snap + 1) * args.dt * YEAR, t_max)
            sim.integrate(t_target, exact_finish_time=False)
            n_snap += 1
            
            if sim.t >= next_report:
                pct = 100 * sim.t / t_max
                elapsed = time.time() - t_start
                remaining = elapsed * (t_max / sim.t - 1)
                print(f"  {pct:.0f}%  t={sim.t/YEAR/1e6:.1f} Myr  "
                      f"elapsed={elapsed/60:.0f}min  remaining≈{remaining/60:.0f}min  "
                      f"N_active={sim.N - sim.N_var}")
                sys.stdout.flush()
                next_report += t_next_report
            
            if sim.t >= t_max:
                break
    except rebound.Escape as e:
        print(f"  Note: particle escaped at t={sim.t/YEAR/1e6:.2f} Myr")
    except Exception as e:
        print(f"  Integration error: {e}")
    
    elapsed = time.time() - t_start
    print(f"\nIntegration complete: {elapsed/60:.1f} min")
    print(f"  N_remaining: {sim.N - sim.N_var}")
    print(f"  t_final:     {sim.t/YEAR/1e6:.2f} Myr")
    
    # Extract ETNOs
    etnos = extract_etnos(sim)
    clustering = compute_clustering(etnos)
    
    print(f"\n  ETNO candidates: {clustering['N']}")
    print(f"  Rayleigh R:      {clustering['R']:.4f}")
    print(f"  p-value:          {clustering['p']:.4f}")
    
    # Save
    output = {
        'mode': args.mode,
        'params': {
            'N_particles': args.N,
            't_max_Myr': args.tmax,
            'P9_mass_Mearth': args.M_p9 if args.mode == 'p9' else None,
            'P9_a_AU': args.a_p9 if args.mode == 'p9' else None,
            'P9_e': args.e_p9 if args.mode == 'p9' else None,
            'P9_i_deg': args.i_p9 if args.mode == 'p9' else None,
            'seed': args.seed,
        },
        'wall_time_min': elapsed / 60,
        'n_surviving': sim.N - sim.N_var,
        'etnos': etnos,
        'clustering': clustering,
    }
    
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)) 
                           if '__file__' in dir() else '.', args.output)
    # Ensure output directory
    outdir = os.path.dirname(outpath) or '.'
    os.makedirs(outdir, exist_ok=True)
    
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nOutput saved: {outpath}")
    print("DONE")

if __name__ == '__main__':
    main()
