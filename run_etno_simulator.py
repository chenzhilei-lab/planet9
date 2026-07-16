#!/usr/bin/env python3
"""
run_etno_simulator.py — CLI entry point for the ETNO survey simulator.

Usage:
    python run_etno_simulator.py --n-boot 50000 --n-fpr 5000
    python run_etno_simulator.py --seed 20260713 --output results.txt
"""

import argparse, sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from etno_simulator import SurveySimulator

def main():
    parser = argparse.ArgumentParser(description="ETNO Survey Simulator")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--n-boot", type=int, default=50000)
    parser.add_argument("--n-fpr", type=int, default=5000)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--data", type=str,
                        default=os.path.join(SCRIPT_DIR, "etno_complete.json"))
    args = parser.parse_args()

    sim = SurveySimulator(args.data, seed=args.seed)
    print(f"Loaded {sim.N} ETNOs from {args.data}")
    print(f"R_obs = {sim.R_obs:.4f}  |  circ_mean = {sim.circ_mean_deg:.1f} deg")
    print()

    results = sim.run_all(n_boot=args.n_boot, n_fpr=args.n_fpr)

    print("--- Core p-values (N=19) ---")
    for k, v in results["core_pvalues"].items():
        print(f"  {k}: p = {v}")
    print(f"\nFPR = {results['fpr_pct']:.1f}%  ({results['fpr_count']}/{results['fpr_trials']})")

    if args.output:
        with open(args.output, "w") as f:
            f.write(sim.summary())
        print(f"\nSaved -> {args.output}")

if __name__ == "__main__":
    main()
