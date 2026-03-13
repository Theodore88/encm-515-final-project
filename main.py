"""
Entry point for the Drone Sensor Fusion Accelerator Design Simulation.

Runs:
  1. Multi-stream sensor simulation (5 seconds of flight)
  2. EKF-based sensor fusion
  3. Pipeline characteristic analysis
  4. Datapath visualisation (8 PNG plots)
  5. Console report
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from dataflow_simulator import MultiStreamSimulator
from pipeline_analysis   import analyse, print_report
from visualisation       import generate_all_plots


def main():
    parser = argparse.ArgumentParser(
        description="Drone Sensor Fusion Pipeline Simulation")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Simulation duration in seconds (default: 5)")
    parser.add_argument("--seed",     type=int,   default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--outdir",   type=str,   default="./output",
                        help="Output directory for plots (default: ./output)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    args = parser.parse_args()

    print("=" * 60)
    print("  Drone Sensor Fusion — Accelerator Design Simulation")
    print("=" * 60)
    print(f"  Duration : {args.duration} s")
    print(f"  Seed     : {args.seed}")
    print(f"  Output   : {args.outdir}")
    print()

    # Run simulation 
    sim    = MultiStreamSimulator(duration_s=args.duration, seed=args.seed)
    result = sim.run(verbose=True)

    # Analyse pipeline 
    report = analyse(result)
    print_report(report)

    #  Visualise 
    if not args.no_plots:
        generate_all_plots(result, output_dir=args.outdir)
    else:
        print("[Visualisation] Skipped (--no-plots).")

    print("\nDone.")


if __name__ == "__main__":
    main()
