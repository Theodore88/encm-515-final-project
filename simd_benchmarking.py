import time
import numpy as np

from kalman_filter import DroneEKF, STATE_DIM
from dataflow_simulator import MultiStreamSimulator
from simd.ekf_functions import covariance_predict_scalar, covariance_predict_simd

import argparse
import os
import matplotlib.pyplot as plt

iterations = 100000
repeats = 5

def get_dt(duration=5.0, seed=42):
    sim = MultiStreamSimulator(duration_s=duration, seed=seed)
    return sim.sim_dt

def generate_F_P_Q_dt(duration, seed, state_dim=STATE_DIM):
    '''
    Generate a list of F, P, Q, dt for several independent predict EKF problems. 
        - F is Jacobian
        - P is covariance
        - Q is process noise covariance
    '''
    rng = np.random.default_rng(seed)
    ekf = DroneEKF()

    dt = get_dt() * rng.uniform(0.5, 2.0) # Randomly jitter the timing to mimic real-world conditions

    base_F = ekf._jacobian_F(ekf.state.x.copy(), dt).copy()
    base_Q = ekf.Q.copy()

    F = np.eye(state_dim, dtype=np.float64)
    F[:STATE_DIM, :STATE_DIM] = base_F

    Q = np.zeros((state_dim, state_dim), dtype=np.float64)
    Q[:STATE_DIM, :STATE_DIM] = base_Q

    random_matrix = rng.normal(size=(state_dim, state_dim)) # Note that STATE_DIM is dependent on our sensor reading values - this is 9 but 
    random_covariance_matrix = random_matrix @ random_matrix.T # Covariance has to be symmetric positive definite (so just multiple by transpose)

    base_P = np.zeros((state_dim, state_dim), dtype=np.float64)
    base_P[:STATE_DIM, :STATE_DIM] = ekf.state.P.copy()

    P = random_covariance_matrix + base_P # Add the current covariance to make sure the new matrix is not too small

    return (
        np.ascontiguousarray(F, dtype=np.float64),
        np.ascontiguousarray(P, dtype=np.float64),
        np.ascontiguousarray(Q, dtype=np.float64),
        dt,
    )

def compare_simd_and_scalar(duration, seed, state_dim=STATE_DIM):
    F, P, Q, dt = generate_F_P_Q_dt(duration, seed, state_dim)

    current_iterations = 100

    scalar_times = []
    simd_times = []
    out_scalar = None
    out_simd = None

    for i in range(repeats): # Not optimized with SIMD
        start_time = time.perf_counter()
        for j in range(current_iterations):
            out_scalar = covariance_predict_scalar(F, P, Q)
        finish_time = time.perf_counter()
        scalar_times.append((finish_time - start_time) / current_iterations)

    for i in range(repeats): # Optimized with SIMD
        start_time = time.perf_counter()
        for j in range(current_iterations):
            out_simd = covariance_predict_simd(F, P, Q)
        finish_time = time.perf_counter()
        simd_times.append((finish_time - start_time) / current_iterations)

    best_scalar_time = min(scalar_times)
    best_simd_time = min(simd_times)

    max_abs_err_scalar_vs_simd = np.max(np.abs(out_scalar - out_simd))

    print(f"EKF Covariance Prediction Comparison (STATE_DIM={state_dim})")
    print(f"Cython scalar computation time: {best_scalar_time:.12f} s")
    print(f"Cython SIMD computation time:   {best_simd_time:.12f} s")
    print(f"Cython SIMD speedup over scalar:       {best_scalar_time / best_simd_time:.2f}x")
    print(f"Absolute difference scalar vs SIMD:  {max_abs_err_scalar_vs_simd:.12e}")

    return {
        "state_dim": state_dim,
        "scalar_time": best_scalar_time,
        "simd_time": best_simd_time,
        "simd_speedup": best_scalar_time / best_simd_time,
        "diff_scalar_simd": max_abs_err_scalar_vs_simd,
        "scalar_times": scalar_times,
        "simd_times": simd_times,
    }

def generate_simd_plots(results: dict):
    output_dir = os.path.join("simd", "output")
    os.makedirs(output_dir, exist_ok=True)

    scalar_time = results["scalar_time"]
    simd_time = results["simd_time"]

    simd_speedup = results["simd_speedup"]

    scalar_times = results["scalar_times"]
    simd_times = results["simd_times"]

    plt.figure()
    plt.bar(["Scalar", "SIMD"], [scalar_time, simd_time])
    plt.ylabel("Execution Time (s)")
    plt.title(f"SIMD Covariance Prediction Runtime (STATE_DIM={results['state_dim']})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"runtime_comparison_{results['state_dim']}.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(["Scalar", "SIMD"], [1.0, simd_speedup])
    plt.ylabel("Speedup vs Scalar")
    plt.title(f"SIMD Covariance Prediction Speedup (STATE_DIM={results['state_dim']})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"speedup_comparison_{results['state_dim']}.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.boxplot([scalar_times, simd_times], labels=["Scalar", "SIMD"])
    plt.ylabel("Execution Time (s)")
    plt.title(f"SIMD Covariance Prediction Runtime Distribution Across Trials (STATE_DIM={results['state_dim']})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"runtime_distribution_{results['state_dim']}.png"), dpi=200)
    plt.close()

    print(f"Saved plots to {output_dir}")

def generate_dimension_plots(all_results: list[dict]):
    output_dir = os.path.join("simd", "output")
    os.makedirs(output_dir, exist_ok=True)

    state_dims = [result["state_dim"] for result in all_results]
    scalar_times = [result["scalar_time"] for result in all_results]
    simd_times = [result["simd_time"] for result in all_results]
    simd_speedups = [result["simd_speedup"] for result in all_results]

    plt.figure()
    plt.plot(state_dims, scalar_times, marker="o", label="Scalar")
    plt.plot(state_dims, simd_times, marker="o", label="SIMD")
    plt.xlabel("State Dimension")
    plt.ylabel("Execution Time (s)")
    plt.title("Covariance Prediction Runtime vs State Dimension")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_vs_dimension.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.plot(state_dims, simd_speedups, marker="o")
    plt.xlabel("State Dimension")
    plt.ylabel("Speedup vs Scalar")
    plt.title("SIMD Speedup vs State Dimension")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_vs_dimension.png"), dpi=200)
    plt.close()

    plt.figure()
    n_cubed = [n**3 for n in state_dims]
    plt.plot(n_cubed, simd_speedups, marker="o")
    plt.xscale("log")
    plt.xlabel("N^3 (log scale)")
    plt.ylabel("Speedup vs Scalar")
    plt.title("SIMD Speedup vs Computational Work (N^3)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_vs_n_cubed.png"), dpi=200)
    plt.close()

    print(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Generate SIMD plots")
    parser.add_argument("--state-dim", type=int, default=None, help="Run a single state dimension")
    args = parser.parse_args()

    duration = 5.0
    seed = 42

    if args.state_dim is not None:
        results = compare_simd_and_scalar(duration, seed, args.state_dim)

        if args.visualize:
            generate_simd_plots(results)
    else:
        state_dims = [9, 24, 100]
        all_results = []

        for state_dim in state_dims:
            results = compare_simd_and_scalar(duration, seed, state_dim)
            all_results.append(results)

            if args.visualize:
                generate_simd_plots(results)

        if args.visualize:
            generate_dimension_plots(all_results)