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

def generate_F_P_Q_dt(duration, seed):
    '''
    Generate a list of F, P, Q, dt for several independent predict EKF problems. 
        - F is Jacobian
        - P is covariance
        - Q is process noise covariance
    '''
    rng = np.random.default_rng(seed)
    ekf = DroneEKF()

    dt = get_dt() * rng.uniform(0.5, 2.0) # Randomly jitter the timing to mimic real-world conditions

    F = ekf._jacobian_F(ekf.state.x.copy(), dt).copy()

    Q = ekf.Q.copy()

    random_matrix = rng.normal(size=(STATE_DIM, STATE_DIM)) # Note that STATE_DIM is dependent on our sensor reading values - this is 9 but 
    random_covariance_matrix = random_matrix @ random_matrix.T # Covariance has to be symmetric positive definite (so just multiple by transpose)
    P = random_covariance_matrix + ekf.state.P.copy() # Add the current covariance to make sure the new matrix is not too small

    return F, P, Q, dt

def compare_simd_and_scalar(duration, seed):
    F, P, Q, dt = generate_F_P_Q_dt(duration, seed)

    scalar_times = []
    simd_times = []
    out_scalar = None
    out_simd = None

    for i in range(repeats): # Not optimized with SIMD
        start_time = time.perf_counter()
        for j in range(iterations):
            out_scalar = covariance_predict_scalar(F, P, Q)
        finish_time = time.perf_counter()
        scalar_times.append((finish_time - start_time) / iterations)

    for i in range(repeats): # Optimized with SIMD
        start_time = time.perf_counter()
        for j in range(iterations):
            out_simd = covariance_predict_simd(F, P, Q)
        finish_time = time.perf_counter()
        simd_times.append((finish_time - start_time) / iterations)

    best_scalar_time = min(scalar_times)
    best_simd_time = min(simd_times)

    max_abs_err_scalar_vs_simd = np.max(np.abs(out_scalar - out_simd))

    print("EKF Covariance Prediction Comparison")
    print(f"Cython scalar computation time: {best_scalar_time:.12f} s")
    print(f"Cython SIMD computation time:   {best_simd_time:.12f} s")
    print(f"Cython SIMD speedup over scalar:       {best_scalar_time / best_simd_time:.2f}x")
    print(f"Absolute difference scalar vs SIMD:  {max_abs_err_scalar_vs_simd:.12e}")

    return {
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
    plt.title("SIMD Covariance Prediction Runtime")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_comparison.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.bar(["Scalar", "SIMD"], [1.0, simd_speedup])
    plt.ylabel("Speedup vs Scalar")
    plt.title("SIMD Covariance Prediction Speedup")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "speedup_comparison.png"), dpi=200)
    plt.close()

    plt.figure()
    plt.boxplot([scalar_times, simd_times], labels=["Scalar", "SIMD"])
    plt.ylabel("Execution Time (s)")
    plt.title("SIMD Covariance Prediction Runtime Distribution Across Trials")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "runtime_distribution.png"), dpi=200)
    plt.close()

    print(f"Saved plots to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="Generate SIMD plots")
    args = parser.parse_args()

    duration = 5.0
    seed = 42
    results = compare_simd_and_scalar(duration, seed)

    if args.visualize:
        generate_simd_plots(results)