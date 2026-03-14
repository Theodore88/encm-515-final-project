'''
For the sake of SIMD benchmarking, we are going to experiment with the "predict" stage of an EKF. This is the predict
function defined in kalman_filter.py

Specifically, we are going to focus on the line: F @ P @ F.T + Q

This line focuses on updating the uncertainty of our new prediction. This is done by matrix multiplying the
current covariance matrix, the Jacobian of the state transition, the transpose of the Jacobian, and the process noise
covariance. Ultimately, this gives us uncertainty in position, velocity, and correlation between them.

This was selected because matrix multiplication has a time complexity of O(n^3), and as a result, is a very
expensive operation that dominates IMU resources. Using SIMD will allow us to accelerate matrix multiplications
by parallelizing dot product computations.

Note that in our original implementation, numpy matrix multiplication is used. This automatically implements
SIMD optimizations under the hood, so we are going to explicitly define an "unoptimized" version to compare against
the optimized version. The unoptimized version will use explicit loops instead of numpy's vectorized operations.
'''

import time
import numpy as np

from kalman_filter import DroneEKF, STATE_DIM
from dataflow_simulator import MultiStreamSimulator

def get_dt(duration=5.0, seed=42):
    # Use the same dt defined in the original simulation
    sim = MultiStreamSimulator(duration_s=duration, seed=seed)
    return sim.sim_dt

def generate_F_P_Q_dt(duration, seed):
    '''
    Generate a list of F, P, Q, dt for several independent predict EKF problems. 

    Recall that:
        - F is Jacobian
        - P is covariance
        - Q is process noise covariance
    '''
    rng = np.random.default_rng(seed)
    ekf = DroneEKF()

    dt = get_dt() * rng.uniform(0.5, 2.0) # Randomly jitter the timing to mimic real-world conditions

    F = ekf._jacobian_F(ekf.state.x.copy(), dt).copy()

    Q = ekf.Q.copy()

    random_matrix = rng.normal(size=(STATE_DIM, STATE_DIM))
    random_covariance_matrix = random_matrix @ random_matrix.T # Covariance has to be symmetric positive definite (so just multiple by transpose)
    P = random_covariance_matrix + ekf.state.P.copy() # Add the current covariance to make sure the new matrix is not too small

    return F, P, Q, dt

def compare_simd_and_scalar(duration, seed):
    F, P, Q, dt = generate_F_P_Q_dt(duration, seed)

    best_scalar_time = float("inf")
    best_simd_time = float("inf")
    out_scalar = None
    out_simd = None
    
    # Run each one three times because there's weird overheads on the system that runs this that are indepedent of the algorithm itself
    for i in range(3):
        start_time = time.perf_counter()
        out_scalar = covariance_predict_scalar(F, P, Q)
        finish_time = time.perf_counter()
        time_passed = finish_time - start_time
        best_scalar_time = min(best_scalar_time, time_passed)

    for j in range(3):
        start_time = time.perf_counter()
        out_simd = F @ P @ F.T + Q
        finish_time = time.perf_counter()
        time_passed = finish_time - start_time
        best_simd_time = min(best_simd_time, time_passed)

    max_abs_err = np.max(np.abs(out_scalar - out_simd))

    print("SIMD Speed-up for EKF Prediction Step")
    print(f"Scalar computation time: {best_scalar_time:.6f} s")
    print(f"Vectorized computation time: {best_simd_time:.6f} s")
    print(f"Speedup: {best_scalar_time / best_simd_time:.2f}x")
    print(f"Absolute difference between scalar and vectorized results: {max_abs_err:.12e}")

if __name__ == "__main__":
    duration = 5.0
    seed = 42
    compare_simd_and_scalar(duration, seed) # Same parameters as in original simulation
