
import time
import numpy as np

from dataflow_simulator import MultiStreamSimulator
from pipeline_analysis   import analyse


def main():

    # Run simulation 
    base_sim    = MultiStreamSimulator(duration_s=10, seed=42, quantized=False, cython=False)

    t0 = time.time() 
    base_result = base_sim.run(verbose=False)
    t1 = time.time()
    base_time = t1 - t0
    base_result = analyse(base_result)
    print(f"\nBase simulation completed in {base_time:.2f} seconds.")
    print(f"Mean Position Error (Base): {base_result['accuracy']['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (Base): {base_result['accuracy']['vel_mean_err_ms']:.6f}")


    quant_sim = MultiStreamSimulator(duration_s=10, seed=42, quantized=True, cython=True)
    quant_sim.qh.X_SCALE = 20 # state values
    quant_sim.qh.P_SCALE = 24  # covariance values (larger dynamic range
    quant_sim.qh.H_SCALE = 20 # measurement matrix values (usually 1, but we use same unit)
    quant_sim.qh.Z_SCALE = 20  # measurement values

    quant_sim.qh.BITS = 64
    t0 = time.time()    
    bits64_result = quant_sim.run(verbose=False)
    t1 = time.time()

    bits64_time = t1 - t0
    bits64_result = analyse(bits64_result)
    print(f"\nQuantized simulation (64-bit) completed in {bits64_time:.2f} seconds.")
    print(f"Mean Position Error (64-bit): {bits64_result["accuracy"]['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (64-bit): {bits64_result["accuracy"]['vel_mean_err_ms']:.6f}")

    # 48 Bits
    quant_sim.qh.BITS = 48

    t0 = time.time()    
    bits48_result = quant_sim.run(verbose=False)
    t1 = time.time()
    bits48_result = analyse(bits48_result)

    bits48_time = t1 - t0

    print(f"\nQuantized simulation (48-bit) completed in {bits48_time:.2f} seconds.")
    print(f"Mean Position Error (48-bit): {bits48_result["accuracy"]['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (48-bit): {bits48_result["accuracy"]['vel_mean_err_ms']:.6f}")


    # 32 Bits
    quant_sim.qh.BITS = 32

    t0 = time.time()    
    bits32_result = quant_sim.run(verbose=False)
    t1 = time.time()
    bits32_result = analyse(bits32_result)

    time_result = t1 - t0

    print(f"\nQuantized simulation (32-bit) completed in {time_result:.2f} seconds.")
    print(f"Mean Position Error (32-bit): {bits32_result["accuracy"]['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (32-bit): {bits32_result["accuracy"]['vel_mean_err_ms']:.6f}")


    # 16 Bits
    quant_sim.qh.BITS = 16

    t0 = time.time()    
    bits16_result = quant_sim.run(verbose=False)
    t1 = time.time()
    bits16_result = analyse(bits16_result)

    time_result = t1 - t0

    print(f"\nQuantized simulation (16-bit) completed in {time_result:.2f} seconds.")
    print(f"Mean Position Error (16-bit): {bits16_result["accuracy"]['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (16-bit): {bits16_result["accuracy"]['vel_mean_err_ms']:.6f}")

    # Scale config 1
    quant_sim.qh.BITS = 64
    quant_sim.qh.X_SCALE = 8 # state values
    quant_sim.qh.P_SCALE = 12  # covariance values (larger dynamic range
    quant_sim.qh.H_SCALE = 8 # measurement matrix values (usually 1, but we use same unit)
    quant_sim.qh.Z_SCALE = 8  # measurement values

    t0 = time.time()    
    result = quant_sim.run(verbose=False)
    t1 = time.time()
    analyse_result = analyse(result)

    time_result = t1 - t0

    print(f"\nQuantized simulation (Scale Config 1) completed in {time_result:.2f} seconds.")
    print(f"Mean Position Error (Scale Config 1): {analyse_result["accuracy"]['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (Scale Config 1): {analyse_result["accuracy"]['vel_mean_err_ms']:.6f}")


    # Scale config 2
    quant_sim.qh.X_SCALE = 10 # state values
    quant_sim.qh.P_SCALE = 14  # covariance values (larger dynamic range
    quant_sim.qh.H_SCALE = 10 # measurement matrix values (usually 1, but we use same unit)
    quant_sim.qh.Z_SCALE = 10  # measurement values

    t0 = time.time()    
    result = quant_sim.run(verbose=False)
    t1 = time.time()
    analyse_result = analyse(result)

    time_result = t1 - t0

    print(f"\nQuantized simulation (Scale Config 2) completed in {time_result:.2f} seconds.")
    print(f"Mean Position Error (Scale Config 2): {analyse_result["accuracy"]['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (Scale Config 2): {analyse_result["accuracy"]['vel_mean_err_ms']:.6f}")


    # Scale config 3
    quant_sim.qh.X_SCALE = 12 # state values
    quant_sim.qh.P_SCALE = 16  # covariance values (larger dynamic range
    quant_sim.qh.H_SCALE = 12 # measurement matrix values (usually 1, but we use same unit)
    quant_sim.qh.Z_SCALE = 12  # measurement values

    t0 = time.time()    
    result = quant_sim.run(verbose=False)
    t1 = time.time()
    analyse_result = analyse(result)

    time_result = t1 - t0

    print(f"\nQuantized simulation (Scale Config 3) completed in {time_result:.2f} seconds.")
    print(f"Mean Position Error (Scale Config 3): {analyse_result["accuracy"]['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (Scale Config 3): {analyse_result["accuracy"]['vel_mean_err_ms']:.6f}")

    # Scale config 4
    quant_sim.qh.X_SCALE = 14 # state values
    quant_sim.qh.P_SCALE = 18  # covariance values (larger dynamic range
    quant_sim.qh.H_SCALE = 14 # measurement matrix values (usually 1, but we use same unit)
    quant_sim.qh.Z_SCALE = 14  # measurement values

    t0 = time.time()    
    result = quant_sim.run(verbose=False)
    t1 = time.time()
    analyse_result = analyse(result)

    time_result = t1 - t0

    print(f"\nQuantized simulation (Scale Config 4) completed in {time_result:.2f} seconds.")
    print(f"Mean Position Error (Scale Config 4): {analyse_result["accuracy"]['pos_mean_err_m']:.6f}")
    print(f"Mean Velocity Error (Scale Config 4): {analyse_result["accuracy"]['vel_mean_err_ms']:.6f}")


# X_SCALE = 10 # state values
# P_SCALE = 14  # covariance values (larger dynamic range)
# H_SCALE = 10 # measurement matrix values (usually 1, but we use same unit)
# Z_SCALE = 10  # measurement values


if __name__ == "__main__":
    main()
