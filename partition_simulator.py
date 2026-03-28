"""
partition_simulator.py
Models CPU/accelerator workload partitioning for the drone EKF pipeline.
Bus transfer costs follow Aung et al. (2013) §V: 10 cycles/write, 18 cycles/read.
Data sizes derived from STATE_DIM = 9 (kalman_filter.py).
"""

import numpy as np
import matplotlib.pyplot as plt

STATE_DIM    = 9   # from kalman_filter.py
GPS_MEAS_DIM = 3   # from kalman_filter.py update methods

CYCLES_PER_WRITE = 10   # Aung et all paper linked in the report
CYCLES_PER_READ  = 18

# stage compute costs in µs, split from TICKS_EXECUTE in dataflow_simulator.py
# proportions based on Aung et al. paper (cov_predict dominates at ~51%)
STAGE_COMPUTE_US = {
    "state_predict": 118 * 0.08,
    "jacobian":      118 * 0.17,
    "cov_predict":   118 * 0.51,
    "kalman_gain":    58 * 1.00,
    "meas_update":    58 * 0.14,
    "cov_correct":    58 * 0.29,
}

# speedup factors when a stage runs on the accelerator vs CPU.
# cov_predict is measured from Cython SIMD benchmarking in simd_benchmarking.py, which was 1.18x
# kalman_gain is taken from Aung et al. paper linked in the report for a comparable EKF FPGA implementation which was 5.7x.
# remaining matrix stages use the measured 1.18x as a conservative estimate.
ACCEL_SPEEDUP = {
    "state_predict": 1.1,
    "jacobian":      1.18,
    "cov_predict":   1.18,
    "kalman_gain":   5.7,
    "meas_update":   1.1,
    "cov_correct":   1.18,
}

STAGE_ORDER = ["state_predict", "jacobian", "cov_predict", "kalman_gain", "meas_update", "cov_correct"]

# word counts transferred at each stage boundary 
BOUNDARY_WORDS = {
    ("state_predict", "jacobian"):    STATE_DIM,
    ("cov_predict",   "kalman_gain"): STATE_DIM * STATE_DIM,
    ("kalman_gain",   "meas_update"): STATE_DIM * GPS_MEAS_DIM,
    ("meas_update",   "cov_correct"): STATE_DIM * GPS_MEAS_DIM + STATE_DIM * STATE_DIM,
    ("cov_correct",   "state_predict"): STATE_DIM * STATE_DIM,
}

SCHEMES = {
    "All-CPU":                dict(state_predict="CPU",  jacobian="CPU",   cov_predict="CPU",   kalman_gain="CPU",   meas_update="CPU",  cov_correct="CPU"),
    "Kalman-Gain-Only":       dict(state_predict="CPU",  jacobian="CPU",   cov_predict="CPU",   kalman_gain="ACCEL", meas_update="CPU",  cov_correct="CPU"),
    "Heavy-Matrix-On-ACCEL":  dict(state_predict="CPU",  jacobian="ACCEL", cov_predict="ACCEL", kalman_gain="ACCEL", meas_update="CPU",  cov_correct="ACCEL"),
    "Predict-Block-On-ACCEL": dict(state_predict="CPU",  jacobian="ACCEL", cov_predict="ACCEL", kalman_gain="CPU",   meas_update="CPU",  cov_correct="CPU"),
}

SCHEME_COLORS = {
    "All-CPU":                "blue",
    "Kalman-Gain-Only":       "green",
    "Heavy-Matrix-On-ACCEL":  "red",
    "Predict-Block-On-ACCEL": "yellow",
}


def compute_transfer_us(scheme, bus_mhz):
    total = 0.0
    for i in range(len(STAGE_ORDER) - 1):
        curr = STAGE_ORDER[i]
        nxt  = STAGE_ORDER[i + 1]
        if scheme[curr] != scheme[nxt]:
            n_words = BOUNDARY_WORDS.get((curr, nxt), 0)
            if scheme[curr] == "CPU":
                total += n_words * CYCLES_PER_WRITE / bus_mhz
            else:
                total += n_words * CYCLES_PER_READ / bus_mhz
    return total


def compute_us(scheme):
    total = 0.0
    for s in STAGE_ORDER:
        if scheme[s] == "ACCEL":
            total += STAGE_COMPUTE_US[s] / ACCEL_SPEEDUP[s]
        else:
            total += STAGE_COMPUTE_US[s]
    return total


def total_latency_us(scheme, bus_mhz):
    return compute_us(scheme) + compute_transfer_us(scheme, bus_mhz)


def n_crossings(scheme):
    count = 0
    for i in range(len(STAGE_ORDER) - 1):
        if scheme[STAGE_ORDER[i]] != scheme[STAGE_ORDER[i + 1]]:
            count += 1
    return count


def print_summary(bus_mhz=100.0):
    baseline = total_latency_us(SCHEMES["All-CPU"], bus_mhz)
    print(f"\nPartition Analysis @ {bus_mhz} MHz bus")
    print(f"{'Scheme':<28} {'Compute':>9} {'Transfer':>10} {'Total':>8} {'Speedup':>9} {'Crossings':>10}")
    print("-" * 76)
    for name, scheme in SCHEMES.items():
        comp     = compute_us(scheme)
        xfer     = compute_transfer_us(scheme, bus_mhz)
        total    = comp + xfer
        speedup  = baseline / total
        crossings = n_crossings(scheme)
        print(f"  {name:<26} {comp:.1f}us   {xfer:.1f}us   {total:.1f}us   {speedup:.2f}x   {crossings}")

    print(f"\nBaseline (All-CPU): {baseline:.1f} us")
    print("\nBoundary crossing detail:")
    for name, scheme in SCHEMES.items():
        lines = []
        for i in range(len(STAGE_ORDER) - 1):
            curr = STAGE_ORDER[i]
            nxt  = STAGE_ORDER[i + 1]
            if scheme[curr] != scheme[nxt]:
                n_words   = BOUNDARY_WORDS.get((curr, nxt), 0)
                direction = scheme[curr] + "->" + scheme[nxt]
                if scheme[curr] == "CPU":
                    t = n_words * CYCLES_PER_WRITE / bus_mhz
                else:
                    t = n_words * CYCLES_PER_READ / bus_mhz
                lines.append(f"    {direction}  {n_words} words  ->  {t:.2f} us  ({curr} -> {nxt})")
        if lines:
            print(f"\n  [{name}]")
            for l in lines:
                print(l)


def plot_latency_vs_bus_freq(bus_freqs):
    fig, ax = plt.subplots()
    for name, scheme in SCHEMES.items():
        ax.plot(bus_freqs, [total_latency_us(scheme, f) for f in bus_freqs], label=name, color=SCHEME_COLORS[name])
    ax.set_xlabel("Bus Frequency (MHz)")
    ax.set_ylabel("Total Latency (us)")
    ax.set_title("EKF Partition Latency vs. Bus Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stacked_breakdown(bus_mhz=100.0):
    names     = list(SCHEMES.keys())
    computes  = [compute_us(s)                   for s in SCHEMES.values()]
    transfers = [compute_transfer_us(s, bus_mhz) for s in SCHEMES.values()]
    x = np.arange(len(names))
    fig, ax = plt.subplots()
    ax.bar(x, computes,  label="Compute",           color="steelblue", alpha=0.85)
    ax.bar(x, transfers, bottom=computes, label="Transfer overhead", color="tomato", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Latency (us)")
    ax.set_title(f"Compute vs. Transfer Overhead (bus = {bus_mhz} MHz)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_optimal_partition_map(bus_freqs):
    scheme_names  = list(SCHEMES.keys())
    all_latencies = np.array([[total_latency_us(s, f) for f in bus_freqs] for s in SCHEMES.values()])
    optimal_idx   = np.argmin(all_latencies, axis=0)
    fig, ax = plt.subplots()
    prev_idx, start = optimal_idx[0], bus_freqs[0]
    for i, (f, idx) in enumerate(zip(bus_freqs, optimal_idx)):
        if idx != prev_idx or i == len(bus_freqs) - 1:
            ax.barh(0, f - start, left=start, height=0.5, color=SCHEME_COLORS[scheme_names[prev_idx]], alpha=0.85)
            prev_idx, start = idx, f
    seen    = {scheme_names[i] for i in optimal_idx}
    patches = [plt.Rectangle((0,0),1,1, color=SCHEME_COLORS[n], label=n) for n in scheme_names if n in seen]
    ax.legend(handles=patches)
    ax.set_xlabel("Bus Frequency (MHz)")
    ax.set_yticks([])
    ax.set_title("Optimal Partition Scheme by Bus Frequency")
    ax.set_xlim(bus_freqs[0], bus_freqs[-1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    BUS_FREQS      = np.linspace(20, 250, 300)
    REPORT_BUS_MHZ = 100.0

    print_summary(bus_mhz=REPORT_BUS_MHZ)
    plot_latency_vs_bus_freq(BUS_FREQS)
    plot_stacked_breakdown(REPORT_BUS_MHZ)
    plot_optimal_partition_map(BUS_FREQS)