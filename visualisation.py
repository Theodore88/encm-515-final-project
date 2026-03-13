"""
visualisation.py
----------------
Generates datapath illustrations and analysis plots.
All figures are saved to disk (no interactive UI).

Plots produced:
  1. Sensor data streams (raw vs estimated)
  2. 3D trajectory: ground truth vs EKF estimate
  3. Position and velocity error over time
  4. Pipeline timing diagram (Gantt-style)
  5. Hazard distribution by sensor
  6. Latency distributions (violin/box)
  7. Throughput vs latency scatter
  8. Datapath block diagram (schematic)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

from dataflow_simulator import SimulationResult, TICK_COST, TICKS_EXECUTE, TICKS_WRITEBACK
from kalman_filter import EXPECTED_RATES_MAP
from pipeline_analysis import analyse


# Colour palette
COLOURS = {
    "IMU":    "#E63946",
    "GPS":    "#457B9D",
    "BARO":   "#2A9D8F",
    "OPFLOW": "#E9C46A",
    "MAG":    "#A8DADC",
    "truth":  "#222222",
    "ekf":    "#F4722B",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _savefig(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 1. Sensor stream overview
# ---------------------------------------------------------------------------

def plot_sensor_streams(result: SimulationResult, out: str):
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("Multi-Stream Sensor Data — Raw Readings", fontsize=13, y=1.01)

    times_truth = np.arange(len(result.truth_trajectory)) * result.sim_dt

    # IMU acceleration magnitude
    ax = axes[0]
    imu_mags = [np.linalg.norm(r.data[:3]) for r in result.sensor_readings["IMU"]]
    imu_t    = [r.timestamp for r in result.sensor_readings["IMU"]]
    ax.plot(imu_t, imu_mags, color=COLOURS["IMU"], lw=0.6, alpha=0.8)
    truth_acc = [np.linalg.norm(tr.acceleration) for tr in result.truth_trajectory]
    ax.plot(times_truth, truth_acc, color=COLOURS["truth"], lw=1.2, ls="--",
            label="Truth |a|")
    ax.set_ylabel("|Accel| (m/s²)")
    ax.set_title(f"IMU  ({EXPECTED_RATES_MAP['IMU']} Hz)")
    ax.legend(fontsize=8)

    # GPS position x, y, z
    ax = axes[1]
    for idx, label, c in zip([0,1,2], ["x","y","z"], ["#c00","#090","#009"]):
        gps_vals = [r.data[idx] for r in result.sensor_readings["GPS"]]
        gps_t    = [r.timestamp for r in result.sensor_readings["GPS"]]
        ax.plot(gps_t, gps_vals, lw=1.0, color=c, label=f"GPS {label}")
    truth_pos_x = [tr.position[0] for tr in result.truth_trajectory]
    ax.plot(times_truth, truth_pos_x, color=COLOURS["truth"], lw=1.0, ls="--",
            label="Truth x")
    ax.set_ylabel("Position (m)")
    ax.set_title(f"GPS  ({EXPECTED_RATES_MAP['GPS']} Hz)")
    ax.legend(fontsize=7, ncol=4)

    # Barometer altitude
    ax = axes[2]
    baro_t   = [r.timestamp for r in result.sensor_readings["BARO"]]
    baro_alt = [r.data[0]   for r in result.sensor_readings["BARO"]]
    truth_alt = [tr.position[2] for tr in result.truth_trajectory]
    ax.plot(baro_t, baro_alt, color=COLOURS["BARO"], lw=0.8, label="Baro alt")
    ax.plot(times_truth, truth_alt, color=COLOURS["truth"], lw=1.2, ls="--",
            label="Truth z")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Barometer  ({EXPECTED_RATES_MAP['BARO']} Hz)")
    ax.legend(fontsize=8)

    # Optical flow vx, vy
    ax = axes[3]
    of_t  = [r.timestamp for r in result.sensor_readings["OPFLOW"]]
    of_vx = [r.data[0]   for r in result.sensor_readings["OPFLOW"]]
    of_vy = [r.data[1]   for r in result.sensor_readings["OPFLOW"]]
    truth_vx = [tr.velocity[0] for tr in result.truth_trajectory]
    ax.plot(of_t, of_vx, color=COLOURS["OPFLOW"], lw=0.7, label="OpFlow vx")
    ax.plot(of_t, of_vy, color="#b8860b", lw=0.7, label="OpFlow vy")
    ax.plot(times_truth, truth_vx, color=COLOURS["truth"], lw=1.0, ls="--",
            label="Truth vx")
    ax.set_ylabel("Vel (m/s)")
    ax.set_title(f"Optical Flow  ({EXPECTED_RATES_MAP['OPFLOW']} Hz)")
    ax.legend(fontsize=7, ncol=3)

    # Magnetometer magnitude
    ax = axes[4]
    mag_t    = [r.timestamp for r in result.sensor_readings["MAG"]]
    mag_mags = [np.linalg.norm(r.data) for r in result.sensor_readings["MAG"]]
    ax.plot(mag_t, mag_mags, color=COLOURS["MAG"], lw=0.8)
    ax.set_ylabel("|B| (norm)")
    ax.set_title(f"Magnetometer  ({EXPECTED_RATES_MAP['MAG']} Hz)")
    ax.set_xlabel("Time (s)")

    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 2. 3D trajectory
# ---------------------------------------------------------------------------

def plot_trajectory_3d(result: SimulationResult, out: str):
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_title("Ground Truth vs EKF Estimated Trajectory")

    tx = [tr.position[0] for tr in result.truth_trajectory]
    ty = [tr.position[1] for tr in result.truth_trajectory]
    tz = [tr.position[2] for tr in result.truth_trajectory]
    ax.plot(tx, ty, tz, color=COLOURS["truth"], lw=2, label="Ground Truth", zorder=3)

    ex = [s.x[0] for s in result.estimated_trajectory]
    ey = [s.x[1] for s in result.estimated_trajectory]
    ez = [s.x[2] for s in result.estimated_trajectory]
    ax.plot(ex, ey, ez, color=COLOURS["ekf"], lw=1.5, ls="--",
            label="EKF Estimate", zorder=2, alpha=0.85)

    # GPS sparse scatter
    gps_x = [r.data[0] for r in result.sensor_readings["GPS"]]
    gps_y = [r.data[1] for r in result.sensor_readings["GPS"]]
    gps_z = [r.data[2] for r in result.sensor_readings["GPS"]]
    ax.scatter(gps_x, gps_y, gps_z, color=COLOURS["GPS"], s=15,
               label="GPS readings", alpha=0.6, zorder=4)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(fontsize=9)
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 3. Position & velocity error over time
# ---------------------------------------------------------------------------

def plot_estimation_error(result: SimulationResult, out: str):
    times = np.arange(len(result.position_errors)) * result.sim_dt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle("EKF Estimation Error Over Time")

    ax1.plot(times, result.position_errors, color="#E63946", lw=0.8)
    ax1.set_ylabel("Position Error (m)")
    ax1.axhline(np.mean(result.position_errors), color="k", ls="--", lw=1,
                label=f"Mean = {np.mean(result.position_errors):.3f} m")
    ax1.legend(fontsize=9)
    ax1.set_title("Position Error |est − truth|")

    ax2.plot(times, result.velocity_errors, color="#457B9D", lw=0.8)
    ax2.set_ylabel("Velocity Error (m/s)")
    ax2.axhline(np.mean(result.velocity_errors), color="k", ls="--", lw=1,
                label=f"Mean = {np.mean(result.velocity_errors):.3f} m/s")
    ax2.legend(fontsize=9)
    ax2.set_title("Velocity Error |est − truth|")
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 4. Pipeline timing Gantt diagram (first 0.05 s)
# ---------------------------------------------------------------------------

def plot_pipeline_gantt(result: SimulationResult, out: str):
    """
    Show the first 50 ms of pipeline scheduling as a Gantt chart.
    Each bar spans [tick_execute, tick_done] for that packet.
    Hazard annotations mark which hazard type caused a stall.
    """
    SHOW_US = 50_000   # first 50 ms in ticks (1 tick = 1 µs)

    sensors_order = ["IMU", "GPS", "BARO", "OPFLOW", "MAG"]
    y_pos = {s: i for i, s in enumerate(sensors_order)}
    bar_h = 0.6

    visible = [p for p in result.packets if p.tick_fetch < SHOW_US]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_title(f"Pipeline Timing Diagram — Fusion Core (first 50 ms)")

    for pkt in visible:
        y      = y_pos.get(pkt.sensor_id, 0)
        colour = COLOURS[pkt.sensor_id]

        # FETCH+DECODE bar (light shade)
        fd_dur = pkt.tick_execute - pkt.tick_fetch
        if fd_dur > 0:
            ax.barh(y, fd_dur, left=pkt.tick_fetch, height=bar_h,
                    color=colour, alpha=0.25, edgecolor="none")

        # EXECUTE bar (full colour)
        exe_dur = TICKS_EXECUTE[pkt.sensor_id]
        ax.barh(y, exe_dur, left=pkt.tick_execute, height=bar_h,
                color=colour, edgecolor="k", linewidth=0.4, alpha=0.9)

        # WRITEBACK bar (dark)
        ax.barh(y, TICKS_WRITEBACK, left=pkt.tick_writeback, height=bar_h,
                color=colour, edgecolor="k", linewidth=0.4, alpha=0.5)

        # Hazard annotation
        if pkt.structural_hazard or pkt.raw_hazard:
            labels = []
            if pkt.structural_hazard: labels.append("S")
            if pkt.raw_hazard:        labels.append("R")
            tag = "/".join(labels)
            ax.annotate(tag,
                        xy=(pkt.tick_execute, y + bar_h/2),
                        xytext=(2, 0), textcoords="offset points",
                        fontsize=6, color="red", va="center", fontweight="bold")

    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels(list(y_pos.keys()))
    ax.set_xlabel("Time (µs)")
    ax.set_xlim(0, SHOW_US)
    ax.grid(axis="x", ls=":", alpha=0.4)

    sensor_patches = [mpatches.Patch(color=COLOURS[s], label=s)
                      for s in sensors_order]
    stage_patches  = [
        mpatches.Patch(color="grey", alpha=0.25, label="FETCH+DECODE"),
        mpatches.Patch(color="grey", alpha=0.9,  label="EXECUTE"),
        mpatches.Patch(color="grey", alpha=0.5,  label="WRITEBACK"),
        mpatches.Patch(color="white", edgecolor="red",
                       label="S=Structural  R=RAW"),
    ]
    ax.legend(handles=sensor_patches + stage_patches,
              fontsize=7, loc="upper right", ncol=3)

    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 5. Hazard distribution
# ---------------------------------------------------------------------------

def plot_hazard_distribution(result: SimulationResult, out: str):
    if not result.packets:
        print("  No packets to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Pipeline Hazard Analysis — Structural, RAW", fontsize=13)

    hazard_types = [
        ("structural_hazard", "Structural\n(EXECUTE contention)", "#E63946"),
        ("raw_hazard",        "RAW\n(stale P read)",              "#457B9D"),
    ]

    for ax, (attr, title, colour) in zip(axes, hazard_types):
        counts = defaultdict(int)
        stalls = defaultdict(list)
        for pkt in result.packets:
            if getattr(pkt, attr):
                counts[pkt.sensor_id] += 1
                stalls[pkt.sensor_id].append(pkt.stall_ticks)

        if not counts:
            ax.text(0.5, 0.5, "No hazards\ndetected",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="#888")
            ax.set_title(title)
            ax.axis("off")
            continue

        sids  = list(counts.keys())
        cnts  = [counts[s] for s in sids]
        bars  = ax.bar(sids, cnts, color=[COLOURS.get(s, "#888") for s in sids],
                       edgecolor="k", linewidth=0.6)
        for bar, cnt in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Sensor")
        ax.set_ylabel("Hazard count")
        ax.grid(axis="y", ls=":", alpha=0.4)

    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 6. Latency distributions
# ---------------------------------------------------------------------------

def plot_latency_distributions(result: SimulationResult, out: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Update Latency Distributions by Sensor Stream")

    data   = []
    labels = []
    colours_list = []
    for sid in EXPECTED_RATES_MAP:
        lats = result.pipeline_metrics[sid].latencies_us
        if lats:
            data.append(lats)
            labels.append(f"{sid}\n({EXPECTED_RATES_MAP[sid]} Hz)")
            colours_list.append(COLOURS[sid])

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", lw=2))
    for patch, c in zip(bp["boxes"], colours_list):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (µs)")
    ax.set_yscale("log")
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_xlabel("Sensor Stream")

    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 7. Throughput vs latency scatter
# ---------------------------------------------------------------------------

def plot_latency_throughput(result: SimulationResult, report: dict, out: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Latency vs Throughput Trade-off per Sensor")

    for sid, d in report["pipeline"].items():
        if "lat_mean_us" not in d:
            continue
        ax.scatter(d["effective_throughput_hz"], d["lat_mean_us"],
                   s=120, color=COLOURS[sid], zorder=3,
                   edgecolors="k", linewidths=0.6)
        ax.annotate(sid,
                    xy=(d["effective_throughput_hz"], d["lat_mean_us"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=9)
        # Error bar for p99
        ax.errorbar(d["effective_throughput_hz"], d["lat_mean_us"],
                    yerr=[[d["lat_mean_us"] - 0],
                          [d["lat_p99_us"] - d["lat_mean_us"]]],
                    fmt="none", color=COLOURS[sid], lw=1.5, alpha=0.7)

    ax.set_xlabel("Effective Throughput (Hz)")
    ax.set_ylabel("Mean Update Latency (µs)  [error bar = p99]")
    ax.set_yscale("log")
    ax.grid(ls=":", alpha=0.4)

    # Annotate ideal vs achieved
    ax.axhline(1e3, color="gray", ls="--", lw=0.8, label="1 ms target")
    ax.legend(fontsize=8)

    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 8. Datapath block diagram
# ---------------------------------------------------------------------------

def plot_datapath_diagram(out: str):
    """
    Schematic illustration of the multi-stream sensor fusion datapath.
    Drawn purely with matplotlib patches — no external diagram library needed.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14); ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Multi-Stream Sensor Fusion Datapath", fontsize=14, pad=12)

    def box(x, y, w, h, label, colour, fontsize=9, sub=""):
        rect = mpatches.FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0.1",
                                       facecolor=colour, edgecolor="#333",
                                       linewidth=1.5, zorder=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.18 if sub else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", zorder=3)
        if sub:
            ax.text(x + w/2, y + h/2 - 0.22, sub,
                    ha="center", va="center", fontsize=7, color="#555", zorder=3)

    def arrow(x1, y1, x2, y2, label="", colour="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=colour, lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.05, my+0.12, label, fontsize=7, color=colour)

    # Sensor boxes (left column)
    sensors = [
        ("IMU",    0.4, 6.8, COLOURS["IMU"],    "400 Hz"),
        ("GPS",    0.4, 5.5, COLOURS["GPS"],    "10 Hz"),
        ("BARO",   0.4, 4.2, COLOURS["BARO"],   "50 Hz"),
        ("OPFLOW", 0.4, 2.9, COLOURS["OPFLOW"], "100 Hz"),
        ("MAG",    0.4, 1.6, COLOURS["MAG"],    "100 Hz"),
    ]
    for sid, x, y, c, rate in sensors:
        box(x, y, 2.0, 0.85, sid, c, sub=rate)

    # Input buffers (small queue)
    for _, x, y, c, _ in sensors:
        box(x+2.3, y+0.1, 1.0, 0.65, "FIFO", "#dde", fontsize=7)
        arrow(x+2.0, y+0.42, x+2.3, y+0.42, colour="#555")

    # Scheduler
    box(4.0, 3.4, 2.2, 1.4, "SCHEDULER", "#f0e0c0", fontsize=10,
        sub="Hazard detection\n& arbitration")
    for _, x, y, c, _ in sensors:
        arrow(x+3.3, y+0.42, 4.0, 4.1, colour="#999")

    # Fusion core (EKF)
    box(7.0, 3.2, 2.6, 1.8, "EKF FUSION\nCORE", "#d0e8d0", fontsize=11,
        sub="Predict / Update\n9-DOF state")
    arrow(6.2, 4.1, 7.0, 4.1, "packets", colour="#333")

    # State vector output
    box(10.5, 3.5, 2.4, 1.2, "STATE\nESTIMATE", "#cce0ff", fontsize=10,
        sub="[pos, vel, att]")
    arrow(9.6, 4.1, 10.5, 4.1, "", colour="#333")

    # Legend note
    ax.text(0.3, 0.5,
            "Structural hazard: new packet arrives while core is executing → stall inserted\n"
            "II (Initiation Interval) = minimum ticks between successive core starts",
            fontsize=8, color="#555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f8f8", edgecolor="#ccc"))

    plt.tight_layout()
    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def generate_all_plots(result: SimulationResult, output_dir: str = "."):
    import os
    os.makedirs(output_dir, exist_ok=True)
    report = analyse(result)

    print("\n[Visualisation] Generating plots ...")
    plot_sensor_streams    (result,         f"{output_dir}/01_sensor_streams.png")
    plot_trajectory_3d     (result,         f"{output_dir}/02_trajectory_3d.png")
    plot_estimation_error  (result,         f"{output_dir}/03_estimation_error.png")
    plot_pipeline_gantt    (result,         f"{output_dir}/04_pipeline_gantt.png")
    plot_hazard_distribution(result,        f"{output_dir}/05_hazard_distribution.png")
    plot_latency_distributions(result,      f"{output_dir}/06_latency_distributions.png")
    plot_latency_throughput(result, report, f"{output_dir}/07_latency_throughput.png")
    plot_datapath_diagram  (                f"{output_dir}/08_datapath_diagram.png")
    print("[Visualisation] All plots saved.")
    return report