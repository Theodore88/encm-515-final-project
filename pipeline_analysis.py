"""
pipeline_analysis.py
--------------------
Computes and reports pipeline characteristics:
  - Initiation Interval (II) per sensor stream
  - Latency distribution (mean, max, p99)
  - Throughput (effective Hz)
  - Structural hazard count and stall budget
  - Latency vs Throughput trade-off summary table
  - Estimation accuracy (RMSE position and velocity)
"""

import numpy as np
from dataflow_simulator import SimulationResult, TICK_COST, CLOCK_MHZ
from kalman_filter import EXPECTED_RATES_MAP


# ---------------------------------------------------------------------------

def compute_initiation_interval(sensor_id: str) -> dict:
    """
    Theoretical II = tick_cost / clock_rate.
    Maximum throughput = 1/II updates per second.
    """
    ticks = TICK_COST[sensor_id]
    ii_us  = ticks / CLOCK_MHZ            # microseconds
    max_tput = 1.0 / (ii_us * 1e-6)       # Hz
    nominal_rate = EXPECTED_RATES_MAP[sensor_id]
    utilisation = (nominal_rate / max_tput) * 100
    return {
        "sensor": sensor_id,
        "tick_cost": ticks,
        "II_us": ii_us,
        "max_throughput_hz": max_tput,
        "nominal_rate_hz": nominal_rate,
        "core_utilisation_pct": utilisation,
    }


def latency_stats(latencies_us: list) -> dict:
    a = np.array(latencies_us)
    return {
        "mean_us":   float(np.mean(a)),
        "median_us": float(np.median(a)),
        "max_us":    float(np.max(a)),
        "p99_us":    float(np.percentile(a, 99)),
        "std_us":    float(np.std(a)),
    }


def analyse(result: SimulationResult) -> dict:
    """
    Full pipeline analysis.  Returns a structured dict ready for printing
    or downstream use.
    """
    report = {}

    # ---- Pipeline characteristics per sensor ----
    report["pipeline"] = {}
    for sid in EXPECTED_RATES_MAP:
        metrics = result.pipeline_metrics[sid]
        ii_info = compute_initiation_interval(sid)
        lat_info = (latency_stats(metrics.latencies_us)
                    if metrics.latencies_us else {})
        report["pipeline"][sid] = {
            **ii_info,
            "actual_updates": metrics.update_count,
            "structural_hazards": metrics.stall_cycles,
            "hazard_rate_pct": (metrics.stall_cycles / max(metrics.update_count, 1)) * 100,
            **{f"lat_{k}": v for k, v in lat_info.items()},
            "effective_throughput_hz": metrics.throughput_hz,
        }

    # ---- Estimation accuracy ----
    pos_arr = np.array(result.position_errors)
    vel_arr = np.array(result.velocity_errors)
    report["accuracy"] = {
        "pos_rmse_m":    float(np.sqrt(np.mean(pos_arr**2))),
        "pos_mean_err_m": float(np.mean(pos_arr)),
        "pos_max_err_m":  float(np.max(pos_arr)),
        "vel_rmse_ms":   float(np.sqrt(np.mean(vel_arr**2))),
        "vel_mean_err_ms": float(np.mean(vel_arr)),
        "vel_max_err_ms":  float(np.max(vel_arr)),
    }

    # Hazard summary 
    total_updates = (result.total_imu_updates + result.total_gps_updates +
                     result.total_baro_updates + result.total_opflow_updates +
                     result.total_mag_updates)
    total_hazards = result.n_structural + result.n_raw

    # Per-sensor hazard breakdown from packet list
    per_sensor_hazards = {sid: {"structural": 0, "raw": 0}
                          for sid in EXPECTED_RATES_MAP}
    for pkt in result.packets:
        d = per_sensor_hazards[pkt.sensor_id]
        if pkt.structural_hazard: d["structural"] += 1
        if pkt.raw_hazard:        d["raw"]        += 1

    # Stall tick budget per sensor
    stall_budgets = {}
    for sid in EXPECTED_RATES_MAP:
        pkts = [p for p in result.packets if p.sensor_id == sid]
        stall_budgets[sid] = {
            "mean_stall_ticks": float(np.mean([p.stall_ticks for p in pkts])) if pkts else 0,
            "max_stall_ticks":  float(np.max ([p.stall_ticks for p in pkts])) if pkts else 0,
            "total_stall_ticks": int(sum(p.stall_ticks for p in pkts)),
        }

    report["hazards"] = {
        "n_structural": result.n_structural,
        "n_raw":        result.n_raw,
        "total_hazards": total_hazards,
        "total_updates": total_updates,
        "overall_hazard_rate_pct": (total_hazards / max(total_updates, 1)) * 100,
        "per_sensor": per_sensor_hazards,
        "stall_budgets": stall_budgets,
    }

    # ---- Aggregate throughput ----
    total_updates = report["hazards"]["total_updates"]
    report["throughput"] = {
        "total_fusion_updates": total_updates,
        "simulation_duration_s": result.duration_s,
        "aggregate_throughput_hz": total_updates / result.duration_s,
        "imu_updates": result.total_imu_updates,
        "gps_updates": result.total_gps_updates,
        "baro_updates": result.total_baro_updates,
        "opflow_updates": result.total_opflow_updates,
        "mag_updates": result.total_mag_updates,
    }

    return report


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_report(report: dict):
    SEP = "=" * 72

    print(f"\n{SEP}")
    print("  DRONE SENSOR FUSION — PIPELINE SIMULATION REPORT")
    print(SEP)

    # ---- Pipeline table ----
    print("\n[1] PIPELINE CHARACTERISTICS PER SENSOR STREAM")
    print("-" * 72)
    hdr = f"{'Sensor':<8} {'Rate':>6} {'II(µs)':>8} {'MaxTP(Hz)':>10} "
    hdr += f"{'Util%':>7} {'Updates':>8} {'Hazards':>8} {'Haz%':>6}"
    print(hdr)
    print("-" * 72)
    for sid, d in report["pipeline"].items():
        print(f"{sid:<8} {d['nominal_rate_hz']:>6.0f} {d['II_us']:>8.1f} "
              f"{d['max_throughput_hz']:>10.1f} {d['core_utilisation_pct']:>7.2f} "
              f"{d['actual_updates']:>8} {d['structural_hazards']:>8} "
              f"{d['hazard_rate_pct']:>6.2f}")
    print()

    # ---- Latency table ----
    print("[2] LATENCY STATISTICS (wall-clock µs per update)")
    print("-" * 72)
    hdr2 = f"{'Sensor':<8} {'Mean':>9} {'Median':>9} {'P99':>9} {'Max':>9} {'Std':>9}"
    print(hdr2)
    print("-" * 72)
    for sid, d in report["pipeline"].items():
        if "lat_mean_us" in d:
            print(f"{sid:<8} {d['lat_mean_us']:>9.2f} {d['lat_median_us']:>9.2f} "
                  f"{d['lat_p99_us']:>9.2f} {d['lat_max_us']:>9.2f} "
                  f"{d['lat_std_us']:>9.2f}")
    print()

    # ---- Throughput ----
    print("[3] THROUGHPUT SUMMARY")
    print("-" * 72)
    tp = report["throughput"]
    print(f"  Total fusion updates  : {tp['total_fusion_updates']}")
    print(f"  Simulation duration   : {tp['simulation_duration_s']:.1f} s")
    print(f"  Aggregate throughput  : {tp['aggregate_throughput_hz']:.1f} Hz")
    print(f"  Breakdown:")
    for k in ["imu_updates","gps_updates","baro_updates","opflow_updates","mag_updates"]:
        label = k.replace("_updates","").upper()
        print(f"    {label:<8}: {tp[k]}")
    print()

    # ---- Hazards ----
    print("[4] HAZARD ANALYSIS")
    print("-" * 72)
    hz = report["hazards"]
    print(f"  {'Hazard type':<22} {'Count':>8}  Description")
    print(f"  {'-'*21:<22} {'-'*7:>8}  {'-'*38}")
    print(f"  {'Structural':<22} {hz['n_structural']:>8}  "
          f"Two packets contend for EXECUTE unit")
    print(f"  {'RAW (read-after-write)':<22} {hz['n_raw']:>8}  "
          f"EXECUTE reads stale P (prior WB not done)")
    print(f"  {'─'*21:<22} {'─'*7:>8}")
    print(f"  {'Total':<22} {hz['total_hazards']:>8}  "
          f"({hz['overall_hazard_rate_pct']:.1f}% of all updates)")
    print()
    print(f"  Per-sensor breakdown:")
    print(f"  {'Sensor':<8} {'Structural':>12} {'RAW':>8} "
          f"{'MeanStall(µs)':>14} {'MaxStall(µs)':>13}")
    print(f"  {'-'*7:<8} {'-'*11:>12} {'-'*7:>8} {'-'*7:>8} "
          f"{'-'*13:>14} {'-'*12:>13}")
    for sid in EXPECTED_RATES_MAP:
        ps = hz["per_sensor"][sid]
        sb = hz["stall_budgets"][sid]
        print(f"  {sid:<8} {ps['structural']:>12} {ps['raw']:>8} "
              f"{sb['mean_stall_ticks']:>14.1f} {sb['max_stall_ticks']:>13.0f}")
    print()

    # ---- Accuracy ----
    print("[5] ESTIMATION ACCURACY")
    print("-" * 72)
    acc = report["accuracy"]
    print(f"  Position RMSE         : {acc['pos_rmse_m']:.4f} m")
    print(f"  Position mean error   : {acc['pos_mean_err_m']:.4f} m")
    print(f"  Position max error    : {acc['pos_max_err_m']:.4f} m")
    print(f"  Velocity RMSE         : {acc['vel_rmse_ms']:.4f} m/s")
    print(f"  Velocity mean error   : {acc['vel_mean_err_ms']:.4f} m/s")
    print(f"  Velocity max error    : {acc['vel_max_err_ms']:.4f} m/s")
    print()

    # ---- Latency vs Throughput trade-off ----
    print("[6] LATENCY vs THROUGHPUT TRADE-OFF TABLE")
    print("-" * 72)
    print(f"  {'Sensor':<8} {'Nom.Rate':>9} {'EffTP':>9} {'TP_ratio':>9} "
          f"{'MeanLat':>9}  Note")
    print(f"  {'-'*7:<8} {'-'*8:>9} {'-'*8:>9} {'-'*8:>9} {'-'*8:>9}  {'-'*25}")
    for sid, d in report["pipeline"].items():
        nom  = d["nominal_rate_hz"]
        eff  = d["effective_throughput_hz"]
        ratio = eff / nom if nom > 0 else 0
        lat  = d.get("lat_mean_us", 0)
        note = ("OK" if ratio >= 0.9
                else "DEGRADED" if ratio >= 0.5
                else "BOTTLENECK")
        print(f"  {sid:<8} {nom:>9.1f} {eff:>9.1f} {ratio:>9.3f} {lat:>9.2f}µs  {note}")
    print()
    print(SEP)