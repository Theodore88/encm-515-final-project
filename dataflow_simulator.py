"""
Multi-stream dataflow simulation.

Models the asynchronous arrival of sensor packets at different rates,
the scheduling of EKF updates, and pipeline hazards when multiple
sensor updates arrive within the same time-step.

Pipeline model:
    Each EKF update is treated as occupying the EKF fusion core for some
    number of 'ticks' (modelled at 1 MHz equivalent).  If a new packet
    arrives while the core is busy, a structural hazard (stall) occurs.

    Initiation Interval (II)  = ticks between successive updates to core
    Latency                   = ticks from packet arrival to state update
    Throughput                = updates / second
"""

import numpy as np
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from sensor_models import (
    SensorReading,
    IMUSensor, GPSSensor, BarometerSensor,
    OpticalFlowSensor, MagnetometerSensor,
    generate_trajectory,
)
from kalman_filter import DroneEKF, PipelineMetrics, KalmanState, _EXPECTED_RATES_MAP


TICK_COST = { # Estimated number of cycles each sensor update takes
    "IMU":    120,  
    "GPS":     60,
    "BARO":    20,
    "OPFLOW":  30,
    "MAG":     25,
}

CLOCK_MHZ = 1.0     # simulated clock rate


@dataclass
class PacketEvent:
    """A sensor packet ready for fusion core processing."""
    sensor_id: str
    reading: SensorReading
    arrival_tick: int


@dataclass
class SimulationResult:
    """Full simulation results for reporting."""
    duration_s: float
    sim_dt: float
    truth_trajectory: list
    estimated_trajectory: list[KalmanState]
    pipeline_metrics: dict[str, PipelineMetrics]
    hazard_log: list[dict]
    sensor_readings: dict[str, list[SensorReading]]
   
   # Aggregates for report summary
    total_imu_updates: int = 0
    total_gps_updates: int = 0
    total_baro_updates: int = 0
    total_opflow_updates: int = 0
    total_mag_updates: int = 0
    total_hazards: int = 0
    position_errors: list = field(default_factory=list)
    velocity_errors: list = field(default_factory=list)

# Main simulator

class MultiStreamSimulator:
    """
    Full simulation loop:
      1. Step the ground truth trajectory.
      2. At each sensor's update rate, generate a reading.
      3. Schedule the reading into the fusion core pipeline.
      4. Resolve structural hazards (stalls).
      5. Run the EKF predict / update.
      6. Record metrics.
    """

    def __init__(self, duration_s: float = 5.0, seed: int = 42):
        self.duration_s  = duration_s
        self.sim_dt      = 1.0 / IMUSensor.UPDATE_RATE_HZ 
        self.rng         = np.random.default_rng(seed)

        # Sensors
        self.imu     = IMUSensor(self.rng)
        self.gps     = GPSSensor(self.rng)
        self.baro    = BarometerSensor(self.rng)
        self.opflow  = OpticalFlowSensor(self.rng)
        self.mag     = MagnetometerSensor(self.rng)

        # EKF
        self.ekf = DroneEKF()

        # Per-sensor metrics
        self.metrics = {sid: PipelineMetrics(sensor_id=sid)
                        for sid in _EXPECTED_RATES_MAP}

        # Pipeline state
        self._core_busy_until_tick = 0    
        self._hazard_log: list[dict] = []

    def _sensor_fires(self, sensor_id: str, step: int) -> bool:
        """Return True if sensor fires at this master-clock step."""
        ratio = int(IMUSensor.UPDATE_RATE_HZ / _EXPECTED_RATES_MAP[sensor_id])
        return (step % ratio) == 0

    def _schedule_update(self, packet: PacketEvent,
                         current_tick: int) -> tuple[int, bool]:
        """
        Schedule packet for core execution.
        If core is busy, packet is stalled until core is free (structural hazard).
        """
        hazard = current_tick < self._core_busy_until_tick
        start_tick = max(current_tick, self._core_busy_until_tick)
        cost = TICK_COST[packet.sensor_id]
        self._core_busy_until_tick = start_tick + cost
        return start_tick, hazard


    def run(self, verbose: bool = True) -> SimulationResult:
        trajectory = generate_trajectory(self.duration_s, self.sim_dt)
        n_steps    = len(trajectory)
        ticks_per_step = int(CLOCK_MHZ * 1e6 * self.sim_dt)  # ticks per master step

        estimated_states: list[KalmanState] = []
        all_readings: dict[str, list] = defaultdict(list)
        hazard_log: list[dict] = []
        pos_errors: list = []
        vel_errors: list = []

        # Pipeline tick counter
        global_tick = 0

        # per-sensor update counters 
        update_counts = defaultdict(int)

        print(f"[Simulation] Starting: {self.duration_s}s, {n_steps} steps, "
              f"dt={self.sim_dt*1000:.2f}ms")
        wall_start = time.perf_counter()

        for step, truth in enumerate(trajectory):
            t = step * self.sim_dt
            current_tick = global_tick + step * ticks_per_step

            # IMU always runs cause it follows master clock 
            t0 = time.perf_counter()
            imu_r = self.imu.read(truth, t)
            all_readings["IMU"].append(imu_r)

            start_tick, hazard = self._schedule_update(
                PacketEvent("IMU", imu_r, current_tick), current_tick)

            if hazard:
                stall_ticks = start_tick - current_tick
                hazard_log.append({"t": t, "sensor": "IMU",
                                   "stall_ticks": stall_ticks})
                self.metrics["IMU"].stall_cycles += 1

            state = self.ekf.predict(imu_r, self.sim_dt)
            self.metrics["IMU"].record(t0, t)
            update_counts["IMU"] += 1

            #  GPS 
            if self._sensor_fires("GPS", step):
                t0 = time.perf_counter()
                gps_r = self.gps.read(truth, t)
                all_readings["GPS"].append(gps_r)

                start_tick, hazard = self._schedule_update(
                    PacketEvent("GPS", gps_r, current_tick), current_tick)
                if hazard:
                    stall_ticks = start_tick - current_tick
                    hazard_log.append({"t": t, "sensor": "GPS",
                                       "stall_ticks": stall_ticks})
                    self.metrics["GPS"].stall_cycles += 1

                state = self.ekf.update_gps(gps_r)
                self.metrics["GPS"].record(t0, t)
                update_counts["GPS"] += 1

            #  Barometer 
            if self._sensor_fires("BARO", step):
                t0 = time.perf_counter()
                baro_r = self.baro.read(truth, t)
                all_readings["BARO"].append(baro_r)

                start_tick, hazard = self._schedule_update(
                    PacketEvent("BARO", baro_r, current_tick), current_tick)
                if hazard:
                    stall_ticks = start_tick - current_tick
                    hazard_log.append({"t": t, "sensor": "BARO",
                                       "stall_ticks": stall_ticks})
                    self.metrics["BARO"].stall_cycles += 1

                state = self.ekf.update_baro(baro_r)
                self.metrics["BARO"].record(t0, t)
                update_counts["BARO"] += 1

            #  Optical Flow 
            if self._sensor_fires("OPFLOW", step):
                t0 = time.perf_counter()
                of_r = self.opflow.read(truth, t)
                all_readings["OPFLOW"].append(of_r)

                start_tick, hazard = self._schedule_update(
                    PacketEvent("OPFLOW", of_r, current_tick), current_tick)
                if hazard:
                    stall_ticks = start_tick - current_tick
                    hazard_log.append({"t": t, "sensor": "OPFLOW",
                                       "stall_ticks": stall_ticks})
                    self.metrics["OPFLOW"].stall_cycles += 1

                state = self.ekf.update_optical_flow(of_r)
                self.metrics["OPFLOW"].record(t0, t)
                update_counts["OPFLOW"] += 1

            #  Magnetometer 
            if self._sensor_fires("MAG", step):
                t0 = time.perf_counter()
                mag_r = self.mag.read(truth, t)
                all_readings["MAG"].append(mag_r)

                start_tick, hazard = self._schedule_update(
                    PacketEvent("MAG", mag_r, current_tick), current_tick)
                if hazard:
                    stall_ticks = start_tick - current_tick
                    hazard_log.append({"t": t, "sensor": "MAG",
                                       "stall_ticks": stall_ticks})
                    self.metrics["MAG"].stall_cycles += 1

                state = self.ekf.update_magnetometer(mag_r)
                self.metrics["MAG"].record(t0, t)
                update_counts["MAG"] += 1

            #  Record estimation error 
            est_pos = state.x[0:3]
            est_vel = state.x[3:6]
            pos_err = np.linalg.norm(est_pos - truth.position)
            vel_err = np.linalg.norm(est_vel - truth.velocity)
            pos_errors.append(pos_err)
            vel_errors.append(vel_err)
            estimated_states.append(state)

            if verbose and step % (n_steps // 10) == 0:
                pct = 100 * step / n_steps
                print(f"  [{pct:5.1f}%] t={t:.2f}s  pos_err={pos_err:.3f}m  "
                      f"hazards_so_far={len(hazard_log)}")

        wall_elapsed = time.perf_counter() - wall_start
        print(f"[Simulation] Done in {wall_elapsed:.3f}s wall time.")

        result = SimulationResult(
            duration_s=self.duration_s,
            sim_dt=self.sim_dt,
            truth_trajectory=trajectory,
            estimated_trajectory=estimated_states,
            pipeline_metrics=self.metrics,
            hazard_log=hazard_log,
            sensor_readings=dict(all_readings),
            total_imu_updates=update_counts["IMU"],
            total_gps_updates=update_counts["GPS"],
            total_baro_updates=update_counts["BARO"],
            total_opflow_updates=update_counts["OPFLOW"],
            total_mag_updates=update_counts["MAG"],
            total_hazards=len(hazard_log),
            position_errors=pos_errors,
            velocity_errors=vel_errors,
        )
        return result
