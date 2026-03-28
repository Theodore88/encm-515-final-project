"""
Multi-stream dataflow simulation with 4-stage pipeline

Pipeline stages per packet:
    FETCH     (F) : read packet from sensor FIFO          — 1 tick
    DECODE    (D) : identify sensor type, load H matrix   — 1 tick
    EXECUTE   (E) : matrix multiply / Kalman gain compute — sensor-dependent
    WRITEBACK (W) : commit updated x and P to registers   — 1 tick

Possible Pipeline Hazards:
    Structural : Two packets both need the EXECUTE unit in the same tick window.
                 Only one hardware multiplier exists — one must stall.

    RAW (data) : An update's EXECUTE reads P/x while a prior operation's
                 WRITEBACK has not yet committed. The update would use a
                 stale covariance matrix.

Pipeline Metrics Recorded:
    Initiation Interval (II) -  minimum ticks between successive EXECUTE starts
    Latency - ticks from FETCH to end of WRITEBACK
    Throughput - completed updates per simulated second
"""

import numpy as np
import time
from collections import defaultdict
from dataclasses import dataclass, field
import copy
from sensor_models import (
    SensorReading,
    IMUSensor, GPSSensor, BarometerSensor,
    OpticalFlowSensor, MagnetometerSensor,
    generate_trajectory,
)

from kalman_filter import DroneEKF as DroneEKFFloat
from kalman_filter_quantize import DroneEKF as DroneEKFQuant
from kalman_filter import PipelineMetrics, KalmanState, EXPECTED_RATES_MAP
import quantize_helpers as qh_module
# Pipeline stage costs (ticks at CLOCK_MHZ)

CLOCK_MHZ = 1.0   # 1 tick = 1 µs

# Fixed stage costs identical for all sensors
TICKS_FETCH     = 1
TICKS_DECODE    = 1
TICKS_WRITEBACK = 1

# EXECUTE cost is sensor-dependent (reflects matrix dimension and op count)
TICKS_EXECUTE = {
    "IMU":    118,  # predict: DCM rotation + full 9x9 covariance propagation
    "GPS":     58,  # 3-measurement update: 3x9 H, Kalman gain, 9x9 P update
    "BARO":    18,  # 1-measurement update
    "OPFLOW":  28,  # 2-measurement update
    "MAG":     23,  # 1-measurement update
}

# Total ideal latency per sensor
TICK_COST = {sid: TICKS_FETCH + TICKS_DECODE + e + TICKS_WRITEBACK
             for sid, e in TICKS_EXECUTE.items()}


# ---------------------------------------------------------------------------
# Pipeline packet — records per-stage tick timestamps and hazard flags
# ---------------------------------------------------------------------------

@dataclass
class PipelinePacket:
    """Tracks a sensor reading through each pipeline stage."""
    sensor_id:      str
    reading:        object          
    tick_fetch:     int = 0
    tick_decode:    int = 0
    tick_execute:   int = 0
    tick_writeback: int = 0
    tick_done:      int = 0

    structural_hazard: bool = False  # EXECUTE unit busy on arrival
    raw_hazard:        bool = False  # P/x not yet committed when EXECUTE starts

    @property
    def total_latency_ticks(self) -> int:
        return self.tick_done - self.tick_fetch

    @property
    def stall_ticks(self) -> int:
        ideal = TICKS_FETCH + TICKS_DECODE + TICKS_EXECUTE[self.sensor_id] + TICKS_WRITEBACK
        return max(0, self.total_latency_ticks - ideal)


@dataclass
class SimulationResult:
    duration_s:           float
    sim_dt:               float
    truth_trajectory:     list
    estimated_trajectory: list
    pipeline_metrics:     dict
    packets:              list      
    sensor_readings:      dict
    total_imu_updates:    int = 0
    total_gps_updates:    int = 0
    total_baro_updates:   int = 0
    total_opflow_updates: int = 0
    total_mag_updates:    int = 0
    position_errors:      list = field(default_factory=list)
    velocity_errors:      list = field(default_factory=list)
    n_structural:         int = 0
    n_raw:                int = 0

class FourStagePipeline:
    """
    Models four stage pipeline with one shared EXECUTE unit
    and one shared register file for P and x (WRITEBACK target).

    _execute_free_tick   : earliest tick the EXECUTE unit accepts a new packet
    _writeback_done_tick : tick at which the last committed WRITEBACK finishes
                           (the earliest tick P and x are safe to read)
    """

    def __init__(self):
        self._execute_free_tick   = 0
        self._writeback_done_tick = 0

    def schedule(self, sensor_id: str, arrival_tick: int) -> PipelinePacket:
        """
        Schedule a packet through F → D → E → W, recording stalls as needed.
        Returns a PipelinePacket with all timestamps and hazard flags set.
        """
        pkt = PipelinePacket(sensor_id=sensor_id, reading=None,
                             tick_fetch=arrival_tick)

        # FETCH and DECODE have no resource conflict — start immediately
        pkt.tick_decode = pkt.tick_fetch  + TICKS_FETCH
        exe_desired     = pkt.tick_decode + TICKS_DECODE

        # Structural hazard 
        # EXECUTE unit is shared across all sensor streams.
        # If it is still busy from a prior packet, this packet must stall.
        if exe_desired < self._execute_free_tick:
            pkt.structural_hazard = True
            exe_desired = self._execute_free_tick   # stall until unit free

        # RAW (read-after-write) data hazard 
        # EXECUTE reads the current P matrix and state vector x.
        # If a prior packet's WRITEBACK hasn't committed yet, reading now
        # would see a stale P  insert a stall and record hazard until the write completes.
        if exe_desired < self._writeback_done_tick:
            pkt.raw_hazard = True
            exe_desired = self._writeback_done_tick  # stall until P/x safe

        pkt.tick_execute        = exe_desired
        exe_cost                = TICKS_EXECUTE[sensor_id]
        self._execute_free_tick = pkt.tick_execute + exe_cost  # Update time when EXECUTE unit will be free again

        wb_desired = pkt.tick_execute + exe_cost

        pkt.tick_writeback        = wb_desired
        pkt.tick_done             = wb_desired + TICKS_WRITEBACK
        self._writeback_done_tick = pkt.tick_done

        return pkt



class MultiStreamSimulator:
    """
    Drives the full simulation loop:
      1. Step the ground truth trajectory.
      2. At each sensor's update rate, generate a reading.
      3. Push the packet through the 4-stage pipeline scheduler.
      4. Detect and classify structural / RAW hazards.
      5. Run the EKF prediction and update.
      6. Record metrics.
    """

    def __init__(self, duration_s: float = 5.0, seed: int = 42, quantized: bool = False, simd: bool = False):
        self.duration_s = duration_s
        self.sim_dt     = 1.0 / IMUSensor.UPDATE_RATE_HZ
        self.rng        = np.random.default_rng(seed)

        self.imu    = IMUSensor(self.rng)
        self.gps    = GPSSensor(self.rng)
        self.baro   = BarometerSensor(self.rng)
        self.opflow = OpticalFlowSensor(self.rng)
        self.mag    = MagnetometerSensor(self.rng)
        self.quantized = quantized
        self.simd = simd


        self.qh = qh_module
        if quantized:
            self.ekf = DroneEKFQuant(cython=True)
        else:
            self.ekf = DroneEKFFloat(simd)
        self.pipeline = FourStagePipeline()
        self.metrics  = {sid: PipelineMetrics(sensor_id=sid)
                         for sid in EXPECTED_RATES_MAP}

    def _sensor_fires(self, sensor_id: str, step: int) -> bool:
        ratio = int(IMUSensor.UPDATE_RATE_HZ / EXPECTED_RATES_MAP[sensor_id])
        return (step % ratio) == 0

    def _process(self, sensor_id: str, reading: SensorReading,
                 current_tick: int, t: float,
                 all_readings: dict, packets: list) -> KalmanState:
        """Schedule one sensor reading and run the corresponding EKF step."""
        all_readings[sensor_id].append(reading)

        t0  = time.perf_counter()
        pkt = self.pipeline.schedule(sensor_id, current_tick)
        pkt.reading = reading
        packets.append(pkt)

        if sensor_id == "IMU":
            state = self.ekf.predict(reading, self.sim_dt)
        elif sensor_id == "GPS":
            state = self.ekf.update_gps(reading)
        elif sensor_id == "BARO":
            state = self.ekf.update_baro(reading)
        elif sensor_id == "OPFLOW":
            state = self.ekf.update_optical_flow(reading)
        else: 
            state = self.ekf.update_magnetometer(reading)

        self.metrics[sensor_id].record(t0, t)
        return state

    def run(self, verbose: bool = True) -> SimulationResult:
        trajectory     = generate_trajectory(self.duration_s, self.sim_dt)
        n_steps        = len(trajectory)
        ticks_per_step = int(CLOCK_MHZ * 1e6 * self.sim_dt)

        estimated_states: list = []
        all_readings:     dict = defaultdict(list)
        packets:          list = []
        pos_errors:       list = []
        vel_errors:       list = []
        update_counts          = defaultdict(int)

        print(f"[Simulation] Starting: {self.duration_s}s, {n_steps} steps, "
              f"dt={self.sim_dt*1000:.2f}ms")
        wall_start = time.perf_counter()

        for step, truth in enumerate(trajectory):
            t            = step * self.sim_dt
            current_tick = step * ticks_per_step

            # IMU fires every step (master clock)
            imu_r = self.imu.read(truth, t)
            state = self._process("IMU", imu_r, current_tick, t,
                                  all_readings, packets)
            update_counts["IMU"] += 1

            if self._sensor_fires("GPS", step):
                state = self._process("GPS", self.gps.read(truth, t),
                                      current_tick, t, all_readings, packets)
                update_counts["GPS"] += 1

            if self._sensor_fires("BARO", step):
                state = self._process("BARO", self.baro.read(truth, t),
                                      current_tick, t, all_readings, packets)
                update_counts["BARO"] += 1

            if self._sensor_fires("OPFLOW", step):
                state = self._process("OPFLOW", self.opflow.read(truth, t),
                                      current_tick, t, all_readings, packets)
                update_counts["OPFLOW"] += 1

            if self._sensor_fires("MAG", step):
                state = self._process("MAG", self.mag.read(truth, t),
                                      current_tick, t, all_readings, packets)
                update_counts["MAG"] += 1

            new_state = copy.deepcopy(state)
            if self.quantized:
                new_state.x = self.qh.dequantize_array(new_state.x, scale=self.qh.X_SCALE)
                new_state.P = self.qh.dequantize_array(new_state.P, scale=self.qh.P_SCALE)
            pos_err = np.linalg.norm(new_state.x[0:3] - truth.position)
            vel_err = np.linalg.norm(new_state.x[3:6] - truth.velocity)
            pos_errors.append(pos_err)
            vel_errors.append(vel_err)
            estimated_states.append(new_state)

            if verbose and step % (n_steps // 10) == 0:
                n_s = sum(1 for p in packets if p.structural_hazard)
                n_r = sum(1 for p in packets if p.raw_hazard)
                print(f"  [{100*step/n_steps:5.1f}%] t={t:.2f}s  "
                      f"pos_err={pos_err:.3f}m  "
                      f"structural={n_s}  RAW={n_r}")

        wall_elapsed = time.perf_counter() - wall_start
        print(f"[Simulation] Done in {wall_elapsed:.3f}s wall time.")

        return SimulationResult(
            duration_s=self.duration_s,
            sim_dt=self.sim_dt,
            truth_trajectory=trajectory,
            estimated_trajectory=estimated_states,
            pipeline_metrics=self.metrics,
            packets=packets,
            sensor_readings=dict(all_readings),
            total_imu_updates=update_counts["IMU"],
            total_gps_updates=update_counts["GPS"],
            total_baro_updates=update_counts["BARO"],
            total_opflow_updates=update_counts["OPFLOW"],
            total_mag_updates=update_counts["MAG"],
            position_errors=pos_errors,
            velocity_errors=vel_errors,
            n_structural=sum(1 for p in packets if p.structural_hazard),
            n_raw=sum(1 for p in packets if p.raw_hazard),
        )