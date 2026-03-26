import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from sensor_models import SensorReading

STATE_DIM = 9   # State vector (9-DOF): x = [px, py, pz, vx, vy, vz, roll, pitch, yaw] (Position, Velocity, Attitude)

@dataclass
class KalmanState:
    x: np.ndarray               # state mean (STATE_DIM,)
    P: np.ndarray               # covariance  (STATE_DIM, STATE_DIM), confidence/uncertainty in state estimate
    timestamp: float = 0.0
    innovation: Optional[np.ndarray] = None   # last measurement residual


class DroneEKF:
    """
    Extended Kalman Filter fusing IMU, GPS, Baro, OpticalFlow, Mag.

    Goal is to best estimate the position, velocity, and orientation of the drone by:
        1. Use IMU readings to predict the next state based on physics/kinematics (i.e., given acceleration and gyro, where do I think the drone moved)
        2. With new sensor measurements, how should I correct my prediction/guess from the step before
    """

    # Process noise standard deviations - amount of uncertainty added per prediction step
    # Noise can be measurement based (i.e., GPS error) or general noise (i.e., vibrations, wind gusts, unmodelled aerodynamics)
    _Q_POS_STD   = 0.01    # m (~1 cm of uncertainty in positioning)
    _Q_VEL_STD   = 0.05    # m/s (~5 cm/s of uncertainty in velocity, which is reasonable for a small drone with good IMU)
    _Q_ATT_STD   = 0.005   # rad (~0.3 degrees of uncertainty in attitude, which is reasonable for a small drone with good IMU)

    def __init__(self, cython: bool = False):
        if cython:
            import quantize_helpers as qh_module
        else:
            import quantize_helpers_python as qh_module

        self.qh = qh_module
        # Initial state: drone at origin, stationary
        self.state = KalmanState(
            x= self.qh.quantize_array(np.zeros(STATE_DIM), scale= self.qh.X_SCALE),
            P= self.qh.quantize_array(np.eye(STATE_DIM) * 1.0, scale= self.qh.P_SCALE),
        )
        self._build_Q()

    # Helper functions for EKF prediction and update steps
    def _build_Q(self):
        """Creates process noise covariance matrix Q based on standard deviations."""
        """Tells filter how much confidence it should lose in mathematical predictions over time due to vibrations, IMU drift and unmodeled physics."""
        q = np.zeros(STATE_DIM)
        q[0:3] = self._Q_POS_STD**2 # uncertainty for x, y, z
        q[3:6] = self._Q_VEL_STD**2 # uncertainty for vx, vy, vz
        q[6:9] = self._Q_ATT_STD**2 # uncertainty for roll, pitch, yaw
        self.Q = self.qh.quantize_array(np.diag(q), scale= self.qh.P_SCALE) # Diagonalize to create covariance matrix where each state variable's noise is independent of the others

    def _state_transition(self, x_q: np.ndarray, dt_q: float,
                          accel_q: np.ndarray, gyro_q: np.ndarray) -> np.ndarray:
        """Non-linear state transition function. Takes current state, IMU readings, and time delta to predict next state."""
        # 1) Extract/angle for rotation (use float trig; keep full q path for everything else)
        x_f = self.qh.dequantize_array(x_q, scale= self.qh.X_SCALE)
        roll, pitch, yaw = x_f[6], x_f[7], x_f[8]

        cr = np.cos(roll); sr = np.sin(roll)
        cp = np.cos(pitch); sp = np.sin(pitch)
        cy = np.cos(yaw); sy = np.sin(yaw)

        # quantize rotation terms into Q_BITS
        cr_q = self.qh.quantize(cr, scale= self.qh.Q_BITS)
        sr_q = self.qh.quantize(sr, scale= self.qh.Q_BITS)
        cp_q = self.qh.quantize(cp, scale= self.qh.Q_BITS)
        sp_q = self.qh.quantize(sp, scale= self.qh.Q_BITS)
        cy_q = self.qh.quantize(cy, scale= self.qh.Q_BITS)
        sy_q = self.qh.quantize(sy, scale= self.qh.Q_BITS)

        # 2) Build R_q in a consistent fixed-point scale (Q_BITS)
        def m(qp, qq): return self.qh.q_mul(qp, qq, self.qh.Q_BITS, self.qh.Q_BITS, self.qh.Q_BITS)
        R_q = np.array([
            [m(cy_q, cp_q), m(m(cy_q, sp_q), sr_q) - m(sy_q, cr_q), m(m(cy_q, sp_q), cr_q) + m(sy_q, sr_q)],
            [m(sy_q, cp_q), m(m(sy_q, sp_q), sr_q) + m(cy_q, cr_q), m(m(sy_q, sp_q), cr_q) - m(cy_q, sr_q)],
            [-sp_q,         m(cp_q, sr_q),                             m(cp_q, cr_q)]
        ], dtype=np.int64)

        # 3) Accel world in H_SCALE; convert R_q*accel_q;
        accel_world_q = self.qh.q_mat_mul(R_q, accel_q.reshape(3,1),
                                    a_scale= self.qh.Q_BITS, b_scale= self.qh.H_SCALE,
                                    out_scale= self.qh.H_SCALE).flatten()
        # Gravity in H_SCALE:
        grav_q = np.array([0, 0, self.qh.quantize(-9.81, scale= self.qh.H_SCALE)], dtype=np.int64)
        acc_world_q = accel_world_q + grav_q

        # 4) Quantized kinematics (keeping X_SCALE for positions/velocity/update)
        px_q, py_q, pz_q = x_q[0], x_q[1], x_q[2]
        vx_q, vy_q, vz_q = x_q[3], x_q[4], x_q[5]
        roll_q, pitch_q, yaw_q = self.qh.quantize(roll, self.qh.X_SCALE), self.qh.quantize(pitch, self.qh.X_SCALE), self.qh.quantize(yaw, self.qh.X_SCALE)

        # dt as int fixed-point Q_BITS, velocities in X_SCALE
        vx_dt_q = self.qh.q_mul(vx_q, dt_q, self.qh.X_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)
        vy_dt_q = self.qh.q_mul(vy_q, dt_q, self.qh.X_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)
        vz_dt_q = self.qh.q_mul(vz_q, dt_q, self.qh.X_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)

        ax_q, ay_q, az_q = acc_world_q[0], acc_world_q[1], acc_world_q[2]
        ax_dt2_q = self.qh.q_mul(self.qh.q_mul(ax_q, dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.H_SCALE),
                            dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)
        ay_dt2_q = self.qh.q_mul(self.qh.q_mul(ay_q, dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.H_SCALE),
                            dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)
        az_dt2_q = self.qh.q_mul(self.qh.q_mul(az_q, dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.H_SCALE),
                            dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)

        # x_{k+1} = p + vdt + 0.5 a dt^2
        # 0.5 factor via >>1 (in X_SCALE)
        x_new0_q = px_q + vx_dt_q + (ax_dt2_q >> 1)
        x_new1_q = py_q + vy_dt_q + (ay_dt2_q >> 1)
        x_new2_q = pz_q + vz_dt_q + (az_dt2_q >> 1)

        # velocities
        x_new3_q = vx_q + self.qh.q_mul(ax_q, dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)
        x_new4_q = vy_q + self.qh.q_mul(ay_q, dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)
        x_new5_q = vz_q + self.qh.q_mul(az_q, dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)

        # attitude updates: gyro in H_SCALE -> to X_SCALE
        gyro_x_q = self.qh.q_mul(gyro_q, dt_q, self.qh.H_SCALE, self.qh.Q_BITS, self.qh.X_SCALE)
        x_new6_q = roll_q + gyro_x_q[0]
        x_new7_q = pitch_q + gyro_x_q[1]
        x_new8_q = yaw_q + gyro_x_q[2]

        x_new_q = np.array([
            x_new0_q, x_new1_q, x_new2_q,
            x_new3_q, x_new4_q, x_new5_q,
            x_new6_q, x_new7_q, x_new8_q
        ], dtype=np.int64)

        return x_new_q

    def _jacobian_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculates jacobian of state transition function. This is used to linearize the non-linear result from _state_transition.
        Mathematically, this is like taking a tangent to a curved surface. It's a local linear approximation around the current state estimate since Kalman filters only work linearly.
        
        This also tells us how sensitive the next state is to each current state variable. 
        For example, if we have a high velocity in the x direction, then our position in the x direction will be more sensitive to errors in that velocity estimate.
        
        Also note that this is a simplified EKF Jacobian. One that is more complex would have things like velocity relying on vehicle attitude and acceleration (i.e., roll) and angular rates.
        """
        F = self.qh.quantize_array(np.eye(STATE_DIM), scale= self.qh.Q_BITS   ) # 9x9 identity matrix that signifies that each state depends on it's previous value
        # Add simple kinematics dependency (position depends on velocity)
        F[0, 3] = dt # position x depends on velocity x
        F[1, 4] = dt # position y depends on velocity y
        F[2, 5] = dt # position z depends on velocity z
        return F

    # Prediction and update functions for EKF
    def predict(self, imu_reading: SensorReading, dt: float) -> KalmanState:
        """IMU-driven prediction step."""
        accel_q = self.qh.quantize_array(imu_reading.data[:3], scale= self.qh.H_SCALE)
        gyro_q = self.qh.quantize_array(imu_reading.data[3:6], scale= self.qh.H_SCALE)
        dt_q = self.qh.quantize(dt, scale= self.qh.Q_BITS)

        x_pred_q = self._state_transition(self.state.x, dt_q, accel_q, gyro_q)
        F_q = self._jacobian_F(self.state.x, dt_q)  # use float dt here
        #F_f = self.qh.dequantize_array(F_q, scale= self.qh.Q_BITS) # Quantize Jacobian to measurement scale since it will be used in the update step which is in measurement scale
        
        # x_pred_f = self.qh.dequantize_array(x_pred_q, scale= self.qh.X_SCALE)
        # P_f = self.qh.dequantize_array(self.state.P, scale= self.qh.P_SCALE)
        # Q_f = self.qh.dequantize_array(self.Q, scale= self.qh.P_SCALE)

        #P_pred_f = F_f @ P_f @ F_f.T + Q_f
        P_pred_q = self.qh.q_mat_mul(
            self.qh.q_mat_mul(F_q, self.state.P, a_scale= self.qh.Q_BITS, b_scale= self.qh.P_SCALE, out_scale= self.qh.P_SCALE),
            F_q.T, a_scale= self.qh.P_SCALE, b_scale= self.qh.Q_BITS, out_scale= self.qh.P_SCALE
        ) + self.Q
        #P_pred_q = self.qh.quantize_array(P_pred_f, scale= self.qh.P_SCALE)

        self.state = KalmanState(x=x_pred_q,
                                 P=P_pred_q,
                                 timestamp=imu_reading.timestamp)
        return self.state

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray,
                timestamp: float) -> KalmanState:
        """EKF update function to correct the predicted state based on sensor measurements."""
        x_q = self.state.x
        P_q = self.state.P

        z_q = self.qh.quantize_array(z, scale= self.qh.Z_SCALE)
        H_q = self.qh.quantize_array(H, scale= self.qh.H_SCALE)
        R_q = self.qh.quantize_array(R, scale= self.qh.P_SCALE)

        # Innovation: y = z - H*x
        Hx_q = self.qh.q_mat_mul(H_q, x_q.reshape(-1,1),
                            a_scale= self.qh.H_SCALE, b_scale= self.qh.X_SCALE,
                            out_scale= self.qh.Z_SCALE).flatten()
        y_q = z_q - Hx_q

        # Innovation covariance: S = HPH^T + R
        HP_q = self.qh.q_mat_mul(H_q, P_q,
                            a_scale= self.qh.H_SCALE, b_scale= self.qh.P_SCALE,
                            out_scale= self.qh.P_SCALE)
        S_q = self.qh.q_mat_mul(HP_q, H_q.T,
                            a_scale= self.qh.P_SCALE, b_scale= self.qh.H_SCALE,
                            out_scale= self.qh.P_SCALE) + R_q

        # Inversion in float, then re-quantize
        S_f = self.qh.dequantize_array(S_q, scale= self.qh.P_SCALE)
        S_inv_f = np.linalg.pinv(S_f)
        S_inv_q = self.qh.quantize_array(S_inv_f, scale= self.qh.P_SCALE)

        # Kalman gain: K = P H^T S^{-1}
        PHt_q = self.qh.q_mat_mul(P_q, H_q.T,
                             a_scale= self.qh.P_SCALE, b_scale= self.qh.H_SCALE,
                             out_scale= self.qh.P_SCALE)
        K_q = self.qh.q_mat_mul(PHt_q, S_inv_q,
                           a_scale= self.qh.P_SCALE, b_scale= self.qh.P_SCALE,
                           out_scale= self.qh.P_SCALE)

        # State update
        K_y_q = self.qh.q_mat_mul(K_q, y_q.reshape(-1,1),
                              a_scale= self.qh.P_SCALE, b_scale= self.qh.Z_SCALE,
                              out_scale= self.qh.X_SCALE).flatten()
        x_upd_q = x_q + K_y_q

        # Covariance update
        K_H_q = self.qh.q_mat_mul(K_q, H_q,
                              a_scale= self.qh.P_SCALE, b_scale= self.qh.H_SCALE,
                              out_scale= self.qh.P_SCALE)
        I_q = self.qh.quantize_array(np.eye(STATE_DIM), scale= self.qh.P_SCALE)
        P_upd_q = self.qh.q_mat_mul((I_q - K_H_q), P_q,
                               a_scale= self.qh.P_SCALE, b_scale= self.qh.P_SCALE,
                               out_scale= self.qh.P_SCALE)

        self.state = KalmanState(x=x_upd_q, P=P_upd_q, timestamp=timestamp,
                                 innovation= self.qh.quantize_array(y_q, scale= self.qh.Z_SCALE))
        return self.state

    def update_gps(self, reading: SensorReading) -> KalmanState:
        H = np.zeros((3, STATE_DIM)); H[0,0]=1; H[1,1]=1; H[2,2]=1 # Matrix to indicate that GPS directly measures position states
        R = np.eye(3) * reading.noise_std**2 # Matrix to describe measurement noise based on GPS noise characteristics
        return self._update(reading.data, H, R, reading.timestamp)

    def update_baro(self, reading: SensorReading) -> KalmanState:
        H = np.zeros((1, STATE_DIM)); H[0, 2] = 1 # Matrix to indicate that Barometer directly measures altitude (z position)
        R = np.array([[reading.noise_std**2]])
        return self._update(reading.data, H, R, reading.timestamp)

    def update_optical_flow(self, reading: SensorReading) -> KalmanState:
        H = np.zeros((2, STATE_DIM)); H[0,3]=1; H[1,4]=1 # Matrix to indicate that Optical Flow directly measures velocity states
        R = np.eye(2) * reading.noise_std**2
        return self._update(reading.data, H, R, reading.timestamp)

    def update_magnetometer(self, reading: SensorReading) -> KalmanState:
        mx, my = reading.data[0], reading.data[1]
        yaw_meas = np.arctan2(my, mx) # Calculate yaw measurement from magnetometer readings
        H = np.zeros((1, STATE_DIM)); H[0, 8] = 1 # Matrix to indicate that Magnetometer directly measures yaw 
        R = np.array([[reading.noise_std**2]])
        return self._update(np.array([yaw_meas]), H, R, reading.timestamp)


# Functions for analyzing pipeline performance and estimation accuracy after simulation runs.
@dataclass
class PipelineMetrics:
    """Tracks latency, throughput, and hazard statistics per sensor stream."""
    sensor_id: str
    latencies_us: list = field(default_factory=list)   # microseconds
    update_count: int  = 0
    stall_cycles: int  = 0    # cycles where sensor data arrived but pipeline busy
    last_update_t: float = -1.0

    def record(self, wall_time_start: float, sim_time: float):
        """Function that's called every time a sensor update is processed to record latency and check for stalls."""
        elapsed_us = (time.perf_counter() - wall_time_start) * 1e6
        self.latencies_us.append(elapsed_us)
        self.update_count += 1
        if self.last_update_t >= 0 and (sim_time - self.last_update_t) > (1.0 / EXPECTED_RATES_MAP.get(self.sensor_id, 100)) * 1.5: # Check if time since last updates is longer than 1.5x expected interval, if so then count as stall
            self.stall_cycles += 1
        self.last_update_t = sim_time

    @property
    def mean_latency_us(self) -> float:
        return float(np.mean(self.latencies_us)) if self.latencies_us else 0.0

    @property
    def max_latency_us(self) -> float:
        return float(np.max(self.latencies_us)) if self.latencies_us else 0.0

    @property
    def throughput_hz(self) -> float:
        if len(self.latencies_us) < 2: return 0.0
        total_s = sum(self.latencies_us) / 1e6
        return self.update_count / total_s if total_s > 0 else 0.0


EXPECTED_RATES_MAP = {"IMU": 400, "GPS": 10, "BARO": 50, "OPFLOW": 100, "MAG": 100} # Expected update rates for each sensor stream
