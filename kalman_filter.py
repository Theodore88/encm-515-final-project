import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from sensor_models import SensorReading

from simd.ekf_compare import covariance_predict_scalar, covariance_predict_simd

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

    def __init__(self, simd = False):        
        # Initial state: drone at origin, stationary
        self.state = KalmanState(
            x=np.zeros(STATE_DIM), # Current state vector: [px, py, pz, vx, vy, vz, roll, pitch, yaw]
            P=np.eye(STATE_DIM) * 1.0, # Covariance matrix; the amount of uncertainty in the estimate
        )
        self._build_Q()
        self.simd = simd

    # Helper functions for EKF prediction and update steps
    def _build_Q(self):
        """Creates process noise covariance matrix Q based on standard deviations."""
        """Tells filter how much confidence it should lose in mathematical predictions over time due to vibrations, IMU drift and unmodeled physics."""
        q = np.zeros(STATE_DIM)
        q[0:3] = self._Q_POS_STD**2 # uncertainty for x, y, z
        q[3:6] = self._Q_VEL_STD**2 # uncertainty for vx, vy, vz
        q[6:9] = self._Q_ATT_STD**2 # uncertainty for roll, pitch, yaw
        self.Q = np.diag(q) # Diagonalize to create covariance matrix where each state variable's noise is independent of the others

    def _state_transition(self, x: np.ndarray, dt: float,
                          accel: np.ndarray, gyro: np.ndarray) -> np.ndarray:
        """Non-linear state transition function. Takes current state, IMU readings, and time delta to predict next state."""
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        roll, pitch, yaw = x[6], x[7], x[8]

        # Convert body-frame accelerations to world-frame using current attitude
        '''
        Note that when an accelerometer measures a drone's acceleration, it is relative to the drone frame. This doesn't necessarily
        align with the world frame, because as a drone moves, it tilts to achieve aerodynamic stability. As a result of this, we have
        to use trig to find the x and y components of acceleration (using the vehicle attitude).
        '''
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        
        # Construct DCM (ZYX rotation matrix) to convert body-frame acceleration to world-frame (direct cosine matrix for a ZYX Euler-angle rotation)
        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])
        g = np.array([0, 0, -9.81]) # Gravity vector in world frame (only in the z-axis; up and down)
        acc_world = R @ accel + g # total acceleration in world frame (including gravity)

        # Update state using kinematics equations
        '''
        new_position = old_position + velocity * dt + 0.5 * acceleration * dt^2)
        new_velocity = old_velocity + acceleration * dt
        new_attitude = old_attitude + angular_rate * dt 
        '''
        x_new = np.array([
            px + vx*dt + 0.5*acc_world[0]*dt**2,
            py + vy*dt + 0.5*acc_world[1]*dt**2,
            pz + vz*dt + 0.5*acc_world[2]*dt**2,
            vx + acc_world[0]*dt,
            vy + acc_world[1]*dt,
            vz + acc_world[2]*dt,
            roll  + gyro[0]*dt,
            pitch + gyro[1]*dt,
            yaw   + gyro[2]*dt,
        ])
        return x_new

    def _jacobian_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculates jacobian of state transition function. This is used to linearize the non-linear result from _state_transition.
        Mathematically, this is like taking a tangent to a curved surface. It's a local linear approximation around the current state estimate since Kalman filters only work linearly.
        
        This also tells us how sensitive the next state is to each current state variable. 
        For example, if we have a high velocity in the x direction, then our position in the x direction will be more sensitive to errors in that velocity estimate.
        
        Also note that this is a simplified EKF Jacobian. One that is more complex would have things like velocity relying on vehicle attitude and acceleration (i.e., roll) and angular rates.
        """
        F = np.eye(STATE_DIM) # 9x9 identity matrix that signifies that each state depends on it's previous value
        # Add simple kinematics dependency (position depends on velocity)
        F[0, 3] = dt # position x depends on velocity x
        F[1, 4] = dt # position y depends on velocity y
        F[2, 5] = dt # position z depends on velocity z
        return F

    # Prediction and update functions for EKF
    def predict(self, imu_reading: SensorReading, dt: float) -> KalmanState:
        """IMU-driven prediction step."""
        accel = imu_reading.data[:3]
        gyro  = imu_reading.data[3:6]

        x_pred = self._state_transition(self.state.x, dt, accel, gyro) # Get predicted state based on previous state and IMU readings
        F      = self._jacobian_F(self.state.x, dt) # Get jacobian of state transition function to linearize the non-linear x_pred
        
        F = np.ascontiguousarray(F, dtype=np.float64)
        P = np.ascontiguousarray(self.state.P, dtype=np.float64)
        Q = np.ascontiguousarray(self.Q, dtype=np.float64)

        # predict new covariance (uncertainty in state) based on previous covariance and process noise
        if self.simd:
            P_pred = covariance_predict_simd(F, P, Q)
        else:
            P_pred = covariance_predict_scalar(F, P, Q)

        self.state = KalmanState(x=x_pred, P=P_pred, # Update filter state with new predicted mean and covariance
                                 timestamp=imu_reading.timestamp)
        return self.state

    def _update(self, z: np.ndarray, H: np.ndarray, R: np.ndarray,
                timestamp: float) -> KalmanState:
        """EKF update function to correct the predicted state based on sensor measurements."""
        y = z - H @ self.state.x                          # Calculate innovation (Difference between measurement and predicted measurement)
        S = H @ self.state.P @ H.T + R                    # Calculate innovation covariance (uncertainty in innovation)
        K = self.state.P @ H.T @ np.linalg.inv(S)         # Kalman gain (How much to trust sensor measurement vs prediction)
        x_upd = self.state.x + K @ y # Update/correct state based on innovation and Kalman gain
        P_upd = (np.eye(STATE_DIM) - K @ H) @ self.state.P # Update/correct covariance based on Kalman gain and measurement model
        self.state = KalmanState(x=x_upd, P=P_upd,
                                 timestamp=timestamp, innovation=y)
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
