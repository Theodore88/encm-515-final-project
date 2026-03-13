"""
Simulates sensor data for drone sensor fusion.
Models IMU, GPS, Barometer, and Optical Flow sensors with
realistic noise characteristics and update rates.
"""
import numpy as np
from dataclasses import dataclass

# Note that all angular rates are in rads/s, angles in radians, and positions in metres.

@dataclass
class SensorReading:
    """
    Generic container for a sensor reading with timestamp.
    Contains "IMU", "GPS", "BARO", "OPFLOW", or "MAG" as sensor_id, the timestamp of the reading, the data as a numpy array, and the noise standard deviation for that sensor.
    """

    sensor_id: str
    timestamp: float          # seconds
    data: np.ndarray
    noise_std: float


@dataclass
class DroneGroundTruth:
    """
    True drone state used to generate synthetic sensor readings.
    State vector: [x, y, z, vx, vy, vz, roll, pitch, yaw, ax, ay, az]
    """
    position: np.ndarray      # [x, y, z] metres
    velocity: np.ndarray      # [vx, vy, vz] m/s
    attitude: np.ndarray      # [roll, pitch, yaw] radians
    acceleration: np.ndarray  # [ax, ay, az] m/s^2


class IMUSensor:
    """
    Inertial Measurement Unit — high rate (400 Hz).
    Outputs: [ax, ay, az, gx, gy, gz]
    """
    UPDATE_RATE_HZ = 400
    ACCEL_NOISE_STD = 0.02    # m/s^2
    GYRO_NOISE_STD  = 0.001   # rad/s
    ACCEL_BIAS      = np.array([0.01, -0.005, 0.008])
    GYRO_BIAS       = np.array([0.002, -0.001, 0.003])

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def read(self, truth: DroneGroundTruth, t: float) -> SensorReading:
        accel = (truth.acceleration
                 + self.ACCEL_BIAS
                 + self.rng.normal(0, self.ACCEL_NOISE_STD, 3))
        gyro  = (truth.attitude * 0          # simplified: attitude rate ~ 0
                 + self.GYRO_BIAS
                 + self.rng.normal(0, self.GYRO_NOISE_STD, 3))
        return SensorReading("IMU", t, np.concatenate([accel, gyro]),
                             self.ACCEL_NOISE_STD)


class GPSSensor:
    """
    GPS receiver — low rate (10 Hz).
    Outputs: [x, y, z] metres
    
    Additional notes:
        - x: horizontal position along the east-west axis
        - y: horizontal position along the north-south axis
        - z: altitude (above the reference frame - i.e., above ground level (AGL), mean sea level (MSL), etc.)
    """
    UPDATE_RATE_HZ = 10
    POS_NOISE_STD  = 0.5      # metres

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def read(self, truth: DroneGroundTruth, t: float) -> SensorReading:
        pos = truth.position + self.rng.normal(0, self.POS_NOISE_STD, 3)
        return SensorReading("GPS", t, pos, self.POS_NOISE_STD)


class BarometerSensor:
    """
    Barometric altimeter — medium rate (50 Hz).
    Outputs: [altitude] metres
    """
    UPDATE_RATE_HZ = 50
    ALT_NOISE_STD  = 0.1      # metres
    ALT_DRIFT      = 0.002    # m/s drift

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._drift_accum = 0.0

    def read(self, truth: DroneGroundTruth, t: float) -> SensorReading:
        self._drift_accum += self.ALT_DRIFT / self.UPDATE_RATE_HZ
        alt = (truth.position[2]
               + self._drift_accum
               + self.rng.normal(0, self.ALT_NOISE_STD))
        return SensorReading("BARO", t, np.array([alt]), self.ALT_NOISE_STD)


class OpticalFlowSensor:
    """
    Optical flow camera — medium rate (100 Hz).
    Outputs: [vx, vy] m/s (horizontal velocity estimate)
    """
    UPDATE_RATE_HZ = 100
    VEL_NOISE_STD  = 0.05     # m/s

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def read(self, truth: DroneGroundTruth, t: float) -> SensorReading:
        vel_xy = truth.velocity[:2] + self.rng.normal(0, self.VEL_NOISE_STD, 2)
        return SensorReading("OPFLOW", t, vel_xy, self.VEL_NOISE_STD)


class MagnetometerSensor:
    """
    Magnetometer — medium rate (100 Hz).
    Outputs: [mx, my, mz] normalized magnetic field
    """
    UPDATE_RATE_HZ = 100
    MAG_NOISE_STD  = 0.02

    # Earth field reference (normalized)
    EARTH_FIELD = np.array([0.22, 0.0, 0.97])

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def read(self, truth: DroneGroundTruth, t: float) -> SensorReading:
        # Rotate earth field by yaw only (simplified)
        yaw = truth.attitude[2]
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        mag = R @ self.EARTH_FIELD + self.rng.normal(0, self.MAG_NOISE_STD, 3)
        return SensorReading("MAG", t, mag, self.MAG_NOISE_STD)


# ---------------------------------------------------------------------------
# Ground-truth trajectory generator
# ---------------------------------------------------------------------------

def generate_trajectory(duration_s: float, dt: float) -> list[DroneGroundTruth]:
    """
    Generate a synthetic drone trajectory (figure-8 horizontal + sinusoidal climb).
    Returns one DroneGroundTruth per dt step.
    """
    times = np.arange(0, duration_s, dt)
    trajectory = []
    omega = 0.3   # rad/s figure-8 frequency

    for t in times:
        x  =  10.0 * np.sin(omega * t) # Will cause the drone to move side to side
        y  =  10.0 * np.sin(2 * omega * t) / 2 # Will cause the drone to move up and down
        z  =  5.0  + 2.0 * np.sin(0.1 * t) # Will cause the drone to climb and descend
        vx =  10.0 * omega * np.cos(omega * t) # Velocity in the x axis
        vy =  10.0 * omega * np.cos(2 * omega * t) # Velocity in the y axis
        vz =  2.0  * 0.1   * np.cos(0.1 * t) # Velocity in the z axis
        ax = -10.0 * omega**2 * np.sin(omega * t) # Acceleration in the x axis
        ay = -20.0 * omega**2 * np.sin(2 * omega * t) # Acceleration in the y axis
        az = -2.0  * 0.01     * np.sin(0.1 * t) # Acceleration in the z axis
        roll  = 0.1 * np.sin(omega * t)
        pitch = 0.1 * np.cos(omega * t)
        yaw   = np.arctan2(vy, vx)

        trajectory.append(DroneGroundTruth(
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
            attitude=np.array([roll, pitch, yaw]),
            acceleration=np.array([ax, ay, az]),
        ))
    return trajectory
