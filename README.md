# `sensor_models.py`
Contains five sensor classes with realistic noise/drift/bias:
- IMU at 400 Hz (accel + gyro, bias + Gaussian noise)
- GPS at 10 Hz (0.5m position noise)
- Barometer at 50 Hz (0.1m noise + slow drift)
- Optical Flow at 100 Hz (velocity noise)
- Magnetometer at 100 Hz (yaw estimation via B-field rotation)
- Synthetic figure-8 + sinusoidal climb trajectory as ground truth

# `kalman_filter.py`
9-DOF Extended Kalman Filter
- Predict step driven by IMU (DCM rotation, kinematics integration)
- Separate update methods per sensor with correct H matrices and R covariances
- `PipelineMetrics` dataclass records per-update wall latency, stall count, throughput

# `dataflow_simulator.py`
Asynchronous multi-stream pipeline model simulator
- Master clock at IMU rate (400 Hz); other sensors fire at their sub-rates
- Each update gets scheduled on a single fusion core with a TICK_COST per sensor type
- Structural hazards are detected and logged when a packet arrives while the core is busy

# `pipeline_analysis.py`
Computes and prints analysis of pipeline
- Initiation Interval (II) and max theoretical throughput per stream
- Latency stats (mean, median, p99, max)
- Hazard count and hazard rate per sensor
- Latency vs Throughput trade-off table

# `visualisation.py` 
Outputs 8 saved plots including: 
- Sensor stream overview 
- 3D trajectory comparison 
- Estimation error over time 
- Gantt pipeline timing diagram 
- Hazard distributions 
- Latency boxplots 
- L-vs-T scatter 
- Datapath block diagram

# `main.py`
Main file for running simulation as a whole.

Usage:
    `python main.py [--duration 10] [--seed 42] [--outdir ./output]`