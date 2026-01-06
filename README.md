# Webots ROS2 Autonomous Driving

Compact demonstration stack combining Webots and ROS 2 for lane-following
and sign-aware longitudinal control. Useful as a reproducible research
prototype or teaching artifact for perception + control integration.

## Overview
This repository contains a small modular Python ROS2 package that runs a
simulated vehicle in Webots with the following responsibilities:
- Lateral control: `road_follower` detects the lane center and publishes steering.
- Sign detection: `sign_detector` performs template-matching and publishes sign tokens.
- Bridge: `webots_bridge` exposes Webots cameras, publishes `/clock` and `/odom`, and
	applies `/control/speed` and `/control/steering` to the simulated actuators.
- High-level control: `car_controller` translates sign tokens into speed commands
	(STOP, yield, speed limits) and enforces safety checks.


## Prerequisites
- ROS 2 (tested on Foxy/Galactic — adapt commands for your distro).
- Webots (matching the project world file; `webots` must be on PATH).
- Python 3 with required packages: `rclpy`, `cv2` (OpenCV), `cv_bridge`, `numpy`.

## Quickstart (developer)
From a ROS 2 workspace root:

```bash
# install runtime deps (adjust for your ROS distro)
rosdep install --from-paths src --ignore-src -r -y

# build package
colcon build --packages-select car_pkg
. install/setup.bash

# launch demo (starts Webots by default)
ros2 launch car_pkg launch.py start_webots:=true
```

Notes:
- The launch file runs `scripts/create_augmented.py` once to ensure pre-scaled
	templates are available in `resources/`.


## What's in the package
- `car_pkg/car_pkg/webots_bridge.py` — Bridge node: publishes
	`/car_camera/image`, `/road_camera/image`, `/clock`, and `/odom`; subscribes
	to `/control/speed` and `/control/steering`.
- `car_pkg/car_pkg/road_follower.py` — Lane detection and `steering` publishing.
- `car_pkg/car_pkg/sign_detector.py` — Template-matching sign detector publishing
	`/traffic_sign` tokens (e.g., `stop`, `yield`, `speed_limit_50`).
- `car_pkg/car_pkg/car_controller.py` — High-level logic: subscribes to
	`/traffic_sign`, `/clock`, `/odom` and publishes `/control/speed` commands.
- `scripts/create_augmented.py` — Utility to generate `_aug_...` and scaled
	template files in `resources/` for deterministic template matching.

## Topics (summary)
- `/car_camera/image` (sensor_msgs/Image) — car-facing camera frames.
- `/road_camera/image` (sensor_msgs/Image) — road-facing camera frames.
- `/traffic_sign` (std_msgs/String) — detected sign tokens.
- `/control/speed` (std_msgs/Float32) — commanded forward/reverse speed.
- `/control/steering` (std_msgs/Float32) — steering setpoint.
- `/odom` (nav_msgs/Odometry) — twist-only speed estimate used for STOP detection.
- `/clock` (rosgraph_msgs/Clock) — simulation time.


## Configuration & tuning notes
- `speed_conversion_factor` (in `car_controller.py`): maps controller speed
	units to motor units used by `webots_bridge`.
- `brake_strength`, `brake_duration`: tune how strongly the controller requests
	negative speed for active braking and how long a short timed brake lasts.
- `wheel_radius` and `flow_scale` (in `webots_bridge.py`) affect the `/odom`
	speed estimate; adjust to match the vehicle geometry and camera mounting.

## Troubleshooting
- Webots warnings about requested velocity exceeding `maxVelocity` indicate
	published speed commands are out of actuator range. The bridge clamps
	requested values to motor limits; tune `brake_strength` and
	`speed_conversion_factor` in the controller to avoid repeated clamping.
- If STOP behavior finishes while the vehicle is still moving, ensure
	`/odom` is publishing correct speeds (check `use_optical_flow` and
	`flow_scale` in `webots_bridge.py`) or increase `physical_hold_time` in
	`car_controller.py`.
- To regenerate templates manually (for development):

```bash
python3 scripts/create_augmented.py
```

## Contributing
Please open issues or pull requests. For changes that affect behavior,
include testing notes and which simulator/world you used.

## License
This project is licensed under the MIT License. The full license text is
provided in the repository root at `LICENSE` for easy discovery.

License metadata is also declared in the package files: see
[package.xml](package.xml) and [setup.py](setup.py) in the `car_pkg`
package.
