# webots-ros2-autonomous-driving

Professional, modular demonstration package for autonomous driving research and prototyping using Webots and ROS 2.

Overview
--------
This repository provides a compact, well-structured stack demonstrating lane following and sign-aware longitudinal control in simulation. It is designed for researchers and engineers who need a reproducible demo combining perception, control, and simulator integration.

Key features
------------
- Lightweight lane detection (`road_follower`) with a proven PID-based lateral controller.
- Template-based traffic sign detection (`sign_detector`) with pre-scaled assets for robust matching.
- Webots/ROS bridge (`webots_bridge`) providing sensors and actuators integration and measured vehicle feedback.
- Controller node (`car_controller`) implementing sign-triggered behaviors (STOP, speed limits) and safety-oriented stop confirmation.
- Utility script (`create_augmented.py`) to generate and package scaled sign templates for consistent detection across distances.

Installation (developer)
------------------------
Clone the repository into your ROS 2 workspace, then build and source the install overlay:

```bash
# from workspace root
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select car_pkg
. install/setup.bash
```

Alternatively, for editable installation of Python dependencies from this package:

```bash
cd src/car_pkg
python3 -m pip install -e .
```

Running the demo
----------------
Launch the demo suite (optionally starts Webots):

```bash
ros2 launch car_pkg launch.py start_webots:=true
```

This launches perception, control, and the bridge components; the launch file arranges template preprocessing when needed.

Project layout
--------------
- `car_pkg/` — Python ROS 2 package (nodes and modules).
- `resources/` — sign templates and image assets.
- `launch/` — launch descriptions for starting the demo.
- `scripts/create_augmented.py` — template augmentation utility.

License & authors
-----------------
Licensed under the MIT license.

Contributing
------------
Contributions are welcome. Please open a PR with a clear description of changes and testing steps.