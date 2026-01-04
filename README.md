# webots-ros2-autonomous-driving (car_pkg)

Quick setup and run instructions for Ubuntu (ROS2 + Webots).

**System prerequisites**
- ROS 2 (install appropriate distro for your system) and `colcon` build tools.
- Webots (install per Webots official instructions).
 - Webots (install per Webots official instructions).

**Python dependencies**
Install Python packages into the same Python environment used by your ROS2 installation:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

OCR is not used by default; speed limits are recognized by template matching.

**ROS2 package setup**
From the workspace root (the folder that contains `src`):

```bash
# Install ROS package dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace (only car_pkg if preferred)
colcon build --packages-select car_pkg

# Source the install overlay
. install/setup.bash
```

**Run the demo**
Launch the package (this will run the augmentor and nodes; optionally starts Webots):

```bash
ros2 launch car_pkg launch.py start_webots:=true
```

The sign detector recognizes speed-limit signs by template matching and template filenames; OCR is not used.

**Notes**
- Do not pip-install ROS core packages like `rclpy`; those are provided by your ROS2 distro.
- On Ubuntu, ensure you run the pip install commands in the same environment/shell where you source ROS2.

**Files added**
- `requirements.txt` — Python dependencies for this project.
