# webots-ros2-autonomous-driving (car_pkg)

Quick setup and run instructions for Ubuntu (ROS2 + Webots).

**System prerequisites**
- ROS 2 (install appropriate distro for your system) and `colcon` build tools.
- Webots (install per Webots official instructions).
- Tesseract OCR (optional, recommended for speed-limit digits):
  - `sudo apt install tesseract-ocr libtesseract-dev`

**Python dependencies**
Install Python packages into the same Python environment used by your ROS2 installation:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

If you do not want OCR, `pytesseract` is optional; the detector will fall back to template parsing.

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

If `tesseract` is installed and `pytesseract` is available in Python, the sign detector will attempt OCR on speed-limit signs. Otherwise it will fall back to template filename parsing.

**Notes**
- Do not pip-install ROS core packages like `rclpy`; those are provided by your ROS2 distro.
- On Ubuntu, ensure you run the pip install commands in the same environment/shell where you source ROS2.

**Files added**
- `requirements.txt` — Python dependencies for this project.
