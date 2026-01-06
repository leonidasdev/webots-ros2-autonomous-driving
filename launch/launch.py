#!/usr/bin/env python3

"""ROS2 launch description for the autonomous driving demo.

Description:
    This module provides a `generate_launch_description` function that
    composes a simple launch description to prepare templates, optionally
    start the Webots simulator, and launch the core ROS nodes that form
    the demonstration stack (bridge, perception, controller, road follower).

Behaviour:
    - Executes `scripts/create_augmented.py` once at launch to produce
        pre-scaled sign templates in `resources/` when needed.
    - Optionally starts the Webots simulator using the `start_webots`
        launch argument (defaults to true).
    - Launches the package's console-entrypoint nodes: `webots_bridge`,
        `road_follower`, `sign_detector`, and `car_controller`.

Notes:
    This launch file is intentionally minimal; node-level parameters and
    advanced lifecycle handling are expected to be managed by individual
    node configurations or a more feature-rich launch description.
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import sys


def generate_launch_description():
    """Create and return the ROS2 `LaunchDescription` for the demo.

    Returns:
        launch.LaunchDescription: A launch description that executes the
                augmentation helper, optionally starts Webots, and launches
                the core ROS nodes for the demo.

    Behaviour:
        - Runs `create_augmented.py` via `ExecuteProcess` using the same
            Python interpreter as the launcher to avoid environment issues.
        - Declares a `start_webots` launch argument that controls whether
            the Webots simulator process is started.
        - Adds Node actions for the package's main executables.
    """

    # Path to the augmentation helper inside the package (now under scripts/)
    script_path = os.path.join(get_package_share_directory('car_pkg'), 'scripts', 'create_augmented.py')

    pre_generate = ExecuteProcess(cmd=[sys.executable, script_path], output='screen')

    # Launch argument: optionally start Webots (default: true)
    declare_webots_arg = DeclareLaunchArgument(
        'start_webots', default_value='true', description='Start Webots simulator')

    # Webots world file (package share)
    world_file = os.path.join(get_package_share_directory('car_pkg'), 'world', 'city_traffic.wbt')
    webots_cmd = ['webots', world_file]

    webots_proc = ExecuteProcess(
        cmd=webots_cmd,
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_webots'))
    )

    # Launch core nodes. Executable names map to console scripts defined in
    # setup.py / entry_points when the package is installed.
    nodes = [
        Node(package='car_pkg', executable='webots_bridge', name='webots_bridge'),
        Node(package='car_pkg', executable='road_follower', name='road_follower'),
        Node(package='car_pkg', executable='sign_detector', name='sign_detector'),
        Node(package='car_pkg', executable='car_controller', name='car_controller'),
    ]

    return LaunchDescription([declare_webots_arg, pre_generate, webots_proc] + nodes)
