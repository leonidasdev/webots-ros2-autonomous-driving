#!/usr/bin/env python3

"""ROS2 launch file for the autonomous driving demo.

This launch description performs three responsibilities:
1. Run the local `create_augmented.py` once to ensure pre-scaled templates
   exist in `resources/`.
2. Optionally start the Webots simulator with the provided world file.
3. Launch the ROS nodes that implement the robot (bridge, perception,
   control and lane-following).

The launch is intentionally simple — node parameters and more advanced
composition are left to individual development tasks.
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
    """Create and return the launch description.

    The augmentation script is executed via `ExecuteProcess` so that the
    same Python interpreter used to launch ROS will run it (avoids PATH
    differences). Webots is started conditionally based on the
    `start_webots` launch argument.
    """

    # Path to the augmentation helper inside the package
    script_path = os.path.join(get_package_share_directory('car_pkg'), 'create_augmented.py')

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
