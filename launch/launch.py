#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import sys


def generate_launch_description():
    # Run the augmentation script once at launch (uses the same Python executable)
    script_path = os.path.join(get_package_share_directory('car_pkg'), 'create_augmented.py')

    pre_generate = ExecuteProcess(
        cmd=[sys.executable, script_path],
        output='screen'
    )

    # Launch argument to optionally start Webots
    declare_webots_arg = DeclareLaunchArgument(
        'start_webots', default_value='true', description='Start Webots simulator')

    # Webots world path
    world_file = os.path.join(get_package_share_directory('car_pkg'), 'world', 'city_traffic.wbt')
    webots_cmd = ['webots', world_file]

    webots_proc = ExecuteProcess(
        cmd=webots_cmd,
        output='screen',
        condition=IfCondition(LaunchConfiguration('start_webots'))
    )

    return LaunchDescription([
        declare_webots_arg,
        pre_generate,
        webots_proc,
        Node(
            package='car_pkg',
            executable='webots_bridge',
            name='webots_bridge'
        ),
        Node(
            package='car_pkg',
            executable='road_follower',
            name='road_follower'
        ),
        Node(
            package='car_pkg',
            executable='sign_detector',
            name='sign_detector'
        ),
        Node(
            package='car_pkg',
            executable='car_controller',
            name='car_controller'
        ),
    ])
