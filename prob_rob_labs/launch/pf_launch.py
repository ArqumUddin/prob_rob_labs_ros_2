#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('prob_rob_labs')

    pf_node = Node(
        package='prob_rob_labs',
        executable='particle_filter',
        name='particle_filter',
        output='screen'
    )

    ld = LaunchDescription()
    ld.add_action(pf_node)

    return ld
