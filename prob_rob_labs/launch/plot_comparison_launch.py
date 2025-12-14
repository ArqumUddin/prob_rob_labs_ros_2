#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='prob_rob_labs',
            executable='plot_comparison',
            name='filter_comparison_plotter',
            output='screen'
        )
    ])
