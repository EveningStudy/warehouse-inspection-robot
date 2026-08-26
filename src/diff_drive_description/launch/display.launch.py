import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory(
        "diff_drive_description"
    )

    state_publisher_launch_file = os.path.join(
        package_share,
        "launch",
        "state_publisher.launch.py",
    )

    rviz_config_file = os.path.join(
        package_share,
        "rviz",
        "display.rviz",
    )

    state_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            state_publisher_launch_file
        )
    )

    joint_state_publisher_gui_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=[
            "-d",
            rviz_config_file,
        ],
    )

    return LaunchDescription(
        [
            state_publisher_launch,
            joint_state_publisher_gui_node,
            rviz_node,
        ]
    )