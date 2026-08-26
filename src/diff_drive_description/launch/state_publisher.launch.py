import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    enable_dataset_camera = LaunchConfiguration(
        "enable_dataset_camera"
    )

    package_share = get_package_share_directory(
        "diff_drive_description"
    )

    xacro_file = os.path.join(
        package_share,
        "urdf",
        "diff_drive.urdf.xacro",
    )

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                xacro_file,
                " enable_dataset_camera:=",
                enable_dataset_camera,
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": ParameterValue(
                    use_sim_time,
                    value_type=bool,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use the Gazebo simulation clock",
            ),
            DeclareLaunchArgument(
                "enable_dataset_camera",
                default_value="false",
                description=(
                    "Add the semantic segmentation camera used for "
                    "synthetic dataset generation"
                ),
            ),
            robot_state_publisher_node,
        ]
    )
