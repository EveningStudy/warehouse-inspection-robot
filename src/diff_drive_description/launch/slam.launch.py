import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    package_share = get_package_share_directory(
        "diff_drive_description"
    )

    slam_toolbox_share = get_package_share_directory(
        "slam_toolbox"
    )

    simulation_launch_file = os.path.join(
        package_share,
        "launch",
        "simulation.launch.py",
    )

    slam_launch_file = os.path.join(
        slam_toolbox_share,
        "launch",
        "online_async_launch.py",
    )

    slam_params_file = os.path.join(
        package_share,
        "config",
        "slam_toolbox.yaml",
    )

    slam_rviz_config_file = os.path.join(
        package_share,
        "rviz",
        "slam.rviz",
    )

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            simulation_launch_file
        ),
        launch_arguments={
            "rviz_config": slam_rviz_config_file,
            # Static map generation should not be corrupted by wheel slip in
            # the simplified fixed-caster simulation. Navigation keeps using
            # the default wheel-derived /gz_odom in navigation.launch.py.
            "odom_input_topic": "/ground_truth_odom",
        }.items(),
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            slam_launch_file
        ),
        launch_arguments={
            "use_sim_time": "true",
            "slam_params_file": slam_params_file,
        }.items(),
    )

    return LaunchDescription(
        [
            simulation,
            slam_toolbox,
        ]
    )
