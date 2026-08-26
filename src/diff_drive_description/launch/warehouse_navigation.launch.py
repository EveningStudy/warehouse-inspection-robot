import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    package_share = get_package_share_directory(
        "diff_drive_description"
    )
    nav2_bringup_share = get_package_share_directory(
        "nav2_bringup"
    )

    simulation_launch_file = os.path.join(
        package_share,
        "launch",
        "simulation.launch.py",
    )
    nav2_bringup_launch_file = os.path.join(
        nav2_bringup_share,
        "launch",
        "bringup_launch.py",
    )
    map_file = os.path.join(
        package_share,
        "maps",
        "inspection_warehouse_map.yaml",
    )
    nav2_params_file = os.path.join(
        package_share,
        "config",
        "nav2_params.yaml",
    )
    rviz_config_file = os.path.join(
        package_share,
        "rviz",
        "nav2.rviz",
    )

    # AMCL normally waits for RViz's /initialpose message. For this fixed
    # simulation starting area, provide the known map pose automatically while
    # keeping the parameters visible and editable in this launch file.
    configured_nav2_params = RewrittenYaml(
        source_file=nav2_params_file,
        param_rewrites={
            "amcl.ros__parameters.set_initial_pose": "true",
            "amcl.ros__parameters.initial_pose.x": "-7.45",
            "amcl.ros__parameters.initial_pose.y": "-5.85",
            "amcl.ros__parameters.initial_pose.z": "0.0",
            "amcl.ros__parameters.initial_pose.yaw": "0.0",
        },
        convert_types=True,
    )

    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(simulation_launch_file),
        launch_arguments={
            "world": "inspection_warehouse.sdf",
            "world_name": "inspection_warehouse",
            "spawn_x": "-7.45",
            "spawn_y": "-5.85",
            "spawn_z": "0.30",
            "spawn_yaw": "0.0",
            # Use Gazebo's measured world pose for repeatable autonomous
            # navigation validation. The differential-drive plugin's wheel
            # odometry accumulates substantial simulated slip over this long
            # patrol and can make AMCL lose the robot even though Nav2 is
            # otherwise behaving correctly.
            "odom_input_topic": "/ground_truth_odom",
            "headless": LaunchConfiguration("headless"),
            "use_rviz": LaunchConfiguration("use_rviz"),
            "rviz_config": rviz_config_file,
            "enable_dataset_camera": LaunchConfiguration(
                "enable_dataset_camera"
            ),
        }.items(),
    )

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_bringup_launch_file),
        launch_arguments={
            "slam": "False",
            "map": map_file,
            "use_sim_time": "True",
            "params_file": configured_nav2_params,
            "autostart": "True",
            "use_composition": "False",
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run Gazebo without its GUI window",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz with the Nav2 configuration",
            ),
            DeclareLaunchArgument(
                "enable_dataset_camera",
                default_value="false",
                description=(
                    "Enable Gazebo semantic ground-truth images for "
                    "dataset collection"
                ),
            ),
            simulation,
            navigation,
        ]
    )
