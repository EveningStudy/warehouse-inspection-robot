import os
from pathlib import Path
import signal
import time

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    IncludeLaunchDescription,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _cleanup_collection_gazebo(context, world_file):
    """Stop an orphaned headless Gazebo process for this exact world."""
    matching_pids = []
    for proc_entry in Path('/proc').iterdir():
        if not proc_entry.name.isdigit():
            continue
        try:
            arguments = [
                value.decode(errors='replace')
                for value in (proc_entry / 'cmdline').read_bytes().split(b'\0')
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue

        command = ' '.join(arguments)
        launches_gazebo = (
            'gz sim' in command
            and world_file in command
            and ' -s ' in f' {command} '
        )
        if launches_gazebo:
            matching_pids.append(int(proc_entry.name))

    for process_id in matching_pids:
        try:
            os.kill(process_id, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + 2.0
    remaining = set(matching_pids)
    while remaining and time.monotonic() < deadline:
        for process_id in tuple(remaining):
            if not Path(f'/proc/{process_id}').exists():
                remaining.remove(process_id)
        if remaining:
            time.sleep(0.05)

    for process_id in remaining:
        try:
            os.kill(process_id, signal.SIGKILL)
        except ProcessLookupError:
            pass

    if matching_pids:
        return [
            LogInfo(
                msg=(
                    'Stopped stale dataset Gazebo processes: '
                    + ', '.join(str(pid) for pid in matching_pids)
                )
            )
        ]
    return []


def generate_launch_description():
    description_share = get_package_share_directory(
        'diff_drive_description'
    )
    missions_share = get_package_share_directory('autonomy_missions')
    warehouse_world = os.path.join(
        description_share,
        'worlds',
        'inspection_warehouse.sdf',
    )

    warehouse_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                description_share,
                'launch',
                'warehouse_navigation.launch.py',
            )
        ),
        launch_arguments={
            'headless': LaunchConfiguration('headless'),
            'use_rviz': LaunchConfiguration('use_rviz'),
            'enable_dataset_camera': 'true',
        }.items(),
    )

    collector = Node(
        package='robot_perception',
        executable='yolo_dataset_collector',
        name='yolo_dataset_collector',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'output_root': LaunchConfiguration('output_root'),
                'positive_sample_period': LaunchConfiguration(
                    'positive_sample_period'
                ),
                'negative_sample_period': LaunchConfiguration(
                    'negative_sample_period'
                ),
            }
        ],
    )

    route_file = os.path.join(
        missions_share,
        'config',
        'yolo_collection_route.yaml',
    )
    dataset_patrol = Node(
        package='autonomy_missions',
        executable='patrol',
        name='patrol_node',
        output='screen',
        parameters=[
            {
                'use_sim_time': True,
                'map_profile': 'dataset',
                'waypoint_file': route_file,
                'wait_seconds': 1.5,
                'max_retries_per_waypoint': 1,
                'continue_on_failure': True,
                'max_goal_rejection_retries': 30,
                'goal_rejection_retry_seconds': 1.0,
                'waypoint_timeout_seconds': 120.0,
                'feedback_log_interval_seconds': 3.0,
            }
        ],
    )

    finish_when_patrol_exits = RegisterEventHandler(
        OnProcessExit(
            target_action=dataset_patrol,
            on_exit=[
                # ros_gz_sim starts Gazebo through a Ruby shell wrapper.  The
                # wrapper may exit while its gz-sim child remains alive, so
                # stop both before asking launch to shut the rest down.
                OpaqueFunction(
                    function=_cleanup_collection_gazebo,
                    args=[warehouse_world],
                ),
                LogInfo(
                    msg=(
                        'Dataset route finished. Finalizing train/val/test '
                        'and preview before shutdown.'
                    )
                ),
                EmitEvent(
                    event=Shutdown(
                        reason='Synthetic YOLO dataset route completed'
                    )
                ),
            ],
        )
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'headless',
                default_value='true',
                description=(
                    'Run Gazebo server-only with EGL rendering for faster '
                    'data generation'
                ),
            ),
            DeclareLaunchArgument(
                'use_rviz',
                default_value='false',
                description='Open RViz while collecting data',
            ),
            DeclareLaunchArgument(
                'output_root',
                default_value=os.path.expanduser(
                    '~/.ros/autonomy_lab/datasets/inspection_panel'
                ),
                description='Root directory for timestamped dataset runs',
            ),
            DeclareLaunchArgument(
                'positive_sample_period',
                default_value='0.5',
                description='Seconds of simulation time between positives',
            ),
            DeclareLaunchArgument(
                'negative_sample_period',
                default_value='2.0',
                description='Seconds of simulation time between negatives',
            ),
            # If an earlier collection was force-closed, remove only a
            # headless Gazebo process using this exact warehouse world before
            # starting the new run.
            OpaqueFunction(
                function=_cleanup_collection_gazebo,
                args=[warehouse_world],
            ),
            warehouse_navigation,
            collector,
            dataset_patrol,
            finish_when_patrol_exits,
        ]
    )
