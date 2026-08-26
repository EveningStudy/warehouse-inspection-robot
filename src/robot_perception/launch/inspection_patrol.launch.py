import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    perception_share = get_package_share_directory('robot_perception')
    description_share = get_package_share_directory(
        'diff_drive_description'
    )
    missions_share = get_package_share_directory('autonomy_missions')

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
            # Semantic labels were only ground truth for dataset generation.
            # Runtime perception receives the ordinary RGB-D camera only.
            'enable_dataset_camera': 'false',
        }.items(),
    )

    yolo_detector = Node(
        package='robot_perception',
        executable='yolo_target_detector',
        name='yolo_target_detector_node',
        output='screen',
        parameters=[
            os.path.join(
                perception_share,
                'config',
                'yolo_detector.yaml',
            ),
            {
                'use_sim_time': True,
                'model_path': LaunchConfiguration('model_path'),
                'confidence_threshold': LaunchConfiguration(
                    'confidence_threshold'
                ),
                'maximum_inference_rate': LaunchConfiguration(
                    'maximum_inference_rate'
                ),
            },
        ],
    )

    inspection_patrol = Node(
        package='autonomy_missions',
        executable='inspection_patrol',
        name='inspection_patrol_node',
        output='screen',
        parameters=[
            os.path.join(
                missions_share,
                'config',
                'inspection_patrol.yaml',
            ),
            {
                'use_sim_time': True,
                'waypoint_file': os.path.join(
                    missions_share,
                    'config',
                    'warehouse_patrol.yaml',
                ),
                'record_directory': LaunchConfiguration(
                    'record_directory'
                ),
            },
        ],
    )

    debug_image_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='yolo_debug_image_view',
        arguments=['/perception/yolo_debug_image'],
        condition=IfCondition(LaunchConfiguration('show_debug_image')),
        output='screen',
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'headless',
                default_value='false',
                description='Run Gazebo without its GUI window',
            ),
            DeclareLaunchArgument(
                'use_rviz',
                default_value='true',
                description='Start RViz with the warehouse map',
            ),
            DeclareLaunchArgument(
                'show_debug_image',
                default_value='true',
                description='Open rqt_image_view with annotated YOLO RGB',
            ),
            DeclareLaunchArgument(
                'model_path',
                default_value=os.path.join(
                    perception_share,
                    'models',
                    'inspection_panel_yolo26n.pt',
                ),
                description='Inspection-panel YOLO checkpoint',
            ),
            DeclareLaunchArgument(
                'confidence_threshold',
                default_value='0.45',
                description='Fixed deployment confidence threshold',
            ),
            DeclareLaunchArgument(
                'maximum_inference_rate',
                default_value='5.0',
                description='Maximum GPU inference callbacks per second',
            ),
            DeclareLaunchArgument(
                'record_directory',
                default_value=os.path.expanduser(
                    '~/.ros/autonomy_lab/inspection_records'
                ),
                description='Parent directory for timestamped mission logs',
            ),
            warehouse_navigation,
            yolo_detector,
            inspection_patrol,
            debug_image_view,
        ]
    )
