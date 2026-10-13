import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, package_share, ros_gz_sim_share):
    """Create actions after launch arguments have concrete string values."""
    world = LaunchConfiguration("world").perform(context)
    world_name = LaunchConfiguration("world_name").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    headless = (
        LaunchConfiguration("headless").perform(context).lower()
        in {"1", "true", "yes", "on"}
    )
    use_rviz = (
        LaunchConfiguration("use_rviz").perform(context).lower()
        in {"1", "true", "yes", "on"}
    )
    odom_input_topic = LaunchConfiguration(
        "odom_input_topic"
    ).perform(context)

    # Only accept a filename from this package's worlds directory. Keeping the
    # file and Gazebo world name separate also makes bridge topics explicit.
    if os.path.basename(world) != world:
        raise RuntimeError(
            "The 'world' argument must be a filename from the package's "
            "worlds directory, for example inspection_warehouse.sdf"
        )

    world_file = os.path.join(package_share, "worlds", world)
    if not os.path.isfile(world_file):
        raise RuntimeError(f"Gazebo world file does not exist: {world_file}")

    # The world name is part of several Gazebo transport topics. Detect a
    # filename/name mismatch here instead of silently creating broken bridges.
    try:
        sdf_root = ET.parse(world_file).getroot()
        sdf_world = sdf_root.find("world")
        actual_world_name = (
            sdf_world.get("name") if sdf_world is not None else None
        )
    except ET.ParseError as error:
        raise RuntimeError(
            f"Cannot parse Gazebo world file {world_file}: {error}"
        ) from error

    if actual_world_name != world_name:
        raise RuntimeError(
            f"world_name='{world_name}' does not match <world "
            f"name='{actual_world_name}'> in {world}"
        )

    models_directory = os.path.join(package_share, "models")
    state_publisher_launch_file = os.path.join(
        package_share,
        "launch",
        "state_publisher.launch.py",
    )

    # Gazebo resolves model://warehouse_shelf and the other local model URIs
    # through GZ_SIM_RESOURCE_PATH.
    gazebo_resource_path = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        models_directory,
    )

    gazebo_arguments = "-r -v 4"
    if headless:
        gazebo_arguments = "-s -r -v 3"

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                ros_gz_sim_share,
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={
            "gz_args": f"{gazebo_arguments} {world_file}",
        }.items(),
    )

    state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(state_publisher_launch_file),
        launch_arguments={
            "use_sim_time": "true",
            "enable_dataset_camera": LaunchConfiguration(
                "enable_dataset_camera"
            ),
        }.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_diff_drive_robot",
        output="screen",
        arguments=[
            "-world",
            world_name,
            "-topic",
            "/robot_description",
            "-name",
            "diff_drive_robot",
            "-x",
            LaunchConfiguration("spawn_x"),
            "-y",
            LaunchConfiguration("spawn_y"),
            "-z",
            LaunchConfiguration("spawn_z"),
            "-Y",
            LaunchConfiguration("spawn_yaw"),
        ],
    )

    joint_state_topic = (
        f"/world/{world_name}/model/diff_drive_robot/joint_state"
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            (
                "/model/diff_drive_robot/odometry"
                "@nav_msgs/msg/Odometry["
                "gz.msgs.Odometry"
            ),
            (
                "/model/diff_drive_robot/ground_truth_odometry"
                "@nav_msgs/msg/Odometry["
                "gz.msgs.Odometry"
            ),
            (
                f"{joint_state_topic}"
                "@sensor_msgs/msg/JointState["
                "gz.msgs.Model"
            ),
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            (
                "/camera/depth_image"
                "@sensor_msgs/msg/Image["
                "gz.msgs.Image"
            ),
            (
                "/camera/camera_info"
                "@sensor_msgs/msg/CameraInfo["
                "gz.msgs.CameraInfo"
            ),
            (
                "/camera/segmentation/labels_map"
                "@sensor_msgs/msg/Image["
                "gz.msgs.Image"
            ),
            (
                "/camera/segmentation/colored_map"
                "@sensor_msgs/msg/Image["
                "gz.msgs.Image"
            ),
        ],
        remappings=[
            ("/model/diff_drive_robot/odometry", "/gz_odom"),
            (
                "/model/diff_drive_robot/ground_truth_odometry",
                "/ground_truth_odom",
            ),
            (joint_state_topic, "/joint_states"),
        ],
    )

    # Keep command callbacks independent of RGB-D traffic. Start this bridge
    # after spawning so the DiffDrive transport subscriber already exists.
    command_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="command_bridge",
        output="screen",
        arguments=[
            "/model/diff_drive_robot/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
        ],
        remappings=[("/model/diff_drive_robot/cmd_vel", "/cmd_vel")],
    )
    command_bridge_after_spawn = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[command_bridge])
    )

    odom_adapter = Node(
        package="diff_drive_description",
        executable="odom_adapter.py",
        name="odom_adapter",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "input_topic": odom_input_topic,
            }
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
    )

    actions = [
        gazebo_resource_path,
    ]

    if headless:
        # Ogre2 renders RGB-D and lidar sensors through EGL without opening a
        # Gazebo GUI window. This substantially reduces WSL desktop overhead.
        actions.append(
            SetEnvironmentVariable("GZ_SIM_HEADLESS_RENDERING", "1")
        )

    actions.extend(
        [
            gazebo,
            state_publisher,
            command_bridge_after_spawn,
            spawn_robot,
            bridge,
            odom_adapter,
        ]
    )

    if use_rviz:
        actions.append(rviz)

    return actions


def generate_launch_description():
    package_share = get_package_share_directory("diff_drive_description")
    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")

    default_rviz_config_file = os.path.join(
        package_share,
        "rviz",
        "display.rviz",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value="inspection_warehouse.sdf",
                description="SDF filename in diff_drive_description/worlds",
            ),
            DeclareLaunchArgument(
                "world_name",
                default_value="inspection_warehouse",
                description=(
                    "Name inside the SDF <world> tag; must match the file"
                ),
            ),
            DeclareLaunchArgument(
                "spawn_x",
                default_value="-7.45",
                description="Robot initial X position in metres",
            ),
            DeclareLaunchArgument(
                "spawn_y",
                default_value="-5.85",
                description="Robot initial Y position in metres",
            ),
            DeclareLaunchArgument(
                "spawn_z",
                default_value="0.30",
                description="Robot initial Z position in metres",
            ),
            DeclareLaunchArgument(
                "spawn_yaw",
                default_value="0.0",
                description="Robot initial yaw in radians",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=default_rviz_config_file,
                description="RViz configuration file to load",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description=(
                    "Run only the Gazebo server and use EGL rendering"
                ),
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="true",
                description="Start RViz together with the simulation",
            ),
            DeclareLaunchArgument(
                "odom_input_topic",
                default_value="/gz_odom",
                description=(
                    "Gazebo odometry topic converted into ROS /odom and TF"
                ),
            ),
            DeclareLaunchArgument(
                "enable_dataset_camera",
                default_value="false",
                description=(
                    "Enable the co-located semantic camera and its ROS "
                    "image bridges"
                ),
            ),
            OpaqueFunction(
                function=_launch_setup,
                args=[package_share, ros_gz_sim_share],
            ),
        ]
    )
