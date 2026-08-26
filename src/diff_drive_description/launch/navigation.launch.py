import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # 找到我们自己的 ROS 2 包安装位置。
    package_share = get_package_share_directory(
        "diff_drive_description"
    )

    # 找到官方 Nav2 bringup 包的安装位置。
    nav2_bringup_share = get_package_share_directory(
        "nav2_bringup"
    )

    # 已有的 Gazebo 仿真启动文件。
    simulation_launch_file = os.path.join(
        package_share,
        "launch",
        "simulation.launch.py",
    )

    # Nav2 官方总启动文件。
    nav2_bringup_launch_file = os.path.join(
        nav2_bringup_share,
        "launch",
        "bringup_launch.py",
    )

    # 我们保存的静态地图。
    map_file = os.path.join(
        package_share,
        "maps",
        "room_map.yaml",
    )

    # 我们刚复制并修改的 Nav2 参数。
    nav2_params_file = os.path.join(
        package_share,
        "config",
        "nav2_params.yaml",
    )

    # Nav2 专用 RViz 界面。
    nav2_rviz_config_file = os.path.join(
        package_share,
        "rviz",
        "nav2.rviz",
    )

    # 启动现有 Gazebo 仿真，并让它加载 Nav2 的 RViz 配置。
    simulation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            simulation_launch_file
        ),
        launch_arguments={
            # This launch file still uses room_map, so deliberately start the
            # original matching world rather than the new warehouse.
            "world": "robot_world.sdf",
            "world_name": "robot_world",
            "spawn_x": "0.0",
            "spawn_y": "0.0",
            "spawn_yaw": "0.0",
            "rviz_config": nav2_rviz_config_file,
        }.items(),
    )

    # 启动地图服务器、AMCL 和整个 Nav2 导航系统。
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            nav2_bringup_launch_file
        ),
        launch_arguments={
            # false 表示不重新建图，而是使用 AMCL 和现成地图。
            "slam": "False",

            # 交给 Map Server 加载的地图文件。
            "map": map_file,

            # Gazebo 使用仿真时钟，因此 Nav2 也必须使用 /clock。
            "use_sim_time": "True",

            # 使用我们自己的机器人参数。
            "params_file": nav2_params_file,

            # 自动把 Nav2 生命周期节点切换到 active 状态。
            "autostart": "True",

            # 初学阶段让各节点作为独立进程运行，方便观察和排错。
            "use_composition": "False",
        }.items(),
    )

    return LaunchDescription(
        [
            simulation,
            navigation,
        ]
    )
