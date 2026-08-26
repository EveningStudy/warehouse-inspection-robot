import sys

import rclpy

from .patrol_node import PatrolNode


def main(args=None):
    """Start PatrolNode with the warehouse waypoint profile by default."""
    ros_args = list(sys.argv[1:] if args is None else args)

    # Preserve any user-provided ROS arguments. Only inject the warehouse
    # profile when the caller did not explicitly choose another profile.
    has_profile = any(
        argument.startswith("map_profile:=")
        or argument == "map_profile"
        for argument in ros_args
    )
    if not has_profile:
        ros_args.extend(
            ["--ros-args", "-p", "map_profile:=warehouse"]
        )

    rclpy.init(args=ros_args)
    node = PatrolNode()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Warehouse patrol stopped by user.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
