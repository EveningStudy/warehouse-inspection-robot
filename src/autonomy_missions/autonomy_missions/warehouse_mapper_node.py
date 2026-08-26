import math

import rclpy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def normalize_angle(angle):
    """Wrap an angle to the range [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(quaternion):
    """Extract planar yaw from a geometry_msgs quaternion."""
    sin_yaw = 2.0 * (
        quaternion.w * quaternion.z
        + quaternion.x * quaternion.y
    )
    cos_yaw = 1.0 - 2.0 * (
        quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )
    return math.atan2(sin_yaw, cos_yaw)


class WarehouseMapper(Node):
    """Drive a repeatable ground-truth-relative SLAM collection route."""

    def __init__(self):
        super().__init__("warehouse_mapper")

        self.declare_parameter("maximum_linear_speed", 0.22)
        self.declare_parameter("maximum_angular_speed", 0.40)
        self.declare_parameter("waypoint_tolerance", 0.18)
        self.declare_parameter("front_stop_distance", 0.48)
        self.declare_parameter("self_filter_distance", 0.28)
        self.declare_parameter("pose_topic", "/ground_truth_odom")
        self.declare_parameter("route_mode", "warehouse")
        self.declare_parameter("start_waypoint", 0)
        self.declare_parameter("use_current_pose_as_origin", True)
        self.declare_parameter("origin_x", 0.0)
        self.declare_parameter("origin_y", 0.0)
        self.declare_parameter("origin_yaw", 0.0)

        self.maximum_linear_speed = float(
            self.get_parameter("maximum_linear_speed").value
        )
        self.maximum_angular_speed = float(
            self.get_parameter("maximum_angular_speed").value
        )
        self.waypoint_tolerance = float(
            self.get_parameter("waypoint_tolerance").value
        )
        self.front_stop_distance = float(
            self.get_parameter("front_stop_distance").value
        )
        self.self_filter_distance = float(
            self.get_parameter("self_filter_distance").value
        )
        self.pose_topic = str(
            self.get_parameter("pose_topic").value
        )
        self.route_mode = str(
            self.get_parameter("route_mode").value
        )
        self.start_waypoint = int(
            self.get_parameter("start_waypoint").value
        )
        self.use_current_pose_as_origin = bool(
            self.get_parameter("use_current_pose_as_origin").value
        )
        self.configured_origin = (
            float(self.get_parameter("origin_x").value),
            float(self.get_parameter("origin_y").value),
            float(self.get_parameter("origin_yaw").value),
        )

        # Coordinates are metres in the robot's initial ground-truth frame.
        # The route covers the outer loop, service bay, north room, both shelf
        # aisles, workbenches and then returns to the charging area.
        self.waypoints = [
            (3.50, 0.25),
            (8.00, 0.25),
            (14.70, 0.25),
            (14.70, 5.10),
            (15.30, 5.10),
            (15.30, 8.90),
            (15.30, 5.10),
            (10.80, 5.10),
            (7.80, 5.10),
            (7.60, 9.15),
            (7.60, 12.20),
            (7.60, 11.45),
            (10.00, 11.45),
            (10.00, 12.20),
            (13.20, 12.20),
            (14.85, 12.20),
            (10.00, 12.20),
            (10.00, 11.45),
            (7.60, 11.45),
            (7.60, 12.20),
            (-0.05, 12.20),
            (0.55, 9.50),
            (0.55, 8.20),
            (0.55, 3.35),
            (-0.05, 2.85),
            (-0.05, 0.35),
            (2.55, 2.85),
            (2.55, 8.85),
            (4.75, 8.85),
            (4.75, 2.85),
            (7.45, 2.45),
            (11.65, 2.45),
            (11.75, 5.85),
            (7.45, 5.85),
            (4.95, 5.85),
            (7.45, 5.85),
            (11.75, 5.85),
            (11.75, 9.25),
            (7.45, 9.25),
            (4.75, 8.85),
            (4.55, 8.85),
            (4.55, 5.00),
            (4.55, 3.80),
            (4.75, 2.85),
            (4.75, 0.35),
            (0.00, 0.35),
            (0.00, 0.00),
        ]

        if self.route_mode == "calibration_square":
            self.waypoints = [
                (2.00, 0.00),
                (2.00, 2.00),
                (0.00, 2.00),
                (0.00, 0.00),
            ]
        elif self.route_mode != "warehouse":
            raise ValueError(
                "route_mode must be 'warehouse' or "
                "'calibration_square'"
            )

        self.velocity_publisher = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )
        self.create_subscription(
            Odometry,
            self.pose_topic,
            self.odom_callback,
            20,
        )
        self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.start_pose = None
        self.current_pose = None
        self.front_range = math.inf
        if not 0 <= self.start_waypoint < len(self.waypoints):
            raise ValueError(
                "start_waypoint must be between 0 and "
                f"{len(self.waypoints) - 1}"
            )
        self.waypoint_index = self.start_waypoint
        self.hold_until = None
        self.finished = False
        self.blocked_cycles = 0

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info(
            f"Warehouse mapping route contains {len(self.waypoints)} "
            "ground-truth-relative waypoints; "
            f"mode={self.route_mode}; pose topic={self.pose_topic}"
        )

    def odom_callback(self, message):
        position = message.pose.pose.position
        yaw = yaw_from_quaternion(message.pose.pose.orientation)
        pose = (position.x, position.y, yaw)

        if self.start_pose is None:
            if self.use_current_pose_as_origin:
                self.start_pose = pose
            else:
                self.start_pose = self.configured_origin
            self.get_logger().info(
                "Mapping origin: "
                f"x={self.start_pose[0]:.2f}, "
                f"y={self.start_pose[1]:.2f}, "
                f"yaw={self.start_pose[2]:.2f}; "
                f"current odom=({pose[0]:.2f}, {pose[1]:.2f})"
            )

        self.current_pose = pose

    def scan_callback(self, message):
        if not message.ranges:
            return

        half_angle = math.radians(22.0)
        valid_ranges = []
        for index, distance in enumerate(message.ranges):
            angle = message.angle_min + index * message.angle_increment
            if abs(normalize_angle(angle)) > half_angle:
                continue
            minimum_range = max(
                message.range_min,
                self.self_filter_distance,
            )
            if math.isfinite(distance) and minimum_range < distance:
                valid_ranges.append(distance)

        self.front_range = min(valid_ranges, default=math.inf)

    def relative_pose(self):
        start_x, start_y, start_yaw = self.start_pose
        current_x, current_y, current_yaw = self.current_pose
        delta_x = current_x - start_x
        delta_y = current_y - start_y

        cosine = math.cos(start_yaw)
        sine = math.sin(start_yaw)
        relative_x = cosine * delta_x + sine * delta_y
        relative_y = -sine * delta_x + cosine * delta_y
        relative_yaw = normalize_angle(current_yaw - start_yaw)
        return relative_x, relative_y, relative_yaw

    def stop_robot(self):
        self.velocity_publisher.publish(Twist())

    def control_loop(self):
        if self.finished or self.current_pose is None:
            return

        now_nanoseconds = self.get_clock().now().nanoseconds
        if self.hold_until is not None:
            if now_nanoseconds < self.hold_until:
                self.stop_robot()
                return
            self.hold_until = None

        if self.waypoint_index >= len(self.waypoints):
            self.stop_robot()
            self.finished = True
            self.get_logger().info(
                "MAPPING_ROUTE_COMPLETE: robot returned to the start area"
            )
            return

        current_x, current_y, current_yaw = self.relative_pose()
        target_x, target_y = self.waypoints[self.waypoint_index]
        delta_x = target_x - current_x
        delta_y = target_y - current_y
        distance = math.hypot(delta_x, delta_y)

        if distance <= self.waypoint_tolerance:
            self.stop_robot()
            self.get_logger().info(
                f"Reached waypoint {self.waypoint_index + 1}/"
                f"{len(self.waypoints)} at "
                f"({current_x:.2f}, {current_y:.2f})"
            )
            self.waypoint_index += 1
            self.hold_until = now_nanoseconds + 700_000_000
            return

        target_heading = math.atan2(delta_y, delta_x)
        heading_error = normalize_angle(target_heading - current_yaw)

        command = Twist()

        # Rotate before translating when the next path segment lies far to the
        # side. This produces clean scans and avoids wide corner-cutting.
        if abs(heading_error) > 0.32:
            command.angular.z = max(
                -self.maximum_angular_speed,
                min(self.maximum_angular_speed, 1.5 * heading_error),
            )
            self.blocked_cycles = 0
        elif self.front_range < self.front_stop_distance:
            # The route was designed with clearance. If something is directly
            # ahead, stop rather than forcing the robot through a collision.
            self.blocked_cycles += 1
            if self.blocked_cycles % 40 == 1:
                self.get_logger().warning(
                    f"Obstacle {self.front_range:.2f} m ahead near waypoint "
                    f"{self.waypoint_index + 1}; waiting"
                )
        else:
            self.blocked_cycles = 0
            command.linear.x = min(
                self.maximum_linear_speed,
                max(0.08, 0.45 * distance),
            )
            command.angular.z = max(
                -self.maximum_angular_speed,
                min(self.maximum_angular_speed, 1.8 * heading_error),
            )

        self.velocity_publisher.publish(command)


def main(args=None):
    rclpy.init(args=args)
    node = WarehouseMapper()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node.stop_robot()
            except Exception:
                # SIGINT can invalidate the publisher context between the
                # rclpy.ok() check and publish. The simulator's controller
                # already times out stale commands, so cleanup can continue.
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
