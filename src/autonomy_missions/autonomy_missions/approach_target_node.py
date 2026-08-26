import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


class ApproachTargetNode(Node):
    """Navigate to a safe pose in front of a perceived target."""

    def __init__(self):
        super().__init__("approach_target_node")

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("standoff_distance", 0.8)

        self.target_frame = str(
            self.get_parameter("target_frame").value
        )
        self.robot_base_frame = str(
            self.get_parameter("robot_base_frame").value
        )
        self.standoff_distance = float(
            self.get_parameter("standoff_distance").value
        )

        if self.standoff_distance <= 0.0:
            raise ValueError("standoff_distance must be positive")

        # TF gives the robot's current pose in the map frame.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        # Nav2 exposes the long-running navigation task as an Action.
        self.action_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose",
        )

        # The perception node publishes the detected target in map.
        self.target_subscription = self.create_subscription(
            PointStamped,
            "/perception/target_point",
            self.target_callback,
            10,
        )

        # Publish the computed standoff pose for inspection in ROS / RViz.
        self.approach_goal_publisher = self.create_publisher(
            PoseStamped,
            "/perception/approach_goal",
            10,
        )

        self.latest_target = None
        self.goal_sent = False
        self.goal_handle = None
        self.finished = False
        self.feedback_counter = 0

        # Target messages, TF, and Nav2 may become ready at different times.
        self.start_timer = self.create_timer(
            0.5,
            self.try_start_navigation,
        )

        self.get_logger().info(
            "Approach target node started. "
            f"Waiting for a target in {self.target_frame}."
        )

    def target_callback(self, message):
        """Keep the newest valid target until one approach goal is sent."""
        if self.goal_sent or self.finished:
            return

        if message.header.frame_id != self.target_frame:
            self.get_logger().warning(
                "Ignoring target in unexpected frame: "
                f"'{message.header.frame_id}', expected "
                f"'{self.target_frame}'."
            )
            return

        if not (
            math.isfinite(message.point.x)
            and math.isfinite(message.point.y)
        ):
            self.get_logger().warning(
                "Ignoring target with non-finite coordinates."
            )
            return

        self.latest_target = message

    def try_start_navigation(self):
        """Wait for perception, TF, and Nav2, then send one goal."""
        if self.goal_sent or self.finished:
            return

        if self.latest_target is None:
            return

        if not self.action_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().info(
                "Waiting for Nav2 navigate_to_pose server..."
            )
            return

        try:
            robot_transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.robot_base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exception:
            self.get_logger().warning(
                "Waiting for robot pose in "
                f"{self.target_frame}: {exception}"
            )
            return

        calculation = self.calculate_approach_pose(
            robot_x=robot_transform.transform.translation.x,
            robot_y=robot_transform.transform.translation.y,
            target_x=self.latest_target.point.x,
            target_y=self.latest_target.point.y,
            target_frame=self.target_frame,
            stamp=self.get_clock().now().to_msg(),
            standoff_distance=self.standoff_distance,
        )

        if calculation is None:
            self.get_logger().error(
                "Robot and target positions are effectively identical; "
                "an approach direction cannot be calculated."
            )
            self.finished = True
            return

        goal_pose, target_distance, goal_yaw = calculation

        self.goal_sent = True
        self.start_timer.cancel()

        self.approach_goal_publisher.publish(goal_pose)

        self.get_logger().info(
            "Approach goal calculated: "
            f"target_distance={target_distance:.2f} m, "
            f"standoff={self.standoff_distance:.2f} m, "
            f"goal=({goal_pose.pose.position.x:.2f}, "
            f"{goal_pose.pose.position.y:.2f}), "
            f"yaw={goal_yaw:.2f} rad"
        )

        goal_message = NavigateToPose.Goal()
        goal_message.pose = goal_pose

        send_goal_future = self.action_client.send_goal_async(
            goal_message,
            feedback_callback=self.feedback_callback,
        )
        send_goal_future.add_done_callback(
            self.goal_response_callback
        )

    @staticmethod
    def calculate_approach_pose(
        robot_x,
        robot_y,
        target_x,
        target_y,
        target_frame,
        stamp,
        standoff_distance,
    ):
        """Return a pose standoff_distance before and facing the target."""
        delta_x = target_x - robot_x
        delta_y = target_y - robot_y
        target_distance = math.hypot(delta_x, delta_y)

        if target_distance < 1.0e-6:
            return None

        direction_x = delta_x / target_distance
        direction_y = delta_y / target_distance

        approach_x = target_x - standoff_distance * direction_x
        approach_y = target_y - standoff_distance * direction_y

        # The final robot heading points from the approach pose to the target.
        goal_yaw = math.atan2(
            target_y - approach_y,
            target_x - approach_x,
        )

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = target_frame
        goal_pose.header.stamp = stamp
        goal_pose.pose.position.x = float(approach_x)
        goal_pose.pose.position.y = float(approach_y)
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.z = math.sin(goal_yaw / 2.0)
        goal_pose.pose.orientation.w = math.cos(goal_yaw / 2.0)

        return goal_pose, target_distance, goal_yaw

    def goal_response_callback(self, future):
        """Handle Nav2 accepting or rejecting the approach request."""
        try:
            self.goal_handle = future.result()
        except Exception as exception:
            self.get_logger().error(
                f"Failed to send approach goal: {exception}"
            )
            self.finished = True
            return

        if self.goal_handle is None or not self.goal_handle.accepted:
            self.get_logger().error(
                "Nav2 rejected the approach goal."
            )
            self.finished = True
            return

        self.get_logger().info(
            "Nav2 accepted the approach goal."
        )

        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_message):
        """Report progress without flooding the terminal."""
        self.feedback_counter += 1
        if self.feedback_counter % 10 != 0:
            return

        feedback = feedback_message.feedback
        self.get_logger().info(
            "Approach progress: "
            f"{feedback.distance_remaining:.2f} m remaining, "
            f"recoveries={feedback.number_of_recoveries}"
        )

    def result_callback(self, future):
        """Handle the final Nav2 action result and finish this one-shot node."""
        try:
            wrapped_result = future.result()
        except Exception as exception:
            self.get_logger().error(
                f"Failed to receive approach result: {exception}"
            )
            self.finished = True
            return

        status = wrapped_result.status
        result = wrapped_result.result

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                "Robot reached the safe approach pose."
            )
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning(
                "Approach navigation was canceled."
            )
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(
                "Approach navigation was aborted: "
                f"error_code={result.error_code}, "
                f"error_msg='{result.error_msg}'"
            )
        else:
            self.get_logger().error(
                "Approach navigation ended with status "
                f"{status}: error_code={result.error_code}, "
                f"error_msg='{result.error_msg}'"
            )

        self.finished = True


def main(args=None):
    rclpy.init(args=args)
    node = ApproachTargetNode()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Approach node stopped.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
