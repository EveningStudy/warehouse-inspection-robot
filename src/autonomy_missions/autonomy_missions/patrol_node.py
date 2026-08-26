import math
import os
import time

import yaml

from ament_index_python.packages import get_package_share_directory

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class PatrolNode(Node):
    def __init__(self):
        super().__init__("patrol_node")

        # 每到达一个巡逻点后，停留多长时间。
        self.declare_parameter("wait_seconds", 2.0)
        self.wait_seconds = float(
            self.get_parameter("wait_seconds").value
        )

        # A patrol should not disappear because one navigation attempt fails.
        # Retry a failed waypoint once, then optionally skip it and continue.
        self.declare_parameter("max_retries_per_waypoint", 1)
        self.max_retries_per_waypoint = max(
            0,
            int(
                self.get_parameter(
                    "max_retries_per_waypoint"
                ).value
            ),
        )
        self.declare_parameter("continue_on_failure", True)
        self.continue_on_failure = bool(
            self.get_parameter("continue_on_failure").value
        )

        # The NavigateToPose action server can be discoverable while Nav2 is
        # still moving its lifecycle nodes into the active state.  A goal sent
        # during that short window is rejected even though navigation will be
        # ready a moment later.  Retry only rejected submissions, with a
        # separate bounded counter so a real configuration error cannot loop
        # forever.
        self.declare_parameter("max_goal_rejection_retries", 30)
        self.max_goal_rejection_retries = max(
            0,
            int(
                self.get_parameter(
                    "max_goal_rejection_retries"
                ).value
            ),
        )
        self.declare_parameter("goal_rejection_retry_seconds", 1.0)
        self.goal_rejection_retry_seconds = max(
            0.1,
            float(
                self.get_parameter(
                    "goal_rejection_retry_seconds"
                ).value
            ),
        )

        # Bound each attempt using wall-clock time.  This is intentionally not
        # simulation time: when Gazebo runs slowly, an unreachable goal should
        # still be canceled in a predictable amount of real time.
        self.declare_parameter("waypoint_timeout_seconds", 120.0)
        self.waypoint_timeout_seconds = max(
            0.0,
            float(
                self.get_parameter(
                    "waypoint_timeout_seconds"
                ).value
            ),
        )

        # Nav2 can publish feedback many times per second.  Logging every few
        # callbacks wastes CPU and makes the useful mission events hard to see.
        self.declare_parameter("feedback_log_interval_seconds", 2.0)
        self.feedback_log_interval_seconds = max(
            0.0,
            float(
                self.get_parameter(
                    "feedback_log_interval_seconds"
                ).value
            ),
        )

        self.declare_parameter("map_profile", "legacy")
        self.declare_parameter("waypoint_file", "")
        map_profile = str(
            self.get_parameter("map_profile").value
        )
        waypoint_file = str(
            self.get_parameter("waypoint_file").value
        )

        # 巡逻点列表。
        #
        # 每一项依次是：
        # (名称, x, y, yaw)
        #
        # x、y 单位是米，属于 map 坐标系。
        # yaw 单位是弧度。
        legacy_waypoints = [
            ("Point A", 1.5, 1.5, 0.0),
            ("Point B", -1.5, 1.5, 1.5708),
            ("Point C", -1.5, -1.5, 3.1416),
            ("Point D", 1.5, -1.5, -1.5708),
            ("Home", 0.0, 0.0, 0.0),
        ]

        if map_profile == "warehouse" and not waypoint_file:
            waypoint_file = os.path.join(
                get_package_share_directory("autonomy_missions"),
                "config",
                "warehouse_patrol.yaml",
            )

        if waypoint_file:
            self.waypoints = self.load_waypoints(waypoint_file)
        else:
            self.waypoints = legacy_waypoints

        self.get_logger().info(
            f"Loaded {len(self.waypoints)} patrol waypoints "
            f"(profile={map_profile})"
        )

        # 当前正在执行第几个巡逻点。
        self.current_index = 0

        # 当前 Nav2 目标的句柄。
        self.goal_handle = None

        # 整个巡逻任务是否已经结束。
        self.finished = False

        # Wall-clock state for feedback throttling and the waypoint watchdog.
        self.goal_started_at = None
        self.last_feedback_log_at = 0.0
        self.timeout_cancel_requested = False

        # 当前巡逻点已经重试了多少次。
        self.retry_count = 0

        # Number of consecutive action-server rejections for the current
        # waypoint.  This is distinct from a goal that was accepted and later
        # failed while navigating.
        self.goal_rejection_count = 0

        # 到达目标后使用的等待定时器。
        self.wait_timer = None

        # 创建 NavigateToPose Action Client。
        self.action_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose",
        )

        # 每秒检查一次 Nav2 是否已经准备好。
        self.server_timer = self.create_timer(
            1.0,
            self.wait_for_nav2,
        )

        self.get_logger().info(
            f"Patrol node started with "
            f"{len(self.waypoints)} waypoints."
        )

    def load_waypoints(self, waypoint_file):
        """Load (name, x, y, yaw) tuples from a ROS parameter YAML file."""
        if not os.path.isfile(waypoint_file):
            raise FileNotFoundError(
                f"Patrol waypoint file does not exist: {waypoint_file}"
            )

        with open(waypoint_file, "r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream) or {}

        try:
            entries = document["patrol_node"]["ros__parameters"][
                "waypoints"
            ]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Waypoint YAML must contain "
                "patrol_node.ros__parameters.waypoints"
            ) from error

        waypoints = []
        for index, entry in enumerate(entries):
            try:
                waypoints.append(
                    (
                        str(entry["name"]),
                        float(entry["x"]),
                        float(entry["y"]),
                        float(entry["yaw"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid waypoint entry at index {index}: {entry}"
                ) from error

        if not waypoints:
            raise ValueError(
                f"Waypoint file contains no waypoints: {waypoint_file}"
            )

        return waypoints

    def wait_for_nav2(self):
        if not self.action_client.wait_for_server(
            timeout_sec=0.0
        ):
            self.get_logger().info(
                "NavigateToPose server is not ready yet..."
            )
            return

        # Nav2 已经准备好，不再重复检查。
        self.server_timer.cancel()

        self.get_logger().info(
            "Nav2 is ready. Starting patrol."
        )

        self.send_current_waypoint()

    def send_current_waypoint(self):
        # 检查所有巡逻点是否都已经完成。
        if self.current_index >= len(self.waypoints):
            self.get_logger().info(
                "Patrol completed."
            )
            self.finished = True
            return

        name, x, y, yaw = self.waypoints[
            self.current_index
        ]

        goal_msg = NavigateToPose.Goal()

        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.position.z = 0.0

        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(
            yaw / 2.0
        )
        goal_msg.pose.pose.orientation.w = math.cos(
            yaw / 2.0
        )

        now = time.monotonic()
        self.goal_started_at = now
        self.last_feedback_log_at = 0.0
        self.timeout_cancel_requested = False

        self.get_logger().info(
            f"Sending waypoint "
            f"{self.current_index + 1}/"
            f"{len(self.waypoints)}: "
            f"{name}, "
            f"x={x:.2f}, "
            f"y={y:.2f}, "
            f"yaw={yaw:.2f}"
        )

        send_goal_future = (
            self.action_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback,
            )
        )

        send_goal_future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):
        try:
            self.goal_handle = future.result()
        except Exception as exception:
            self.get_logger().error(
                f"Failed to send waypoint: {exception}"
            )
            self.finished = True
            return

        if (
            self.goal_handle is None
            or not self.goal_handle.accepted
        ):
            self.goal_handle = None
            if (
                self.goal_rejection_count
                < self.max_goal_rejection_retries
            ):
                self.goal_rejection_count += 1
                self.get_logger().warning(
                    "Nav2 rejected the waypoint while it may still be "
                    "activating. Retrying the same waypoint in "
                    f"{self.goal_rejection_retry_seconds:.1f} seconds "
                    f"(rejection retry {self.goal_rejection_count}/"
                    f"{self.max_goal_rejection_retries})."
                )
                self.wait_timer = self.create_timer(
                    self.goal_rejection_retry_seconds,
                    self.start_next_waypoint,
                )
                return

            self.get_logger().error(
                "Nav2 kept rejecting the waypoint after "
                f"{self.max_goal_rejection_retries} retries. "
                "Stopping the patrol."
            )
            self.finished = True
            return

        self.goal_rejection_count = 0

        name, _, _, _ = self.waypoints[
            self.current_index
        ]

        self.get_logger().info(
            f"Nav2 accepted {name}."
        )

        result_future = (
            self.goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.result_callback
        )

    def feedback_callback(self, feedback_message):
        now = time.monotonic()

        if (
            self.waypoint_timeout_seconds > 0.0
            and self.goal_started_at is not None
            and not self.timeout_cancel_requested
            and now - self.goal_started_at
            >= self.waypoint_timeout_seconds
            and self.goal_handle is not None
        ):
            name, _, _, _ = self.waypoints[
                self.current_index
            ]
            self.timeout_cancel_requested = True
            self.get_logger().error(
                f"Navigation to {name} exceeded "
                f"{self.waypoint_timeout_seconds:.1f} seconds. "
                "Canceling this attempt."
            )
            self.goal_handle.cancel_goal_async()
            return

        if (
            self.feedback_log_interval_seconds > 0.0
            and now - self.last_feedback_log_at
            < self.feedback_log_interval_seconds
        ):
            return

        self.last_feedback_log_at = now

        feedback = feedback_message.feedback

        name, _, _, _ = self.waypoints[
            self.current_index
        ]

        estimated_seconds = (
            feedback.estimated_time_remaining.sec
            + feedback.estimated_time_remaining.nanosec
            / 1_000_000_000
        )

        self.get_logger().info(
            f"{name}: "
            f"distance remaining="
            f"{feedback.distance_remaining:.2f} m, "
            f"estimated time="
            f"{estimated_seconds:.1f} s, "
            f"recoveries="
            f"{feedback.number_of_recoveries}"
        )

    def result_callback(self, future):
        try:
            wrapped_result = future.result()
        except Exception as exception:
            self.get_logger().error(
                f"Failed to receive result: {exception}"
            )
            self.finished = True
            return

        status = wrapped_result.status
        result = wrapped_result.result

        name, _, _, _ = self.waypoints[
            self.current_index
        ]

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                f"Reached {name}."
            )

            # 移向列表中的下一个巡逻点。
            self.retry_count = 0
            self.current_index += 1

            # 如果所有点都完成，立即结束。
            if self.current_index >= len(
                self.waypoints
            ):
                self.get_logger().info(
                    "All waypoints completed. "
                    "Patrol finished."
                )
                self.finished = True
                return

            self.get_logger().info(
                f"Waiting {self.wait_seconds:.1f} "
                f"seconds before next waypoint."
            )

            # 创建定时器，在停留时间结束后发送下一个目标。
            self.wait_timer = self.create_timer(
                self.wait_seconds,
                self.start_next_waypoint,
            )
            return

        if status == GoalStatus.STATUS_CANCELED:
            if self.timeout_cancel_requested:
                self.get_logger().error(
                    f"Navigation to {name} timed out."
                )
            else:
                self.get_logger().warning(
                    f"Navigation to {name} was canceled."
                )
                self.finished = True
                return

        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(
                f"Navigation to {name} was aborted. "
                f"error_code={result.error_code}, "
                f"error_msg='{result.error_msg}'"
            )

        else:
            self.get_logger().error(
                f"Navigation to {name} ended with "
                f"status={status}, "
                f"error_code={result.error_code}, "
                f"error_msg='{result.error_msg}'"
            )

        if self.retry_count < self.max_retries_per_waypoint:
            self.retry_count += 1
            self.get_logger().warning(
                f"Retrying {name} after {self.wait_seconds:.1f} "
                f"seconds (retry {self.retry_count}/"
                f"{self.max_retries_per_waypoint})."
            )
            self.wait_timer = self.create_timer(
                self.wait_seconds,
                self.start_next_waypoint,
            )
            return

        if self.continue_on_failure:
            self.get_logger().error(
                f"Skipping {name} after "
                f"{self.retry_count} retries and continuing patrol."
            )
            self.retry_count = 0
            self.current_index += 1

            if self.current_index >= len(self.waypoints):
                self.get_logger().info(
                    "Patrol ended after the final waypoint failure."
                )
                self.finished = True
                return

            self.wait_timer = self.create_timer(
                self.wait_seconds,
                self.start_next_waypoint,
            )
            return

        self.finished = True

    def start_next_waypoint(self):
        # ROS 2 Timer 默认会周期性执行。
        # 我们只需要执行一次，所以立即取消它。
        if self.wait_timer is not None:
            self.wait_timer.cancel()
            self.wait_timer = None

        self.send_current_waypoint()


def main(args=None):
    rclpy.init(args=args)

    node = PatrolNode()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )
    except (KeyboardInterrupt, ExternalShutdownException):
        if rclpy.ok():
            node.get_logger().info(
                "Patrol node stopped by user."
            )
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
