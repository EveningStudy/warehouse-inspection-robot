import math

import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


class SingleGoalNode(Node):
    def __init__(self):
        super().__init__("single_goal_node")

        # 声明可以从命令行修改的目标参数。
        self.declare_parameter("goal_x", 1.0)
        self.declare_parameter("goal_y", 1.0)
        self.declare_parameter("goal_yaw", 0.0)

        self.goal_x = float(
            self.get_parameter("goal_x").value
        )
        self.goal_y = float(
            self.get_parameter("goal_y").value
        )
        self.goal_yaw = float(
            self.get_parameter("goal_yaw").value
        )

        # 创建 NavigateToPose Action Client。
        #
        # NavigateToPose 是接口类型。
        # navigate_to_pose 是 Nav2 提供的 Action 名称。
        self.action_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose",
        )

        self.goal_sent = False
        self.goal_handle = None
        self.finished = False
        self.feedback_counter = 0

        # 每秒检查一次 Nav2 Action Server 是否已经准备好。
        self.server_timer = self.create_timer(
            1.0,
            self.try_send_goal,
        )

        self.get_logger().info(
            "Single goal node started. Waiting for Nav2..."
        )

    def try_send_goal(self):
        # 防止同一个目标被重复发送。
        if self.goal_sent:
            return

        # 非阻塞检查 Nav2 的 Action Server 是否存在。
        if not self.action_client.wait_for_server(
            timeout_sec=0.0
        ):
            self.get_logger().info(
                "NavigateToPose server is not ready yet..."
            )
            return

        self.goal_sent = True
        self.server_timer.cancel()

        goal_msg = NavigateToPose.Goal()

        # 目标坐标是 map 坐标系中的位置。
        goal_msg.pose.header.frame_id = "map"

        # 使用当前节点的时钟填写时间戳。
        goal_msg.pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        goal_msg.pose.pose.position.x = self.goal_x
        goal_msg.pose.pose.position.y = self.goal_y
        goal_msg.pose.pose.position.z = 0.0

        # ROS 姿态使用四元数。
        # 对于只绕 Z 轴旋转的地面机器人：
        #
        # z = sin(yaw / 2)
        # w = cos(yaw / 2)
        goal_msg.pose.pose.orientation.x = 0.0
        goal_msg.pose.pose.orientation.y = 0.0
        goal_msg.pose.pose.orientation.z = math.sin(
            self.goal_yaw / 2.0
        )
        goal_msg.pose.pose.orientation.w = math.cos(
            self.goal_yaw / 2.0
        )

        self.get_logger().info(
            "Sending goal: "
            f"x={self.goal_x:.2f}, "
            f"y={self.goal_y:.2f}, "
            f"yaw={self.goal_yaw:.2f}"
        )

        # 异步发送目标。
        send_goal_future = (
            self.action_client.send_goal_async(
                goal_msg,
                feedback_callback=self.feedback_callback,
            )
        )

        # Nav2 接受或拒绝目标后，调用 goal_response_callback。
        send_goal_future.add_done_callback(
            self.goal_response_callback
        )

    def goal_response_callback(self, future):
        try:
            self.goal_handle = future.result()
        except Exception as exception:
            self.get_logger().error(
                f"Failed to send goal: {exception}"
            )
            self.finished = True
            return

        if not self.goal_handle.accepted:
            self.get_logger().error(
                "Nav2 rejected the goal."
            )
            self.finished = True
            return

        self.get_logger().info(
            "Nav2 accepted the goal."
        )

        # 请求最终结果。
        result_future = (
            self.goal_handle.get_result_async()
        )

        result_future.add_done_callback(
            self.result_callback
        )

    def feedback_callback(self, feedback_message):
        feedback = feedback_message.feedback

        # Nav2 会频繁发送反馈。
        # 每收到 10 次，只打印一次，避免刷屏。
        self.feedback_counter += 1
        if self.feedback_counter % 10 != 0:
            return

        estimated_seconds = (
            feedback.estimated_time_remaining.sec
            + feedback.estimated_time_remaining.nanosec
            / 1_000_000_000
        )

        self.get_logger().info(
            "Navigation feedback: "
            f"distance remaining="
            f"{feedback.distance_remaining:.2f} m, "
            f"estimated time remaining="
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

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(
                "Navigation succeeded."
            )

        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warning(
                "Navigation was canceled."
            )

        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error(
                "Navigation was aborted. "
                f"error_code={result.error_code}, "
                f"error_msg='{result.error_msg}'"
            )

        else:
            self.get_logger().error(
                f"Navigation ended with status={status}, "
                f"error_code={result.error_code}, "
                f"error_msg='{result.error_msg}'"
            )

        self.finished = True


def main(args=None):
    rclpy.init(args=args)

    node = SingleGoalNode()

    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(
                node,
                timeout_sec=0.1,
            )
    except KeyboardInterrupt:
        node.get_logger().info(
            "Single goal node stopped by user."
        )
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()