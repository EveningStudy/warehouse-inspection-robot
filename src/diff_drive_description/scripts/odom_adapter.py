#!/usr/bin/env python3

import rclpy

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomAdapter(Node):

    def __init__(self):
        super().__init__("odom_adapter")

        self.declare_parameter("input_topic", "/gz_odom")
        self.input_topic = str(
            self.get_parameter("input_topic").value
        )

        self.odom_publisher = self.create_publisher(
            Odometry,
            "/odom",
            10,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            self.input_topic,
            self.odom_callback,
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        self.get_logger().info(
            f"Converting {self.input_topic} to /odom and publishing "
            "odom -> base_footprint TF"
        )

    def odom_callback(self, message):
        message.header.frame_id = "odom"
        message.child_frame_id = "base_footprint"

        self.odom_publisher.publish(message)

        transform = TransformStamped()

        transform.header.stamp = message.header.stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_footprint"

        transform.transform.translation.x = (
            message.pose.pose.position.x
        )
        transform.transform.translation.y = (
            message.pose.pose.position.y
        )
        transform.transform.translation.z = (
            message.pose.pose.position.z
        )

        transform.transform.rotation = (
            message.pose.pose.orientation
        )

        self.tf_broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)

    node = OdomAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
