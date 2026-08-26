import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class RgbdInspectorNode(Node):
    def __init__(self):
        super().__init__("rgbd_inspector_node")

        # CvBridge 负责 ROS Image 和 OpenCV/NumPy 之间的转换。
        self.bridge = CvBridge()

        # 相机内参。收到 CameraInfo 后再赋值。
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_received = False
        self.last_log_time_ns = None

        # 订阅相机内参。
        self.camera_info_subscription = (
            self.create_subscription(
                CameraInfo,
                "/camera/camera_info",
                self.camera_info_callback,
                qos_profile_sensor_data,
            )
        )

        # message_filters.Subscriber 不直接调用我们的回调。
        # 它会先把消息交给下面的时间同步器。
        self.rgb_subscription = (
            message_filters.Subscriber(
                self,
                Image,
                "/camera/image",
                qos_profile=qos_profile_sensor_data,
            )
        )

        self.depth_subscription = (
            message_filters.Subscriber(
                self,
                Image,
                "/camera/depth_image",
                qos_profile=qos_profile_sensor_data,
            )
        )

        # 在彩色图和深度图时间戳足够接近时，
        # 才一起调用 rgbd_callback。
        self.synchronizer = (
            message_filters.ApproximateTimeSynchronizer(
                [
                    self.rgb_subscription,
                    self.depth_subscription,
                ],
                queue_size=10,
                slop=0.1,
            )
        )

        self.synchronizer.registerCallback(
            self.rgbd_callback
        )

        # 发布带中心点和深度文字的调试图。
        self.debug_image_publisher = (
            self.create_publisher(
                Image,
                "/perception/debug_image",
                10,
            )
        )

        self.get_logger().info(
            "RGB-D inspector started. "
            "Waiting for camera messages..."
        )

    def camera_info_callback(self, message):
        # CameraInfo 的 K 是 3×3 相机内参矩阵：
        #
        # [fx  0 cx]
        # [ 0 fy cy]
        # [ 0  0  1]
        self.fx = message.k[0]
        self.fy = message.k[4]
        self.cx = message.k[2]
        self.cy = message.k[5]

        # 内参通常不会持续变化，因此只打印第一次。
        if not self.camera_info_received:
            self.camera_info_received = True

            self.get_logger().info(
                "Camera info received: "
                f"frame={message.header.frame_id}, "
                f"size={message.width}x{message.height}, "
                f"fx={self.fx:.2f}, "
                f"fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, "
                f"cy={self.cy:.2f}"
            )

    def rgbd_callback(
        self,
        rgb_message,
        depth_message,
    ):
        try:
            # ROS 彩色图转换成 OpenCV BGR 数组。
            #
            # copy() 确保我们可以安全地在图像上画内容。
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_message,
                desired_encoding="bgr8",
            ).copy()

            # 保留深度图原始编码。
            #
            # 常见形式：
            #   32FC1：每个像素是以米为单位的 float
            #   16UC1：每个像素通常是以毫米为单位的整数
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding="passthrough",
            )

        except Exception as exception:
            self.get_logger().error(
                f"Image conversion failed: {exception}"
            )
            return

        rgb_height, rgb_width = rgb_image.shape[:2]
        depth_height, depth_width = depth_image.shape[:2]

        # 找到彩色图的中心像素。
        center_u = rgb_width // 2
        center_v = rgb_height // 2

        # 如果 RGB 和深度分辨率不同，
        # 按比例转换成深度图中的像素坐标。
        depth_u = int(
            center_u * depth_width / rgb_width
        )
        depth_v = int(
            center_v * depth_height / rgb_height
        )

        # 防止取到数组边界之外。
        depth_u = min(
            max(depth_u, 0),
            depth_width - 1,
        )
        depth_v = min(
            max(depth_v, 0),
            depth_height - 1,
        )

        raw_depth = float(
            depth_image[depth_v, depth_u]
        )

        # 根据编码统一转换成米。
        if depth_message.encoding in (
            "16UC1",
            "mono16",
        ):
            depth_metres = raw_depth / 1000.0
        else:
            depth_metres = raw_depth

        depth_is_valid = (
            np.isfinite(depth_metres)
            and depth_metres > 0.0
        )

        # 在 RGB 图像中心画一个红色圆圈。
        cv2.circle(
            rgb_image,
            (center_u, center_v),
            radius=6,
            color=(0, 0, 255),
            thickness=2,
        )

        if depth_is_valid:
            depth_text = (
                f"center depth: {depth_metres:.2f} m"
            )
        else:
            depth_text = "center depth: invalid"

        # 把深度显示在图像左上角。
        cv2.putText(
            rgb_image,
            depth_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        # OpenCV 图像重新转换成 ROS Image。
        debug_message = (
            self.bridge.cv2_to_imgmsg(
                rgb_image,
                encoding="bgr8",
            )
        )

        # 保留原始 RGB 图像的时间戳和坐标系。
        debug_message.header = rgb_message.header

        self.debug_image_publisher.publish(
            debug_message
        )

        # 每秒最多打印一次，避免十帧每秒持续刷屏。
        now_ns = self.get_clock().now().nanoseconds

        should_log = (
            self.last_log_time_ns is None
            or now_ns - self.last_log_time_ns
            >= 1_000_000_000
        )

        if should_log:
            self.last_log_time_ns = now_ns

            if depth_is_valid:
                self.get_logger().info(
                    "RGB-D frame: "
                    f"rgb={rgb_width}x{rgb_height}, "
                    f"depth="
                    f"{depth_width}x{depth_height}, "
                    f"encoding={depth_message.encoding}, "
                    f"center_depth="
                    f"{depth_metres:.2f} m"
                )
            else:
                self.get_logger().warning(
                    "RGB-D frame received, but "
                    "the centre depth is invalid."
                )


def main(args=None):
    rclpy.init(args=args)

    node = RgbdInspectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            "RGB-D inspector stopped by user."
        )
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()