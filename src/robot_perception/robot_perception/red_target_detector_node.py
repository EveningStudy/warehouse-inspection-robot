import cv2
import message_filters
import numpy as np
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker


class RedTargetDetectorNode(Node):
    """Detect the red-orange target and locate it in a requested TF frame."""

    def __init__(self):
        super().__init__("red_target_detector_node")

        self.declare_parameter("target_frame", "map")
        self.declare_parameter("minimum_area", 200.0)

        self.target_frame = str(
            self.get_parameter("target_frame").value
        )
        self.minimum_area = float(
            self.get_parameter("minimum_area").value
        )

        self.bridge = CvBridge()

        # Intrinsic camera parameters from CameraInfo.K.
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.last_log_time_ns = None

        # Buffer stores transforms received from /tf and /tf_static.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/camera/camera_info",
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        # RGB and depth are synchronized before processing.
        self.rgb_subscription = message_filters.Subscriber(
            self,
            Image,
            "/camera/image",
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_subscription = message_filters.Subscriber(
            self,
            Image,
            "/camera/depth_image",
            qos_profile=qos_profile_sensor_data,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [
                self.rgb_subscription,
                self.depth_subscription,
            ],
            queue_size=10,
            slop=0.1,
        )
        self.synchronizer.registerCallback(self.rgbd_callback)

        self.target_point_publisher = self.create_publisher(
            PointStamped,
            "/perception/target_point",
            10,
        )
        self.marker_publisher = self.create_publisher(
            Marker,
            "/perception/target_marker",
            10,
        )
        self.debug_image_publisher = self.create_publisher(
            Image,
            "/perception/target_debug_image",
            10,
        )

        self.get_logger().info(
            "Red target detector started. "
            f"Target frame: {self.target_frame}"
        )

    def camera_info_callback(self, message):
        """Store the pinhole camera intrinsics."""
        self.fx = message.k[0]
        self.fy = message.k[4]
        self.cx = message.k[2]
        self.cy = message.k[5]

    def rgbd_callback(self, rgb_message, depth_message):
        """Detect, back-project, transform, and publish one target."""
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_message,
                desired_encoding="bgr8",
            ).copy()
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding="passthrough",
            )
        except Exception as exception:
            self.get_logger().error(
                f"Image conversion failed: {exception}"
            )
            return

        if any(
            value is None
            for value in (self.fx, self.fy, self.cx, self.cy)
        ):
            self.draw_text(
                rgb_image,
                "waiting for CameraInfo",
                (10, 25),
                (0, 0, 255),
            )
            self.publish_debug_image(rgb_image, rgb_message)
            return

        detection = self.detect_red_target(rgb_image)
        if detection is None:
            self.draw_text(
                rgb_image,
                "target not found",
                (10, 25),
                (0, 0, 255),
            )
            self.publish_debug_image(rgb_image, rgb_message)
            return

        box_x, box_y, box_width, box_height, contour_area = detection
        if contour_area < self.minimum_area:
            self.draw_text(
                rgb_image,
                "red region too small",
                (10, 25),
                (0, 0, 255),
            )
            self.publish_debug_image(rgb_image, rgb_message)
            return

        center_u = box_x + box_width // 2
        center_v = box_y + box_height // 2

        depth_metres = self.convert_depth_to_metres(
            depth_image,
            depth_message.encoding,
        )
        target_depth = self.estimate_target_depth(
            depth_metres,
            rgb_image.shape[:2],
            center_u,
            center_v,
            box_width,
            box_height,
        )

        if target_depth is None:
            self.draw_detection(
                rgb_image,
                box_x,
                box_y,
                box_width,
                box_height,
                center_u,
                center_v,
                (0, 0, 255),
            )
            self.draw_text(
                rgb_image,
                "target found, depth invalid",
                (10, 25),
                (0, 0, 255),
            )
            self.publish_debug_image(rgb_image, rgb_message)
            return

        point_in_camera = self.project_pixel_to_camera(
            rgb_message,
            center_u,
            center_v,
            target_depth,
        )

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                point_in_camera.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.2),
            )
            point_in_target = do_transform_point(
                point_in_camera,
                transform,
            )
        except TransformException as exception:
            self.draw_detection(
                rgb_image,
                box_x,
                box_y,
                box_width,
                box_height,
                center_u,
                center_v,
                (0, 255, 255),
            )
            self.draw_text(
                rgb_image,
                "target found, TF unavailable",
                (10, 25),
                (0, 255, 255),
            )
            if self.should_log():
                self.get_logger().warning(
                    "Could not transform "
                    f"{point_in_camera.header.frame_id} "
                    f"to {self.target_frame}: {exception}"
                )
            self.publish_debug_image(rgb_image, rgb_message)
            return

        self.target_point_publisher.publish(point_in_target)
        self.publish_target_marker(point_in_target)

        self.draw_detection(
            rgb_image,
            box_x,
            box_y,
            box_width,
            box_height,
            center_u,
            center_v,
            (0, 255, 0),
        )
        self.draw_text(
            rgb_image,
            f"depth: {target_depth:.2f} m",
            (10, 25),
            (0, 255, 0),
        )
        self.draw_text(
            rgb_image,
            (
                f"{self.target_frame}: "
                f"({point_in_target.point.x:.2f}, "
                f"{point_in_target.point.y:.2f}, "
                f"{point_in_target.point.z:.2f})"
            ),
            (10, 50),
            (0, 255, 0),
        )
        self.publish_debug_image(rgb_image, rgb_message)

        if self.should_log():
            self.get_logger().info(
                "Target located: "
                f"pixel=({center_u}, {center_v}), "
                f"depth={target_depth:.2f} m, "
                f"{self.target_frame}="
                f"({point_in_target.point.x:.2f}, "
                f"{point_in_target.point.y:.2f}, "
                f"{point_in_target.point.z:.2f})"
            )

    @staticmethod
    def detect_red_target(image):
        """Return the largest red-orange bounding box and its area."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask_red_orange = cv2.inRange(
            hsv_image,
            np.array([0, 80, 60], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8),
        )
        mask_red = cv2.inRange(
            hsv_image,
            np.array([170, 80, 60], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        )
        colour_mask = cv2.bitwise_or(mask_red_orange, mask_red)

        kernel = np.ones((5, 5), dtype=np.uint8)
        colour_mask = cv2.morphologyEx(
            colour_mask,
            cv2.MORPH_OPEN,
            kernel,
        )
        colour_mask = cv2.morphologyEx(
            colour_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        contours, _ = cv2.findContours(
            colour_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        x, y, width, height = cv2.boundingRect(largest_contour)
        return x, y, width, height, area

    @staticmethod
    def convert_depth_to_metres(depth_image, encoding):
        """Normalize common Gazebo / depth camera encodings to metres."""
        depth_metres = depth_image.astype(np.float32)
        if encoding in ("16UC1", "mono16"):
            depth_metres = depth_metres / 1000.0
        return depth_metres

    @staticmethod
    def estimate_target_depth(
        depth_metres,
        rgb_shape,
        center_u,
        center_v,
        box_width,
        box_height,
    ):
        """Estimate robust median depth from the centre of the box."""
        rgb_height, rgb_width = rgb_shape
        depth_height, depth_width = depth_metres.shape[:2]

        depth_center_u = int(center_u * depth_width / rgb_width)
        depth_center_v = int(center_v * depth_height / rgb_height)
        depth_box_width = max(
            int(box_width * depth_width / rgb_width),
            1,
        )
        depth_box_height = max(
            int(box_height * depth_height / rgb_height),
            1,
        )

        half_roi_width = max(depth_box_width // 4, 2)
        half_roi_height = max(depth_box_height // 4, 2)

        left = max(depth_center_u - half_roi_width, 0)
        right = min(depth_center_u + half_roi_width, depth_width)
        top = max(depth_center_v - half_roi_height, 0)
        bottom = min(depth_center_v + half_roi_height, depth_height)

        roi = depth_metres[top:bottom, left:right]
        valid_depths = roi[
            np.isfinite(roi)
            & (roi > 0.15)
            & (roi < 8.0)
        ]
        if valid_depths.size == 0:
            return None

        return float(np.median(valid_depths))

    def project_pixel_to_camera(
        self,
        rgb_message,
        pixel_u,
        pixel_v,
        depth,
    ):
        """Back-project one RGB pixel into the optical camera frame."""
        optical_x = (pixel_u - self.cx) * depth / self.fx
        optical_y = (pixel_v - self.cy) * depth / self.fy
        optical_z = depth

        point = PointStamped()
        point.header = rgb_message.header
        if not point.header.frame_id:
            point.header.frame_id = "camera_optical_link"
        point.point.x = float(optical_x)
        point.point.y = float(optical_y)
        point.point.z = float(optical_z)
        return point

    def publish_target_marker(self, target_point):
        """Publish a short-lived red sphere at the detected target."""
        marker = Marker()
        marker.header = target_point.header
        marker.ns = "detected_target"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = target_point.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.20
        marker.scale.y = 0.20
        marker.scale.z = 0.20
        marker.color.r = 1.0
        marker.color.g = 0.1
        marker.color.b = 0.0
        marker.color.a = 0.9
        marker.lifetime.sec = 1
        self.marker_publisher.publish(marker)

    def publish_debug_image(self, image, source_message):
        """Convert an annotated OpenCV image back to a ROS image."""
        debug_message = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )
        debug_message.header = source_message.header
        self.debug_image_publisher.publish(debug_message)

    @staticmethod
    def draw_detection(
        image,
        box_x,
        box_y,
        box_width,
        box_height,
        center_u,
        center_v,
        colour,
    ):
        """Draw a bounding box and its centre pixel."""
        cv2.rectangle(
            image,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            colour,
            2,
        )
        cv2.circle(
            image,
            (center_u, center_v),
            5,
            (0, 0, 255),
            -1,
        )

    @staticmethod
    def draw_text(image, text, position, colour):
        """Draw readable status text on an OpenCV image."""
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )

    def should_log(self):
        """Limit recurring status logs to once per simulated second."""
        now_ns = self.get_clock().now().nanoseconds
        if (
            self.last_log_time_ns is None
            or now_ns - self.last_log_time_ns >= 1_000_000_000
        ):
            self.last_log_time_ns = now_ns
            return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = RedTargetDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Red target detector stopped.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
