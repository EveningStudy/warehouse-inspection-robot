from collections import deque
import json
import os
import time

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import message_filters
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


class YoloTargetDetectorNode(Node):
    """Detect inspection panels and localize them with RGB-D and TF."""

    def __init__(self):
        super().__init__('yolo_target_detector_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', 0.45)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('device', '0')
        self.declare_parameter('maximum_detections', 5)
        self.declare_parameter('maximum_inference_rate', 5.0)
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('camera_frame_fallback', 'camera_optical_link')
        self.declare_parameter('rgb_topic', '/camera/image')
        self.declare_parameter('depth_topic', '/camera/depth_image')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('minimum_depth', 0.15)
        self.declare_parameter('maximum_depth', 8.0)
        self.declare_parameter('depth_roi_fraction', 0.35)
        self.declare_parameter('minimum_depth_samples', 6)

        self.model_path = os.path.expanduser(
            str(self.get_parameter('model_path').value)
        )
        self.confidence_threshold = float(
            self.get_parameter('confidence_threshold').value
        )
        self.image_size = int(self.get_parameter('image_size').value)
        self.device = str(self.get_parameter('device').value)
        self.maximum_detections = int(
            self.get_parameter('maximum_detections').value
        )
        self.maximum_inference_rate = float(
            self.get_parameter('maximum_inference_rate').value
        )
        self.target_frame = str(
            self.get_parameter('target_frame').value
        )
        self.camera_frame_fallback = str(
            self.get_parameter('camera_frame_fallback').value
        )
        self.minimum_depth = float(
            self.get_parameter('minimum_depth').value
        )
        self.maximum_depth = float(
            self.get_parameter('maximum_depth').value
        )
        self.depth_roi_fraction = float(
            self.get_parameter('depth_roi_fraction').value
        )
        self.minimum_depth_samples = int(
            self.get_parameter('minimum_depth_samples').value
        )

        # The ROS package can still expose and test its RGB-D geometry helpers
        # with the system Python. Heavy ML dependencies are loaded only when
        # this executable is started through the .venv-yolo wrapper.
        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exception:
            raise RuntimeError(
                'YOLO runtime dependencies are unavailable. Start this node '
                'with the installed yolo_target_detector wrapper so it uses '
                'the workspace .venv-yolo environment.'
            ) from exception
        self.torch = torch
        self.yolo_class = YOLO

        self.validate_parameters()

        self.bridge = CvBridge()
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.last_inference_wall_time = 0.0
        self.last_log_wall_time = 0.0
        self.inference_times_ms = deque(maxlen=30)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        camera_info_topic = str(
            self.get_parameter('camera_info_topic').value
        )
        rgb_topic = str(self.get_parameter('rgb_topic').value)
        depth_topic = str(self.get_parameter('depth_topic').value)

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )
        self.rgb_subscription = message_filters.Subscriber(
            self,
            Image,
            rgb_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.depth_subscription = message_filters.Subscriber(
            self,
            Image,
            depth_topic,
            qos_profile=qos_profile_sensor_data,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_subscription, self.depth_subscription],
            queue_size=5,
            slop=0.1,
        )
        self.synchronizer.registerCallback(self.rgbd_callback)

        self.target_point_publisher = self.create_publisher(
            PointStamped,
            '/perception/target_point',
            10,
        )
        self.detections_publisher = self.create_publisher(
            String,
            '/perception/yolo_detections',
            10,
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/perception/yolo_markers',
            10,
        )
        self.debug_image_publisher = self.create_publisher(
            Image,
            '/perception/yolo_debug_image',
            10,
        )

        self.get_logger().info(
            f'Loading YOLO weights: {self.model_path}'
        )
        self.model = self.yolo_class(self.model_path)
        self.warm_up_model()
        self.get_logger().info(
            'YOLO RGB-D detector ready: '
            f'device={self.device}, conf={self.confidence_threshold:.2f}, '
            f'max_rate={self.maximum_inference_rate:.1f} Hz, '
            f'target_frame={self.target_frame}'
        )

    def validate_parameters(self):
        """Reject configuration errors before ROS subscriptions start."""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f'YOLO model does not exist: {self.model_path}'
            )
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                'confidence_threshold must be in the interval (0, 1]'
            )
        if self.image_size <= 0:
            raise ValueError('image_size must be positive')
        if self.maximum_detections <= 0:
            raise ValueError('maximum_detections must be positive')
        if self.maximum_inference_rate <= 0.0:
            raise ValueError('maximum_inference_rate must be positive')
        if not 0.0 < self.depth_roi_fraction <= 1.0:
            raise ValueError('depth_roi_fraction must be in (0, 1]')
        if self.minimum_depth <= 0.0:
            raise ValueError('minimum_depth must be positive')
        if self.maximum_depth <= self.minimum_depth:
            raise ValueError(
                'maximum_depth must be greater than minimum_depth'
            )
        if self.minimum_depth_samples <= 0:
            raise ValueError('minimum_depth_samples must be positive')
        if self.device != 'cpu' and not self.torch.cuda.is_available():
            raise RuntimeError(
                'CUDA inference was requested, but PyTorch cannot access CUDA'
            )

    def warm_up_model(self):
        """Initialize CUDA kernels before the first camera callback."""
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(
            source=dummy_image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            max_det=self.maximum_detections,
            classes=[0],
            verbose=False,
        )

    def camera_info_callback(self, message):
        """Store the pinhole-camera intrinsic matrix."""
        self.fx = float(message.k[0])
        self.fy = float(message.k[4])
        self.cx = float(message.k[2])
        self.cy = float(message.k[5])

    def rgbd_callback(self, rgb_message, depth_message):
        """Run detection, depth localization, TF, and publication."""
        now = time.monotonic()
        minimum_period = 1.0 / self.maximum_inference_rate
        if now - self.last_inference_wall_time < minimum_period:
            return
        self.last_inference_wall_time = now

        try:
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_message,
                desired_encoding='bgr8',
            ).copy()
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_message,
                desired_encoding='passthrough',
            )
        except Exception as exception:
            self.get_logger().error(
                f'Image conversion failed: {exception}'
            )
            return

        if any(
            value is None
            for value in (self.fx, self.fy, self.cx, self.cy)
        ):
            self.draw_status(
                rgb_image,
                'Waiting for /camera/camera_info',
                (0, 0, 255),
            )
            self.publish_debug_image(rgb_image, rgb_message)
            return

        inference_started = time.perf_counter()
        try:
            prediction = self.model.predict(
                source=rgb_image,
                imgsz=self.image_size,
                conf=self.confidence_threshold,
                device=self.device,
                max_det=self.maximum_detections,
                classes=[0],
                verbose=False,
            )[0]
        except Exception as exception:
            self.get_logger().error(f'YOLO inference failed: {exception}')
            return
        inference_ms = (
            time.perf_counter() - inference_started
        ) * 1000.0
        self.inference_times_ms.append(inference_ms)

        depth_metres = self.convert_depth_to_metres(
            depth_image,
            depth_message.encoding,
        )
        localized_detections = []

        boxes = prediction.boxes
        if boxes is not None:
            ordered_indices = sorted(
                range(len(boxes)),
                key=lambda index: float(boxes.conf[index]),
                reverse=True,
            )
            for marker_id, index in enumerate(ordered_indices):
                confidence = float(boxes.conf[index])
                x1, y1, x2, y2 = [
                    float(value)
                    for value in boxes.xyxy[index].cpu().tolist()
                ]
                detection = self.localize_detection(
                    rgb_message=rgb_message,
                    depth_metres=depth_metres,
                    rgb_shape=rgb_image.shape[:2],
                    box=(x1, y1, x2, y2),
                    confidence=confidence,
                    marker_id=marker_id,
                )
                self.draw_detection(rgb_image, detection)
                if detection['localized']:
                    localized_detections.append(detection)

        self.publish_detection_message(
            localized_detections,
            rgb_message,
            inference_ms,
        )
        self.publish_markers(localized_detections, rgb_message)
        if localized_detections:
            self.target_point_publisher.publish(
                localized_detections[0]['point_message']
            )

        average_ms = sum(self.inference_times_ms) / len(
            self.inference_times_ms
        )
        status = (
            f'panels={len(localized_detections)}  '
            f'inference={inference_ms:.1f} ms  avg={average_ms:.1f} ms'
        )
        self.draw_status(
            rgb_image,
            status,
            (0, 255, 0) if localized_detections else (0, 200, 255),
        )
        self.publish_debug_image(rgb_image, rgb_message)

        if localized_detections and now - self.last_log_wall_time >= 1.0:
            self.last_log_wall_time = now
            best = localized_detections[0]
            target = best['target']
            self.get_logger().info(
                'Panel located: '
                f"confidence={best['confidence']:.3f}, "
                f"depth={best['depth_m']:.2f} m, "
                f"map=({target['x']:.2f}, {target['y']:.2f}, "
                f"{target['z']:.2f}), inference={inference_ms:.1f} ms"
            )

    def localize_detection(
        self,
        rgb_message,
        depth_metres,
        rgb_shape,
        box,
        confidence,
        marker_id,
    ):
        """Add robust depth and a map-frame point to one YOLO box."""
        rgb_height, rgb_width = rgb_shape
        x1 = max(0.0, min(float(box[0]), rgb_width - 1.0))
        y1 = max(0.0, min(float(box[1]), rgb_height - 1.0))
        x2 = max(x1 + 1.0, min(float(box[2]), float(rgb_width)))
        y2 = max(y1 + 1.0, min(float(box[3]), float(rgb_height)))
        center_u = int(round((x1 + x2) / 2.0))
        center_v = int(round((y1 + y2) / 2.0))
        depth = self.estimate_target_depth(
            depth_metres,
            rgb_shape,
            (x1, y1, x2, y2),
            self.depth_roi_fraction,
            self.minimum_depth,
            self.maximum_depth,
            self.minimum_depth_samples,
        )

        detection = {
            'marker_id': marker_id,
            'class_id': 0,
            'class_name': 'inspection_panel',
            'confidence': confidence,
            'bbox': {
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
            },
            'center': {'u': center_u, 'v': center_v},
            'depth_m': depth,
            'localized': False,
            'target': None,
            'point_message': None,
            'localization_error': None,
        }
        if depth is None:
            detection['localization_error'] = 'invalid_depth'
            return detection

        point_in_camera = self.project_pixel_to_camera(
            rgb_message,
            center_u,
            center_v,
            depth,
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
            detection['localization_error'] = f'tf: {exception}'
            return detection

        detection['localized'] = True
        detection['point_message'] = point_in_target
        detection['target'] = {
            'x': float(point_in_target.point.x),
            'y': float(point_in_target.point.y),
            'z': float(point_in_target.point.z),
        }
        return detection

    @staticmethod
    def convert_depth_to_metres(depth_image, encoding):
        """Normalize common RGB-D encodings to metres."""
        depth_metres = depth_image.astype(np.float32)
        if encoding in ('16UC1', 'mono16'):
            depth_metres = depth_metres / 1000.0
        return depth_metres

    @staticmethod
    def estimate_target_depth(
        depth_metres,
        rgb_shape,
        box,
        roi_fraction,
        minimum_depth,
        maximum_depth,
        minimum_samples,
    ):
        """Return median depth from the central part of a detection box."""
        rgb_height, rgb_width = rgb_shape
        depth_height, depth_width = depth_metres.shape[:2]
        x1, y1, x2, y2 = box
        center_u = (x1 + x2) / 2.0
        center_v = (y1 + y2) / 2.0
        roi_width = max((x2 - x1) * roi_fraction, 2.0)
        roi_height = max((y2 - y1) * roi_fraction, 2.0)

        left = int(
            max(0.0, (center_u - roi_width / 2.0) * depth_width / rgb_width)
        )
        right = int(
            min(
                float(depth_width),
                (center_u + roi_width / 2.0) * depth_width / rgb_width,
            )
        )
        top = int(
            max(0.0, (center_v - roi_height / 2.0) * depth_height / rgb_height)
        )
        bottom = int(
            min(
                float(depth_height),
                (center_v + roi_height / 2.0) * depth_height / rgb_height,
            )
        )
        if right <= left or bottom <= top:
            return None

        roi = depth_metres[top:bottom, left:right]
        valid = roi[
            np.isfinite(roi)
            & (roi >= minimum_depth)
            & (roi <= maximum_depth)
        ]
        if valid.size < minimum_samples:
            return None
        return float(np.median(valid))

    def project_pixel_to_camera(
        self,
        rgb_message,
        pixel_u,
        pixel_v,
        depth,
    ):
        """Back-project one RGB pixel into the optical camera frame."""
        point = PointStamped()
        point.header = rgb_message.header
        if not point.header.frame_id:
            point.header.frame_id = self.camera_frame_fallback
        point.point.x = float((pixel_u - self.cx) * depth / self.fx)
        point.point.y = float((pixel_v - self.cy) * depth / self.fy)
        point.point.z = float(depth)
        return point

    def publish_detection_message(
        self,
        detections,
        source_message,
        inference_ms,
    ):
        """Publish localized detections as documented JSON."""
        serializable_detections = []
        for detection in detections:
            serializable_detections.append(
                {
                    key: value
                    for key, value in detection.items()
                    if key not in ('point_message', 'marker_id')
                }
            )
        payload = {
            'schema_version': 1,
            'stamp': {
                'sec': int(source_message.header.stamp.sec),
                'nanosec': int(source_message.header.stamp.nanosec),
            },
            'source_frame': (
                source_message.header.frame_id
                or self.camera_frame_fallback
            ),
            'target_frame': self.target_frame,
            'inference_ms': float(inference_ms),
            'confidence_threshold': self.confidence_threshold,
            'detections': serializable_detections,
        }
        message = String()
        message.data = json.dumps(payload, separators=(',', ':'))
        self.detections_publisher.publish(message)

    def publish_markers(self, detections, source_message):
        """Show localized panel positions as short-lived blue spheres."""
        marker_array = MarkerArray()
        for detection in detections:
            marker = Marker()
            marker.header.frame_id = self.target_frame
            marker.header.stamp = source_message.header.stamp
            marker.ns = 'yolo_inspection_panels'
            marker.id = int(detection['marker_id'])
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            target = detection['target']
            marker.pose.position.x = target['x']
            marker.pose.position.y = target['y']
            marker.pose.position.z = target['z']
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.20
            marker.scale.y = 0.20
            marker.scale.z = 0.20
            marker.color.r = 0.1
            marker.color.g = 0.4
            marker.color.b = 1.0
            marker.color.a = 0.95
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

    def publish_debug_image(self, image, source_message):
        """Publish the annotated RGB frame for RViz or rqt_image_view."""
        message = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        message.header = source_message.header
        self.debug_image_publisher.publish(message)

    @staticmethod
    def draw_detection(image, detection):
        """Draw one localized or depth-invalid prediction."""
        box = detection['bbox']
        x1 = int(round(box['x1']))
        y1 = int(round(box['y1']))
        x2 = int(round(box['x2']))
        y2 = int(round(box['y2']))
        colour = (255, 120, 0) if detection['localized'] else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        center = detection['center']
        cv2.circle(
            image,
            (center['u'], center['v']),
            4,
            colour,
            -1,
        )
        label = f"panel {detection['confidence']:.2f}"
        if detection['depth_m'] is not None:
            label += f" {detection['depth_m']:.2f}m"
        else:
            label += ' depth-invalid'
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def draw_status(image, text, colour):
        """Draw a readable black-backed status line."""
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(
            image,
            text,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            colour,
            2,
            cv2.LINE_AA,
        )


def main(args=None):
    rclpy.init(args=args)
    node = YoloTargetDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('YOLO target detector stopped by user.')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
