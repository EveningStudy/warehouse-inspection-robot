from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import shutil

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
import yaml


class YoloDatasetCollectorNode(Node):
    """Synchronize RGB and semantic images and build a YOLO dataset run."""

    def __init__(self):
        super().__init__('yolo_dataset_collector')

        default_output_root = os.path.expanduser(
            '~/.ros/autonomy_lab/datasets/inspection_panel'
        )
        self.declare_parameter('output_root', default_output_root)
        self.declare_parameter('rgb_topic', '/camera/image')
        self.declare_parameter(
            'label_topic',
            '/camera/segmentation/labels_map',
        )
        self.declare_parameter('gazebo_label_id', 1)
        self.declare_parameter('class_name', 'inspection_panel')
        self.declare_parameter('positive_sample_period', 0.5)
        self.declare_parameter('negative_sample_period', 2.0)
        self.declare_parameter('sync_slop', 0.08)
        self.declare_parameter('min_label_pixels', 40)
        self.declare_parameter('jpeg_quality', 95)
        self.declare_parameter('split_seed', 42)

        self.output_root = Path(
            os.path.expanduser(
                str(self.get_parameter('output_root').value)
            )
        ).resolve()
        self.rgb_topic = str(self.get_parameter('rgb_topic').value)
        self.label_topic = str(self.get_parameter('label_topic').value)
        self.gazebo_label_id = int(
            self.get_parameter('gazebo_label_id').value
        )
        self.class_name = str(
            self.get_parameter('class_name').value
        )
        self.positive_sample_period = max(
            0.0,
            float(
                self.get_parameter('positive_sample_period').value
            ),
        )
        self.negative_sample_period = max(
            0.0,
            float(
                self.get_parameter('negative_sample_period').value
            ),
        )
        self.sync_slop = max(
            0.0,
            float(self.get_parameter('sync_slop').value),
        )
        self.min_label_pixels = max(
            1,
            int(self.get_parameter('min_label_pixels').value),
        )
        self.jpeg_quality = min(
            100,
            max(1, int(self.get_parameter('jpeg_quality').value)),
        )
        self.split_seed = int(
            self.get_parameter('split_seed').value
        )

        session_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = self.output_root / 'runs' / session_name
        self.raw_images_dir = self.session_dir / 'raw' / 'images'
        self.raw_labels_dir = self.session_dir / 'raw' / 'labels'
        self.raw_images_dir.mkdir(parents=True, exist_ok=False)
        self.raw_labels_dir.mkdir(parents=True, exist_ok=False)

        self.bridge = CvBridge()
        self.records = []
        self.sample_count = 0
        self.positive_count = 0
        self.negative_count = 0
        self.last_positive_stamp = -math.inf
        self.last_negative_stamp = -math.inf
        self.last_status_count = 0
        self.finalized = False

        image_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.rgb_subscriber = message_filters.Subscriber(
            self,
            Image,
            self.rgb_topic,
            qos_profile=image_qos,
        )
        self.label_subscriber = message_filters.Subscriber(
            self,
            Image,
            self.label_topic,
            qos_profile=image_qos,
        )
        self.synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_subscriber, self.label_subscriber],
            queue_size=20,
            slop=self.sync_slop,
        )
        self.synchronizer.registerCallback(self.image_callback)

        self.preview_publisher = self.create_publisher(
            Image,
            '/dataset/annotation_preview',
            10,
        )

        self.get_logger().info(
            'YOLO dataset collector started. '
            f'RGB={self.rgb_topic}, labels={self.label_topic}, '
            f'run={self.session_dir}'
        )

    @staticmethod
    def stamp_seconds(message):
        return (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) / 1_000_000_000.0
        )

    def extract_boxes(self, label_image):
        if label_image.ndim == 2:
            label_plane = label_image
        elif label_image.ndim == 3:
            # Gazebo semantic labels are repeated over RGB channels in the ROS
            # image. The official Gazebo byte layout stores the label in the
            # final channel, so use that channel explicitly.
            label_plane = label_image[:, :, -1]
        else:
            raise ValueError(
                f'Unsupported label image shape: {label_image.shape}'
            )

        mask = (label_plane == self.gazebo_label_id).astype(np.uint8)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        boxes = []
        for component in range(1, component_count):
            x, y, width, height, area = stats[component]
            if int(area) < self.min_label_pixels:
                continue
            boxes.append(
                {
                    'x': int(x),
                    'y': int(y),
                    'width': int(width),
                    'height': int(height),
                    'area': int(area),
                }
            )

        return boxes

    @staticmethod
    def box_to_yolo(box, image_width, image_height):
        x_center = (box['x'] + box['width'] / 2.0) / image_width
        y_center = (box['y'] + box['height'] / 2.0) / image_height
        width = box['width'] / image_width
        height = box['height'] / image_height
        return x_center, y_center, width, height

    def should_save(self, stamp, is_positive):
        if is_positive:
            if (
                stamp - self.last_positive_stamp
                < self.positive_sample_period
            ):
                return False
            self.last_positive_stamp = stamp
            return True

        if stamp - self.last_negative_stamp < self.negative_sample_period:
            return False
        self.last_negative_stamp = stamp
        return True

    def image_callback(self, rgb_message, label_message):
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(
                rgb_message,
                desired_encoding='bgr8',
            )
            label_image = self.bridge.imgmsg_to_cv2(
                label_message,
                desired_encoding='passthrough',
            )
        except Exception as error:
            self.get_logger().error(
                f'Could not convert synchronized images: {error}'
            )
            return

        if rgb_image.shape[:2] != label_image.shape[:2]:
            self.get_logger().error(
                'RGB and semantic image sizes differ: '
                f'{rgb_image.shape[:2]} vs {label_image.shape[:2]}'
            )
            return

        try:
            boxes = self.extract_boxes(label_image)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        stamp = self.stamp_seconds(rgb_message)
        is_positive = bool(boxes)
        if not self.should_save(stamp, is_positive):
            return

        self.sample_count += 1
        if is_positive:
            self.positive_count += 1
        else:
            self.negative_count += 1

        stamp_nanoseconds = (
            int(rgb_message.header.stamp.sec) * 1_000_000_000
            + int(rgb_message.header.stamp.nanosec)
        )
        stem = f'frame_{self.sample_count:06d}_{stamp_nanoseconds}'
        image_path = self.raw_images_dir / f'{stem}.jpg'
        label_path = self.raw_labels_dir / f'{stem}.txt'

        saved = cv2.imwrite(
            str(image_path),
            rgb_image,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not saved:
            self.get_logger().error(f'Failed to write {image_path}')
            return

        image_height, image_width = rgb_image.shape[:2]
        label_lines = []
        for box in boxes:
            x_center, y_center, width, height = self.box_to_yolo(
                box,
                image_width,
                image_height,
            )
            label_lines.append(
                '0 '
                f'{x_center:.6f} {y_center:.6f} '
                f'{width:.6f} {height:.6f}'
            )
        label_path.write_text(
            '\n'.join(label_lines) + ('\n' if label_lines else ''),
            encoding='utf-8',
        )

        record = {
            'stem': stem,
            'image': str(image_path),
            'label': str(label_path),
            'sim_time': stamp,
            'positive': is_positive,
            'boxes': boxes,
            'width': image_width,
            'height': image_height,
        }
        self.records.append(record)

        annotated = self.draw_annotation(rgb_image, boxes)
        preview_message = self.bridge.cv2_to_imgmsg(
            annotated,
            encoding='bgr8',
        )
        preview_message.header = rgb_message.header
        self.preview_publisher.publish(preview_message)

        if self.sample_count - self.last_status_count >= 20:
            self.last_status_count = self.sample_count
            self.get_logger().info(
                f'Saved {self.sample_count} samples '
                f'({self.positive_count} positive, '
                f'{self.negative_count} negative).'
            )

    def draw_annotation(self, image, boxes):
        annotated = image.copy()
        for box in boxes:
            x1 = box['x']
            y1 = box['y']
            x2 = x1 + box['width']
            y2 = y1 + box['height']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                self.class_name,
                (x1, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        if not boxes:
            cv2.putText(
                annotated,
                'negative sample',
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def split_records(self):
        rng = random.Random(self.split_seed)
        split_map = {'train': [], 'val': [], 'test': []}

        for group in (
            [record for record in self.records if record['positive']],
            [record for record in self.records if not record['positive']],
        ):
            rng.shuffle(group)
            count = len(group)
            if count == 0:
                continue
            if count == 1:
                split_map['train'].extend(group)
                continue
            if count == 2:
                split_map['train'].append(group[0])
                split_map['val'].append(group[1])
                continue

            train_count = max(1, int(round(count * 0.70)))
            val_count = max(1, int(round(count * 0.20)))
            if train_count + val_count >= count:
                train_count = count - 2
                val_count = 1

            split_map['train'].extend(group[:train_count])
            split_map['val'].extend(
                group[train_count:train_count + val_count]
            )
            split_map['test'].extend(
                group[train_count + val_count:]
            )

        for split_records in split_map.values():
            rng.shuffle(split_records)
        return split_map

    def build_preview(self, preview_path):
        positives = [
            record for record in self.records if record['positive']
        ]
        negatives = [
            record for record in self.records if not record['positive']
        ]
        rng = random.Random(self.split_seed)
        rng.shuffle(positives)
        rng.shuffle(negatives)
        selected = positives[:8] + negatives[:4]
        rng.shuffle(selected)
        if not selected:
            return False

        tile_width = 320
        tile_height = 264
        columns = 4
        rows = int(math.ceil(len(selected) / columns))
        montage = np.full(
            (rows * tile_height, columns * tile_width, 3),
            35,
            dtype=np.uint8,
        )

        for index, record in enumerate(selected):
            image = cv2.imread(record['image'], cv2.IMREAD_COLOR)
            if image is None:
                continue
            annotated = self.draw_annotation(image, record['boxes'])
            annotated = cv2.resize(
                annotated,
                (tile_width, tile_height - 24),
                interpolation=cv2.INTER_AREA,
            )
            tile = np.full(
                (tile_height, tile_width, 3),
                35,
                dtype=np.uint8,
            )
            tile[24:, :] = annotated
            kind = 'positive' if record['positive'] else 'negative'
            cv2.putText(
                tile,
                f"{record['stem'][:18]}  {kind}",
                (6, 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            row = index // columns
            column = index % columns
            montage[
                row * tile_height:(row + 1) * tile_height,
                column * tile_width:(column + 1) * tile_width,
            ] = tile

        return cv2.imwrite(str(preview_path), montage)

    def update_latest_link(self):
        latest_link = self.output_root / 'latest'
        latest_text = self.output_root / 'LATEST.txt'
        self.output_root.mkdir(parents=True, exist_ok=True)

        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            self.get_logger().warning(
                f'Cannot replace non-symlink path {latest_link}; '
                'LATEST.txt will still be updated.'
            )
            latest_text.write_text(
                str(self.session_dir) + '\n',
                encoding='utf-8',
            )
            return

        relative_target = os.path.relpath(
            self.session_dir,
            latest_link.parent,
        )
        latest_link.symlink_to(relative_target, target_is_directory=True)
        latest_text.write_text(
            str(self.session_dir) + '\n',
            encoding='utf-8',
        )

    def finalize_dataset(self):
        if self.finalized:
            return
        self.finalized = True

        if not self.records:
            if rclpy.ok():
                self.get_logger().error(
                    'No synchronized samples were collected. '
                    f'Raw run directory: {self.session_dir}'
                )
            return

        dataset_dir = self.session_dir / 'dataset'
        split_map = self.split_records()
        for split, records in split_map.items():
            image_dir = dataset_dir / 'images' / split
            label_dir = dataset_dir / 'labels' / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            for record in records:
                shutil.copy2(
                    record['image'],
                    image_dir / f"{record['stem']}.jpg",
                )
                shutil.copy2(
                    record['label'],
                    label_dir / f"{record['stem']}.txt",
                )

        dataset_yaml = dataset_dir / 'dataset.yaml'
        dataset_yaml.write_text(
            yaml.safe_dump(
                {
                    'path': str(dataset_dir),
                    'train': 'images/train',
                    'val': 'images/val',
                    'test': 'images/test',
                    'names': {0: self.class_name},
                },
                sort_keys=False,
            ),
            encoding='utf-8',
        )

        preview_path = self.session_dir / 'preview.jpg'
        preview_created = self.build_preview(preview_path)
        summary = {
            'session_dir': str(self.session_dir),
            'dataset_yaml': str(dataset_yaml),
            'preview': str(preview_path) if preview_created else None,
            'samples': self.sample_count,
            'positive_samples': self.positive_count,
            'negative_samples': self.negative_count,
            'split_counts': {
                split: len(records)
                for split, records in split_map.items()
            },
            'class_names': {0: self.class_name},
            'gazebo_label_id': self.gazebo_label_id,
            'split_seed': self.split_seed,
        }
        summary_path = self.session_dir / 'summary.json'
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        self.update_latest_link()

        if rclpy.ok():
            self.get_logger().info(
                'DATASET READY: '
                f'{self.sample_count} samples '
                f'({self.positive_count} positive, '
                f'{self.negative_count} negative). '
                f'YAML={dataset_yaml}; preview={preview_path}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = YoloDatasetCollectorNode()
    rclpy.get_default_context().on_shutdown(node.finalize_dataset)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finalize_dataset()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
