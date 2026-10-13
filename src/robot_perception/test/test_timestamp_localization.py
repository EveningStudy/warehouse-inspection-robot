"""Regression tests for capture-time RGB-D geometry; no ML weights required."""
import types
import unittest
import numpy as np
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from robot_perception.yolo_target_detector_node import YoloTargetDetectorNode


class FakeBuffer:
    def __init__(self):
        self.request = None

    def lookup_transform(self, target, source, timestamp, timeout):
        self.request = target, source, timestamp.nanoseconds
        transform = TransformStamped()
        transform.header.frame_id = target
        transform.child_frame_id = source
        transform.transform.rotation.w = 1.0
        transform.transform.translation.x = 1.0
        return transform


class LocalizationTests(unittest.TestCase):
    def test_transform_uses_image_exposure_not_latest_pose(self):
        buffer = FakeBuffer()
        detector = types.SimpleNamespace(
            fx=100.0, fy=100.0, cx=50.0, cy=50.0,
            target_frame='map', camera_frame_fallback='camera_optical_link',
            depth_roi_fraction=.35, minimum_depth=.15, maximum_depth=8.0,
            minimum_depth_samples=6, tf_buffer=buffer,
            estimate_target_depth=YoloTargetDetectorNode.estimate_target_depth,
        )
        detector.project_pixel_to_camera = types.MethodType(
            YoloTargetDetectorNode.project_pixel_to_camera, detector)
        message = Image()
        message.header.frame_id = 'camera_optical_link'
        message.header.stamp.sec = 42
        message.header.stamp.nanosec = 123000000
        result = YoloTargetDetectorNode.localize_detection(
            detector, message, np.full((100,100),2.0,np.float32),
            (100,100), (40,40,60,60), .95, 0)
        self.assertEqual(buffer.request, ('map','camera_optical_link',42123000000))
        self.assertTrue(result['localized'])
        self.assertAlmostEqual(result['target']['x'],1.0)
        self.assertAlmostEqual(result['target']['z'],2.0)

    def test_invalid_depth_never_produces_target(self):
        depth = np.full((20,20),np.nan,np.float32)
        self.assertIsNone(YoloTargetDetectorNode.estimate_target_depth(
            depth,(20,20),(2,2,18,18),.35,.15,8.0,6))


if __name__ == '__main__':
    unittest.main()
