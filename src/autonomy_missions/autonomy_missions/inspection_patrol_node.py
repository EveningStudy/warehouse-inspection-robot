from datetime import datetime, timezone
import json
import math
import os
import time

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker
import yaml


class InspectionPatrolNode(Node):
    """Own Nav2 while patrol is interrupted by confirmed panel detections."""

    def __init__(self):
        super().__init__('inspection_patrol_node')

        self.declare_parameter('map_profile', 'warehouse')
        self.declare_parameter('waypoint_file', '')
        self.declare_parameter('wait_seconds', 1.5)
        self.declare_parameter('standoff_distance', 0.9)
        self.declare_parameter('inspection_seconds', 3.0)
        self.declare_parameter('confirmation_count', 3)
        self.declare_parameter('confirmation_radius', 0.45)
        self.declare_parameter('confirmation_timeout', 2.0)
        self.declare_parameter('deduplication_radius', 1.5)
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('robot_base_frame', 'base_footprint')
        self.declare_parameter(
            'detection_topic',
            '/perception/yolo_detections',
        )
        self.declare_parameter('record_directory', '')
        self.declare_parameter('waypoint_timeout_seconds', 120.0)
        self.declare_parameter('approach_timeout_seconds', 90.0)
        self.declare_parameter('max_retries_per_waypoint', 1)
        self.declare_parameter('continue_on_waypoint_failure', True)
        self.declare_parameter('max_goal_rejection_retries', 30)
        self.declare_parameter('goal_rejection_retry_seconds', 1.0)
        self.declare_parameter('feedback_log_interval_seconds', 3.0)
        self.declare_parameter('maximum_tf_retries', 20)

        self.wait_seconds = max(
            0.0,
            float(self.get_parameter('wait_seconds').value),
        )
        self.standoff_distance = float(
            self.get_parameter('standoff_distance').value
        )
        self.inspection_seconds = max(
            0.0,
            float(self.get_parameter('inspection_seconds').value),
        )
        self.confirmation_count_required = max(
            1,
            int(self.get_parameter('confirmation_count').value),
        )
        self.confirmation_radius = float(
            self.get_parameter('confirmation_radius').value
        )
        self.confirmation_timeout = float(
            self.get_parameter('confirmation_timeout').value
        )
        self.deduplication_radius = float(
            self.get_parameter('deduplication_radius').value
        )
        self.target_frame = str(
            self.get_parameter('target_frame').value
        )
        self.robot_base_frame = str(
            self.get_parameter('robot_base_frame').value
        )
        self.waypoint_timeout_seconds = max(
            0.0,
            float(
                self.get_parameter('waypoint_timeout_seconds').value
            ),
        )
        self.approach_timeout_seconds = max(
            0.0,
            float(
                self.get_parameter('approach_timeout_seconds').value
            ),
        )
        self.max_retries_per_waypoint = max(
            0,
            int(
                self.get_parameter('max_retries_per_waypoint').value
            ),
        )
        self.continue_on_waypoint_failure = bool(
            self.get_parameter('continue_on_waypoint_failure').value
        )
        self.max_goal_rejection_retries = max(
            0,
            int(
                self.get_parameter(
                    'max_goal_rejection_retries'
                ).value
            ),
        )
        self.goal_rejection_retry_seconds = max(
            0.1,
            float(
                self.get_parameter(
                    'goal_rejection_retry_seconds'
                ).value
            ),
        )
        self.feedback_log_interval_seconds = max(
            0.0,
            float(
                self.get_parameter(
                    'feedback_log_interval_seconds'
                ).value
            ),
        )
        self.maximum_tf_retries = max(
            0,
            int(self.get_parameter('maximum_tf_retries').value),
        )
        self.validate_parameters()

        map_profile = str(self.get_parameter('map_profile').value)
        waypoint_file = os.path.expanduser(
            str(self.get_parameter('waypoint_file').value)
        )
        if map_profile == 'warehouse' and not waypoint_file:
            waypoint_file = os.path.join(
                get_package_share_directory('autonomy_missions'),
                'config',
                'warehouse_patrol.yaml',
            )
        self.waypoints = self.load_waypoints(waypoint_file)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.action_client = ActionClient(
            self,
            NavigateToPose,
            'navigate_to_pose',
        )

        detection_topic = str(
            self.get_parameter('detection_topic').value
        )
        self.detection_subscription = self.create_subscription(
            String,
            detection_topic,
            self.detection_callback,
            10,
        )
        self.approach_goal_publisher = self.create_publisher(
            PoseStamped,
            '/perception/approach_goal',
            10,
        )
        self.status_publisher = self.create_publisher(
            String,
            '/inspection/mission_status',
            10,
        )
        self.inspected_marker_publisher = self.create_publisher(
            Marker,
            '/inspection/inspected_markers',
            10,
        )

        self.state = 'WAITING_FOR_NAV2'
        self.finished = False
        self.current_index = 0
        self.retry_count = 0
        self.goal_rejection_count = 0
        self.goal_sequence = 0
        self.active_goal_handle = None
        self.active_goal_context = None
        self.goal_started_at = None
        self.last_feedback_log_at = 0.0
        self.cancel_reason = None
        self.cancel_requested = False
        self.pending_detection = None
        self.confirmation_candidate = None
        self.confirmation_count = 0
        self.confirmation_last_at = 0.0
        self.handled_targets = []
        self.successful_inspections = 0
        self.failed_inspections = 0
        self.approach_tf_retries = 0
        self.transition_timer = None
        self.summary_written = False

        self.record_directory = self.create_record_directory()
        self.events_path = os.path.join(
            self.record_directory,
            'mission_events.jsonl',
        )
        self.inspections_path = os.path.join(
            self.record_directory,
            'inspections.jsonl',
        )
        self.started_at = self.utc_now()

        self.server_timer = self.create_timer(
            1.0,
            self.wait_for_nav2,
        )
        self.watchdog_timer = self.create_timer(
            0.5,
            self.navigation_watchdog,
        )

        self.record_event(
            'mission_started',
            waypoint_count=len(self.waypoints),
            record_directory=self.record_directory,
        )
        self.get_logger().info(
            'Inspection patrol ready: '
            f'{len(self.waypoints)} waypoints, '
            f'confirmation={self.confirmation_count_required} frames, '
            f'standoff={self.standoff_distance:.2f} m, '
            f'records={self.record_directory}'
        )

    def validate_parameters(self):
        """Validate the geometry and confirmation configuration."""
        positive_values = {
            'standoff_distance': self.standoff_distance,
            'confirmation_radius': self.confirmation_radius,
            'confirmation_timeout': self.confirmation_timeout,
            'deduplication_radius': self.deduplication_radius,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f'{name} must be positive')

    @staticmethod
    def load_waypoints(waypoint_file):
        """Load the existing patrol YAML format."""
        if not waypoint_file or not os.path.isfile(waypoint_file):
            raise FileNotFoundError(
                f'Patrol waypoint file does not exist: {waypoint_file}'
            )
        with open(waypoint_file, 'r', encoding='utf-8') as stream:
            document = yaml.safe_load(stream) or {}
        try:
            entries = document['patrol_node']['ros__parameters'][
                'waypoints'
            ]
        except (KeyError, TypeError) as error:
            raise ValueError(
                'Waypoint YAML must contain '
                'patrol_node.ros__parameters.waypoints'
            ) from error

        waypoints = []
        for index, entry in enumerate(entries):
            try:
                waypoints.append(
                    (
                        str(entry['name']),
                        float(entry['x']),
                        float(entry['y']),
                        float(entry['yaw']),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f'Invalid waypoint entry at index {index}: {entry}'
                ) from error
        if not waypoints:
            raise ValueError('Waypoint file contains no waypoints')
        return waypoints

    def create_record_directory(self):
        """Create one timestamped mission directory without overwriting."""
        configured_root = os.path.expanduser(
            str(self.get_parameter('record_directory').value)
        )
        if not configured_root:
            configured_root = os.path.expanduser(
                '~/.ros/autonomy_lab/inspection_records'
            )
        run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_directory = os.path.join(configured_root, run_name)
        suffix = 1
        while os.path.exists(run_directory):
            run_directory = os.path.join(
                configured_root,
                f'{run_name}_{suffix:02d}',
            )
            suffix += 1
        os.makedirs(run_directory, exist_ok=False)
        return run_directory

    def wait_for_nav2(self):
        """Start only after the NavigateToPose action server exists."""
        if not self.action_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().info(
                'Waiting for Nav2 navigate_to_pose server...'
            )
            return
        self.server_timer.cancel()
        self.record_event('nav2_available')
        self.send_current_waypoint()

    def send_current_waypoint(self):
        """Send or resume the current patrol waypoint."""
        self.cancel_transition_timer()
        if self.current_index >= len(self.waypoints):
            self.finish_mission()
            return
        name, x, y, yaw = self.waypoints[self.current_index]
        pose = self.create_pose(x, y, yaw)
        self.send_navigation_goal(
            kind='patrol',
            pose=pose,
            name=name,
            waypoint_index=self.current_index,
        )

    def send_navigation_goal(
        self,
        kind,
        pose,
        name,
        waypoint_index=None,
    ):
        """Send one Nav2 goal and attach sequence-safe callbacks."""
        self.goal_sequence += 1
        sequence = self.goal_sequence
        self.active_goal_handle = None
        self.active_goal_context = {
            'sequence': sequence,
            'kind': kind,
            'pose': pose,
            'name': name,
            'waypoint_index': waypoint_index,
        }
        self.goal_started_at = time.monotonic()
        self.last_feedback_log_at = 0.0
        self.cancel_reason = None
        self.cancel_requested = False
        self.state = (
            'SENDING_PATROL' if kind == 'patrol'
            else 'SENDING_APPROACH'
        )

        goal = NavigateToPose.Goal()
        goal.pose = pose
        future = self.action_client.send_goal_async(
            goal,
            feedback_callback=(
                lambda message, goal_sequence=sequence:
                self.feedback_callback(message, goal_sequence)
            ),
        )
        future.add_done_callback(
            lambda response, goal_sequence=sequence:
            self.goal_response_callback(response, goal_sequence)
        )
        self.record_event(
            'navigation_goal_sent',
            kind=kind,
            name=name,
            waypoint_index=waypoint_index,
            goal=self.pose_to_dict(pose),
        )
        self.get_logger().info(
            f"Sending {kind} goal '{name}': "
            f'({pose.pose.position.x:.2f}, '
            f'{pose.pose.position.y:.2f})'
        )

    def goal_response_callback(self, future, sequence):
        """Handle Nav2 accepting or temporarily rejecting a goal."""
        if not self.is_current_sequence(sequence):
            return
        try:
            goal_handle = future.result()
        except Exception as exception:
            self.handle_goal_submission_failure(
                f'exception while sending goal: {exception}'
            )
            return

        if goal_handle is None or not goal_handle.accepted:
            self.handle_goal_rejection()
            return

        self.active_goal_handle = goal_handle
        self.goal_rejection_count = 0
        kind = self.active_goal_context['kind']
        self.state = 'PATROLLING' if kind == 'patrol' else 'APPROACHING'
        self.record_event(
            'navigation_goal_accepted',
            kind=kind,
            name=self.active_goal_context['name'],
        )
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda result, goal_sequence=sequence:
            self.navigation_result_callback(result, goal_sequence)
        )

    def handle_goal_rejection(self):
        """Retry startup-time Nav2 rejections without losing mission state."""
        context = self.active_goal_context
        if self.goal_rejection_count < self.max_goal_rejection_retries:
            self.goal_rejection_count += 1
            self.get_logger().warning(
                f"Nav2 rejected {context['kind']} goal "
                f"'{context['name']}'. Retrying in "
                f'{self.goal_rejection_retry_seconds:.1f} s '
                f'({self.goal_rejection_count}/'
                f'{self.max_goal_rejection_retries}).'
            )
            self.schedule_transition(
                self.goal_rejection_retry_seconds,
                self.resend_active_context,
            )
            return
        self.handle_goal_submission_failure(
            'Nav2 kept rejecting the goal'
        )

    def resend_active_context(self):
        """Resend a rejected goal using its stored context."""
        context = dict(self.active_goal_context)
        self.send_navigation_goal(
            kind=context['kind'],
            pose=context['pose'],
            name=context['name'],
            waypoint_index=context['waypoint_index'],
        )

    def handle_goal_submission_failure(self, reason):
        """Route a permanent submission failure to the correct recovery."""
        context = self.active_goal_context
        self.get_logger().error(reason)
        if context and context['kind'] == 'approach':
            self.complete_inspection(
                success=False,
                status='goal_submission_failed',
                error_message=reason,
            )
        else:
            self.handle_patrol_failure(reason)

    def feedback_callback(self, feedback_message, sequence):
        """Throttle feedback while the independent watchdog tracks timeout."""
        if not self.is_current_sequence(sequence):
            return
        now = time.monotonic()
        if (
            self.feedback_log_interval_seconds > 0.0
            and now - self.last_feedback_log_at
            < self.feedback_log_interval_seconds
        ):
            return
        self.last_feedback_log_at = now
        feedback = feedback_message.feedback
        context = self.active_goal_context
        self.get_logger().info(
            f"{context['kind']} '{context['name']}': "
            f'{feedback.distance_remaining:.2f} m remaining, '
            f'recoveries={feedback.number_of_recoveries}'
        )

    def navigation_watchdog(self):
        """Cancel a navigation goal using predictable wall-clock time."""
        if self.state not in ('PATROLLING', 'APPROACHING'):
            return
        if self.active_goal_handle is None or self.goal_started_at is None:
            return
        context = self.active_goal_context
        timeout = (
            self.waypoint_timeout_seconds
            if context['kind'] == 'patrol'
            else self.approach_timeout_seconds
        )
        if timeout <= 0.0:
            return
        if time.monotonic() - self.goal_started_at < timeout:
            return
        self.get_logger().error(
            f"{context['kind']} goal '{context['name']}' exceeded "
            f'{timeout:.1f} wall-clock seconds; canceling.'
        )
        self.request_active_goal_cancel('timeout')

    def request_active_goal_cancel(self, reason):
        """Request cancellation once and retain the reason for result routing."""
        if self.active_goal_handle is None or self.cancel_requested:
            return
        self.cancel_requested = True
        self.cancel_reason = reason
        self.state = (
            'CANCELING_FOR_INSPECTION'
            if reason == 'inspection'
            else 'CANCELING_TIMEOUT'
        )
        future = self.active_goal_handle.cancel_goal_async()
        future.add_done_callback(self.cancel_response_callback)
        self.record_event(
            'navigation_cancel_requested',
            reason=reason,
            kind=self.active_goal_context['kind'],
            name=self.active_goal_context['name'],
        )

    def cancel_response_callback(self, future):
        """Report a failed cancellation request while awaiting final status."""
        try:
            response = future.result()
            canceling = response.goals_canceling
        except Exception as exception:
            self.get_logger().error(
                f'Cancel request failed: {exception}'
            )
            return
        if not canceling:
            self.get_logger().warning(
                'Nav2 did not accept the cancel request; awaiting the '
                'navigation result.'
            )

    def navigation_result_callback(self, future, sequence):
        """Continue patrol, start inspection, or recover after Nav2 result."""
        if not self.is_current_sequence(sequence):
            return
        context = self.active_goal_context
        try:
            wrapped = future.result()
            status = wrapped.status
            result = wrapped.result
            error_code = int(result.error_code)
            error_message = str(result.error_msg)
        except Exception as exception:
            status = GoalStatus.STATUS_UNKNOWN
            error_code = -1
            error_message = str(exception)

        self.active_goal_handle = None
        self.goal_started_at = None
        self.record_event(
            'navigation_result',
            kind=context['kind'],
            name=context['name'],
            status=int(status),
            error_code=error_code,
            error_message=error_message,
            cancel_reason=self.cancel_reason,
        )

        if context['kind'] == 'patrol':
            self.handle_patrol_result(
                status,
                error_code,
                error_message,
            )
        else:
            self.handle_approach_result(
                status,
                error_code,
                error_message,
            )

    def handle_patrol_result(self, status, error_code, error_message):
        """Advance, inspect after cancellation, or retry a waypoint."""
        name = self.active_goal_context['name']
        if (
            self.cancel_reason == 'inspection'
            and self.pending_detection is not None
        ):
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().info(
                    f"Reached patrol waypoint '{name}' while its cancel "
                    'request was completing.'
                )
                self.retry_count = 0
                self.current_index += 1
            self.get_logger().info(
                f"Patrol paused at '{name}' for a confirmed panel."
            )
            self.approach_tf_retries = 0
            self.try_send_approach_goal()
            return

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"Reached patrol waypoint '{name}'.")
            self.retry_count = 0
            self.current_index += 1
            if self.current_index >= len(self.waypoints):
                self.finish_mission()
                return
            self.state = 'WAITING_PATROL'
            self.schedule_transition(
                self.wait_seconds,
                self.send_current_waypoint,
            )
            return

        reason = (
            f'status={status}, error_code={error_code}, '
            f"error_message='{error_message}'"
        )
        self.handle_patrol_failure(reason)

    def handle_patrol_failure(self, reason):
        """Retry or skip one patrol waypoint using existing project policy."""
        if self.current_index >= len(self.waypoints):
            self.finish_mission()
            return
        name = self.waypoints[self.current_index][0]
        if self.retry_count < self.max_retries_per_waypoint:
            self.retry_count += 1
            self.get_logger().warning(
                f"Patrol waypoint '{name}' failed: {reason}. Retrying "
                f'({self.retry_count}/{self.max_retries_per_waypoint}).'
            )
            self.state = 'WAITING_PATROL'
            self.schedule_transition(
                self.wait_seconds,
                self.send_current_waypoint,
            )
            return
        if self.continue_on_waypoint_failure:
            self.get_logger().error(
                f"Skipping patrol waypoint '{name}' after failure: {reason}"
            )
            self.record_event(
                'patrol_waypoint_skipped',
                name=name,
                waypoint_index=self.current_index,
                reason=reason,
            )
            self.retry_count = 0
            self.current_index += 1
            self.state = 'WAITING_PATROL'
            self.schedule_transition(
                self.wait_seconds,
                self.send_current_waypoint,
            )
            return
        self.finish_mission(completed=False, reason=reason)

    def detection_callback(self, message):
        """Confirm a new map-space target before interrupting patrol."""
        if self.state not in ('PATROLLING', 'WAITING_PATROL'):
            return
        try:
            payload = json.loads(message.data)
            detections = payload.get('detections', [])
        except (json.JSONDecodeError, AttributeError, TypeError) as exception:
            self.get_logger().warning(
                f'Ignoring invalid YOLO detection message: {exception}'
            )
            return
        if payload.get('target_frame') != self.target_frame:
            return

        candidate = self.select_unhandled_detection(detections)
        if candidate is None:
            self.reset_confirmation()
            return
        if not self.update_confirmation(candidate):
            return

        confirmed = dict(self.confirmation_candidate)
        self.reset_confirmation()
        self.begin_inspection(confirmed)

    def select_unhandled_detection(self, detections):
        """Choose the highest-confidence valid target not inspected before."""
        candidates = []
        for detection in detections:
            try:
                target = detection['target']
                x = float(target['x'])
                y = float(target['y'])
                z = float(target['z'])
                confidence = float(detection['confidence'])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x, y, z)):
                continue
            if self.is_handled_target(x, y):
                continue
            candidates.append(
                {
                    'x': x,
                    'y': y,
                    'z': z,
                    'confidence': confidence,
                    'depth_m': detection.get('depth_m'),
                    'bbox': detection.get('bbox'),
                }
            )
        if not candidates:
            return None
        return max(candidates, key=lambda item: item['confidence'])

    def update_confirmation(self, candidate):
        """Require nearby detections across multiple inference callbacks."""
        now = time.monotonic()
        reset = (
            self.confirmation_candidate is None
            or now - self.confirmation_last_at
            > self.confirmation_timeout
            or self.planar_distance(
                candidate,
                self.confirmation_candidate,
            ) > self.confirmation_radius
        )
        if reset:
            self.confirmation_candidate = dict(candidate)
            self.confirmation_count = 1
        else:
            count = self.confirmation_count
            for key in ('x', 'y', 'z'):
                self.confirmation_candidate[key] = (
                    self.confirmation_candidate[key] * count
                    + candidate[key]
                ) / (count + 1)
            self.confirmation_candidate['confidence'] = max(
                self.confirmation_candidate['confidence'],
                candidate['confidence'],
            )
            self.confirmation_candidate['depth_m'] = candidate.get(
                'depth_m'
            )
            self.confirmation_candidate['bbox'] = candidate.get('bbox')
            self.confirmation_count += 1
        self.confirmation_last_at = now

        if self.confirmation_count < self.confirmation_count_required:
            return False
        self.get_logger().info(
            'Confirmed a new panel after '
            f'{self.confirmation_count} detections at '
            f"({self.confirmation_candidate['x']:.2f}, "
            f"{self.confirmation_candidate['y']:.2f})."
        )
        return True

    def begin_inspection(self, detection):
        """Pause patrol now, or use an existing patrol wait period."""
        self.pending_detection = detection
        self.record_event(
            'panel_confirmed',
            detection=detection,
            interrupted_waypoint_index=self.current_index,
        )
        if self.state == 'WAITING_PATROL':
            self.cancel_transition_timer()
            self.approach_tf_retries = 0
            self.try_send_approach_goal()
            return
        self.request_active_goal_cancel('inspection')

    def try_send_approach_goal(self):
        """Calculate a safe standoff pose after the patrol goal is stopped."""
        if self.pending_detection is None:
            self.resume_patrol()
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.robot_base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exception:
            if self.approach_tf_retries < self.maximum_tf_retries:
                self.approach_tf_retries += 1
                self.get_logger().warning(
                    'Waiting for robot pose before approach '
                    f'({self.approach_tf_retries}/'
                    f'{self.maximum_tf_retries}): {exception}'
                )
                self.schedule_transition(
                    0.5,
                    self.try_send_approach_goal,
                )
                return
            self.complete_inspection(
                success=False,
                status='tf_unavailable',
                error_message=str(exception),
            )
            return

        target = self.pending_detection
        calculation = self.calculate_approach_pose(
            robot_x=transform.transform.translation.x,
            robot_y=transform.transform.translation.y,
            target_x=target['x'],
            target_y=target['y'],
            target_frame=self.target_frame,
            stamp=self.get_clock().now().to_msg(),
            standoff_distance=self.standoff_distance,
        )
        if calculation is None:
            self.complete_inspection(
                success=False,
                status='invalid_approach_geometry',
                error_message='robot and target positions are identical',
            )
            return
        pose, distance, yaw = calculation
        self.pending_detection['approach_goal'] = self.pose_to_dict(pose)
        self.pending_detection['distance_before_approach'] = distance
        self.pending_detection['approach_yaw'] = yaw
        self.approach_goal_publisher.publish(pose)
        self.send_navigation_goal(
            kind='approach',
            pose=pose,
            name='inspection panel',
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
        """Return a pose before the target, facing the target."""
        delta_x = target_x - robot_x
        delta_y = target_y - robot_y
        distance = math.hypot(delta_x, delta_y)
        if distance < 1.0e-6:
            return None
        direction_x = delta_x / distance
        direction_y = delta_y / distance
        approach_x = target_x - standoff_distance * direction_x
        approach_y = target_y - standoff_distance * direction_y
        yaw = math.atan2(
            target_y - approach_y,
            target_x - approach_x,
        )
        pose = PoseStamped()
        pose.header.frame_id = target_frame
        pose.header.stamp = stamp
        pose.pose.position.x = float(approach_x)
        pose.pose.position.y = float(approach_y)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose, distance, yaw

    def handle_approach_result(self, status, error_code, error_message):
        """Record inspection outcome and resume the interrupted waypoint."""
        success = status == GoalStatus.STATUS_SUCCEEDED
        if success:
            self.get_logger().info(
                'Reached the inspection standoff pose. Recording result.'
            )
            outcome = 'succeeded'
        else:
            self.get_logger().error(
                'Panel approach failed: '
                f'status={status}, error_code={error_code}, '
                f"error_message='{error_message}'"
            )
            outcome = 'navigation_failed'
        self.complete_inspection(
            success=success,
            status=outcome,
            error_code=error_code,
            error_message=error_message,
        )

    def complete_inspection(
        self,
        success,
        status,
        error_code=0,
        error_message='',
    ):
        """Persist one handled target and suppress repeat approaches."""
        detection = dict(self.pending_detection or {})
        if all(key in detection for key in ('x', 'y', 'z')):
            handled = {
                'x': detection['x'],
                'y': detection['y'],
                'z': detection['z'],
                'success': bool(success),
            }
            self.handled_targets.append(handled)
            self.publish_inspected_marker(handled)

        record = {
            'timestamp': self.utc_now(),
            'inspection_number': len(self.handled_targets),
            'success': bool(success),
            'status': status,
            'error_code': int(error_code),
            'error_message': str(error_message),
            'target': detection,
            'resume_waypoint_index': self.current_index,
        }
        self.append_json_line(self.inspections_path, record)
        self.record_event('inspection_recorded', **record)
        if success:
            self.successful_inspections += 1
        else:
            self.failed_inspections += 1

        self.pending_detection = None
        self.state = 'INSPECTING'
        self.schedule_transition(
            self.inspection_seconds,
            self.resume_patrol,
        )

    def resume_patrol(self):
        """Re-send the waypoint that was active before inspection."""
        self.reset_confirmation()
        if self.current_index >= len(self.waypoints):
            self.finish_mission()
            return
        name = self.waypoints[self.current_index][0]
        self.get_logger().info(
            f"Inspection complete. Resuming patrol at '{name}'."
        )
        self.record_event(
            'patrol_resumed',
            waypoint_index=self.current_index,
            name=name,
        )
        self.send_current_waypoint()

    def is_handled_target(self, x, y):
        """Cluster repeat observations of a previously handled panel."""
        candidate = {'x': x, 'y': y}
        return any(
            self.planar_distance(candidate, target)
            <= self.deduplication_radius
            for target in self.handled_targets
        )

    def reset_confirmation(self):
        self.confirmation_candidate = None
        self.confirmation_count = 0
        self.confirmation_last_at = 0.0

    def schedule_transition(self, delay_seconds, callback):
        """Create a one-shot ROS timer."""
        self.cancel_transition_timer()

        def run_once():
            self.cancel_transition_timer()
            callback()

        self.transition_timer = self.create_timer(
            max(0.01, delay_seconds),
            run_once,
        )

    def cancel_transition_timer(self):
        if self.transition_timer is not None:
            self.transition_timer.cancel()
            self.transition_timer = None

    def finish_mission(self, completed=True, reason='route_finished'):
        """Finish the route and write a machine-readable summary."""
        if self.finished:
            return
        self.finished = True
        self.state = 'FINISHED'
        self.cancel_transition_timer()
        self.record_event(
            'mission_finished',
            completed=bool(completed),
            reason=reason,
            successful_inspections=self.successful_inspections,
            failed_inspections=self.failed_inspections,
        )
        self.write_summary(completed, reason)
        self.get_logger().info(
            'Inspection patrol finished: '
            f'inspected={len(self.handled_targets)}, '
            f'successful={self.successful_inspections}, '
            f'failed={self.failed_inspections}, '
            f'records={self.record_directory}'
        )

    def write_summary(self, completed, reason):
        """Write final or interrupted mission state once."""
        if self.summary_written:
            return
        summary = {
            'schema_version': 1,
            'started_at': self.started_at,
            'finished_at': self.utc_now(),
            'completed': bool(completed),
            'reason': reason,
            'state': self.state,
            'waypoints_total': len(self.waypoints),
            'next_waypoint_index': self.current_index,
            'handled_targets': self.handled_targets,
            'successful_inspections': self.successful_inspections,
            'failed_inspections': self.failed_inspections,
            'events_file': self.events_path,
            'inspections_file': self.inspections_path,
        }
        path = os.path.join(self.record_directory, 'mission_summary.json')
        with open(path, 'w', encoding='utf-8') as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
        self.summary_written = True

    def record_event(self, event, **fields):
        """Append an event and publish the same JSON on a status topic."""
        record = {
            'timestamp': self.utc_now(),
            'event': event,
            'state': self.state,
            **fields,
        }
        self.append_json_line(self.events_path, record)
        message = String()
        message.data = json.dumps(
            record,
            ensure_ascii=False,
            separators=(',', ':'),
        )
        self.status_publisher.publish(message)

    @staticmethod
    def append_json_line(path, record):
        with open(path, 'a', encoding='utf-8') as stream:
            stream.write(
                json.dumps(record, ensure_ascii=False, separators=(',', ':'))
                + '\n'
            )

    def publish_inspected_marker(self, target):
        """Leave a persistent green marker at each handled target."""
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'inspected_panels'
        marker.id = len(self.handled_targets) - 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target['x']
        marker.pose.position.y = target['y']
        marker.pose.position.z = target['z']
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.28
        marker.scale.y = 0.28
        marker.scale.z = 0.28
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 0.95
        self.inspected_marker_publisher.publish(marker)

    def is_current_sequence(self, sequence):
        return (
            self.active_goal_context is not None
            and sequence == self.active_goal_context['sequence']
        )

    def create_pose(self, x, y, yaw):
        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(yaw / 2.0)
        pose.pose.orientation.w = math.cos(yaw / 2.0)
        return pose

    @staticmethod
    def pose_to_dict(pose):
        return {
            'frame_id': pose.header.frame_id,
            'position': {
                'x': float(pose.pose.position.x),
                'y': float(pose.pose.position.y),
                'z': float(pose.pose.position.z),
            },
            'orientation': {
                'x': float(pose.pose.orientation.x),
                'y': float(pose.pose.orientation.y),
                'z': float(pose.pose.orientation.z),
                'w': float(pose.pose.orientation.w),
            },
        }

    @staticmethod
    def planar_distance(first, second):
        return math.hypot(
            float(first['x']) - float(second['x']),
            float(first['y']) - float(second['y']),
        )

    @staticmethod
    def utc_now():
        return datetime.now(timezone.utc).isoformat()

    def destroy_node(self):
        if not self.summary_written:
            self.write_summary(False, 'node_stopped')
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = InspectionPatrolNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        if rclpy.ok():
            node.get_logger().info(
                'Inspection patrol stopped by user.'
            )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
