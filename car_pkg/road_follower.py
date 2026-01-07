#!/usr/bin/env python3

"""Road follower node.

Description:
    Node that detects the lateral road center from `/road_camera/image`
    frames and publishes steering commands to keep the vehicle centered
    in the lane. Detection uses a column-intensity analysis and a small
    PID controller operating on pixel error. The controller includes
    protections against integral windup and simple recovery strategies
    for short detection outages.

Publishes:
    /control/steering (std_msgs/Float32): Steering setpoint for the vehicle.

Subscribes:
    /road_camera/image (sensor_msgs/Image): Camera frames used for lane detection.
"""

import rclpy
from rclpy.node import Node
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import time


class RoadFollower(Node):
    """ROS2 node that detects the road center and publishes steering.

    Detection strategy:
    - Compute a column-wise brightness profile, smooth it, and locate the
        dominant peak as the lane/road center.
    - Scenes are classified into `LINE`, `NO_LINE`, or `CROSSWALK` to decide
        how to behave when clear lane markings are missing.
    - For brief detection outages, the node reuses the last confident center
        to avoid oscillatory steering commands.
    """

    def __init__(self):
        """Initialize the `RoadFollower` node and controller state.

        Side effects:
            - Creates ROS2 publisher `/control/steering` and subscriber
                `/road_camera/image`.
            - Initializes PID gains, internal integrators, and persistence
                counters used for detection smoothing and recovery.
        """
        super().__init__('road_follower')
        self.bridge = CvBridge()
        
        # Publishers
        self.steering_pub = self.create_publisher(Float32, '/control/steering', 10)

        # Subscribers
        self.image_sub = self.create_subscription(Image, '/road_camera/image', self.image_callback, 10)

        # Image center (pixels). Many simulator frames are 512 px wide.
        # This value is updated per-frame in `image_callback` when image
        # metadata is available so the controller operates on correct size.
        self.image_center = 256

        # Last observed centers and persistence counters used to smooth
        # behavior when detections are weak or temporarily missing.
        self.last_known_road_center = self.image_center
        self.last_confident_center = self.image_center
        self.no_line_persist_counter = 0
        # Number of consecutive frames to reuse last confident center
        # before attempting fresh recovery. At ~30 FPS, 15 frames ≈ 0.5 s.
        self.no_line_persist_max = 30

        # PID tuning (operates on pixel error). Values were chosen to be
        # conservative for the simulator; tune as needed in-sim.
        self.Kp = 0.012
        self.Ki = 0.000001
        self.Kd = 0.0005

        # Controller state
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

        # Clamp for integral term to limit long-term bias
        self.integral_max = 50.0
        # Maximum absolute steering command (radians or simulator units)
        self.max_steering = 0.25

        # Track previous detection confidence to reset controller state on
        # transitions from confident -> unconfident views.
        self.prev_confidence = True

    def image_callback(self, msg):
        """Handle incoming camera frames, detect road center, and publish steering.

        Args:
            msg (sensor_msgs.msg.Image): ROS image message from the road camera.

        Behaviour:
            - Converts the incoming ROS image to OpenCV format.
            - Updates `self.image_center` to match the current frame width.
            - Runs `follow_road` to obtain a candidate road center and
              a confidence flag.
            - Computes a steering command via `calculate_pid` and publishes
              it on `/control/steering`.

        Side effects:
            Updates controller internal state (`last_known_road_center`
            and PID integrators) and publishes steering messages.
        """
        try:

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Update image center for the current frame so PID uses correct
            # pixel coordinates (handles variable-width frames).
            try:
                self.image_center = cv_image.shape[1] // 2
            except Exception:
                # Keep previous value if image metadata unavailable
                pass

            road_center, confidence = self.follow_road(cv_image)

            # Store the latest detected center (integer pixels) and compute
            # a steering command based on the controller state.
            self.last_known_road_center = int(road_center)
            steering = self.calculate_pid(self.last_known_road_center, confidence)

            steering_msg = Float32()
            steering_msg.data = steering
            self.steering_pub.publish(steering_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def follow_road(self, image):
        """Top-level detection pipeline: classify scene and detect center.

        Args:
            image (numpy.ndarray): BGR image array from the camera.

        Returns:
            tuple: (road_center_x (int), confidence (bool)).

        Behaviour:
            - Classifies the scene using `_classify_scene`.
            - For `NO_LINE` or `CROSSWALK`, attempts to reuse a recent
              confident center for a short persistence window, otherwise
              performs a relaxed detection (`_handle_line(no_yellow=True)`).
            - For `LINE`, uses the normal detection pipeline.

        Side effects:
            May update `self.last_confident_center` and persistence counters
            when confident detections are found.
        """
        image_center = image.shape[1] // 2
        try:
            scene, yellow_ratio = self._classify_scene(image)

            # CROSSWALK and NO_LINE share a recovery strategy: reuse a
            # recently confident center for a short persistence window to
            # avoid jitter, otherwise attempt a weaker detection pass.
            if scene in ('NO_LINE', 'CROSSWALK'):
                if self.no_line_persist_counter < self.no_line_persist_max and self.last_confident_center is not None:
                    # Reuse previous confident center for stability
                    self.no_line_persist_counter += 1
                    return int(self.last_confident_center), False

                # Try a relaxed detection pass that ignores strong yellow
                # markers (useful when paint is faded or partially occluded).
                center, conf = self._handle_line(image, no_yellow=True)
                if conf:
                    self.last_confident_center = int(center)
                    self.no_line_persist_counter = 0
                    return center, conf

                # No reliable detection: fall back to last confident center
                if self.last_confident_center is not None:
                    return int(self.last_confident_center), False
                self.no_line_persist_counter = 0
                return image_center, False

            # LINE: normal detection path
            self.no_line_persist_counter = 0
            center, conf = self._handle_line(image, no_yellow=False)
            if conf:
                self.last_confident_center = int(center)
            return center, conf
        except Exception:
            # On unexpected errors, return the image center and mark as
            # unconfident so the controller avoids acting aggressively.
            return image_center, False

    def _classify_scene(self, image):
        """Classify scene type using a heuristic yellow-paint ratio.

        Args:
            image (numpy.ndarray): BGR image array from the camera.

        Returns:
            tuple: (scene_str (str), yellow_ratio (float)).

        Scene labels:
            - 'CROSSWALK' when a large fraction of the image contains yellow.
            - 'NO_LINE' when there is very little yellow present.
            - 'LINE' otherwise.
        """
        try:
            h, w = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            yellow_lower = np.array([15, 100, 100])
            yellow_upper = np.array([35, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            yellow_ratio = float(np.sum(yellow_mask > 0)) / float(max(1, h * w))
            if yellow_ratio > 0.25:
                return 'CROSSWALK', yellow_ratio
            if yellow_ratio < 0.07:
                return 'NO_LINE', yellow_ratio
            return 'LINE', yellow_ratio
        except Exception:
            return 'LINE', 0.0

    def _handle_line(self, image, no_yellow=False):
        """Detect lane center via smoothed column brightness and heuristics.

        Args:
            image (numpy.ndarray): BGR image array from the camera.
            no_yellow (bool): If True, be conservative about accepting
                very strong bright peaks (used during relaxed recovery).

        Returns:
            tuple: (x (int), confidence (bool)).

        Behaviour:
            - Computes a column-wise brightness profile and smooths it.
            - Selects the dominant peak as a candidate lane center when
              it meets conservative confidence thresholds.
            - If the peak is weak, searches locally near
              `self.last_known_road_center` for a plausible center.

        Side effects:
            None beyond returning a candidate center and confidence flag.
        """
        try:
            # Prepare a stable brightness signal
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Column-wise average brightness
            column_means = np.mean(blurred, axis=0)

            # Smooth the 1D signal to avoid single-pixel noise creating
            # false peaks. Kernel size is small relative to image width.
            SMOOTH_K = 11
            kernel = np.ones(SMOOTH_K) / SMOOTH_K
            pad = SMOOTH_K // 2
            padded = np.pad(column_means, pad, mode='reflect')
            column_means_smooth = np.convolve(padded, kernel, mode='valid')

            max_idx = int(np.argmax(column_means_smooth))
            max_val = column_means_smooth[max_idx]

            # A strong peak relative to nearby values indicates a confident
            # lane marking. Thresholds here are intentionally conservative.
            if max_val > 30:
                window = 40
                start = max(0, max_idx - window)
                end = min(column_means_smooth.shape[0] - 1, max_idx + window)
                if end > start:
                    local_avg = np.mean(column_means_smooth[start:end])
                    if max_val > local_avg * 1.15:
                        if no_yellow:
                            # In relaxed/no-yellow mode we don't accept
                            # very strong bright peaks as confident.
                            pass
                        else:
                            return max_idx, True

            # Weak peak: attempt a localized search near the last known
            # center to recover from partial occlusions or faded markings.
            search_center = self.last_known_road_center
            search_window = 180
            start = max(0, search_center - search_window // 2)
            end = min(column_means_smooth.shape[0] - 1, search_center + search_window // 2)
            if end > start:
                local_max = np.argmax(column_means_smooth[start:end]) + start
                if column_means_smooth[local_max] > 20:
                    # Found a weaker but plausible center
                    return local_max, False

            # No reliable detection: return image center and mark as
            # unconfident so the controller avoids active corrections.
            return image.shape[1] // 2, False
        except Exception:
            return image.shape[1] // 2, False

    def calculate_pid(self, road_center, confidence):
        """PID controller for lateral error (in pixels).

        Args:
            road_center (int): Detected center x-coordinate (pixels).
            confidence (bool): Whether detection is considered reliable.
                When False the controller reduces gains and resets
                integral/derivative state to avoid windup.

        Returns:
            float: Steering command clamped to [-self.max_steering, self.max_steering].

        Side effects:
            Updates PID internal state: `self.integral`, `self.prev_error`,
            `self.prev_time`, and `self.prev_confidence`.
        """

        # If the vehicle is centered and the detection is unconfident,
        # explicitly avoid sending a steering command.
        if road_center == self.image_center and not confidence:
            return 0.0

        image_center = self.image_center
        current_time = time.time()
        dt = max(0.001, min(current_time - self.prev_time, 0.1))

        # Lateral error (pixels): positive means center is to the right.
        error = road_center - image_center
        filtered_error = error

        # Effective proportional gain depends on detection confidence.
        Kp_eff = self.Kp if confidence else 0.0
        P = Kp_eff * filtered_error

        # On transition to unconfident, reset integral/derivative to avoid
        # windup and sudden corrective actions.
        if not confidence and self.prev_confidence:
            self.integral = 0.0
            self.prev_error = 0.0

        Ki_eff = self.Ki if confidence else 0.0
        Kd_eff = self.Kd if confidence else 0.0
        integral_gain = 1.0 if confidence else 0.0
        self.integral += filtered_error * dt * integral_gain
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        I = Ki_eff * self.integral

        if dt > 0.01:
            derivative = (filtered_error - self.prev_error) / dt
            derivative = np.clip(derivative, -15, 15)
            D = Kd_eff * derivative
        else:
            derivative = 0.0
            D = 0.0

        steering = P + I + D
        steering = max(-self.max_steering, min(self.max_steering, steering))

        # When detection is unreliable do not issue corrections; this keeps
        # the actuator command neutral while preserving internal state.
        if not confidence:
            steering = 0.0

        # Update controller state for the next iteration
        self.prev_error = filtered_error
        self.prev_time = current_time
        self.prev_confidence = confidence

        return steering

def main(args=None):
    """Module entry point to start the `RoadFollower` node.

    Args:
        args (list or None): Optional arguments forwarded to `rclpy.init`.

    Behaviour:
        Initializes the ROS2 client library, creates the `RoadFollower`
        node, and spins until shutdown. Ensures proper cleanup on exit.
    """

    rclpy.init(args=args)
    node = RoadFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
