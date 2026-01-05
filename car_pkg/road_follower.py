#!/usr/bin/env python3

"""Road follower node: detect lane/road center and publish steering.

Subscriptions:
- `/road_camera/image` (sensor_msgs/Image): camera frames used for
    lane/road-center detection.

Publications:
- `/control/steering` (std_msgs/Float32): steering command published
    as a single float (simulator/motor steering units).

Summary:
This node finds a lateral road center using a lightweight
column-intensity approach, converts pixel error to a steering command
via a simple PID controller, and publishes the command. The node is
designed to be robust in simulation: it avoids integral windup when
detections are unreliable and falls back to recent confident
detections for short outages.

Design goals:
- Keep the detector simple and robust for a simulation environment.
- Avoid control windup when visual detections are unreliable.
- Fall back to recent confident detections for short outages.

The PID controller operates on pixel error (pixels between detected
center and image center). When `confidence` is False the controller
resets or ignores integral/derivative contributions to prevent
unwanted steering from noisy measurements.
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
        super().__init__('road_follower')
        self.bridge = CvBridge()

        # Publishers
        self.steering_pub = self.create_publisher(Float32, '/control/steering', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, '/road_camera/image', self.image_callback, 10)

        # Image center (pixels). Many simulator frames are 512 px wide; store
        # a default here for cases where image metadata isn't available.
        self.image_center = 256

        # Last observed centers and persistence counters used to smooth
        # behavior when detections are weak or temporarily missing.
        self.last_known_road_center = self.image_center
        self.no_line_persist_counter = 0
        # Number of consecutive frames to reuse last confident center
        # before attempting fresh recovery. At ~30 FPS, 15 frames ≈ 0.5 s.
        self.no_line_persist_max = 15
        self.last_confident_center = self.image_center

        # PID tuning (operates on pixel error). Values were chosen to be
        # conservative for the simulator; tune as needed in-sim.
        self.Kp = 0.005
        self.Ki = 0.0003
        self.Kd = 0.0002
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        # Clamp for integral term to limit long-term bias
        self.integral_max = 100.0
        # Maximum absolute steering command (radians or simulator units)
        self.max_steering = 0.25

        # Track previous detection confidence to reset controller state on
        # transitions from confident -> unconfident views.
        self.prev_confidence = True

    def image_callback(self, msg):
        """ROS subscriber callback: convert image and run the pipeline.

        Receives a `sensor_msgs/Image`, converts it to an OpenCV BGR image,
        runs detection, computes steering via the PID controller, and
        publishes a `std_msgs/Float32` with the steering command.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

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
        """Top-level detection pipeline.

        Returns:
            (road_center_x, confidence)

        The pipeline first classifies the scene then delegates to handlers
        that implement recovery strategies for missing or ambiguous lane
        markings. When confidence is False the returned center should be
        treated as advisory and the controller will avoid aggressive
        corrections.
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
        """Quick heuristic to decide scene type using yellow-paint ratio.

        Returns (scene_str, yellow_ratio).
        - 'CROSSWALK' when a large fraction of the scene contains yellow
            (crosswalk markings).
        - 'NO_LINE' when very little yellow is present.
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
        """Detect the lane center using a smoothed column brightness profile.

        Returns (x, confidence). If `no_yellow` is True the detector avoids
        accepting a very strong bright peak (useful for recovering when
        yellow paint or reflections could be misleading).
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
            road_center: Detected center x-coordinate (pixels).
            confidence: Boolean indicating whether detection is considered
                        reliable. When False the controller reduces gains and
                        resets integral/derivative state to avoid windup.

        Returns:
            Steering command (float) clamped to [-self.max_steering, self.max_steering].
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
