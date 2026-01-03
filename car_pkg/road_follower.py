#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32
import cv2
import time


class RoadFollower(Node):
    """ROS2 node that detects the road center from a forward camera and
    publishes steering commands.

    Key points:
      - Detection runs on per-frame column intensity (grayscale) and
        finds a locally-prominent column as the road center. If ambiguous
        the detector returns the image center so the vehicle keeps a
        straight heading.
      - Reflect-padding is used for the smoothing convolution to avoid
        edge bias.
      - PID acts on pixel error; integral and derivative are disabled and
        reset when detection is unreliable to prevent windup and spikes.
    """

    def __init__(self):
        super().__init__('road_follower')
        self.bridge = CvBridge()
        # keep default logger level (no temporary debug overrides)

        # Subscriptions / publishers
        self.image_sub = self.create_subscription(Image, '/road_camera/image', self.image_callback, 10)
        self.steering_pub = self.create_publisher(Float32, '/control/steering', 10)

        # State
        self.last_known_road_center = 256  # x coordinate in pixels (0..511)

        # Persistence for NO_LINE scenes: reuse last confident center for
        # a small number of frames to avoid margin-of-error jitter.
        self.no_line_persist_counter = 0
        self.no_line_persist_max = 10
        self.last_confident_center = 256

        # PID gains and state
        self.Kp = 0.005
        self.Ki = 0.0003
        self.Kd = 0.0002
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.integral_max = 100.0
        self.max_steering = 0.25

        # Track previous detection confidence to detect transitions
        self.prev_confidence = True

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            road_center, confidence = self.follow_road(cv_image)

            # Use instantaneous detection; store last seen center. The
            # detector already returns `image_center` for ambiguous
            # frames, so we do not use `last_known_road_center` to
            # override ambiguous results.
            self.last_known_road_center = int(road_center)
            steering = self.calculate_pid(self.last_known_road_center, confidence)

            steering_msg = Float32()
            steering_msg.data = steering
            self.steering_pub.publish(steering_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def follow_road(self, image):
        """Return (road_center_x, confidence).

        This method has been refactored into a small scene classifier and
        three handlers for clarity: CROSSWALK, NO_LINE and LINE. The
        external behavior and thresholds are preserved.
        """
        image_center = image.shape[1] // 2
        try:
            scene, yellow_ratio = self._classify_scene(image)

            # Treat CROSSWALK like NO_LINE for persistence: prefer using the
            # last confident center for a few frames rather than issuing
            # steering immediately. This avoids margin errors after turns.
            if scene in ('NO_LINE', 'CROSSWALK'):
                # reuse last confident center for a few frames
                if self.no_line_persist_counter < self.no_line_persist_max and self.last_confident_center is not None:
                    self.no_line_persist_counter += 1
                    return int(self.last_confident_center), False

                if scene == 'NO_LINE':
                    center, conf = self._handle_line(image, no_yellow=True)
                    if conf:
                        self.last_confident_center = int(center)
                        self.no_line_persist_counter = 0
                    return center, conf

                # CROSSWALK (persistence exhausted): do not apply steering
                # — return center with confidence False so PID stays idle.
                self.no_line_persist_counter = 0
                return image_center, False

            # Normal LINE case: reset persistence and run regular handler.
            self.no_line_persist_counter = 0
            center, conf = self._handle_line(image, no_yellow=False)
            if conf:
                self.last_confident_center = int(center)
            return center, conf
        except Exception:
            return image_center, False

    def _classify_scene(self, image):
        """Classify the scene as 'CROSSWALK', 'NO_LINE', or 'LINE'.

        Returns (scene_str, yellow_ratio).
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
        """Detect the bright/grayscale center line and return (x, confidence).

        `no_yellow` True disables acceptance of a confident bright peak.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            column_means = np.mean(blurred, axis=0)
            SMOOTH_K = 11
            kernel = np.ones(SMOOTH_K) / SMOOTH_K
            pad = SMOOTH_K // 2
            padded = np.pad(column_means, pad, mode='reflect')
            column_means_smooth = np.convolve(padded, kernel, mode='valid')

            max_idx = int(np.argmax(column_means_smooth))
            max_val = column_means_smooth[max_idx]

            if max_val > 30:
                window = 40
                start = max(0, max_idx - window)
                end = min(column_means_smooth.shape[0] - 1, max_idx + window)
                if end > start:
                    local_avg = np.mean(column_means_smooth[start:end])
                    if max_val > local_avg * 1.15:
                        if no_yellow:
                            pass
                        else:
                            return max_idx, True

            # Weak peak: try searching near the last known center
            search_center = self.last_known_road_center
            search_window = 180
            start = max(0, search_center - search_window // 2)
            end = min(column_means_smooth.shape[0] - 1, search_center + search_window // 2)
            if end > start:
                local_max = np.argmax(column_means_smooth[start:end]) + start
                if column_means_smooth[local_max] > 20:
                    return local_max, False

            return image.shape[1] // 2, False
        except Exception:
            return image.shape[1] // 2, False

    def calculate_pid(self, road_center, confidence):
        """PID controller for lateral error (in pixels).

        Behavior notes:
          - `road_center` and `image_center` are pixel coordinates.
          - When `confidence` is False the controller zeros `Kp/Ki/Kd`
            (effective gains) or resets integral/derivative state to avoid
            steering from unreliable measurements.
          - The integral term is clamped by `integral_max` to limit bias.
        """

        # If ambiguous and centered, explicitly return zero steering
        if road_center == 256 and not confidence:
            return 0.0

        image_center = 256
        current_time = time.time()
        dt = max(0.001, min(current_time - self.prev_time, 0.1))

        # instantaneous lateral error in pixels
        error = road_center - image_center
        filtered_error = error

        # Use Kp only when detection is confident
        Kp_eff = self.Kp if confidence else 0.0
        P = Kp_eff * filtered_error

        # reset on loss of confidence (prevent windup/spikes)
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

        # Force zero steering when detection is not confident to avoid
        # acting on unreliable measurements (actuator dynamics may still
        # carry the vehicle, but no new steering command is issued).
        if not confidence:
            steering = 0.0

        # update state
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
