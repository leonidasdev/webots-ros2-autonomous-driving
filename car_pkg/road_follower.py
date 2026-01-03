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

        # Subscriptions / publishers
        self.image_sub = self.create_subscription(Image, '/road_camera/image', self.image_callback, 10)
        self.steering_pub = self.create_publisher(Float32, '/control/steering', 10)

        # State
        self.last_known_road_center = 256  # x coordinate in pixels (0..511)

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

            if cv_image.shape[0] == 16 and cv_image.shape[1] == 512:
                road_center, confidence = self.detect_road_center(cv_image)

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

    def detect_road_center(self, image):
        """Return (road_center_x, confidence).

        Returns a pixel x coordinate and a boolean `confidence`. A
        confident detection is a locally-prominent peak; otherwise the
        function attempts a local search near the last known center and
        finally returns the image center when still ambiguous.
        """
        try:
            image_center = image.shape[1] // 2
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            column_means = np.mean(blurred, axis=0)
            SMOOTH_K = 11
            kernel = np.ones(SMOOTH_K) / SMOOTH_K
            pad = SMOOTH_K // 2
            # reflect-pad to avoid edge bias from zero-padding
            padded = np.pad(column_means, pad, mode='reflect')
            column_means_smooth = np.convolve(padded, kernel, mode='valid')

            # primary peak detection
            max_idx = np.argmax(column_means_smooth)
            max_val = column_means_smooth[max_idx]

            # Debug: report primary peak location and value (debug-level)
            try:
                self.get_logger().debug(f"detect: max_idx={int(max_idx)} max_val={float(max_val):.2f} image_center={image_center}")
            except Exception:
                pass

            # Accept peak only if it is sufficiently prominent vs. local avg
            if max_val > 30:
                window = 40
                start = max(0, max_idx - window)
                end = min(511, max_idx + window)
                if end > start:
                    local_avg = np.mean(column_means_smooth[start:end])
                    # Debug: local context values
                    try:
                        self.get_logger().debug(f"detect: local window [{start}:{end}] local_avg={local_avg:.2f}")
                    except Exception:
                        pass
                    if max_val > local_avg * 1.15:
                        try:
                            self.get_logger().debug(f"detect CONFIDENT: center={int(max_idx)} val={float(max_val):.2f} local_avg={local_avg:.2f}")
                        except Exception:
                            pass
                        return max_idx, True

            # Weak peak: try searching near the last known center before
            # declaring ambiguity. This allows short gaps in detection to be
            # bridged but returns confidence=False (controller disables Ki/Kd).
            search_center = self.last_known_road_center
            search_window = 180
            start = max(0, search_center - search_window // 2)
            end = min(511, search_center + search_window // 2)
            if end > start:
                local_max = np.argmax(column_means_smooth[start:end]) + start
                if column_means_smooth[local_max] > 20:
                    try:
                        self.get_logger().debug(f"detect WEAK_LOCAL: chosen={int(local_max)} val={float(column_means_smooth[local_max]):.2f} last_known={search_center}")
                    except Exception:
                        pass
                    return local_max, False

            # Ambiguous: return image center so the controller drives straight
            try:
                self.get_logger().debug("detect AMBIGUOUS: returning image_center")
            except Exception:
                pass
            return image_center, False
        except Exception:
            return image_center, False

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
        # Debug: report PID components and steering before optional forcing
        try:
            self.get_logger().debug(f"PID: error={filtered_error:.2f} dt={dt:.4f} P={P:.6f} I={I:.6f} D={D:.6f} steering={steering:.6f} confidence={confidence}")
        except Exception:
            pass

        if not confidence:
            try:
                self.get_logger().debug("PID: low confidence -> forcing steering=0.0")
            except Exception:
                pass
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
