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
    def __init__(self):
        super().__init__('road_follower')
        self.bridge = CvBridge()

        # subscribe to road camera images
        self.image_sub = self.create_subscription(
            Image,
            '/road_camera/image',
            self.image_callback,
            10
        )

        # publish steering commands
        self.steering_pub = self.create_publisher(Float32, '/control/steering', 10)

        # last known road center (x coordinate, 0..511)
        self.last_known_road_center = 256

        # PID gains and state
        self.Kp = 0.005
        self.Ki = 0.0003
        self.Kd = 0.0002
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.integral_max = 100.0
        self.max_steering = 0.25

        # history/smoothing removed: controller uses instantaneous detections

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if cv_image.shape[0] == 16 and cv_image.shape[1] == 512:
                # detect center (grayscale pipeline)
                road_center, confidence = self.detect_road_center(cv_image)

                if not confidence:
                    # fall back to last known center when detection fails
                    road_center = self.last_known_road_center

                # use instantaneous detection
                self.last_known_road_center = int(road_center)
                steering = self.calculate_pid(self.last_known_road_center, confidence)

                steering_msg = Float32()
                steering_msg.data = steering
                self.steering_pub.publish(steering_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def detect_road_center(self, image):
        """Detect road center based on intensity."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            column_means = np.mean(blurred, axis=0)
            SMOOTH_K = 11
            column_means_smooth = np.convolve(column_means, np.ones(SMOOTH_K)/SMOOTH_K, mode='same')

            # primary peak detection
            max_idx = np.argmax(column_means_smooth)
            max_val = column_means_smooth[max_idx]

            if max_val > 30:
                window = 40
                start = max(0, max_idx - window)
                end = min(511, max_idx + window)
                if end > start:
                    local_avg = np.mean(column_means_smooth[start:end])
                    if max_val > local_avg * 1.15:
                        return max_idx, True

            # fallback: search near last known center
            search_center = self.last_known_road_center
            search_window = 180
            start = max(0, search_center - search_window//2)
            end = min(511, search_center + search_window//2)
            if end > start:
                local_max = np.argmax(column_means_smooth[start:end]) + start
                if column_means_smooth[local_max] > 20:
                    return local_max, False

            # centroid fallback
            try:
                thresh = max_val * 0.5
                mask = column_means_smooth > thresh
                if np.any(mask):
                    cols = np.arange(column_means_smooth.shape[0])
                    weighted_center = int(np.sum(cols[mask] * column_means_smooth[mask]) / np.sum(column_means_smooth[mask]))
                    return weighted_center, False
            except Exception:
                pass

            return self.last_known_road_center, False
        except Exception:
            return self.last_known_road_center, False

    def calculate_pid(self, road_center, confidence):
        """PID controller for steering using instantaneous lateral error."""
        if road_center == 256 and not confidence:
            return 0.0

        image_center = 256
        current_time = time.time()
        dt = max(0.001, min(current_time - self.prev_time, 0.1))

        # instantaneous lateral error
        error = road_center - image_center
        filtered_error = error

        # proportional
        P = self.Kp * filtered_error

        # integral (reduced gain when low confidence)
        integral_gain = 0.3 if not confidence else 1.0
        self.integral += filtered_error * dt * integral_gain
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        I = self.Ki * self.integral

        # derivative (disabled if low confidence)
        if dt > 0.01:
            derivative = (filtered_error - self.prev_error) / dt
            derivative = np.clip(derivative, -15, 15)
            if not confidence:
                derivative = 0.0
            D = self.Kd * derivative
        else:
            D = 0.0

        steering = P + I + D
        if not confidence:
            steering *= 0.5

        steering = max(-self.max_steering, min(self.max_steering, steering))

        # update state
        self.prev_error = filtered_error
        self.prev_time = current_time

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
