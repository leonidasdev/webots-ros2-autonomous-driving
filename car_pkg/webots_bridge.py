#!/usr/bin/env python3

"""Bridge node between Webots and ROS2.

This node exposes Webots devices to ROS2 topics and applies control
commands coming from ROS. It publishes camera images and subscribes to
speed/steering commands issued by the controller.
"""

import rclpy
from rclpy.node import Node
from controller import Robot
import sys
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock


class WebotsBridge(Node):
    """Expose Webots sensors/actuators as ROS2 topics.

    - Publishes `/car_camera/image` and `/road_camera/image` (sensor_msgs/Image)
    - Subscribes to `/control/speed` and `/control/steering` (Float32)
    - Applies speed to rear-wheel motors and steering positions to steer motors
    """

    def __init__(self):
        super().__init__('webots_bridge')

        self.get_logger().info("Connecting to Webots...")

        # Connect to Webots controller
        try:
            self.robot = Robot()
            self.timestep = int(self.robot.getBasicTimeStep())
            self.get_logger().info("Connected to Webots")
        except Exception as e:
            self.get_logger().error(f"Error connecting to Webots: {e}")
            sys.exit(1)

        # Acquire camera devices
        try:
            self.car_camera = self.robot.getDevice('car_camera')
            self.road_camera = self.robot.getDevice('road_camera')
            self.get_logger().info("Cameras found")
        except Exception:
            self.get_logger().error("Could not find cameras")
            sys.exit(1)

        # Configure expected motors
        self.motors = {}
        motor_names = ['right_steer', 'left_steer', 'right_rear_wheel', 'left_rear_wheel']

        for name in motor_names:
            try:
                motor = self.robot.getDevice(name)
                self.motors[name] = motor
                # Wheels are velocity-controlled (infinite position)
                if 'wheel' in name:
                    motor.setPosition(float('inf'))
                    motor.setVelocity(0.0)
                self.get_logger().info(f"Configured motor: {name}")
            except Exception as e:
                self.get_logger().error(f"Error configuring motor {name}: {e}")
                sys.exit(1)

        # Enable camera sampling at the Webots basic time step
        self.car_camera.enable(self.timestep)
        self.road_camera.enable(self.timestep)
        self.bridge = CvBridge()

        # --- Publishers (expose sensors / measured values) ---
        # Publishers for camera images
        self.car_camera_pub = self.create_publisher(Image, '/car_camera/image', 10)
        self.road_camera_pub = self.create_publisher(Image, '/road_camera/image', 10)
        # Publisher for measured vehicle speed (average rear-wheel velocity)
        self.speed_actual_pub = self.create_publisher(Float32, '/vehicle/speed_actual', 10)
        # Publisher for simulation clock so ROS nodes can use sim-time
        try:
            self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        except Exception:
            self.clock_pub = None

        # --- Subscribers (control commands) ---
        self.speed_sub = self.create_subscription(Float32, '/control/speed', self.speed_callback, 10)
        self.steering_sub = self.create_subscription(Float32, '/control/steering', self.steering_callback, 10)

        # --- Runtime state ---
        # Current commanded speed and steering (applied to Webots devices)
        self.current_speed = 0.0
        self.current_steering = 0.0

        # Timer to publish sensor data (convert ms -> s)
        self.timer = self.create_timer(self.timestep / 1000.0, self.publish_data)
        
        self.get_logger().info("Webots Bridge ready")

    def speed_callback(self, msg):
        """Handle incoming speed commands and apply them to rear wheels.

        Args:
            msg (std_msgs.msg.Float32): Commanded speed (simulator motor units).

        The callback clamps the commanded speed to a safe positive range and
        applies it to the rear-wheel motors using the Webots motor API.
        """
        # Clamp commanded speed to [0, 100]
        self.current_speed = max(0.0, min(float(msg.data), 100.0))
        # Apply velocity to rear wheels
        try:
            self.motors['right_rear_wheel'].setVelocity(self.current_speed)
            self.motors['left_rear_wheel'].setVelocity(self.current_speed)
        except Exception:
            # If motors are not available or API fails, log and continue
            self.get_logger().error("Failed to set rear-wheel velocity")

    def steering_callback(self, msg):
        """Handle incoming steering commands and apply to steer motors.

        Args:
            msg (std_msgs.msg.Float32): Steering position/angle for steer motors.

        The value is clamped to a safe range before being applied.
        """
        # Clamp steering to [-0.5, 0.5]
        self.current_steering = max(-0.5, min(float(msg.data), 0.5))
        # Apply steering position to both steering motors
        try:
            self.motors['right_steer'].setPosition(self.current_steering)
            self.motors['left_steer'].setPosition(self.current_steering)
        except Exception:
            self.get_logger().error("Failed to set steering position")

    def publish_data(self):
        """Advance the simulation one step and publish sensor topics.

        This method reads camera images from Webots, converts them to ROS
        `sensor_msgs/Image` messages and publishes them. It also publishes
        the simulation clock and an observed rear-wheel velocity when
        available.
        """
        if self.robot.step(self.timestep) == -1:
            return

        try:
            # Publish `car_camera` image
            car_image_data = self.car_camera.getImage()
            if car_image_data:
                car_image = np.frombuffer(car_image_data, np.uint8).reshape(
                    (self.car_camera.getHeight(), self.car_camera.getWidth(), 4))
                car_image_rgb = cv2.cvtColor(car_image, cv2.COLOR_BGRA2BGR)
                car_msg = self.bridge.cv2_to_imgmsg(car_image_rgb, "bgr8")
                self.car_camera_pub.publish(car_msg)

            # Publish `road_camera` image
            road_image_data = self.road_camera.getImage()
            if road_image_data:
                road_image = np.frombuffer(road_image_data, np.uint8).reshape(
                    (self.road_camera.getHeight(), self.road_camera.getWidth(), 4))
                road_image_rgb = cv2.cvtColor(road_image, cv2.COLOR_BGRA2BGR)
                road_msg = self.bridge.cv2_to_imgmsg(road_image_rgb, "bgr8")
                self.road_camera_pub.publish(road_msg)

            # Publish simulation clock so ROS nodes can use sim-time
            if self.clock_pub is not None:
                sim_t = float(self.robot.getTime())
                sec = int(sim_t)
                nsec = int((sim_t - float(sec)) * 1e9)
                clk = Clock()
                clk.clock.sec = sec
                clk.clock.nanosec = nsec
                self.clock_pub.publish(clk)

                # Publish measured rear-wheel velocity so controllers can
                # observe physical inertia. Use available motor API.
                try:
                    vr = self.motors['right_rear_wheel'].getVelocity()
                    vl = self.motors['left_rear_wheel'].getVelocity()
                    avg_v = float((vr + vl) / 2.0)
                    vmsg = Float32()
                    vmsg.data = avg_v
                    self.speed_actual_pub.publish(vmsg)
                except Exception:
                    # If motors don't expose velocity, skip publishing
                    pass
        except Exception as e:
            self.get_logger().error(f"Error publishing sensor data: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WebotsBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
