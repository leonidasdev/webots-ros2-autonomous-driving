#!/usr/bin/env python3

"""Webots ↔ ROS2 bridge node.

Provides `WebotsBridge`, a ROS2 node that exposes selected Webots devices
as ROS topics and applies incoming control commands to simulated actuators.
Published topics include camera images, a simulation `/clock`, and a
minimal `/odom` twist estimate used for speed monitoring and STOP
detection. The node subscribes to `/control/speed` and
`/control/steering` to apply actuator commands.

The implementation is lightweight and supplies a twist-only `/odom`
message rather than a full pose-based odometry estimate.
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
from nav_msgs.msg import Odometry


class WebotsBridge(Node):
    """Expose Webots sensors/actuators as ROS2 topics.

    - Publishes `/car_camera/image` and `/road_camera/image` (sensor_msgs/Image)
    - Subscribes to `/control/speed` and `/control/steering` (Float32)
    - Applies speed to rear-wheel motors and steering positions to steer motors
    """

    def __init__(self):
        """Create the bridge node and configure Webots devices and topics.

        The initializer attempts to connect to the Webots controller, enables
        cameras, configures motors, and registers ROS publishers/subscribers.

        Raises:
            SystemExit: If required Webots devices (cameras or motors) cannot
                be found or configured.
        """

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

        # Acquire camera devices from Webots
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

        # Publishers
        # Publishers for camera images
        self.car_camera_pub = self.create_publisher(Image, '/car_camera/image', 10)
        self.road_camera_pub = self.create_publisher(Image, '/road_camera/image', 10)
        # Publisher for simulation clock so ROS nodes can use sim-time
        try:
            self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        except Exception:
            self.clock_pub = None

        # Subscribers
        self.speed_sub = self.create_subscription(Float32, '/control/speed', self.speed_callback, 10)
        self.steering_sub = self.create_subscription(Float32, '/control/steering', self.steering_callback, 10)

        # Current commanded speed and steering (applied to Webots devices)
        self.current_speed = 0.0
        self.current_steering = 0.0

        # Wheel radius (m) used to convert wheel angular velocity to linear
        # speed for a simple odometry/twist publisher. Tune `wheel_radius`
        # to match the vehicle geometry in your Webots world.
        self.wheel_radius = 0.1

        # Odometry publisher (twist-only estimate published on `/odom`)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)

        # Optical flow-based speed estimation settings.
        # If enabled, optical flow between consecutive `road_camera` frames
        # is used to estimate forward speed. `flow_scale` maps pixels/sec
        # to meters/sec and must be tuned for the camera mounting and FOV.
        self.use_optical_flow = True
        self.flow_scale = 0.001
        self.prev_road_gray = None

        # Timer to publish sensor data (convert ms -> s)
        self.timer = self.create_timer(self.timestep / 1000.0, self.publish_data)
        
        self.get_logger().info("Webots Bridge ready")

    def speed_callback(self, msg):
        """Apply a requested forward speed to the rear wheels.

        Args:
            msg (std_msgs.msg.Float32): Commanded forward speed in simulator
                motor units. The value is clamped to [-500, 100]. Negative
                values request reverse torque for active braking.
        """
        # Clamp commanded speed to the supported actuator range. Query the
        # motors for their `maxVelocity` and ensure we do not request a
        # speed that exceeds the hardware/robot limits (avoids Webots
        # warnings like 'requested velocity ... exceeds maxVelocity').
        req = float(msg.data)
        try:
            mr = abs(self.motors['right_rear_wheel'].getMaxVelocity())
            ml = abs(self.motors['left_rear_wheel'].getMaxVelocity())
            motor_max = min(mr, ml)
        except Exception:
            motor_max = 100.0

        # Positive acceleration cap remains conservative (100); negative
        # braking can use up to the motor max velocity in magnitude.
        pos_cap = min(100.0, motor_max)
        neg_cap = -motor_max

        if req > pos_cap or req < neg_cap:
            self.get_logger().warning(
                f"Requested speed {req:.1f} outside motor limits [{neg_cap:.1f},{pos_cap:.1f}] - clamping"
            )

        self.current_speed = max(neg_cap, min(req, pos_cap))
        # Apply velocity to rear wheels
        try:
            self.motors['right_rear_wheel'].setVelocity(self.current_speed)
            self.motors['left_rear_wheel'].setVelocity(self.current_speed)
        except Exception:
            # If motors are not available or API fails, log and continue
            self.get_logger().error("Failed to set rear-wheel velocity")

    def steering_callback(self, msg):
        """Apply a steering setpoint to the steer motors.

        Args:
            msg (std_msgs.msg.Float32): Steering position or angle. The
                value is clamped to [-0.5, 0.5] before being applied to the
                steering motor positions.
        """
        # Clamp steering to [-0.5, 0.5].
        self.current_steering = max(-0.5, min(float(msg.data), 0.5))
        # Apply steering position to both steering motors
        try:
            self.motors['right_steer'].setPosition(self.current_steering)
            self.motors['left_steer'].setPosition(self.current_steering)
        except Exception:
            self.get_logger().error("Failed to set steering position")

    def publish_data(self):
        """Advance the simulation and publish sensor topics.

        This method performs a single simulation step, then publishes the
        following topics when data is available:
        - `/car_camera/image` (sensor_msgs/Image)
        - `/road_camera/image` (sensor_msgs/Image)
        - `/clock` (rosgraph_msgs/Clock)
        - `/odom` (nav_msgs/Odometry) — twist-only estimate

        The `/odom` message is produced from dense optical flow computed on
        the `road_camera` frames when `use_optical_flow` is enabled; a
        wheel-velocity-based fallback is used otherwise.
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

            # Publish simulation clock so ROS nodes can use sim-time.
            if self.clock_pub is not None:
                sim_t = float(self.robot.getTime())
                sec = int(sim_t)
                nsec = int((sim_t - float(sec)) * 1e9)
                clk = Clock()
                clk.clock.sec = sec
                clk.clock.nanosec = nsec
                self.clock_pub.publish(clk)

                # Publish a minimal `/odom` message. Prefer optical-flow-based
                # speed estimation (computed from `road_camera`) and fall back
                # to wheel velocities when necessary. The produced Odometry
                # message contains a zero pose and a populated `twist`.
                try:
                    # Prefer optical-flow-based speed estimation if enabled.
                    linear_speed = None
                    if self.use_optical_flow and 'road_image_rgb' in locals():
                        try:
                            gray = cv2.cvtColor(road_image_rgb, cv2.COLOR_BGR2GRAY)
                            if self.prev_road_gray is not None:
                                # Farneback dense optical flow
                                flow = cv2.calcOpticalFlowFarneback(
                                    self.prev_road_gray, gray, None,
                                    0.5, 3, 15, 3, 5, 1.2, 0)
                                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                                mean_mag = float(np.mean(mag))
                                # convert pixels/frame -> pixels/sec
                                dt = float(self.timestep) / 1000.0
                                pixels_per_sec = mean_mag / max(dt, 1e-6)
                                linear_speed = pixels_per_sec * float(self.flow_scale)
                            # store for next iteration
                            self.prev_road_gray = gray
                        except Exception:
                            linear_speed = None

                    # Fallback: use wheel velocities (may be commanded value)
                    if linear_speed is None:
                        vr = float(self.motors['right_rear_wheel'].getVelocity())
                        vl = float(self.motors['left_rear_wheel'].getVelocity())
                        avg_w = (abs(vr) + abs(vl)) / 2.0
                        linear_speed = avg_w * float(self.wheel_radius)

                    odom = Odometry()
                    odom.header.stamp.sec = sec
                    odom.header.stamp.nanosec = nsec
                    odom.header.frame_id = 'odom'
                    odom.child_frame_id = 'base_link'
                    odom.twist.twist.linear.x = float(linear_speed)
                    odom.twist.twist.linear.y = 0.0
                    odom.twist.twist.linear.z = 0.0
                    odom.twist.twist.angular.x = 0.0
                    odom.twist.twist.angular.y = 0.0
                    odom.twist.twist.angular.z = 0.0
                    self.odom_pub.publish(odom)
                except Exception:
                    # If wheels/velocity read fails or other error, skip odom.
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
