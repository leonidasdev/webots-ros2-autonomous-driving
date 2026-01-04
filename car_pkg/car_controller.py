#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import time

class CarController(Node):
    def __init__(self):
        super().__init__('car_controller')
        
        # Configuration
        self.base_speed = 80.0
        self.current_speed = self.base_speed
        self.max_speed = self.base_speed
        
        # Conversion factor from internal speed units (e.g. km/h-like) to
        # the simulator/motor units used by Webots. Tweak to slow the car
        # in simulation. Example: 80 * 0.3 => 24 (motor units).
        self.speed_conversion_factor = 0.3
        
        # Control state
        self.steering = 0.0
        self.last_sign = None
        self.last_sign_time = 0
        self.sign_cooldown = 1.5
        
        # STOP state
        self.is_stopped = False
        self.stop_start_time = 0
        self.stop_duration = 1.0
        
        # Yield state
        self.yield_speed_active = False
        
        # Subscribers
        self.steering_sub = self.create_subscription(
            Float32, '/control/steering', self.steering_callback, 10)
        self.sign_sub = self.create_subscription(
            String, '/traffic_sign', self.sign_callback, 10)
            
        # Publishers
        self.speed_pub = self.create_publisher(Float32, '/control/speed', 10)
        # Controller should not publish steering; the road_follower node
        # publishes `/control/steering` directly. Removing redundant publish
        # avoids feedback loops.
        
        # Control loop timer. Use a short period for responsive control
        self.control_dt = 0.01
        self.timer = self.create_timer(self.control_dt, self.control_loop)

        # Speed increment per tick
        # (1.0 per 0.05s => 20 units/s)
        self._speed_step_per_sec = 1.0 / 0.05
        self.speed_step = self._speed_step_per_sec * self.control_dt
        
        self.get_logger().info("Car Controller iniciado")
        
    def steering_callback(self, msg):
        """Receive steering commands from the road follower."""
        self.steering = msg.data
        
    def sign_callback(self, msg):
        """Handle incoming traffic-sign messages from the sign detector."""
        current_time = time.time()
        sign_data = msg.data
        # Verify cooldown: allow processing if the sign is different (override)
        if current_time - self.last_sign_time < self.sign_cooldown:
            if sign_data == self.last_sign:
                return

        # Process whether the sign is new
        if sign_data != self.last_sign:
            self.last_sign = sign_data
            self.last_sign_time = current_time

            # Override yield: enable only if the new sign is a yield, otherwise disable
            if sign_data.startswith('yield'):
                self.yield_speed_active = True
            else:
                self.yield_speed_active = False

            # Process the sign
            self.handle_traffic_sign()
            
    def handle_traffic_sign(self):
        """Dispatch to specific sign handlers based on last_sign."""
        
        if self.last_sign == 'yield':
            self.handle_yield()
            
        elif self.last_sign == 'stop':
            self.handle_stop()
            
        elif self.last_sign.startswith('speed_limit'):
            self.handle_speed_limit()
            
    def handle_yield(self):
        """Handle YIELD: enable reduced max speed (half base)."""
        self.yield_speed_active = True
        self.max_speed = self.base_speed / 2.0
        self.get_logger().info(f"YIELD received: max_speed set to {self.max_speed}")
        
    def handle_stop(self):
        """Handle STOP: stop for configured duration."""
        self.current_speed = 0.0
        self.is_stopped = True
        self.stop_start_time = time.time()
        # disable yield while stopped
        self.yield_speed_active = False
        self.get_logger().info(f"STOP received: stopping for {self.stop_duration}s")
        
    def handle_speed_limit(self):
        """Handle SPEED_LIMIT: parse and apply new max speed."""
        try:
            # Extraer número de velocidad
            parts = self.last_sign.split('_')
            speed_number = int(parts[-1])

            # Store limit in internal units (e.g. km/h-like). Conversion to
            # motor/simulator units is applied when publishing.
            self.max_speed = speed_number
            
            # Limitar valores razonables
            if self.max_speed < 10:
                self.max_speed = 10
            elif self.max_speed > 100:
                self.max_speed = 100
                
            # If yield is active, cap max speed to half the limit
            if self.yield_speed_active:
                self.max_speed = self.max_speed / 2.0
                
            # If not stopped, ensure current speed does not exceed the new limit
            if not self.is_stopped:
                self.current_speed = min(self.current_speed, self.max_speed)
            self.get_logger().info(f"SPEED_LIMIT received: set max_speed to {self.max_speed}")
                
        except (ValueError, IndexError):
            # If the `speed_limit` message does not include a numeric value,
            # the detector must provide the value.
            # Keep existing limits unchanged and log the event.
            self.get_logger().info("SPEED_LIMIT without numeric value received: no change applied (detector should send explicit value)")
            return
            
    def check_stop_timer(self):
        """Check whether the STOP duration has elapsed and resume if so."""
        if self.is_stopped and (time.time() - self.stop_start_time) >= self.stop_duration:
            self.resume_after_stop()
        
    def resume_after_stop(self):
        """Resume motion after STOP: restore appropriate max/current speed."""
        self.is_stopped = False

        if self.yield_speed_active:
            self.max_speed = self.base_speed / 2.0
        elif self.last_sign and self.last_sign.startswith('speed_limit'):
            self.handle_speed_limit()
        else:
            self.max_speed = self.base_speed

        self.current_speed = self.max_speed
        self.get_logger().info(f"Resuming after STOP: max_speed={self.max_speed}, current_speed={self.current_speed}")
        
    def control_loop(self):
        """Main control loop: maintain speed and publish controls."""

        # If stopped, check whether to resume
        if self.is_stopped:
            self.check_stop_timer()

        if not self.is_stopped:
            # Smoothly move current_speed toward max_speed
            if self.current_speed > self.max_speed:
                self.current_speed = max(self.max_speed, self.current_speed - self.speed_step)
            elif self.current_speed < self.max_speed:
                self.current_speed = min(self.max_speed, self.current_speed + self.speed_step)

            # Publish converted speed for Webots
            speed_msg = Float32()
            speed_msg.data = self.current_speed * self.speed_conversion_factor
            self.speed_pub.publish(speed_msg)
            # Steering is forwarded by the road_follower node; controller

        else:
            # While stopped: publish zero speed but keep steering value
            speed_msg = Float32()
            speed_msg.data = 0.0
            self.speed_pub.publish(speed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CarController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
