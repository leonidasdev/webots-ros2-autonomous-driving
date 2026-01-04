#!/usr/bin/env python3

"""Car controller node: apply speed commands based on detected signs.

This node subscribes to `/traffic_sign` (std_msgs/String) and `/control/steering`
and publishes speed commands to `/control/speed` (std_msgs/Float32).

Behavior rules implemented:
- `base_speed` is the nominal default and initializes `max_speed`.
- `yield` halves the effective maximum speed (modifier only).
- `speed_limit_N` sets the nominal `max_speed` to N.
- `stop` halts motion for `stop_duration` seconds, then resumes toward
    the effective maximum.

The node prefers simulated time (via `get_clock().now().nanoseconds`) for
accurate STOP timing when available, with a wall-clock fallback.
"""

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
        # Use simulation clock nanoseconds for accurate sim-time STOP timing.
        # Stored as integer nanoseconds. Fallback to wall-clock if needed.
        self.stop_start_time_ns = 0
        # TODO For testing we use 3 seconds stop; change to 1.0 for production
        self.stop_duration = 3.0
        
        # Yield state
        # When True the effective maximum speed is halved
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
        
        self.get_logger().info("Car Controller started")
        
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

            # Yield is a transient modifier: active only while the last seen
            # sign is a `yield`. The effective max speed will be computed from
            # `self.max_speed` and the `yield` flag (halved when active).
            self.yield_speed_active = sign_data.startswith('yield')

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
        """Handle YIELD: mark yield active so effective max is halved.

        The halving is applied dynamically when computing the effective
        maximum speed; we do not mutate the sign-provided `self.max_speed`
        so that subsequent `speed_limit` signs can update the nominal
        limit and the yield modifier will apply on top of it.
        """
        self.yield_speed_active = True
        self.get_logger().info("YIELD received: effective max speed will be halved")
        
    def handle_stop(self):
        """Handle STOP: stop for configured duration."""
        self.current_speed = 0.0
        self.is_stopped = True
        # Record stop start time in simulation time (ns) when available.
        try:
            self.stop_start_time_ns = self.get_clock().now().nanoseconds
        except Exception:
            # Fallback: store wall-clock time as nanoseconds
            self.stop_start_time_ns = int(time.time() * 1e9)
        # Do not change yield state on STOP. When resuming the controller
        # will move toward the effective max speed which already accounts
        # for a possible active `yield`.
        # Detect whether simulated time appears active. Prefer the explicit
        # `use_sim_time` parameter if set; otherwise look for a `/clock` topic.
        sim_time_active = False
        try:
            if self.has_parameter('use_sim_time'):
                try:
                    sim_time_active = bool(self.get_parameter('use_sim_time').value)
                except Exception:
                    sim_time_active = False
            else:
                # Check published topics for /clock
                topics = [t for (t, _) in self.get_topic_names_and_types()]
                sim_time_active = '/clock' in topics
        except Exception:
            sim_time_active = False

        self.get_logger().info(f"STOP received: stopping for {self.stop_duration}s (sim_time={'ON' if sim_time_active else 'OFF'})")
        
    def handle_speed_limit(self):
        """Handle `speed_limit_N`: parse numeric value and apply nominal max.

        The nominal `self.max_speed` is updated; the effective maximum used
        by the controller considers the `yield` modifier separately.
        """
        try:
            # Extract numeric speed value from the message `speed_limit_N`.
            parts = self.last_sign.split('_')
            speed_number = int(parts[-1])

            # Store nominal maximum speed (internal units). Conversion to
            # simulator motor units is applied when publishing.
            # Set nominal maximum speed (from sign payload).
            self.max_speed = float(speed_number)
            
            # Clamp to reasonable bounds
            if self.max_speed < 10:
                self.max_speed = 10
            elif self.max_speed > 100:
                self.max_speed = 100
                
            # Do not change the yield modifier here; it is applied on top of
            # the nominal `max_speed`. If currently stopped, the resume will
            # move toward the effective maximum when the stop ends.
            # If not stopped, ensure current speed does not exceed the
            # effective maximum (taking yield into account).
            if not self.is_stopped:
                eff = self.get_effective_max()
                self.current_speed = min(self.current_speed, eff)
            self.get_logger().info(f"SPEED_LIMIT received: set max_speed to {self.max_speed}")
                
        except (ValueError, IndexError):
            # If the `speed_limit` message does not include a numeric value,
            # the detector must provide the value.
            # Keep existing limits unchanged and log the event.
            self.get_logger().info("SPEED_LIMIT without numeric value received: no change applied (detector should send explicit value)")
            return
            
    def check_stop_timer(self):
        """Check whether the STOP duration has elapsed and resume if so."""
        if not self.is_stopped:
            return

        # Prefer simulation clock (nanoseconds). If get_clock() isn't
        # publishing sim time, fall back to wall-clock comparison.
        try:
            now_ns = self.get_clock().now().nanoseconds
            if (now_ns - self.stop_start_time_ns) >= int(self.stop_duration * 1e9):
                self.resume_after_stop()
        except Exception:
            # Fallback to wall-clock seconds: stored ns -> seconds
            try:
                start_s = float(self.stop_start_time_ns) / 1e9
                if (time.time() - start_s) >= self.stop_duration:
                    self.resume_after_stop()
            except Exception:
                pass
        
    def get_effective_max(self):
        """Return the effective maximum speed, accounting for `yield`.

        The nominal `self.max_speed` is set by `speed_limit` signs or by
        the `base_speed`. When `yield` is active the effective maximum is
        halved (with a sensible lower bound).
        """
        eff = float(self.max_speed)
        if self.yield_speed_active:
            eff = max(10.0, eff / 2.0)
        return eff
        
    def resume_after_stop(self):
        """Resume motion after STOP: restore appropriate current speed with the max speed."""
        self.is_stopped = False
        # Resume toward the effective maximum (accounts for yield).
        self.current_speed = self.get_effective_max()
        self.get_logger().info(f"Resuming after STOP: max_speed={self.max_speed}, yield_active={self.yield_speed_active}, current_speed={self.current_speed}")
        
    def control_loop(self):
        """Main control loop: maintain speed and publish controls."""

        # If stopped, check whether to resume
        if self.is_stopped:
            self.check_stop_timer()

        if not self.is_stopped:
            # Smoothly move current_speed toward max_speed
            # Target is the effective maximum which considers yield.
            eff_max = self.get_effective_max()
            if self.current_speed > eff_max:
                self.current_speed = max(eff_max, self.current_speed - self.speed_step)
            elif self.current_speed < eff_max:
                self.current_speed = min(eff_max, self.current_speed + self.speed_step)

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
