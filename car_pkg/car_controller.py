#!/usr/bin/env python3

"""Car controller node: apply speed commands based on detected signs.

This node subscribes to `/traffic_sign` (std_msgs/String) and publishes
speed commands to `/control/speed` (std_msgs/Float32).

Behavior rules:
- `base_speed` is the nominal default and initializes `max_speed`.
- `yield` halves the effective maximum speed.
- `speed_limit_N` sets the nominal `max_speed` to N.
- `stop` halts motion for 1 second (using simulation time from /clock),
  then resumes toward the effective maximum.

The node uses simulation time from /clock for accurate STOP timing.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry


class CarController(Node):
    def __init__(self):
        super().__init__('car_controller')
        
        # Publishers
        self.speed_pub = self.create_publisher(Float32, '/control/speed', 10)

        # Subscribers
        self.sign_sub = self.create_subscription(String, '/traffic_sign', self.sign_callback, 10)
        self.clock_sub = self.create_subscription(Clock, '/clock', self.clock_callback, 10)
        # Subscribe to odometry and derive measured speed from linear velocity
        self.measured_speed = None
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # Configuration
        self.base_speed = 80.0
        self.current_speed = self.base_speed
        self.max_speed = self.base_speed
        
        # Conversion factor from internal speed units (e.g. km/h-like) to
        # the simulator/motor units used by Webots. Tweak to slow the car
        # in simulation. Example: 80 * 0.8 => 64 (motor units).
        self.speed_conversion_factor = 0.8
        
        # Sign processing state
        self.last_sign = None
        self.last_sign_time = 0.0
        self.sign_cooldown = 1.5
        
        # STOP state
        self.is_stopped = False
        # Store stop start time using simulation time from /clock
        self.stop_start_sim_s = None
        self.current_sim_time = None
        # Stop duration: 1 second as per requirements
        self.stop_duration = 1.0
        # Active braking settings: apply a short negative command (torque)
        # when a STOP is issued to help the vehicle slow down faster.
        self.brake_active = False
        self.brake_duration = 0.2
        self.brake_end_sim_s = None
        # Brake strength in controller internal units. Tune as needed.
        self.brake_strength = 40.0
        # Braking / STOP coordination
        # Physical-stop detection: require measured speed to be below this
        # threshold for `physical_hold_time` seconds before starting the
        # stop-duration timer. If no measured speed is available we fall
        # back to starting the timer when commanded speed reaches zero.
        self.physical_stop_threshold = 0.1
        self.physical_hold_time = 0.5
        self.physical_below_start_sim_s = None
        self.waiting_for_physical_stop = False
        
        # Yield flag
        self.yield_active = False
        
        # Control loop timer. Use a short period for responsive control
        self.control_dt = 0.01
        self.timer = self.create_timer(self.control_dt, self.control_loop)

        # Note: we no longer ramp speeds in the controller; the simulator
        # handles physical acceleration/braking. Commands are applied
        # immediately.
        
        self.get_logger().info("Car Controller started")

    def clock_callback(self, msg):
        """Handle simulation clock updates from `/clock`.

        Args:
            msg (rosgraph_msgs.msg.Clock): Simulation clock message.

        Stores the latest simulation time in `self.current_sim_time` as
        a floating-point seconds value for STOP timing.
        """
        t = msg.clock
        self.current_sim_time = float(t.sec) + float(t.nanosec) * 1e-9
        
    def sign_callback(self, msg):
        """Process traffic-sign messages and dispatch handlers.

        Args:
            msg (std_msgs.msg.String): Incoming sign token from `/traffic_sign`.

        The method enforces a cooldown between duplicate sign events (based
        on simulation time) and calls the appropriate handler when a new
        sign is accepted.
        """
        # Skip processing if we don't have sim time yet
        if self.current_sim_time is None:
            return

        sign_data = msg.data

        # Verify cooldown: allow processing if the sign is different (override)
        if self.current_sim_time - self.last_sign_time < self.sign_cooldown:
            if sign_data == self.last_sign:
                return

        # Process the new sign
        if sign_data != self.last_sign:
            self.last_sign = sign_data
            self.last_sign_time = self.current_sim_time

            # Process the sign
            self.handle_traffic_sign()
            
    def handle_traffic_sign(self):
        """Dispatch to a specific handler based on the last received sign.

        No arguments. Uses `self.last_sign` to choose the handler.
        """
        if self.last_sign == 'yield':
            self.handle_yield()
        elif self.last_sign == 'stop':
            self.handle_stop()
        elif self.last_sign.startswith('speed_limit'):
            self.handle_speed_limit()
            
    def handle_yield(self):
        """Handle a YIELD sign by halving the configured `max_speed`.

        Note: this implementation mutates `self.max_speed` immediately.
        """
        if not self.yield_active:
            self.max_speed = self.max_speed // 2
            self.get_logger().info(f"YIELD received: max speed halved from {self.max_speed * 2} to {self.max_speed}")
            self.yield_active = True
        else:
            self.get_logger().info("YIELD received but already active: no change applied")
            
    def handle_stop(self):
        """Handle a STOP sign: begin commanded braking and prepare to time stop.

        The controller will step the commanded speed down (see
        `self.braking_step`) until it reaches zero; once commanded zero is
        reached the 1s stop timer will be started (using sim time).
        """

        # Reset yield flag when stopping
        self.yield_active = False

        # Command immediate stop: set commanded speed to zero and mark
        # stopped state. The simulator will handle deceleration.
        self.is_stopped = True
        self.current_speed = 0.0
        # Reset any previous stop timer; we'll start it after the vehicle
        # is observed to be physically stopped (or immediately if no
        # measured speed is available).
        self.stop_start_sim_s = None
        self.physical_below_start_sim_s = None

        # If we have a measured speed and it's above threshold, wait for
        # the physical stop confirmation; otherwise start the timer now.
        if self.measured_speed is None:
            if self.current_sim_time is not None:
                self.stop_start_sim_s = float(self.current_sim_time)
                self.get_logger().info(f"Commanded stop issued at sim {self.stop_start_sim_s:.6f}s; starting {self.stop_duration}s wait (no measured speed)")
            # Apply a short active brake immediately (if we have sim time)
            if self.current_sim_time is not None:
                self.brake_active = True
                self.brake_end_sim_s = float(self.current_sim_time) + float(self.brake_duration)
                self.get_logger().info(f"Active brake applied until sim {self.brake_end_sim_s:.6f}s")
        else:
            if abs(self.measured_speed) <= self.physical_stop_threshold:
                if self.current_sim_time is not None:
                    self.stop_start_sim_s = float(self.current_sim_time)
                    self.get_logger().info(f"Physical stop observed at sim {self.stop_start_sim_s:.6f}s; starting {self.stop_duration}s wait")
            else:
                # Begin waiting-for-physical-stop; `check_stop_timer` will
                # monitor measured speed and start the 1s timer once held
                # below threshold.
                self.waiting_for_physical_stop = True
                # Apply a short active brake while waiting for physical stop
                if self.current_sim_time is not None:
                    self.brake_active = True
                    self.brake_end_sim_s = float(self.current_sim_time) + float(self.brake_duration)
                    self.get_logger().info(f"Active brake applied until sim {self.brake_end_sim_s:.6f}s (waiting for physical stop)")

        sim_str = f"sim={self.current_sim_time:.6f}" if self.current_sim_time is not None else "sim=N/A"
        self.get_logger().info(f"STOP received: commanded immediate stop [{sim_str}]")

    def handle_speed_limit(self):
        """Handle `speed_limit_N` signs by parsing and updating `max_speed`.

        Expects `self.last_sign` to be of the form 'speed_limit_<N>'. If the
        numeric parsing fails the method logs and leaves existing limits.
        """
        
        # Reset yield flag when a new speed limit is received
        self.yield_active = False
        
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
                
            # If not stopped, ensure current speed does not exceed the
            # effective maximum (taking yield into account).
            if not self.is_stopped:
                self.current_speed = min(self.current_speed, self.max_speed)
            self.get_logger().info(f"SPEED_LIMIT received: set max_speed to {self.max_speed}")
                
        except (ValueError, IndexError):
            # If the `speed_limit` message does not include a numeric value,
            # the detector must provide the value.
            # Keep existing limits unchanged and log the event.
            self.get_logger().info("SPEED_LIMIT without numeric value received: no change applied (detector should send explicit value)")
            return
            
    def odom_callback(self, msg):
        """Receive odometry and compute planar speed magnitude.

        Computes speed = sqrt(vx**2 + vy**2) using the `twist.twist.linear`
        components and stores it in `self.measured_speed`.
        """
        try:
            vx = float(msg.twist.twist.linear.x)
            vy = float(msg.twist.twist.linear.y)
            self.measured_speed = (vx * vx + vy * vy) ** 0.5
        except Exception:
            self.measured_speed = None
            
    def check_stop_timer(self):
        """Check if the 1s STOP duration has elapsed (using sim time).

        When the stop timer has been started (see `self.stop_start_sim_s`)
        this method will resume motion after `self.stop_duration` seconds
        of simulation time have elapsed.
        """
        if not self.is_stopped:
            return
        # Clear active brake if its duration expired
        if self.brake_active and self.current_sim_time is not None and self.brake_end_sim_s is not None:
            if self.current_sim_time >= self.brake_end_sim_s:
                self.brake_active = False
                self.brake_end_sim_s = None
        # If we're waiting for the vehicle to physically slow down, check
        # measured speed first and require it to be below threshold for a
        # short hold time before starting the stop-duration timer.
        if self.waiting_for_physical_stop:
            if self.measured_speed is None:
                # No measurement yet; wait.
                return
            # If measured speed is below threshold, start/continue hold timer
            if abs(self.measured_speed) <= self.physical_stop_threshold:
                if self.current_sim_time is None:
                    return
                if self.physical_below_start_sim_s is None:
                    self.physical_below_start_sim_s = float(self.current_sim_time)
                    return
                # If held below threshold long enough, start the 1s stop timer
                if float(self.current_sim_time) - self.physical_below_start_sim_s >= self.physical_hold_time:
                    self.stop_start_sim_s = float(self.current_sim_time)
                    self.waiting_for_physical_stop = False
                    self.physical_below_start_sim_s = None
                    self.get_logger().info(f"Physical stop observed at sim {self.stop_start_sim_s:.6f}s; starting {self.stop_duration}s wait")
                    # Fall through to check the stop timer below
            else:
                # Vehicle still moving; reset hold-start time and continue waiting
                self.physical_below_start_sim_s = None
                return

        # Now ensure we have a valid sim-time start for the stop-duration timer
        if self.current_sim_time is None or self.stop_start_sim_s is None:
            return

        # Check if stop duration elapsed by simulation time
        elapsed = self.current_sim_time - self.stop_start_sim_s
        if elapsed >= self.stop_duration:
            self.get_logger().info(f"STOP complete: waited {elapsed:.2f}s (sim time) - resuming")
            self.resume_after_stop()
        
    def resume_after_stop(self):
        """Resume motion after a completed STOP period.

        Sets `self.is_stopped` to False and updates `self.current_speed`
        toward the configured `max_speed`.
        """
        self.is_stopped = False
        self.current_speed = self.max_speed
        # Ensure brake state cleared on resume
        self.brake_active = False
        self.brake_end_sim_s = None
        sim_str = f"sim={self.current_sim_time:.6f}" if self.current_sim_time is not None else "sim=N/A"
        self.get_logger().info(f"Resuming: speed set to {self.current_speed:.1f} (max={self.max_speed}) [{sim_str}]")
        
    def control_loop(self):
        """Main control loop: maintain commanded speed and publish to `/control/speed`.

        The loop enforces stepped braking when a STOP sign is active, starts
        the stop-duration timer when commanded speed reaches zero, and
        otherwise ramps `self.current_speed` toward `self.max_speed`.
        """
        # Decide current commanded speed depending on state
        speed_to_publish = 0.0

        if self.is_stopped:
            # We command zero immediately and rely on `check_stop_timer`
            # to start the 1s stop-duration once the vehicle is observed
            # to be stationary (or immediately if no measured speed).
            # If an active brake window is in effect, publish a negative
            # speed to apply braking torque for that short duration.
            if self.brake_active and self.current_sim_time is not None and self.brake_end_sim_s is not None and self.current_sim_time < self.brake_end_sim_s:
                self.check_stop_timer()
                speed_to_publish = -abs(self.brake_strength) * self.speed_conversion_factor
            else:
                self.check_stop_timer()
                speed_to_publish = 0.0
        else:
            # Immediately command the configured maximum speed (no ramping).
            self.current_speed = self.max_speed
            speed_to_publish = self.current_speed * self.speed_conversion_factor

        speed_msg = Float32()
        speed_msg.data = float(speed_to_publish)
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
