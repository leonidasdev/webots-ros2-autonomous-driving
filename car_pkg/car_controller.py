#!/usr/bin/env python3

"""Car controller node.

Description:
    Node that manages vehicle speed according to detected traffic signs
    and simulated time. It applies braking when a STOP sign is seen,
    enforces speed limits, and supports a YIELD behaviour.

Publishes:
    /control/speed (std_msgs/Float32): Commanded wheel speed (motor units).

Subscribes:
    /traffic_sign (std_msgs/String): Tokens produced by the sign detector
        (e.g. 'stop', 'yield', 'speed_limit_50').
    /clock (rosgraph_msgs/Clock): Simulation clock used for timing STOPs.
    /odom (nav_msgs/Odometry): Twist-only odometry (used to detect
        physical stop via optical-flow or wheel-based estimates).
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from rosgraph_msgs.msg import Clock
from nav_msgs.msg import Odometry


class CarController(Node):
    """ROS2 node that converts traffic-sign input into speed commands.

    Description:
        The controller listens for high-level sign tokens and publishes
        immediate speed commands to `/control/speed`. STOP handling uses
        odometry to detect a physical halt before initiating the configured
        stop-duration timer.

    Publishes:
        /control/speed (std_msgs/Float32): Commanded forward/reverse
            speed sent to the bridge which applies it to wheel motors.

    Subscribes:
        /traffic_sign (std_msgs/String): Incoming sign tokens.
        /clock (rosgraph_msgs/Clock): Simulation time for precise timing.
        /odom (nav_msgs/Odometry): Measured linear speed used to
            determine physical stop.
    """
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
        # in simulation. Example: 80 * 0.5 => 40 (motor units).
        self.speed_conversion_factor = 0.5
        
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
        self.brake_end_sim_s = None
        # Braking strength (internal units per second), maximum is approximately 174 after conversion
        self.brake_strength = 174/self.speed_conversion_factor
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

        Side effects:
            Updates `self.current_sim_time` (float seconds) used by stop
            timing logic.
        """
        t = msg.clock
        self.current_sim_time = float(t.sec) + float(t.nanosec) * 1e-9
        
    def sign_callback(self, msg):
        """Process traffic-sign messages and dispatch handlers.

        Args:
            msg (std_msgs.msg.String): Incoming sign token from
                `/traffic_sign` (string payload).

        Behaviour:
            Enforces a cooldown window to avoid rapid duplicate processing
            and dispatches to `handle_traffic_sign` when a new sign is
            accepted.
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
        """Dispatch to the appropriate handler for the last received sign.

        Uses `self.last_sign` to select the handler, for example:
        - 'yield' -> `handle_yield`
        - 'stop' -> `handle_stop`
        - 'speed_limit_*' -> `handle_speed_limit`
        """
        if self.last_sign == 'yield':
            self.handle_yield()
        elif self.last_sign == 'stop':
            self.handle_stop()
        elif self.last_sign.startswith('speed_limit'):
            self.handle_speed_limit()
            
    def handle_yield(self):
        """Handle a YIELD sign by halving the configured `max_speed`.

        Side effects:
            Mutates `self.max_speed` to half its current value and sets
            `self.yield_active`.
        """
        if not self.yield_active:
            self.max_speed = self.max_speed / 2.0
            self.get_logger().info(f"YIELD received: max speed halved from {self.max_speed * 2} to {self.max_speed}")
            self.yield_active = True
        else:
            self.get_logger().info("YIELD received but already active: no change applied")
            
    def handle_stop(self):
        """Handle a STOP sign: initiate braking and prepare the stop timer.

        Behaviour:
                - Commands an immediate stop (sets `self.is_stopped` and
                    `self.current_speed = 0.0`).
                - Initiates active braking (short negative commands) while
                    waiting for `self.measured_speed` to fall below
                    `self.physical_stop_threshold` for `self.physical_hold_time`.
                - If odometry is unavailable, may apply a short timed brake
                    as a fallback.

        Side effects:
                Sets `self.is_stopped`, `self.brake_active`, and related
                timing state used by `check_stop_timer`.
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
            # If no measured speed is available we still apply an active
            # brake continuously until a physical-stop measurement is
            # observed. Do NOT use a short timed brake here; rely on the
            # optical-flow / wheel-velocity measurement to detect the
            # physical stop and start the stop-duration timer.
            self.waiting_for_physical_stop = True
            self.brake_active = True
            self.brake_end_sim_s = None
            if self.current_sim_time is not None:
                self.get_logger().info(f"Active braking started at sim {self.current_sim_time:.6f}s; waiting for physical stop (no measured speed yet)")
        else:
            if abs(self.measured_speed) <= self.physical_stop_threshold:
                if self.current_sim_time is not None:
                    self.stop_start_sim_s = float(self.current_sim_time)
                    self.get_logger().info(f"Physical stop observed at sim {self.stop_start_sim_s:.6f}s; starting {self.stop_duration}s wait")
            else:
                # Begin waiting-for-physical-stop; `check_stop_timer` will
                # monitor measured speed and start the 1s timer once held
                # below threshold.
                # Actively brake until measured speed is observed below the
                # physical threshold for `physical_hold_time` seconds.
                self.waiting_for_physical_stop = True
                self.brake_active = True
                # Clear timed end so braking continues until measurement shows
                # the vehicle is physically stopped.
                self.brake_end_sim_s = None
                if self.current_sim_time is not None:
                    self.get_logger().info(f"Active braking started at sim {self.current_sim_time:.6f}s; waiting for physical stop")

        sim_str = f"sim={self.current_sim_time:.6f}" if self.current_sim_time is not None else "sim=N/A"
        self.get_logger().info(f"STOP received: commanded immediate stop [{sim_str}]")

    def handle_speed_limit(self):
        """Handle `speed_limit_N` signs by parsing and updating `max_speed`.

        Expects `self.last_sign` to be of the form ``speed_limit_<N>``.

        Side effects:
            Updates `self.max_speed` (clamped to [10, 100]) and adjusts
            `self.current_speed` if not stopped.
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

        Args:
            msg (nav_msgs.msg.Odometry): Incoming odometry message. The
                callback extracts `twist.twist.linear.x` and `.y` and
                computes the scalar planar speed.

        Side effects:
            Updates `self.measured_speed` (float meters/sec) or sets it to
            `None` on error.
        """
        try:
            vx = float(msg.twist.twist.linear.x)
            vy = float(msg.twist.twist.linear.y)
            self.measured_speed = (vx * vx + vy * vy) ** 0.5
        except Exception:
            self.measured_speed = None
            
    def check_stop_timer(self):
        """Monitor stop/brake timers and transition out of STOP when ready.

        Behaviour:
                - Expires timed active brake windows and starts the stop
                    duration timer if appropriate.
                - When waiting for a physical stop, requires measured speed
                    to remain below `self.physical_stop_threshold` for
                    `self.physical_hold_time` before starting the stop timer.
                - When the configured `stop_duration` has elapsed, calls
                    `resume_after_stop` to resume motion.
        """
        if not self.is_stopped:
            return
        # Handle timed brake expiry: if a timed brake was set (fallback for
        # missing odometry) expire it and start the stop-duration timer.
        if self.brake_active and self.brake_end_sim_s is not None:
            if self.current_sim_time is None:
                return
            if float(self.current_sim_time) >= float(self.brake_end_sim_s):
                self.brake_active = False
                self.brake_end_sim_s = None
                # Start stop-duration timer now that timed brake completed
                self.stop_start_sim_s = float(self.current_sim_time)
                self.get_logger().info(f"Timed brake expired at sim {self.stop_start_sim_s:.6f}s; starting {self.stop_duration}s wait")
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
                    # Stop active braking now that physical stop confirmed
                    self.brake_active = False
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
        """Resume motion after the STOP waiting period.

        Side effects:
            Clears STOP state and sets `self.current_speed = self.max_speed`.
        """
        self.is_stopped = False
        self.current_speed = self.max_speed
            
        # Ensure brake state cleared on resume
        self.brake_active = False
        self.brake_end_sim_s = None
        sim_str = f"sim={self.current_sim_time:.6f}" if self.current_sim_time is not None else "sim=N/A"
        self.get_logger().info(f"Resuming: speed set to {self.current_speed:.1f} (max={self.max_speed}) [{sim_str}]")
        
    def control_loop(self):
        """Main periodic control loop: choose and publish speed commands.

        Behaviour:
            - If in STOP state, publish negative braking commands while
              `brake_active` is True, otherwise publish zero until the
              stop timer completes.
            - If not stopped, immediately command `self.max_speed`.

        Publishes:
            std_msgs/Float32 on `/control/speed` each control tick.
        """
        # Decide current commanded speed depending on state
        speed_to_publish = 0.0

        if self.is_stopped:
            # We command zero immediately and rely on `check_stop_timer`
            # to start the 1s stop-duration once the vehicle is observed
            # to be stationary (or immediately if no measured speed).
            # If an active brake window is in effect, publish a negative
            # speed to apply braking torque for that short duration.
            # Publish braking command continuously while `brake_active` is
            # True. The brake is cleared once a physical stop is observed
            # (see `check_stop_timer`) which will then start the 1s stop
            # timer and resume motion afterward.
            if self.brake_active:
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
