#!/home/adam/miniforge3/envs/camera/bin/python3
# Catheter Driver — ROS2 Jazzy
# Bridges the Trust serial catheter controller to ROS2 topics.
#
# Topics
# ------
#   Sub  /catheter/command           std_msgs/String  JSON command (see below)
#   Pub  /catheter/state             std_msgs/String  JSON state at 10 Hz
#   Pub  /catheter/feedback          std_msgs/String  Raw device response lines
#   Pub  /catheter/position_reached  std_msgs/Bool    True when motion completes
#
# Command JSON schema
# -------------------
#   {"insertion": <cm>, "rotation": <radians>, "relative": <bool>}
#   "relative": true  → add to current position (default)
#   "relative": false → move to absolute position
#
# Device unit conversions (from hardware spec)
#   500  insertion units  = 1 cm
#   2650 rotation units   = 1 full rotation (2π rad)

import json
import math
import threading
import time
import argparse

import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

SERIAL_PORT      = '/dev/ttyUSB0'
SERIAL_BAUD      = 115200
SERIAL_TIMEOUT   = 0.1    # s — readline timeout (short so we can check deadline)
MOTION_TIMEOUT_S = 30.0   # s — max time to wait for POSITION_REACHED before giving up
SILENCE_TIMEOUT  = 3.0    # s — if device goes quiet after acknowledging, assume done

INSERTION_UNITS_PER_CM  = 500
ROTATION_UNITS_PER_REV  = 2650


class CatheterDriverNode(Node):

    def __init__(self, port: str, baud: int):
        super().__init__('catheter_driver')

        # Tracked absolute position (physical units)
        self._total_insertion = 0.0   # cm
        self._total_rotation  = 0.0   # radians
        self._busy            = False
        self._lock            = threading.Lock()

        # --- Publishers ---
        self._state_pub    = self.create_publisher(String, 'catheter/state',            10)
        self._feedback_pub = self.create_publisher(String, 'catheter/feedback',         10)
        self._done_pub     = self.create_publisher(Bool,   'catheter/position_reached', 10)

        # --- Subscriber ---
        self._cmd_sub = self.create_subscription(
            String, 'catheter/command', self._on_command, 10
        )

        # --- Serial ---
        self.get_logger().info(f'Opening {port} @ {baud} baud...')
        self._serial = serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)
        time.sleep(2.0)   # allow device firmware to initialise
        self.get_logger().info('Catheter driver ready.')

        # State published at 10 Hz
        self.create_timer(0.1, self._publish_state)

    # ------------------------------------------------------------------
    def _on_command(self, msg: String):
        """
        Parse a JSON command and dispatch to a background thread.

        Accepted keys:
          insertion  float   cm        (default 0.0)
          rotation   float   radians   (default 0.0)
          relative   bool              (default true)
          reset      bool   if true, ignore insertion/rotation and go to 0,0
        """
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as ex:
            self.get_logger().error(f'Invalid command JSON: {ex}')
            return

        if self._busy:
            self.get_logger().warn('Command queued — waiting for previous action to finish.')
            # Block until the previous action completes instead of dropping the command
            deadline = time.time() + MOTION_TIMEOUT_S + 5.0
            while self._busy and time.time() < deadline:
                time.sleep(0.1)
            if self._busy:
                self.get_logger().error('Previous action never finished — forcing reset of busy flag.')
                self._busy = False

        if cmd.get('reset', False):
            insertion, rotation, relative = 0.0, 0.0, False
        else:
            insertion = float(cmd.get('insertion', 0.0))
            rotation  = float(cmd.get('rotation',  0.0))
            relative  = bool(cmd.get('relative',   True))

        threading.Thread(
            target=self._perform_action,
            args=(insertion, rotation, relative),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    def _perform_action(self, insertion: float, rotation: float, relative: bool):
        with self._lock:
            self._busy = True
            try:
                if relative:
                    self._total_insertion += insertion
                    self._total_rotation  += rotation
                else:
                    self._total_insertion = insertion
                    self._total_rotation  = rotation

                ins_units = self._total_insertion * INSERTION_UNITS_PER_CM
                rot_units = (self._total_rotation / (2 * math.pi)) * ROTATION_UNITS_PER_REV

                cmd_str = f'{ins_units:.2f},{rot_units:.2f}\n'
                self.get_logger().info(f'→ device: {cmd_str.strip()}')

                # Drain any stale serial data before sending
                self._serial.reset_input_buffer()
                self._serial.write(cmd_str.encode('utf-8'))

                # Wait for POSITION_REACHED, with a hard timeout.
                # If the device acknowledges the command ("Parsed") but then
                # goes silent for SILENCE_TIMEOUT seconds, treat that as done
                # — some firmware versions don't send POSITION_REACHED.
                deadline     = time.time() + MOTION_TIMEOUT_S
                confirmed    = False
                acknowledged = False
                last_rx      = time.time()
                while time.time() < deadline:
                    raw = self._serial.readline()
                    if not raw:
                        # No data — check silence timeout after acknowledgment
                        if acknowledged and (time.time() - last_rx) >= SILENCE_TIMEOUT:
                            self.get_logger().info(
                                f'Device silent for {SILENCE_TIMEOUT:.0f} s '
                                'after ack — assuming motion complete.'
                            )
                            confirmed = True
                            break
                        continue
                    line = raw.decode('utf-8', errors='replace').strip()
                    if line:
                        last_rx = time.time()
                        self.get_logger().info(f'← device: {line}')
                        fb = String()
                        fb.data = line
                        self._feedback_pub.publish(fb)
                    if 'POSITION_REACHED' in line:
                        confirmed = True
                        break
                    if 'Parsed' in line or 'Received' in line:
                        acknowledged = True

                if not confirmed:
                    self.get_logger().warn(
                        f'Motion timed out after {MOTION_TIMEOUT_S:.0f} s — '
                        'declaring done anyway.'
                    )

                done = Bool()
                done.data = True
                self._done_pub.publish(done)
                self.get_logger().info(
                    f'Position reached — '
                    f'insertion={self._total_insertion:.4f} cm  '
                    f'rotation={self._total_rotation:.4f} rad'
                )

            except serial.SerialException as ex:
                self.get_logger().error(f'Serial error: {ex}')
            finally:
                self._busy = False

    # ------------------------------------------------------------------
    def _publish_state(self):
        state = {
            'insertion_cm':     self._total_insertion,
            'rotation_rad':     self._total_rotation,
            'insertion_units':  self._total_insertion * INSERTION_UNITS_PER_CM,
            'rotation_units':   (self._total_rotation / (2 * math.pi)) * ROTATION_UNITS_PER_REV,
            'busy':             self._busy,
        }
        msg = String()
        msg.data = json.dumps(state)
        self._state_pub.publish(msg)

    # ------------------------------------------------------------------
    def destroy_node(self):
        self.get_logger().info('Shutting down catheter driver.')
        if self._serial.is_open:
            self._serial.close()
        super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description='Catheter serial driver ROS2 node')
    parser.add_argument('--port', default=SERIAL_PORT,
                        help=f'Serial port (default: {SERIAL_PORT})')
    parser.add_argument('--baud', default=SERIAL_BAUD, type=int,
                        help=f'Baud rate (default: {SERIAL_BAUD})')
    known, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = CatheterDriverNode(known.port, known.baud)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
