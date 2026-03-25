#!/home/adam/miniforge3/envs/camera/bin/python3
# Data Collection Pipeline — ROS2 Jazzy
#
# Orchestrates random catheter movements and image captures for dataset generation.
#
# Required nodes (run separately):
#   python3 ./camera/FLIR.py          (on-demand mode, no --save-all)
#   python3 ./driver.py
#
# Usage:
#   python3 ./collect.py [options]
#
# Dataset format — JSONL, one record per step:
#   <output_dir>/dataset.jsonl
#   <output_dir>/images/<timestamp_ns>.bin   (raw BGR8, loaded via numpy)
#
# Each JSONL record:
# {
#   "step":              int,
#   "trial":            int,
#   "trial_id":         str,           e.g. "trial_0003_step_012"
#   "timestamp_iso":    str,           ISO-8601 wall time at step start
#   "image_path":       str,           absolute path to raw .bin image
#   "image_w":          int,
#   "image_h":          int,
#   "state_before":     { insertion_cm, rotation_rad, insertion_units, rotation_units },
#   "command":          { insertion, rotation, relative },
#   "state_after":      { insertion_cm, rotation_rad, insertion_units, rotation_units }
# }

import argparse
import json
import math
import os
import random
import time
import threading
from datetime import datetime, timezone

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Bool

# ---------------------------------------------------------------------------
# Action limits
# ---------------------------------------------------------------------------
INSERTION_MIN_CM   = -0.1    # cm, relative per step (biased positive)
INSERTION_MAX_CM   =  1.0
ROTATION_MIN_RAD   = -math.pi / 6
ROTATION_MAX_RAD   =  math.pi / 6

INSERTION_BIAS      = 0.7    # probability of taking a non-zero insertion step
ROTATION_BIAS       = 0.3    # probability of taking a non-zero rotation step

# Safety clamps — absolute position
INSERTION_ABS_MAX_CM  = 15
INSERTION_ABS_MIN_CM  = 0.0
ROTATION_ABS_MAX_RAD  =  math.pi * 2
ROTATION_ABS_MIN_RAD  = -math.pi * 2

MOTION_TIMEOUT  = 45.0   # s — must be > driver's 30 s timeout
IMAGE_TIMEOUT   = 10.0   # s — wait for camera save response
IMAGE_RETRIES   = 3      # retry image capture on failure
IDLE_TIMEOUT    = 40.0   # s — max wait for driver to become idle

IMAGE_SAVE_DIR = '/tmp/flir/collect'   # sent as save_request folder


class CollectNode(Node):

    def __init__(self, trials: int, steps: int, output_dir: str, step_delay: float):
        super().__init__('collect')

        self._total_trials  = trials
        self._total_steps   = steps
        self._output_dir    = output_dir
        self._step_delay    = step_delay

        self._images_dir = os.path.join(output_dir, 'images')
        os.makedirs(self._images_dir, exist_ok=True)
        self._jsonl_path = os.path.join(output_dir, 'dataset.jsonl')

        # --- State ---
        self._latest_state: dict | None = None
        self._image_event   = threading.Event()
        self._done_event    = threading.Event()
        self._latest_image: tuple | None = None  # (path, w, h)

        qos_be = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Receive catheter state
        self.create_subscription(String, 'catheter/state', self._on_state, qos_be)

        # Receive position-reached notifications
        self.create_subscription(Bool, 'catheter/position_reached', self._on_done, 10)

        # Receive saved image paths from the camera node
        self.create_subscription(String, 'flir/save_response', self._on_image, 10)

        # Publish commands and save requests
        self._cmd_pub  = self.create_publisher(String, 'catheter/command',    10)
        self._save_pub = self.create_publisher(String, 'flir/save_request',   10)

        self.get_logger().info(
            f'Collection node ready — {trials} trial(s) × {steps} step(s) '
            f'→ {self._jsonl_path}'
        )

    # ------------------------------------------------------------------
    def _on_state(self, msg: String):
        try:
            self._latest_state = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _on_done(self, msg: Bool):
        if msg.data:
            self._done_event.set()

    def _on_image(self, msg: String):
        parts = msg.data.split('|')
        if len(parts) >= 3:
            self._latest_image = (parts[0], int(parts[1]), int(parts[2]))
            self._image_event.set()

    # ------------------------------------------------------------------
    def _request_image(self) -> tuple | None:
        """Request a frame save with retries."""
        for attempt in range(1, IMAGE_RETRIES + 1):
            self._image_event.clear()
            self._latest_image = None
            req = String()
            req.data = self._images_dir
            self._save_pub.publish(req)
            deadline = time.monotonic() + IMAGE_TIMEOUT
            while time.monotonic() < deadline:
                if self._image_event.is_set():
                    return self._latest_image
                rclpy.spin_once(self, timeout_sec=0.05)
            self.get_logger().warn(
                f'Image save timed out (attempt {attempt}/{IMAGE_RETRIES}).'
            )
        return None

    def _send_command(self, insertion: float, rotation: float, relative: bool = True):
        # Ensure driver is idle before sending — prevents command rejection
        self._wait_driver_idle()
        cmd = String()
        cmd.data = json.dumps({
            'insertion': insertion,
            'rotation':  rotation,
            'relative':  relative,
        })
        self._done_event.clear()
        self._cmd_pub.publish(cmd)

    def _wait_done(self) -> bool:
        return self._done_event.is_set()

    def _wait_driver_idle(self):
        """Spin until the driver reports busy=False in its state topic."""
        deadline = time.monotonic() + IDLE_TIMEOUT
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._latest_state and not self._latest_state.get('busy', True):
                return
        self.get_logger().warn('Driver still busy after idle timeout — proceeding anyway.')

    def _current_state_snapshot(self) -> dict:
        if self._latest_state is None:
            return {}
        return {k: self._latest_state[k] for k in
                ('insertion_cm', 'rotation_rad', 'insertion_units', 'rotation_units')
                if k in self._latest_state}

    def _clamp_action(self, ins_rel: float, rot_rel: float) -> tuple[float, float]:
        """Clamp so the resulting absolute position stays within safety limits."""
        cur_ins = self._latest_state.get('insertion_cm', 0.0) if self._latest_state else 0.0
        cur_rot = self._latest_state.get('rotation_rad', 0.0) if self._latest_state else 0.0
        new_ins = max(INSERTION_ABS_MIN_CM, min(INSERTION_ABS_MAX_CM, cur_ins + ins_rel))
        new_rot = max(ROTATION_ABS_MIN_RAD, min(ROTATION_ABS_MAX_RAD, cur_rot + rot_rel))
        return new_ins - cur_ins, new_rot - cur_rot

    # ------------------------------------------------------------------
    def _append_record(self, record: dict):
        with open(self._jsonl_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    # ------------------------------------------------------------------
    def run(self):
        self.get_logger().info('Waiting for first catheter state...')
        deadline = time.monotonic() + 10.0
        while self._latest_state is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._latest_state is None:
            self.get_logger().error('No catheter state received — is driver.py running?')
            return

        for trial in range(self._total_trials):
            self.get_logger().info(f'=== Trial {trial + 1}/{self._total_trials} ===')

            # Reset catheter to home at start of each trial
            self._wait_driver_idle()
            reset_cmd = String()
            reset_cmd.data = json.dumps({'reset': True})
            self._done_event.clear()
            self._cmd_pub.publish(reset_cmd)
            self.get_logger().info('Resetting catheter to home...')
            self._spin_until(self._wait_done, timeout=MOTION_TIMEOUT)

            for step in range(self._total_steps):
                trial_id = f'trial_{trial:04d}_step_{step:03d}'
                self.get_logger().info(f'  {trial_id}')

                try:
                    # 1. Capture image before action
                    img = self._request_image()

                    if img is None:
                        self.get_logger().warn(f'{trial_id}: no image, skipping step.')
                        continue

                    image_path, img_w, img_h = img

                    # 2. Snapshot state before
                    state_before = self._current_state_snapshot()

                    # 3. Random action, biased towards insertion
                    if random.random() < INSERTION_BIAS:
                        ins_rel = random.uniform(INSERTION_MIN_CM, INSERTION_MAX_CM)
                    else:
                        ins_rel = 0.0
                    if random.random() < ROTATION_BIAS:
                        rot_rel = random.uniform(ROTATION_MIN_RAD, ROTATION_MAX_RAD)
                    else:
                        rot_rel = 0.0
                    ins_rel, rot_rel = self._clamp_action(ins_rel, rot_rel)

                    # 4. Send command and wait for completion
                    self._send_command(ins_rel, rot_rel, relative=True)
                    self.get_logger().info(
                        f'    → ins {ins_rel:+.4f} cm  rot {rot_rel:+.4f} rad'
                    )
                    if not self._spin_until(self._wait_done, timeout=MOTION_TIMEOUT):
                        self.get_logger().warn(f'{trial_id}: motion timed out, waiting for driver idle...')
                        self._wait_driver_idle()

                    # 5. Snapshot state after
                    rclpy.spin_once(self, timeout_sec=0.1)
                    state_after = self._current_state_snapshot()

                    # 6. Write record
                    record = {
                        'step':          step,
                        'trial':         trial,
                        'trial_id':      trial_id,
                        'timestamp_iso': datetime.now(timezone.utc).isoformat(),
                        'image_path':    image_path,
                        'image_w':       img_w,
                        'image_h':       img_h,
                        'state_before':  state_before,
                        'command': {
                            'insertion': ins_rel,
                            'rotation':  rot_rel,
                            'relative':  True,
                        },
                        'state_after':   state_after,
                    }
                    self._append_record(record)

                except Exception as ex:
                    self.get_logger().error(f'{trial_id}: unexpected error: {ex}')
                    self._wait_driver_idle()

                if self._step_delay > 0:
                    time.sleep(self._step_delay)

        self.get_logger().info(f'Collection complete. Dataset: {self._jsonl_path}')

    # ------------------------------------------------------------------
    def _spin_until(self, wait_fn, timeout: float):
        """Keep spinning ROS2 callbacks while waiting for a blocking condition."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if wait_fn():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().warn('Timed out waiting for condition.')
        return False


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Catheter data collection pipeline')
    parser.add_argument('--trials',   type=int,   default=10,
                        help='Number of trials (resets between each). Default: 10')
    parser.add_argument('--steps',    type=int,   default=50,
                        help='Steps (random actions) per trial. Default: 100')
    parser.add_argument('--output',   default='/tmp/catheter_dataset',
                        help='Output directory for dataset. Default: /tmp/catheter_dataset')
    parser.add_argument('--delay',    type=float, default=0.0,
                        help='Extra sleep between steps in seconds. Default: 0')
    known, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = CollectNode(
        trials=known.trials,
        steps=known.steps,
        output_dir=known.output,
        step_delay=known.delay,
    )
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
