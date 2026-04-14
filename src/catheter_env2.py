import json
import math
import time
import threading
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import gymnasium as gym
from gymnasium import spaces
from segment_guidewire import segment_guidewire, find_guidewire_tip, detect_dots, generate_goal_points

# Actuation limits
INSERTION_ABS_MAX_CM  = 15
INSERTION_ABS_MIN_CM  = 0.0
ROTATION_ABS_MAX_RAD  =  math.pi * 2
ROTATION_ABS_MIN_RAD  = -math.pi * 2

MOTION_TIMEOUT  = 45.0   # s — must be > driver's 30 s timeout
IMAGE_TIMEOUT   = 10.0   # s — wait for camera save response
IMAGE_RETRIES   = 3      # retry image capture on failure
IDLE_TIMEOUT    = 40.0   # s — max wait for driver to become idle


class CatheterRosBridge(Node):
    def __init__(self, images_dir: str = '/tmp/catheter_images', jsonl_path: str = '/tmp/catheter_log.jsonl'):
        Node.__init__(self, "catheter_bridge_node")

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
        self._cmd_pub  = self.create_publisher(String, 'catheter/command', 10)
        self._save_pub = self.create_publisher(String, 'flir/save_request', 10)

        # Internal state
        self._latest_state: dict | None = None
        self._image_event  = threading.Event()
        self._done_event   = threading.Event()
        self._latest_image: tuple | None = None  # (path, w, h)

        # Paths
        self._images_dir = images_dir
        self._jsonl_path = jsonl_path
        Path(images_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Subscriber callbacks                                                #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    #  Public helpers                                                      #
    # ------------------------------------------------------------------ #

    def request_image(self) -> tuple | None:
        """Request a frame save with retries. Returns (path, w, h) or None."""
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

    def send_command(self, ins_rel: float, rot_rel: float):
        """Publish a relative move command and clear the done flag."""
        self._done_event.clear()
        cmd = String()
        cmd.data = json.dumps({'ins_rel_cm': ins_rel, 'rot_rel_rad': rot_rel})
        self._cmd_pub.publish(cmd)

    def wait_done(self) -> bool:
        return self._done_event.is_set()

    def wait_driver_idle(self):
        """Spin until the driver reports busy=False in its state topic."""
        deadline = time.monotonic() + IDLE_TIMEOUT
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._latest_state and not self._latest_state.get('busy', True):
                return
        self.get_logger().warn('Driver still busy after idle timeout — proceeding anyway.')

    def current_state_snapshot(self) -> dict:
        if self._latest_state is None:
            return {}
        return {k: self._latest_state[k] for k in
                ('insertion_cm', 'rotation_rad', 'insertion_units', 'rotation_units')
                if k in self._latest_state}

    def clamp_action(self, ins_rel: float, rot_rel: float) -> tuple[float, float]:
        """Clamp so the resulting absolute position stays within safety limits."""
        cur_ins = self._latest_state.get('insertion_cm', 0.0) if self._latest_state else 0.0
        cur_rot = self._latest_state.get('rotation_rad', 0.0) if self._latest_state else 0.0
        new_ins = max(INSERTION_ABS_MIN_CM, min(INSERTION_ABS_MAX_CM, cur_ins + ins_rel))
        new_rot = max(ROTATION_ABS_MIN_RAD, min(ROTATION_ABS_MAX_RAD, cur_rot + rot_rel))
        return new_ins - cur_ins, new_rot - cur_rot

    def append_record(self, record: dict):
        with open(self._jsonl_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def spin_until(self, wait_fn, timeout: float) -> bool:
        """Keep spinning ROS2 callbacks while waiting for a blocking condition."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if wait_fn():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().warn('Timed out waiting for condition.')
        return False


# ====================================================================== #

class CatheterEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        max_steps_per_goal: int = 10,
        goal_tolerance_px: float = 15.0,
        crop_size: int = 128,
        goal_distance_mult: float = 1.0,
        images_dir: str = '/tmp/catheter_images',
        jsonl_path: str = '/tmp/catheter_log.jsonl',
    ):
        super().__init__()
        self._bridge             = CatheterRosBridge(images_dir=images_dir, jsonl_path=jsonl_path)
        self._max_steps          = max_steps_per_goal
        self._goal_tolerance_px  = goal_tolerance_px
        self._crop_size          = crop_size
        self._goal_distance_mult = goal_distance_mult

        # Action space: [insertion_rel_cm, rotation_rel_rad]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -math.pi / 4], dtype=np.float32),
            high=np.array([1.0,  math.pi / 4], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: grayscale crop centred on tip + normalised pixel coords
        self.observation_space = spaces.Dict({
            'image':   spaces.Box(low=0, high=255, shape=(crop_size, crop_size, 1), dtype=np.uint8),
            'tip_xy':  spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'goal_xy': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        })

        self.current_step: int           = 0
        self.current_goal: np.ndarray | None = None   # (x, y) in full-image pixels
        self.current_tip:  np.ndarray | None = None   # (x, y) in full-image pixels
        self.current_image: np.ndarray | None = None  # last raw BGR frame
        self._spline_points: list | None = None       # trailing spline from last capture

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment.

        options (optional dict) keys
        ----------------------------
        goal_xy : (x, y) pixel coords to use as the goal.
                  If absent, a point is sampled via generate_goal_points().
        """
        super().reset(seed=seed)
        self.current_step = 0

        self._bridge.wait_driver_idle()

        obs, tip_xy, full_img, spline_pts = self._capture_obs()
        if obs is None:
            raise RuntimeError('reset(): failed to capture initial image.')

        self.current_image  = full_img
        self.current_tip    = tip_xy
        self._spline_points = spline_pts

        if options and 'goal_xy' in options:
            self.current_goal = np.array(options['goal_xy'], dtype=np.float32)
        else:
            self.current_goal = self._sample_goal(full_img, tip_xy, spline_pts)

        info = {
            'tip_xy':  self.current_tip.tolist(),
            'goal_xy': self.current_goal.tolist(),
            'state':   self._bridge.current_state_snapshot(),
        }
        return obs, info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """
        Apply action = [ins_rel_cm, rot_rel_rad], move the catheter, observe.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        self.current_step += 1

        ins_rel, rot_rel = self._bridge.clamp_action(float(action[0]), float(action[1]))

        self._bridge.send_command(ins_rel, rot_rel)
        reached = self._bridge.spin_until(self._bridge.wait_done, MOTION_TIMEOUT)
        if not reached:
            self._bridge.get_logger().warn('step(): motion timed out.')

        obs, tip_xy, full_img, spline_pts = self._capture_obs()
        if obs is None:
            return self._null_obs(), -1.0, False, True, {'error': 'image_capture_failed'}

        prev_tip = self.current_tip.copy() if self.current_tip is not None else tip_xy
        self.current_image  = full_img
        self.current_tip    = tip_xy
        self._spline_points = spline_pts

        reward, dist = self._compute_reward(prev_tip, tip_xy, self.current_goal)
        terminated = dist < self._goal_tolerance_px
        truncated  = (not terminated) and (self.current_step >= self._max_steps)

        info = {
            'tip_xy':      tip_xy.tolist(),
            'goal_xy':     self.current_goal.tolist(),
            'dist_px':     float(dist),
            'ins_rel_cm':  ins_rel,
            'rot_rel_rad': rot_rel,
            'state':       self._bridge.current_state_snapshot(),
            'motion_ok':   reached,
        }

        self._bridge.append_record({
            'step': self.current_step,
            **info,
            'reward':     reward,
            'terminated': terminated,
            'truncated':  truncated,
        })

        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Return an annotated BGR frame (goal = green cross, tip = red circle)."""
        if self.current_image is None:
            return None
        frame = self.current_image.copy()
        if self.current_goal is not None:
            cx, cy = int(self.current_goal[0]), int(self.current_goal[1])
            cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if self.current_tip is not None:
            tx, ty = int(self.current_tip[0]), int(self.current_tip[1])
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)
        return frame

    def close(self):
        self._bridge.destroy_node()

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _capture_obs(
        self,
    ) -> tuple[dict | None, np.ndarray | None, np.ndarray | None, list | None]:
        """
        Request an image from the camera node, run the segmentation pipeline,
        and assemble an observation dict.

        Pipeline
        --------
        segment_guidewire(path)  → boolean skeleton
        find_guidewire_tip(skel) → (tip_position, spline_points)

        Returns (obs_dict, tip_xy, full_bgr_image, spline_points)
        or      (None, None, None, None) on any failure.
        """
        result = self._bridge.request_image()
        if result is None:
            return None, None, None, None

        img_path, _w, _h = result

        full_img = cv2.imread(img_path)
        if full_img is None:
            return None, None, None, None

        # segment_guidewire expects a file path and returns a boolean skeleton
        skeleton = segment_guidewire(img_path)
        if skeleton is None:
            return None, None, None, None

        tip_result = find_guidewire_tip(skeleton)
        if tip_result is None:
            return None, None, None, None

        tip_position, spline_points = tip_result
        tip_xy = np.array(tip_position, dtype=np.float32)

        img_h, img_w = full_img.shape[:2]
        norm = np.array([img_w, img_h], dtype=np.float32)
        obs = {
            'image':   self._crop_around(full_img, tip_xy),
            'tip_xy':  tip_xy / norm,
            'goal_xy': (self.current_goal / norm
                        if self.current_goal is not None
                        else np.zeros(2, dtype=np.float32)),
        }
        return obs, tip_xy, full_img, spline_points

    def _crop_around(self, img: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Return a (crop_size × crop_size × 1) uint8 grayscale patch centred on *centre*."""
        h, w = img.shape[:2]
        half = self._crop_size // 2
        cx, cy = int(centre[0]), int(centre[1])

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, x1 + self._crop_size)
        y2 = min(h, y1 + self._crop_size)

        gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

        pad_b = self._crop_size - gray.shape[0]
        pad_r = self._crop_size - gray.shape[1]
        if pad_b > 0 or pad_r > 0:
            gray = np.pad(gray, ((0, pad_b), (0, pad_r)), mode='edge')

        return gray[..., np.newaxis]  # (H, W, 1)

    def _compute_reward(
        self,
        prev_tip: np.ndarray,
        curr_tip: np.ndarray,
        goal: np.ndarray,
    ) -> tuple[float, float]:
        """
        Dense shaped reward = improvement in distance to goal.
        +1 success bonus, −0.01 living penalty.

        Returns (reward, current_distance_px).
        """
        prev_dist = float(np.linalg.norm(prev_tip - goal))
        curr_dist = float(np.linalg.norm(curr_tip - goal))

        reward  = self._goal_distance_mult * (prev_dist - curr_dist)
        reward -= 0.01

        if curr_dist < self._goal_tolerance_px:
            reward += 1.0

        return reward, curr_dist

    def _sample_goal(
        self,
        img: np.ndarray,
        tip_xy: np.ndarray,
        spline_points: list | None = None,
    ) -> np.ndarray:
        """
        Use generate_goal_points() + detect_dots() (from segment_guidewire.py)
        to pick a goal that avoids the dot-grid obstacles.

        Falls back to a random obstacle-free pixel if generate_goal_points
        returns nothing.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obstacle_mask = detect_dots(gray)  # uint8, 255 inside dots

        tip_position = (int(tip_xy[0]), int(tip_xy[1]))
        pts = spline_points if spline_points else [tip_position]

        goals = generate_goal_points(
            tip_position=tip_position,
            spline_points=pts,
            obstacle_mask=obstacle_mask,
            num_goal_points=1,
        )
        if goals:
            return np.array(goals[0], dtype=np.float32)

        # Fallback: uniform random pixel that is obstacle-free and far enough away
        h, w = img.shape[:2]
        for _ in range(200):
            gx = int(self.np_random.integers(0, w))
            gy = int(self.np_random.integers(0, h))
            if obstacle_mask[gy, gx] == 0:
                candidate = np.array([gx, gy], dtype=np.float32)
                if np.linalg.norm(candidate - tip_xy) >= self._goal_tolerance_px * 2:
                    return candidate

        # Last resort
        return np.array([w - tip_xy[0], h - tip_xy[1]], dtype=np.float32)

    def _null_obs(self) -> dict:
        """Zero-filled observation returned on capture failure."""
        return {
            'image':   np.zeros((self._crop_size, self._crop_size, 1), dtype=np.uint8),
            'tip_xy':  np.zeros(2, dtype=np.float32),
            'goal_xy': np.zeros(2, dtype=np.float32),
        }