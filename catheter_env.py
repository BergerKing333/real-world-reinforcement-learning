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

import gymnasium as gym
from gymnasium import spaces

from segment_guidewire import extract_tip_from_image


class CatheterEnv(Node, gym.Env):
    """
    Gym-style catheter environment.

    Action space:
        [delta_insertion_cm, delta_rotation_rad]

    Observation space:
        New: crop_size x crop_size x 3 BGR image centered on the guidewire tip
        the goal position is drawn as a green dot on the crop so the agent
        can see where it needs to go relative to tip position (center).

        Previous: [tip_x, tip_y, goal_x, goal_y]

    Important note:
    The agent still sends actions in robot control space, but the state and goal
    are both defined in image space. That matches the actual task better than
    using insertion/rotation targets directly.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps=5,                  # changed from 50 - temporary value pre-training
        goal_tolerance_px=20.0,
        image_timeout=10.0,
        motion_timeout=45.0,
        image_shape=(2048, 2448, 3),  
        crop_size=224,                # new - size of the square image observation crop around tip
        goal_dist_px=60.0,            # new - how far ahead on the skeleton to place subgoals (temporary value pre-training)
        goal_dot_radius=8,            # new - size of the goal marker drawn on the crop 
        use_mock_cv=True,
    ):
        Node.__init__(self, "catheter_rl_env")
        gym.Env.__init__(self)

        self.max_steps = max_steps
        self.goal_tolerance_px = goal_tolerance_px
        self.image_timeout = image_timeout
        self.motion_timeout = motion_timeout
        self.image_shape = image_shape
        self.crop_size = crop_size                     
        self.goal_dist_px = goal_dist_px
        self.goal_dot_radius = goal_dot_radius
        self.use_mock_cv = use_mock_cv

        self.current_step = 0

        self.latest_state = None
        self.latest_image_info = None  # (path, width, height)

        self.done_event = threading.Event()
        self.image_event = threading.Event()

        self.goal_xy = None
        self.tip_xy = None

        # cached between steps so get_obs doesnt need to request a new image
        self._last_image_bgr = None
        self._last_skeleton = None

        # Control space 
        self.action_space = spaces.Box(
            low=np.array([-0.1, -math.pi / 6], dtype=np.float32),
            high=np.array([1.0, math.pi / 6], dtype=np.float32),
            dtype=np.float32,
        )

        # changed from 4-float [tip_x tip_y goal_x goal_y] to a crop image
        # crop centered on the tip
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.crop_size, self.crop_size, 3),
            dtype=np.uint8,
        )

        # ROS subscriptions
        self.state_sub = self.create_subscription(
            String, "catheter/state", self.on_state, 10
        )
        self.done_sub = self.create_subscription(
            Bool, "catheter/position_reached", self.on_done, 10
        )
        self.image_sub = self.create_subscription(
            String, "flir/save_response", self.on_image, 10
        )

        # ROS publishers
        self.cmd_pub = self.create_publisher(String, "catheter/command", 10)
        self.save_pub = self.create_publisher(String, "flir/save_request", 10)

        self.image_save_dir = str(Path("/tmp/flir/collect"))

    # ------------------------------------------------------------------
    # ROS callbacks
    # ------------------------------------------------------------------
    def on_state(self, msg):
        try:
            self.latest_state = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def on_done(self, msg):
        if msg.data:
            self.done_event.set()

    def on_image(self, msg):
        parts = msg.data.split("|")
        if len(parts) >= 3:
            self.latest_image_info = (parts[0], int(parts[1]), int(parts[2]))
            self.image_event.set()

    # ------------------------------------------------------------------
    # Wait helpers
    # ------------------------------------------------------------------
    def wait_for_state(self, timeout=10.0):
        start = time.time()
        while self.latest_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                raise TimeoutError("No state received from /catheter/state")

    def wait_done(self, timeout=None):
        if timeout is None:
            timeout = self.motion_timeout

        start = time.time()
        while not self.done_event.is_set():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for /catheter/position_reached")

    def request_image(self):
        self.image_event.clear()
        self.latest_image_info = None

        msg = String()
        msg.data = self.image_save_dir
        self.save_pub.publish(msg)

        start = time.time()
        while not self.image_event.is_set():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - start > self.image_timeout:
                raise TimeoutError("Timed out waiting for /flir/save_response")

        return self.latest_image_info

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------
    def load_raw_bgr_image(self, image_path, width, height):
        """
        Load a raw BGR8 image written by the FLIR pipeline.
        """
        frame = np.fromfile(image_path, dtype=np.uint8)
        expected = height * width * 3

        if frame.size != expected:
            raise ValueError(
                f"Unexpected image size for {image_path}. "
                f"Expected {expected} bytes, got {frame.size}."
            )

        return frame.reshape(height, width, 3)

    # ------------------------------------------------------------------
    # CV hook
    # ------------------------------------------------------------------
    def extract_tip_xy(self, image_bgr):
        """
        Return the current guidewire tip as (x, y) in image coordinates.

        During local development I can still use a mock CV mode so the RL
        pipeline is easy to test even when the real vision path is not ready.

        When use_mock_cv=False, this uses the actual guidewire extraction code.
        """
        if self.use_mock_cv:
            # ammars original mock tip - computes a fake tip from insertion and rotation
            # so the gym loop can be tested without a camera attached
            h, w, _ = image_bgr.shape

            insertion = float(self.latest_state.get("insertion_cm", 0.0)) if self.latest_state else 0.0
            rotation = float(self.latest_state.get("rotation_rad", 0.0)) if self.latest_state else 0.0

            base_x = 150.0
            base_y = h / 2.0

            x = base_x + 80.0 * insertion
            y = base_y + 200.0 * math.sin(rotation)

            x = float(np.clip(x, 0, w - 1))
            y = float(np.clip(y, 0, h - 1))
            return np.array([x, y], dtype=np.float32)

        # changed from _ _ tip_xy - now we also capture the skeleton instead of throwing it away
        # skeleton is stored so _sample_goal can walk along it to pick a reachable subgoal
        wire_mask, skeleton, tip_xy = extract_tip_from_image(image_bgr, top_percent=0.5)
        self._last_skeleton = skeleton

        # If tracking fails on one frame, keep going using the previous tip.
        # That makes the environment less brittle while the CV is still being tuned.
        if tip_xy is None:
            if self.tip_xy is not None:
                return self.tip_xy.copy()
            return np.array([0.0, 0.0], dtype=np.float32)

        return np.array([float(tip_xy[0]), float(tip_xy[1])], dtype=np.float32)

    def get_current_tip_xy(self):
        image_path, w, h = self.request_image()
        image_bgr = self.load_raw_bgr_image(image_path, w, h)
        # cache the full frame so get_obs can build the crop without another camera request
        self._last_image_bgr = image_bgr
        tip_xy = self.extract_tip_xy(image_bgr)
        self.tip_xy = tip_xy
        return tip_xy

    # ------------------------------------------------------------------
    # Goal handling
    # ------------------------------------------------------------------
    def _sample_goal(self, skeleton, tip_xy):
        """
        Pick a subgoal along the guidewire skeleton.

        Real mode: walk the guidewire skeleton and return the point closest
        to goal_dist_px pixels away from the current tip. That gives us a
        local reachable subgoal ~3 moves ahead.

        Mock / fallback: random offset from tip - this is basically ammars original
        sample_goal_xy kept as a safety net for when cv fails or mock mode is on.
        """
        # ammars original random goal logic kept here as fallback
        # fires in mock mode or if the cv pipeline returns an empty skeleton
        if skeleton is None or self.use_mock_cv:
            h, w, _ = self.image_shape
            offset = np.random.uniform(-self.goal_dist_px, self.goal_dist_px, size=2).astype(np.float32)
            return np.clip(tip_xy + offset, [0.0, 0.0], [float(w - 1), float(h - 1)])

        ys, xs = np.where(skeleton > 0)

        # cv ran but found no wire pixels - fall back to random offset same as above
        if len(xs) == 0:
            h, w, _ = self.image_shape
            offset = np.random.uniform(-self.goal_dist_px, self.goal_dist_px, size=2).astype(np.float32)
            return np.clip(tip_xy + offset, [0.0, 0.0], [float(w - 1), float(h - 1)])

        # real subgoal selection - find skeleton point ~goal_dist_px ahead of the tip
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        dists = np.linalg.norm(pts - tip_xy, axis=1)
        idx = int(np.argmin(np.abs(dists - self.goal_dist_px)))
        return pts[idx]

    def set_goal_xy(self, goal_xy):
        self.goal_xy = np.array(goal_xy, dtype=np.float32)

    # ------------------------------------------------------------------
    # Observation / reward
    # ------------------------------------------------------------------
    def _build_crop(self, image_bgr, tip_xy, goal_xy):
        """
        Cut a crop_size x crop_size patch from image_bgr centered on tip_xy.
        If the tip is near the image edge the patch is zero-padded.
        The goal position is drawn as a filled green circle on the patch so
        the network can see where to go relative to the tip.
        """
        h, w, _ = image_bgr.shape
        half = self.crop_size // 2

        cx, cy = int(round(float(tip_xy[0]))), int(round(float(tip_xy[1])))

        # window boundaries in full image coords
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half

        # how much to pad if the window goes outside the image
        pad_left   = max(0, -x0)
        pad_right  = max(0, x1 - w)
        pad_top    = max(0, -y0)
        pad_bottom = max(0, y1 - h)

        # clamp to valid image region then extract
        sx0, sx1 = max(0, x0), min(w, x1)
        sy0, sy1 = max(0, y0), min(h, y1)
        patch = image_bgr[sy0:sy1, sx0:sx1].copy()

        # pad to full crop size if tip was near an edge
        if pad_left or pad_right or pad_top or pad_bottom:
            patch = np.pad(
                patch,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=0,
            )

        # translate goal into crop-local coords and draw it if visible
        gx_crop = int(round(float(goal_xy[0]) - x0))
        gy_crop = int(round(float(goal_xy[1]) - y0))
        if 0 <= gx_crop < self.crop_size and 0 <= gy_crop < self.crop_size:
            cv2.circle(patch, (gx_crop, gy_crop), self.goal_dot_radius, (0, 255, 0), -1)

        return patch

    def get_obs(self):
        if self.tip_xy is None or self.goal_xy is None:
            raise RuntimeError("tip_xy or goal_xy is not set")

        # main observation format is the image crop - _build_crop is what actually runs
        # the zeros branch below is just a defensive guard that should never fire in normal
        # operation because get_current_tip_xy always sets _last_image_bgr before get_obs is called
        if self._last_image_bgr is None:
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)

        return self._build_crop(self._last_image_bgr, self.tip_xy, self.goal_xy)

    @staticmethod
    def pixel_distance(p1, p2):
        return float(
            np.linalg.norm(
                np.array(p1, dtype=np.float32) - np.array(p2, dtype=np.float32)
            )
        )

    def compute_reward(self, old_tip_xy, new_tip_xy):
        old_dist = self.pixel_distance(old_tip_xy, self.goal_xy)
        new_dist = self.pixel_distance(new_tip_xy, self.goal_xy)

        # Reward progress toward the goal in image space
        reward = old_dist - new_dist

        # Tiny penalty so the policy does not learn to jitter forever
        reward -= 0.01

        terminated = new_dist < self.goal_tolerance_px
        if terminated:
            reward += 10.0

        info = {
            "old_distance_px": old_dist,
            "new_distance_px": new_dist,
        }
        return reward, terminated, info

    # ------------------------------------------------------------------
    # Robot command
    # ------------------------------------------------------------------
    def send_command(self, insertion_delta_cm, rotation_delta_rad):
        msg = String()
        msg.data = json.dumps(
            {
                "insertion": float(insertion_delta_cm),
                "rotation": float(rotation_delta_rad),
                "relative": True,
            }
        )
        self.done_event.clear()
        self.cmd_pub.publish(msg)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.wait_for_state()

        # Send the catheter back to home before starting a new episode
        reset_cmd = String()
        reset_cmd.data = json.dumps({"reset": True})
        self.done_event.clear()
        self.cmd_pub.publish(reset_cmd)

        self.wait_done()

        # get current tip - also caches _last_image_bgr and _last_skeleton
        self.tip_xy = self.get_current_tip_xy()

        # changed from random pixel to skeleton-based subgoal
        # options still lets you pass an explicit goal for evaluation
        if options is not None and "goal_xy" in options:
            self.set_goal_xy(options["goal_xy"])
        else:
            self.set_goal_xy(self._sample_goal(self._last_skeleton, self.tip_xy))

        obs = self.get_obs()
        info = {
            "tip_xy": self.tip_xy.copy(),
            "goal_xy": self.goal_xy.copy(),
        }
        return obs, info

    def step(self, action):
        self.current_step += 1

        insertion_delta = float(action[0])
        rotation_delta = float(action[1])

        old_tip_xy = self.tip_xy.copy()

        self.send_command(insertion_delta, rotation_delta)
        self.wait_done()

        # get_current_tip_xy also refreshes _last_image_bgr and _last_skeleton
        new_tip_xy = self.get_current_tip_xy()

        reward, terminated, reward_info = self.compute_reward(old_tip_xy, new_tip_xy)
        truncated = self.current_step >= self.max_steps

        obs = self.get_obs()
        info = {
            "tip_xy": self.tip_xy.copy(),
            "goal_xy": self.goal_xy.copy(),
            **reward_info,
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self.destroy_node()
