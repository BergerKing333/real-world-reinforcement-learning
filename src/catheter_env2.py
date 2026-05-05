import json
import math
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import gymnasium as gym
from gymnasium import spaces
from segmentation_model import segment_guidewire, find_guidewire_tip, detect_dots, generate_goal_points

# Actuation limits
INSERTION_ABS_MAX_CM  = 40
INSERTION_ABS_MIN_CM  = 0
ROTATION_ABS_MAX_RAD  =  np.inf
ROTATION_ABS_MIN_RAD  = -np.inf

MOTION_TIMEOUT  = 45.0   # s — must be > driver's 30 s timeout
IMAGE_TIMEOUT   = 10.0   # s — wait for camera save response
IMAGE_RETRIES   = 3      # retry image capture on failure
IDLE_TIMEOUT    = 40.0   # s — max wait for driver to become idle


class CatheterRosBridge(Node):
    def __init__(self, images_dir: str = 'tmp/catheter_images', jsonl_path: str = 'tmp/catheter_log.jsonl'):
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
        self.cmd_pub  = self.create_publisher(String, 'catheter/command', 10)
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

    def send_command(self, ins_rel: float, rot_rel: float, relative: bool = True):
        """Publish a relative move command and clear the done flag."""
        self._done_event.clear()
        msg = String()
        msg.data = json.dumps({'insertion': ins_rel, 'rotation': rot_rel, 'relative': relative})
        self.cmd_pub.publish(msg)
        # print(msg)

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

    def get_catheter_total_insertion_rotation(self) -> tuple[float, float]:
        """Return the current total insertion (cm) and rotation (rad) from the latest state."""
        if self._latest_state is None:
            return 0.0, 0.0
        return (
            self._latest_state.get('insertion_cm', 0.0),
            self._latest_state.get('rotation_rad', 0.0)
        )

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
def load_image(path: str, w: int, h: int) -> np.ndarray:
    """Load a raw BGR8 .bin file and return RGB uint8 array."""
    time.sleep(1)
    raw = np.fromfile(path, dtype=np.uint8)
    expected = h * w * 3
    if raw.size != expected:
        return np.zeros((h, w, 3), dtype=np.uint8)
    bgr = raw.reshape((h, w, 3))
    return bgr[:, :, ::-1]  # BGR → RGB


class CatheterEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(
        self,
        max_steps_per_goal: int = 30,
        goal_tolerance_px: float = 35.0,
        crop_size: int = 128,
        goal_distance_mult: float = 1.0,
        max_spline_points: int = 2,
        images_dir: str = 'tmp/catheter_images',
        jsonl_path: str = 'tmp/catheter_log.jsonl',
        precomputed_goals: list | None = None,
    ):
        super().__init__()
        self._bridge             = CatheterRosBridge(images_dir=images_dir, jsonl_path=jsonl_path)
        self._max_steps          = max_steps_per_goal
        self._goal_tolerance_px  = goal_tolerance_px
        self._crop_size          = crop_size
        self._goal_distance_mult = goal_distance_mult
        self.max_spline_points = max_spline_points
        self.precomputed_goals = precomputed_goals

        self.last_reward = None

        # Action space: [insertion_rel_cm, rotation_rel_rad]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -math.pi / 4], dtype=np.float32),
            high=np.array([1.0,  math.pi / 4], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: grayscale crop centerd on tip + normalised pixel coords
        self.observation_space = spaces.Dict({
            'image':   spaces.Box(low=0, high=255, shape=(crop_size, crop_size, 1), dtype=np.uint8),
            'tip_xy':  spaces.Box(low=0.0, high=0.0, shape=(2,), dtype=np.float32),
            'goal_xy': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'spline_points': spaces.Box(low=-1.0, high=1.0, shape=(self.max_spline_points, 2), dtype=np.float32),
            'current_insertion_cm': spaces.Box(low=INSERTION_ABS_MIN_CM, high=INSERTION_ABS_MAX_CM, shape=(), dtype=np.float32),
            'current_rotation_rad': spaces.Box(low=ROTATION_ABS_MIN_RAD, high=ROTATION_ABS_MAX_RAD, shape=(), dtype=np.float32),
        })

        self.current_step: int           = 0
        self.current_goal: np.ndarray | None = None   # (x, y) in full-image pixels
        self.current_tip:  np.ndarray | None = None   # (x, y) in full-image pixels
        self.current_image: np.ndarray | None = None  # last raw BGR frame
        self._spline_points: list | None = None       # trailing spline from last capture

        # cv2.namedWindow("gym environment", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("gym environment", 500, 500)

    # ------------------------------------------------------------------ #
    #  Gymnasium API                                                       #
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None) -> tuple[dict, dict]:
        """
        Reset the environment.

        options (optional dict) keys
        ----------------------------
        goal_xy : (x, y) pixel coords to use as the goal.
                  If absent, a point is sampled from a forward cone.
        """
        super().reset(seed=seed)
        self.current_step = 0

        self._bridge.wait_driver_idle()

        # self._bridge.send_command(-1.0, 0.0, relative=False)
        self.visual_home_guidewire()
        # reached = self._bridge.spin_until(self._bridge.wait_done, MOTION_TIMEOUT)
        # if not reached:
        #     self._bridge.get_logger().warn('step(): motion timed out.')


        time.sleep(0.1)

        obs, tip_xy, full_img, spline_pts = self._capture_obs()

        time.sleep(0.1)

        if obs is None:
            raise RuntimeError('reset(): failed to capture initial image.')

        self.current_image  = full_img
        self.current_tip    = tip_xy
        self._spline_points = spline_pts

        if self.precomputed_goals is not None and len(self.precomputed_goals) > 0:
            list_idx = self.np_random.integers(0, len(self.precomputed_goals))
            chosen_list = self.precomputed_goals[list_idx]

            self.goal_list = [np.array(g, dtype=np.float32) for g in chosen_list]
            self.current_goal_idx = 0
            self.current_goal = self.goal_list[self.current_goal_idx]
        elif options and 'goal_xy' in options:
            self.current_goal = np.array(options['goal_xy'], dtype=np.float32)
        else:
            self.current_goal = self._sample_goal(full_img, tip_xy, spline_pts)
        
        img_h, img_w = full_img.shape[:2]
        
        obs['goal_xy'] = (self.current_goal - tip_xy) / np.array([img_w, img_h], dtype=np.float32)

        obs['current_insertion_cm'], obs['current_rotation_rad'] = self._bridge.get_catheter_total_insertion_rotation()

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

        time.sleep(0.2)
        if not reached:
            self._bridge.get_logger().warn('step(): motion timed out.')

        obs, tip_xy, full_img, spline_pts = self._capture_obs()
        if obs is None:
            return self._null_obs(), -1.0, False, True, {'error': 'image_capture_failed'}

        prev_tip = self.current_tip.copy() if self.current_tip is not None else tip_xy
        self.current_image  = full_img
        self.current_tip    = tip_xy
        self._spline_points = spline_pts

        reward, dist = self._compute_reward(prev_tip, tip_xy, self.current_goal, action)

        reached_current_goal = dist < self._goal_tolerance_px
        terminated = False
        if reached_current_goal:
            self.current_goal_idx += 1
            if self.current_goal_idx < len(self.goal_list):
                self.current_goal = self.goal_list[self.current_goal_idx]
                reward += 10.0
                self._bridge.get_logger().info(f"Goal reached! Moving to next goal ({self.current_goal_idx}/{len(self.goal_list)-1}).")
            else:
                terminated = True
                reward += 20.0
                self._bridge.get_logger().info("All goals reached! Episode terminated.")

        truncated = (not terminated) and (self.current_step >= self._max_steps)

        obs['goal_xy'] = (self.current_goal - tip_xy) / np.array([full_img.shape[1], full_img.shape[0]], dtype=np.float32)

        obs['current_insertion_cm'], obs['current_rotation_rad'] = self._bridge.get_catheter_total_insertion_rotation()

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

    def render(self, headless=False) -> np.ndarray | None:
        """Return an annotated BGR frame (goal = green cross, tip = red circle)."""
        if self.current_image is None:
            return None
        frame = self.current_image.copy()
        if self.current_goal is not None:
            cx, cy = int(self.current_goal[0]), int(self.current_goal[1])
            if self._goal_tolerance_px is not None:
                cv2.circle(frame, (cx, cy), int(self._goal_tolerance_px), (0, 255, 0), 2)
            else:
                cv2.drawMarker(frame, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        if self.current_tip is not None:
            tx, ty = int(self.current_tip[0]), int(self.current_tip[1])
            cv2.circle(frame, (tx, ty), 6, (0, 0, 255), -1)
        
        if self._spline_points is not None:
            for pt in self._spline_points:
                pt_img = [pt[0], pt[1]]
                cv2.circle(frame, (int(pt_img[0]), int(pt_img[1])), 4, (255, 0, 0), -1)
        
        if self.last_reward is not None:
            cv2.putText(frame, f"Reward: {self.last_reward}", (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 6)

        if not headless:
            return frame
            cv2.imshow("gym environment", frame)
            cv2.waitKey(1)
        else:
            cv2.imwrite(f"tmp/render_{self.current_step:04d}.png", frame)
            # cv2.waitKey(0)
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

        full_img = load_image(img_path, _w, _h)
        if full_img is None:
            return None, None, None, None
        
        # segment_guidewire expects a file path and returns a boolean skeleton
        skeleton = segment_guidewire(full_img)
        if skeleton is None:
            return None, None, None, None

        tip_result = find_guidewire_tip(skeleton)
        if tip_result is None:
            return None, None, None, None

        tip_position, spline_points = tip_result
        tip_xy = np.array(tip_position, dtype=np.float32)

        if np.isnan(tip_xy).any():
            self._bridge.get_logger().warn('Vision pipeline returned NaN for tip. Treating as capture failure.')
            return None, None, None, None

        img_h, img_w = full_img.shape[:2]
        norm = self._crop_size

        processed_spline = np.zeros((self.max_spline_points, 2), dtype=np.float32)
        if spline_points:
            # Subtract absolute tip_xy to make it relative, then normalize
            sp_arr = (np.array(spline_points, dtype=np.float32) - tip_xy) / np.array([img_w, img_h], dtype=np.float32)
            n_pts = min(len(sp_arr), self.max_spline_points)
            processed_spline[:n_pts] = sp_arr[:n_pts]

        # --- Process Relative Goal ---
        if self.current_goal is not None:
            rel_goal = (self.current_goal - tip_xy) / np.array([img_w, img_h], dtype=np.float32)
        else:
            rel_goal = np.zeros(2, dtype=np.float32)

        obs = {
            'image':   self._crop_around(full_img, tip_xy),
            'tip_xy':  np.zeros(2, dtype=np.float32), # Hardcoded to origin
            'goal_xy': rel_goal,
            'spline_points': processed_spline,
        }

        return obs, tip_xy, full_img, spline_points

    def _crop_around(self, img: np.ndarray, center: np.ndarray) -> np.ndarray:
        """Return a (crop_size × crop_size × 1) uint8 grayscale patch centerd on *center*."""
        h, w = img.shape[:2]
        half = self._crop_size // 2
        if center is None or np.isnan(center).any() or len(center) != 2:
            return np.zeros((self._crop_size, self._crop_size, 1), dtype=np.uint8)
        cx, cy = int(center[0]), int(center[1])
        if cx is None or cy is None:
            return np.zeros((self._crop_size, self._crop_size, 1), dtype=np.uint8)

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
        action: np.ndarray
    ) -> tuple[float, float]:
        """
        Dense shaped reward = improvement in distance to goal.
        +1 success bonus, −0.01 living penalty.

        Returns (reward, current_distance_px).
        """
        prev_dist = float(np.linalg.norm(prev_tip - goal))
        curr_dist = float(np.linalg.norm(curr_tip - goal))

        if np.linalg.norm(curr_tip - prev_tip) > 300:
            print("Camera Blip")
            return 0, curr_dist

        # print(f"prev_dist: {prev_dist:.2f}, curr_dist: {curr_dist:.2f}, action: {action}")

        img_h, img_w = self.current_image.shape[:2]
        sigma = math.sqrt(img_h**2 + img_w**2) * 0.25  # quarter of image diagonal

        prev_well = math.exp(-(prev_dist**2) / (2 * sigma**2))
        curr_well = math.exp(-(curr_dist**2) / (2 * sigma**2))
        gravity_delta = curr_well - prev_well  # range roughly (-1, 1)
        gravity_delta *= 10

        success_bonus = 5.0 if curr_dist < self._goal_tolerance_px else 0.0
        living_penalty = -0.01
        living_penalty = 0.0

        # Drop the action penalty or make it tiny
        reward = gravity_delta + success_bonus + living_penalty
        self._last_reward = reward

        print(f"Reward: {reward:.3f} (gravity {gravity_delta:.3f}, success {success_bonus}, living {living_penalty})")
        return reward, curr_dist

    def _sample_goal(
        self,
        img: np.ndarray,
        tip_xy: np.ndarray,
        spline_points: list | None = None,
    ) -> np.ndarray:
        """
        Pick a goal near the tip using the local wire tangent.

        Falls back to a random workspace pixel if no valid cone sample
        is available.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        obstacle_mask = detect_dots(gray)

        # grow dots for clearance
        clearance_px = max(4, int(self._goal_tolerance_px // 2))
        kernel_size = 2 * clearance_px + 1
        dilated_obstacles = cv2.dilate(
            obstacle_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)),
        )

        # bound goals to the dotted workspace
        ys, xs = np.where(obstacle_mask > 0)
        edge_margin = max(8, int(self._goal_tolerance_px))
        if xs.size > 0 and ys.size > 0:
            wx_min = max(int(xs.min()) + edge_margin, edge_margin)
            wx_max = min(int(xs.max()) - edge_margin, w - edge_margin - 1)
            wy_min = max(int(ys.min()) + edge_margin, edge_margin)
            wy_max = min(int(ys.max()) - edge_margin, h - edge_margin - 1)
        else:
            wx_min, wy_min = edge_margin, edge_margin
            wx_max, wy_max = w - edge_margin - 1, h - edge_margin - 1

        if wx_max <= wx_min or wy_max <= wy_min:
            wx_min, wy_min = edge_margin, edge_margin
            wx_max, wy_max = w - edge_margin - 1, h - edge_margin - 1

        # use spline only for heading
        forward = None
        if spline_points:
            ref = np.mean(
                np.asarray(spline_points[: min(3, len(spline_points))], dtype=np.float32),
                axis=0,
            )
            direction = np.asarray(tip_xy, dtype=np.float32) - ref
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-3:
                forward = direction / direction_norm

        if forward is None:
            angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            forward = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

        # sample in forward or retract cones
        d_min = max(
            self._goal_tolerance_px * 2.0,
            (self._crop_size / 4.0) * self._goal_distance_mult,
        )
        d_max = max(
            d_min + 10.0,
            (self._crop_size / 2.0) * self._goal_distance_mult,
        )
        theta_max = math.radians(60.0)
        retract_goal_prob = 0.25                                                       # tweak later

        # choose forward or retract mode
        if float(self.np_random.uniform(0.0, 1.0)) < retract_goal_prob:
            cone_axis = -forward
        else:
            cone_axis = forward

        for _ in range(200):
            distance = float(self.np_random.uniform(d_min, d_max))
            theta = float(self.np_random.uniform(-theta_max, theta_max))
            cos_theta, sin_theta = math.cos(theta), math.sin(theta)
            dir_vec = np.array(
                [
                    cone_axis[0] * cos_theta - cone_axis[1] * sin_theta,
                    cone_axis[0] * sin_theta + cone_axis[1] * cos_theta,
                ],
                dtype=np.float32,
            )
            candidate = np.asarray(tip_xy, dtype=np.float32) + distance * dir_vec
            gx = int(round(float(candidate[0])))
            gy = int(round(float(candidate[1])))

            if not (wx_min <= gx <= wx_max and wy_min <= gy <= wy_max):
                continue
            if dilated_obstacles[gy, gx] != 0:
                continue
            return candidate

        # fallback inside the workspace
        for _ in range(200):
            gx = int(self.np_random.integers(wx_min, wx_max + 1))
            gy = int(self.np_random.integers(wy_min, wy_max + 1))
            if dilated_obstacles[gy, gx] != 0:
                continue
            candidate = np.array([gx, gy], dtype=np.float32)
            if np.linalg.norm(candidate - np.asarray(tip_xy, dtype=np.float32)) >= d_min:
                return candidate

        # last resort mirror across the image
        return np.array([w - 1 - float(tip_xy[0]), h - 1 - float(tip_xy[1])], dtype=np.float32)

    def _null_obs(self) -> dict:
            """Zero-filled observation returned on capture failure."""
            return {
                'image':   np.zeros((self._crop_size, self._crop_size, 1), dtype=np.uint8),
                'tip_xy':  np.zeros(2, dtype=np.float32),
                'goal_xy': np.zeros(2, dtype=np.float32),
                'spline_points': np.zeros((self.max_spline_points, 2), dtype=np.float32),
            }
    
    def visual_home_guidewire(self, home_x_threshold: float = 200.0):
        self._bridge.get_logger().info("Starting Visual Homing Sequence...")
        
        while True:
            obs, tip_xy, full_img, spline_pts = self._capture_obs()

            if obs is None:
                self._bridge.get_logger().warn("Failed to capture image during homing.")
                break
            
            if tip_xy is None or tip_xy[0] <= home_x_threshold:
                self._bridge.get_logger().info("Catheter has reached physical home!")
                break
                
            # 3. If not home, pull back by 1.0 cm and check again
            self._bridge.send_command(-1.0, 0.0)
            self._bridge.spin_until(self._bridge.wait_done, MOTION_TIMEOUT)
            
            # Allow the camera buffer to clear motion blur
            time.sleep(0.25)
            
        # 4. We are physically at 0.0. Tell the driver to reset its math.
        msg = String()
        msg.data = json.dumps({'set_home': True})
        self._bridge.cmd_pub.publish(msg)
        
        # Give the driver a tiny moment to process the state change
        time.sleep(0.1)

def visualize_obs(obs):
    frame = cv2.cvtColor(obs['image'], cv2.COLOR_GRAY2BGR)
    tip = obs['tip_xy']
    goal = obs['goal_xy']
    spline_points = obs['spline_points']

    h = frame.shape[0]
    w = frame.shape[1]
    tip = [tip[0] * h + h // 2, tip[1] * w + w // 2]
    cv2.circle(frame, (int(tip[0]), int(tip[1])), 2, (0, 0, 255), -1)

    goal = [goal[0] * h + h // 2, goal[1] * w + w // 2]
    cv2.drawMarker(frame, (int(goal[0]), int(goal[1])), (0, 255, 0), cv2.MARKER_CROSS, 5, 2)

    for pt in spline_points:
        pt_img = [pt[0] * h + h // 2, pt[1] * w + w // 2]
        cv2.circle(frame, (int(pt_img[0]), int(pt_img[1])), 2, (255, 0, 0), -1)

    cv2.namedWindow("gym environment", cv2.WINDOW_NORMAL)
    # cv2.imshow("gym environment", frame)
    cv2.resizeWindow("gym environment", 500, 500)
    cv2.waitKey(0)

if __name__ == "__main__":
    rclpy.init()

    gymEnv = CatheterEnv()

    print("Waiting for ROS 2 discovery...")
    time.sleep(1)



    obs, info = gymEnv.reset()
    obs, reward, terminated, truncated, info = gymEnv.step(np.array([0.0, 0.0]))



    # visualize_obs(obs)

    # Render if needed
    gymEnv.render(headless=True)

    # obs, reward, terminated, trunacted, info = gymEnv.step(np.array([1.0, 0.0]))

    gymEnv.close()
    rclpy.shutdown()