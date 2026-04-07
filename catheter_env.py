import json
import time
import math
import threading
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

import gymnasium as gym
from gymnasium import spaces


class CatheterEnv(Node, gym.Env):
    def __init__(self, max_steps=50, goal_tolerance=0.2):
        Node.__init__(self, 'catheter_rl_env')
        gym.Env.__init__(self)

        self.max_steps = max_steps
        self.goal_tolerance = goal_tolerance
        self.current_step = 0

        self.latest_state = None
        self.done_event = threading.Event()

        self.goal_insertion = 3.0
        self.goal_rotation = 0.0

        self.action_space = spaces.Box(
            low=np.array([-0.1, -math.pi / 6], dtype=np.float32),
            high=np.array([1.0,  math.pi / 6], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, -2 * math.pi, 0.0, -2 * math.pi], dtype=np.float32),
            high=np.array([15.0, 2 * math.pi, 15.0, 2 * math.pi], dtype=np.float32),
            dtype=np.float32
        )

        self.state_sub = self.create_subscription(
            String, 'catheter/state', self.on_state, 10
        )

        self.done_sub = self.create_subscription(
            Bool, 'catheter/position_reached', self.on_done, 10
        )

        self.cmd_pub = self.create_publisher(String, 'catheter/command', 10)

    def on_state(self, msg):
        try:
            self.latest_state = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def on_done(self, msg):
        if msg.data:
            self.done_event.set()

    def wait_for_state(self, timeout=10.0):
        start = time.time()
        while self.latest_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                raise TimeoutError("No state received from /catheter/state")

    def get_obs(self):
        insertion = float(self.latest_state.get('insertion_cm', 0.0))
        rotation = float(self.latest_state.get('rotation_rad', 0.0))

        return np.array([
            insertion,
            rotation,
            self.goal_insertion,
            self.goal_rotation
        ], dtype=np.float32)

    def send_command(self, insertion, rotation):
        cmd = String()
        cmd.data = json.dumps({
            'insertion': float(insertion),
            'rotation': float(rotation),
            'relative': True
        })
        self.done_event.clear()
        self.cmd_pub.publish(cmd)

    def wait_done(self, timeout=45.0):
        start = time.time()
        while not self.done_event.is_set():
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for /catheter/position_reached")

    def distance(self, ins, rot):
        return math.sqrt(
            (ins - self.goal_insertion) ** 2 +
            (rot - self.goal_rotation) ** 2
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.wait_for_state()

        reset_cmd = String()
        reset_cmd.data = json.dumps({'reset': True})
        self.done_event.clear()
        self.cmd_pub.publish(reset_cmd)

        self.wait_done()

        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

        obs = self.get_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        obs_before = self.get_obs()
        old_dist = self.distance(obs_before[0], obs_before[1])

        self.send_command(action[0], action[1])
        self.wait_done()

        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

        obs = self.get_obs()
        new_dist = self.distance(obs[0], obs[1])

        reward = old_dist - new_dist

        terminated = new_dist < self.goal_tolerance
        truncated = self.current_step >= self.max_steps

        if terminated:
            reward += 10.0

        info = {
            "distance_to_goal": new_dist
        }

        return obs, reward, terminated, truncated, info

    def close(self):
        self.destroy_node()