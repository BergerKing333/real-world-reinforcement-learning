import rclpy
from catheter_env import CatheterEnv

rclpy.init()

env = CatheterEnv(max_steps=5)

try:
    obs, info = env.reset()
    print("Initial observation:", obs)

    action = env.action_space.sample()
    print("Sample action:", action)

    obs, reward, terminated, truncated, info = env.step(action)
    print("Next observation:", obs)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Info:", info)

finally:
    env.close()
    rclpy.shutdown()