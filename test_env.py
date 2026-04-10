import rclpy
from catheter_env import CatheterEnv

rclpy.init()

env = CatheterEnv(
    max_steps=5,
    goal_tolerance_px=20.0,
    use_mock_cv=True,
)

try:
    obs, info = env.reset()
    print("Initial observation:", obs)
    print("Initial tip_xy:", info["tip_xy"])
    print("Goal_xy:", info["goal_xy"])

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