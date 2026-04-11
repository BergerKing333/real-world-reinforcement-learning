import rclpy
from stable_baselines3 import SAC
from catheter_env import CatheterEnv


def main():
    rclpy.init()

    env = CatheterEnv(
        max_steps=5,
        goal_tolerance_px=20.0,
        use_mock_cv=True,   # False for real CV path testing
    )

    # changed from MlpPolicy - observation is now an image 
    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
    )

    model.learn(total_timesteps=1000)
    model.save("sac_catheter_model")

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()