import os

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from src.catheter_env2 import CatheterEnv


def make_env():
    """
    Create one environment instance and wrap it with Monitor
    so episode rewards and lengths are logged automatically.
    """
    env = CatheterEnv()
    env = Monitor(env)
    return env


def main():
    # Start ROS before creating the environment
    rclpy.init()

    # Make sure output folders exist before training starts
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_env()
    eval_env = make_env()

    try:
        # Use CnnPolicy because the observation is now an image crop
        model = SAC(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
        )

        # Run evaluation every so often and save the best checkpoint
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./models/",
            log_path="./logs/",
            eval_freq=2000,
            deterministic=True,
            render=False,
        )

        # Train the model
        model.learn(
            total_timesteps=20000,
            callback=eval_callback,
        )

        # Save the final trained model
        model.save("models/sac_catheter_env2")

    finally:
        # Close everything cleanly even if training crashes
        env.close()
        eval_env.close()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()