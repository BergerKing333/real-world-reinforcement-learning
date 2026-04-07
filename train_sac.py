import rclpy
from stable_baselines3 import SAC
from catheter_env import CatheterEnv


def main():
    rclpy.init()
    env = CatheterEnv(max_steps=20)

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1
    )

    model.learn(total_timesteps=1000)
    model.save("sac_catheter_model")

    env.close()
    rclpy.shutdown()


if __name__ == "__main__":
    main()