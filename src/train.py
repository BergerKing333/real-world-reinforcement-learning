import os
import cv2
from pathlib import Path
import threading

import rclpy
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback

from gymnasium.wrappers import FilterObservation

from catheter_env2 import CatheterEnv

def save_frame(filename, frame):
    cv2.imwrite(filename, frame)

def make_env():
    """
    Create one environment instance and wrap it with Monitor
    so episode rewards and lengths are logged automatically.
    """
    env = CatheterEnv()
    env = FilterObservation(env, filter_keys=['tip_xy', 'goal_xy', 'spline_points'])
    env = Monitor(env)
    return env

# class RenderCallback(BaseCallback):
#     def __init__(self, render_freq: int, verbose=0):
#         super().__init__(verbose)
#         self.render_freq=render_freq

#     def _on_step(self) -> bool:
#         if self.n_calls % self.render_freq == 0:
#             self.training_env.envs[0].render()
#         else:
#             cv2.waitKey(1)
#         return True

def delete_bin_files(directory_path: str):
    folder = Path(directory_path)

    if not folder.exists() or not folder.is_dir():
        return
    
    count = 0

    for file_path in folder.glob('*.bin'):
        try:
            file_path.unlink()
            count += 1
        except Exception as e:
            print(f"failed to delete {file_path}: {e}")

        print(f"deleted {count} files")

class RecordEpisodeCallback(BaseCallback):
    def __init__(self, record_freq_episodes: int, save_dir: str='./recorded_episodes', verbose=0):
        super().__init__(verbose)
        self.record_freq = record_freq_episodes
        self.save_dir = save_dir
        
        self.episodes_completed = 0
        self.is_recording = True  # We start by recording the very first episode (Episode 0)
        self.step_in_episode = 0
        
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.is_recording:
            env = self.training_env.envs[0]

            frame = env.render()

            if frame is not None:
                ep_folder = os.path.join(self.save_dir, f"episode_{self.episodes_completed:04d}")
                os.makedirs(ep_folder, exist_ok=True)
                
                # Save the image (e.g., "step_000.png", "step_001.png")
                filename = os.path.join(ep_folder, f"step_{self.step_in_episode:03d}.png")
                # cv2.imwrite(filename, frame)
                threading.Thread(target=save_frame, args=(filename, frame)).start()
                
            self.step_in_episode += 1
        else:
            cv2.waitKey(1)

        done = self.locals.get('dones', [False])[0]
        
        if done:
            self.episodes_completed += 1
            self.step_in_episode = 0 # Reset step counter for the next episode
            
            # Decide if the next episode is a recording target
            if self.episodes_completed % self.record_freq == 0:
                self.is_recording = True
                print(f"\n[Callback] Starting to record Episode {self.episodes_completed}...")
                delete_bin_files('tmp/catheter_images')
            else:
                self.is_recording = False
                
        return True

def main():
    # Start ROS before creating the environment
    rclpy.init()

    # Make sure output folders exist before training starts
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = make_env()

    try:
        # Use MultiInputPolicy because the observation is a Dict 
        # (contains both 'image' and 1D arrays like 'goal_xy', 'spline_points')
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.95,
            tau=0.005,
            train_freq=1,
            gradient_steps=4,
            tensorboard_log="./logs/" # Added so you can still track progress
        )

        # Since we removed eval_env to prevent ROS topic collisions, 
        # use CheckpointCallback to save the model periodically during training.
        checkpoint_callback = CheckpointCallback(
            save_freq=2000,
            save_path="./models/",
            name_prefix="sac_catheter"
        )

        record_callback = RecordEpisodeCallback(record_freq_episodes=10, save_dir="./training_videos")

        callbacks = [checkpoint_callback, record_callback]

        # Train the model
        model.learn(
            total_timesteps=10000,
            callback=callbacks
        )

        # Save the final trained model
        model.save("models/sac_catheter_env2_final")

    finally:
        # Close everything cleanly even if training crashes
        env.close()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()