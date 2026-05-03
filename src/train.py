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

# the 8 paths for gym 1
precomputed_goals = [
    [(269, 1019), (383, 1015), (466, 1004), (549, 973), (623, 934), (691, 907), (770, 875), (842, 839), (933, 801), (1024, 739), (1081, 682), (1122, 638), (1170, 591), (1226, 536), (1287, 466), (1340, 413), (1376, 373), (1395, 328), (1412, 292), (1431, 243), (1463, 186), (1484, 137), (1491, 108), (1506, 63)],
    [(301, 1032), (403, 1015), (477, 1017), (540, 1049), (617, 1083), (683, 1113), (748, 1140), (801, 1161), (873, 1178), (924, 1197), (986, 1221), (1050, 1214), (1117, 1212), (1164, 1221), (1268, 1214), (1353, 1216), (1431, 1216), (1491, 1212), (1563, 1208), (1629, 1180), (1692, 1151), (1752, 1125), (1807, 1102), (1851, 1081), (1890, 1064)],
    [(273, 1026), (379, 1015), (470, 1004), (543, 977), (627, 928), (712, 903), (791, 867), (848, 841), (931, 809), (1011, 752), (1058, 697), (1094, 659), (1134, 619), (1187, 566), (1249, 504), (1294, 457), (1332, 419), (1376, 377), (1431, 362), (1482, 341), (1529, 318), (1576, 298), (1637, 273), (1701, 248)], 
    [(284, 1011), (364, 1028), (428, 1019), (502, 1015), (528, 1051), (576, 1074), (642, 1098), (702, 1113), (761, 1136), (801, 1153), (850, 1174), (899, 1180), (960, 1208), (1016, 1225), (1067, 1223), (1141, 1221), (1207, 1223), (1289, 1221), (1357, 1219), (1412, 1223), (1440, 1221), (1497, 1210), (1550, 1208), (1599, 1227), (1644, 1242), (1684, 1261), (1739, 1280), (1788, 1295), (1832, 1312), (1894, 1333)],
    [(256, 1019), (350, 1007), (458, 1009), (566, 968), (636, 930), (725, 892), (791, 862), (871, 828), (952, 790), (1018, 786), (1079, 786), (1143, 786), (1217, 784), (1289, 784), (1357, 786), (1412, 778), (1465, 775), (1514, 773), (1561, 771), (1599, 754), (1644, 729), (1697, 712), (1754, 689), (1809, 667), (1851, 644), (1877, 638)],
    [(238, 1012), (335, 1004), (381, 1014), (447, 1021), (447, 1021), (517, 1037), (574, 1057), (629, 1078), (692, 1110), (759, 1131), (812, 1157), (876, 1182), (950, 1205), (950, 1205), (993, 1223), (1021, 1253), (1060, 1299), (1102, 1332), (1145, 1375), (1196, 1415), (1246, 1459), (1300, 1511), (1348, 1553), (1380, 1581), (1403, 1597), (1447, 1620), (1498, 1629), (1518, 1645), (1578, 1666), (1629, 1685), (1686, 1701), (1719, 1728)],
    [(250, 1030), (345, 1024), (458, 1011), (553, 981), (627, 937), (708, 892), (774, 856), (859, 839), (965, 801), (1026, 792), (1081, 786), (1145, 784), (1232, 788), (1291, 782), (1342, 784), (1393, 773), (1465, 778), (1520, 773), (1563, 773), (1593, 788), (1629, 807), (1669, 826), (1714, 841), (1769, 860), (1824, 884), (1883, 905)],
    [(292, 1024), (360, 1024), (449, 1026), (513, 1043), (572, 1074), (621, 1089), (661, 1104), (723, 1130), (784, 1149), (835, 1174), (893, 1197), (943, 1214), (999, 1238), (1047, 1282), (1073, 1310), (1111, 1337), (1145, 1375), (1181, 1407), (1228, 1450), (1289, 1494), (1327, 1530), (1364, 1566), (1397, 1607), (1419, 1649), (1442, 1693), (1465, 1740), (1482, 1780), (1504, 1819), (1520, 1855), (1535, 1888), (1552, 1918)]
]

def save_frame(filename, frame):
    cv2.imwrite(filename, frame)

def make_env():
    """
    Create one environment instance and wrap it with Monitor
    so episode rewards and lengths are logged automatically.
    """
    env = CatheterEnv(precomputed_goals=precomputed_goals)
    env = FilterObservation(env, filter_keys=['goal_xy', 'spline_points'])
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
            gradient_steps=2,
            tensorboard_log="./logs/"
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