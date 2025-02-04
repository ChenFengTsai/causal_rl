# import os
# import gymnasium as gym
# import numpy as np
# if not hasattr(np, 'bool'):
#     np.bool = bool
# from stable_baselines3 import PPO
# from gymnasium.wrappers import RecordVideo

# # Set environment variables for headless rendering
# os.environ["MUJOCO_GL"] = "egl"  # Try EGL first
# os.environ["PYOPENGL_PLATFORM"] = "egl"

# def record_episodes(policy, save_dir):
#     """
#     Record episodes and save as MP4 videos
#     """
#     print("Starting recording process...")
    
#     # Create output directory
#     os.makedirs(save_dir, exist_ok=True)
#     print(f"Created output directory: {save_dir}")
    
#     try:
#         # Initialize environment with video recording wrapper
#         print("Initializing environment...")
#         env = gym.make('HalfCheetah-v5', render_mode="rgb_array")
#         env = RecordVideo(
#             env, 
#             save_dir,
#             episode_trigger=lambda x: True,  # Record every episode
#             name_prefix="halfcheetah_episode"
#         )
        
#         # Record episodes
#         num_episodes = 5
#         print(f"\nRecording {num_episodes} episodes...")
        
#         for episode in range(num_episodes):
#             print(f"Recording episode {episode + 1}/{num_episodes}")
#             obs, _ = env.reset()
#             episode_reward = 0
            
#             for step in range(1000):  # HalfCheetah typically uses longer episodes
#                 action, _ = policy.predict(obs)
#                 obs, reward, terminated, truncated, info = env.step(action)
#                 done = terminated or truncated
#                 episode_reward += reward
                
#                 if done:
#                     break
            
#             print(f"Completed episode {episode + 1} with total reward: {episode_reward:.2f}")
                    
#         env.close()
#         print("\nFinished recording episodes")
#         print(f"\nVideo files saved in: {save_dir}")
#         print("\nDownload the MP4 files to your local machine using:")
#         print(f"scp -r your_username@your_server:{save_dir}/*.mp4 ./")
        
#         return save_dir
        
#     except Exception as e:
#         print(f"\nError during recording: {e}")
#         # If EGL fails, try software rendering
#         if "EGL" in str(e):
#             print("\nTrying software rendering instead...")
#             os.environ["MUJOCO_GL"] = "osmesa"
#             return record_episodes(policy, save_dir)
#         return None

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model', required=True, type=str)
#     args = parser.parse_args()
#     MODEL_PATH = args.model
#     # Get the directory containing the zip file
#     dir_path = MODEL_PATH.rsplit('/', 1)[0]
#     # Get the timestamp folder name
#     timestamp = dir_path.split('/')[-1]
#     VIDEO_DIR = os.path.expanduser(os.path.join("~/CRL/src/videos/HalfCheetah-v5", timestamp))

#     policy = PPO.load(MODEL_PATH, device='cpu')
#     record_episodes(policy, VIDEO_DIR)


import os
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

if not hasattr(np, 'bool'):
    np.bool = bool
import os
os.environ["MUJOCO_GL"] = "osmesa"


# Modified environment to include the dynamics model
class ModifiedEnv(gym.Env):
    def __init__(self, original_env, extra_feature_dim, dynamics_model):
        self.original_env = original_env
        self.extra_feature_dim = extra_feature_dim
        self.dynamics_model = dynamics_model
        
        # Observation space augmentation
        original_obs_space = original_env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.hstack((original_obs_space.low, [-np.inf] * extra_feature_dim)),
            high=np.hstack((original_obs_space.high, [np.inf] * extra_feature_dim)),
            dtype=np.float32
        )
        self.action_space = original_env.action_space
        self.render_mode = original_env.render_mode

    def reset(self, seed=None, options=None):
        obs, info = self.original_env.reset(seed=seed, options=options)
        dummy_action = np.zeros(self.action_space.shape[0])
        extra_feature = self._generate_extra_feature(obs, dummy_action).flatten()
        combined_obs = np.hstack((obs.flatten(), extra_feature))
        return combined_obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.original_env.step(action)
        extra_feature = self._generate_extra_feature(obs, action).flatten()
        combined_obs = np.hstack((obs.flatten(), extra_feature))
        return combined_obs, reward, done, truncated, info

    def render(self, *args, **kwargs):
        return self.original_env.render(*args, **kwargs)

    def _generate_extra_feature(self, obs, action):
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        next_state_pred = self.dynamics_model(state_tensor, action_tensor)
        return next_state_pred.detach().numpy().squeeze()



def record_episodes(policy, dynamics_model, save_dir):
    """
    Record episodes and save as MP4 videos
    """
    print("Starting recording process...")
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created output directory: {save_dir}")
    
    try:
        print("Initializing environment...")
        # When creating the original environment

        original_env = gym.make('HalfCheetah-v5', render_mode='rgb_array')
        env = ModifiedEnv(original_env, extra_feature_dim=17, dynamics_model=dynamics_model)
        env = RecordVideo(
            env, 
            save_dir,
            episode_trigger=lambda x: True,
            name_prefix="halfcheetah_episode"
        )
        
        num_episodes = 5
        print(f"\nRecording {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            print(f"Recording episode {episode + 1}/{num_episodes}")
            obs, _ = env.reset()
            episode_reward = 0
            
            for step in range(1000):  # HalfCheetah typically uses longer episodes
                action, _ = policy.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
                if done:
                    break
            
            print(f"Completed episode {episode + 1} with total reward: {episode_reward:.2f}")
                    
        env.close()
        print("\nFinished recording episodes")
        print(f"\nVideo files saved in: {save_dir}")
        
        return save_dir
        
    except Exception as e:
        print(f"\nError during recording: {e}")
        if "EGL" in str(e):
            print("\nTrying software rendering instead...")
            os.environ["MUJOCO_GL"] = "osmesa"
            return record_episodes(policy, dynamics_model, save_dir)
        return None
    
# Define the model class
class DynamicsModel(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super(DynamicsModel, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, state_dim)  # Predict next state
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)



if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--policy_model', required=True, type=str, help="Path to the PPO policy model")
    # parser.add_argument('--dynamics_model', required=True, type=str, help="Path to the dynamics model")
    # args = parser.parse_args()
    
    # Load the PPO policy
    policy_model_pth = '/home/richtsai1103/CRL/src/results/HalfCheetah-v5/dyna_ppo_20250122_020609/PPO_HalfCheetah-v5_final.zip'
    dynamics_model_pth = '/home/richtsai1103/CRL/src/results/HalfCheetah-v5/dyna_ppo_20250122_020609/Dyna_HalfCheetah-v5_final.pth'
    policy = PPO.load(policy_model_pth, device='cpu')

    # Load the dynamics model
    # Recreate the model architecture
    state_dim = 17  # Example
    action_dim = 6   # Example
    dynamics_model = DynamicsModel(state_dim, action_dim)

    # Load the saved weights
    dynamics_model.load_state_dict(torch.load(dynamics_model_pth, weights_only=True))

    # Set to evaluation mode (important for inference)
    dynamics_model.eval()
    # dynamics_model = torch.load(dynamics_model_pth)
    # print(dynamics_model)
    # dynamics_model.eval()

    # Set up the video output directory
    dir_path = policy_model_pth.rsplit('/', 1)[0]
    timestamp = dir_path.split('/')[-1]
    VIDEO_DIR = os.path.expanduser(os.path.join("~/CRL/src/videos/HalfCheetah-v5", timestamp))

    record_episodes(policy, dynamics_model, VIDEO_DIR)
