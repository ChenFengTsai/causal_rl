import os
import gymnasium as gym
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

# Set environment variables for headless rendering
os.environ["MUJOCO_GL"] = "egl"  # Try EGL first
os.environ["PYOPENGL_PLATFORM"] = "egl"

def record_episodes(policy, save_dir):
    """
    Record episodes and save as MP4 videos
    """
    print("Starting recording process...")
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    print(f"Created output directory: {save_dir}")
    
    try:
        # Initialize environment with video recording wrapper
        print("Initializing environment...")
        env = gym.make('HalfCheetah-v5', render_mode="rgb_array")
        env = RecordVideo(
            env, 
            save_dir,
            episode_trigger=lambda x: True,  # Record every episode
            name_prefix="halfcheetah_episode"
        )
        
        # Record episodes
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
        print("\nDownload the MP4 files to your local machine using:")
        print(f"scp -r your_username@your_server:{save_dir}/*.mp4 ./")
        
        return save_dir
        
    except Exception as e:
        print(f"\nError during recording: {e}")
        # If EGL fails, try software rendering
        if "EGL" in str(e):
            print("\nTrying software rendering instead...")
            os.environ["MUJOCO_GL"] = "osmesa"
            return record_episodes(policy, save_dir)
        return None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    args = parser.parse_args()
    MODEL_PATH = args.model
    # Get the directory containing the zip file
    dir_path = MODEL_PATH.rsplit('/', 1)[0]
    # Get the timestamp folder name
    timestamp = dir_path.split('/')[-1]
    VIDEO_DIR = os.path.expanduser(os.path.join("~/CRL/src/videos/HalfCheetah-v5", timestamp))

    policy = PPO.load(MODEL_PATH, device='cpu')
    record_episodes(policy, VIDEO_DIR)