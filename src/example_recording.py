import os
import numpy as np
import time
import torch
if not hasattr(np, 'bool'):
    np.bool = bool
from stable_baselines3 import PPO
from causal_world.envs.causalworld import CausalWorld
from causal_world.task_generators.task import generate_task
from causal_world.loggers.data_recorder import DataRecorder

from causal_world.loggers.data_loader import DataLoader
import causal_world.viewers.task_viewer as viewer

def record_policy_episodes(policy, video_dir, device='auto'):
    """
    Record episodes and save video in a specified directory.
    
    Args:
        policy: Trained policy
        video_dir: Directory where video will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(video_dir, exist_ok=True)
    
    # Create a subdirectory for the episode data
    episodes_dir = os.path.join(video_dir, 'episodes_data')
    data_recorder = DataRecorder(output_directory=episodes_dir,
                               rec_dumb_frequency=11)

    # Initialize environment
    task = generate_task(task_generator_id='pushing')
    env = CausalWorld(task=task,
                     enable_visualization=True,
                     data_recorder=data_recorder)

    timestamp = time.time()
    obs = env.reset()
    video_path = os.path.join(video_dir, f"pushing_video_{timestamp}.mp4")
    env.start_recording(video_path)  # Start recording
    
    for step in range(10):
        action, _ = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    
    env.stop_recording()  # Stop recording
    env.close()
    
    return video_path
    
    # # Record episodes
    # for episode in range(5):
    #     obs = env.reset()
    #     for step in range(50):
    #         # Convert observation to tensor on the correct device if necessary
    #         if isinstance(obs, np.ndarray):
    #             obs = torch.FloatTensor(obs).to(device)
            
    #         action, _ = policy.predict(obs, deterministic=True)
    #         obs, reward, done, info = env.step(action)
    #         if done:
    #             break
                
    # env.close()

    # # Load recorded episode and create video
    # data = DataLoader(episode_directory=episodes_dir)
    # episode = data.get_episode(1)  # Get first episode
    
    # # Save video with timestamp to avoid overwriting
    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # video_path = os.path.join(video_dir, f"pushing_video_{timestamp}.mp4")
    
    # viewer.record_video_of_episode(episode=episode, file_name=video_path)
    # print(f"\nVideo saved at: {video_path}")
    # return video_path

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
    # Set video directory in your home folder
    VIDEO_DIR = os.path.expanduser("~/CRL/pushing_videos")
    
    # Load policy
    policy = PPO.load("./pushing/logs/training_logs/models/PPO_001/saved_model_6.zip", device=device)
    
    # Record and save video
    video_path = record_policy_episodes(policy, VIDEO_DIR, device=device)
    
    # Print instructions for downloading
    print("\nTo download the video to your local machine, open a new terminal and run:")
    print(f"scp your_username@your_server:{video_path} ./")


