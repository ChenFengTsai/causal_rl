import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import torch
import os
from IPython.display import HTML
from IPython import display
import glob
from ppo_halfcheetah import *

def visualize_agent(env_name, model_path, num_episodes=3, save_video=True, render_mode='rgb_array'):
    """
    Visualize a trained agent interacting with the environment.
    
    Args:
        env_name (str): Name of the Gymnasium environment
        model_path (str): Path to the saved model weights
        num_episodes (int): Number of episodes to run
        save_video (bool): Whether to save video recordings
        render_mode (str): Rendering mode ('human' or 'rgb_array')
    """
    # Create directories for videos if saving
    if save_video:
        video_dir = f"videos/{env_name}"
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            gym.make(env_name, render_mode='rgb_array'),
            video_dir,
            episode_trigger=lambda x: True  # Record every episode
        )
    else:
        env = gym.make(env_name, render_mode=render_mode)
    
    # Load model
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        act_dim = env.action_space.shape[0]
    elif isinstance(env.action_space, gym.spaces.Discrete):
        act_dim = env.action_space.n
    
    # Create actor-critic model with same architecture as training
    actor_critic = Actor_Critic(
        obs_dim, 
        act_dim,
        hidden_sizes=[64, 64],
        action_space=env.action_space
    )
    
    # Load saved weights
    actor_critic.load_state_dict(torch.load(model_path))
    actor_critic.eval()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                action, _, _ = actor_critic(obs_tensor)
            
            # Take step in environment
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if render_mode == 'human':
                env.render()
        
        print(f'Episode {episode + 1}: Total Reward = {total_reward}, Steps = {steps}')
    
    env.close()
    
    # If videos were saved, display the last one
    if save_video:
        # Get the latest video file
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        if video_files:
            latest_video = max(video_files, key=os.path.getmtime)
            # Display video in notebook if in IPython environment
            try:
                display.clear_output()
                display.display(HTML(f'<video controls><source src="{latest_video}" type="video/mp4"></video>'))
            except:
                print(f"Video saved to: {latest_video}")


    
# After training your PPO agent
if __name__ == '__main__':
    env_name = "CartPole-v1"  # or your environment name
    model_path = "../results/CartPole-v1/ppo_20241204_231532/PPO_CartPole-v1_best.pth"  # path to your saved model
    
    # # To visualize in a window
    # visualize_agent(env_name, model_path, num_episodes=3, render_mode='human')
    
    # To save videos
    visualize_agent(env_name, model_path, num_episodes=3, save_video=True)
    
