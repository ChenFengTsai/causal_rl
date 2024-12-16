import gymnasium
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import torch
import time
import os
from metrics_logger import MetricsLogger
import pandas as pd



class RewardCallback(BaseCallback):
    def __init__(self, metrics_logger, save_every_n_steps=1000):
        super().__init__()
        self.metrics_logger = metrics_logger
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Step counter for deciding when to save
        self.step_counter = 0
        self.save_every_n_steps = save_every_n_steps
        
        # Keep track of the previous step's data
        self.prev_step_info = None
        
    def _on_step(self) -> bool:
        # Increment step counter
        self.step_counter += 1
        
        # Get current state, action, and reward
        try:
            current_state = self.locals['new_obs'][0]  # Use 'new_obs' for observations
        except KeyError:
            current_state = self.locals['obs'][0]  # Fallback to 'obs' if 'new_obs' doesn't exist
            
        current_action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        # Create the current step's information
        current_step_info = {
            'global_step': self.step_counter,
            'episode': len(self.episode_rewards),
            'step': self.current_episode_length,
            'current_state': current_state.tolist(),
            'current_action': current_action.tolist(),
            'reward': reward,
            'done': done
        }
        
        # Save the current step along with the previous step every `save_every_n_steps`
        if self.step_counter % self.save_every_n_steps == 0 and self.prev_step_info is not None:
            row_data = {
                'global_step': self.step_counter,
                'episode': len(self.episode_rewards),
                'current_state': current_step_info['current_state'],
                'current_action': current_step_info['current_action'],
                'current_reward': current_step_info['reward'],
                'done': current_step_info['done'],
                'prev_state': self.prev_step_info['current_state'],
                'prev_action': self.prev_step_info['current_action'],
                'prev_reward': self.prev_step_info['reward']
            }
            self._save_step_data(row_data)
        
        # Update the previous step information
        self.prev_step_info = current_step_info
        
        # Update episode tracking
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Handle episode end
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Original metrics logging (every 5 episodes)
            if len(self.episode_rewards) >= 5:
                approx_kl = self.locals.get('approx_kl_divergence', 0)
                clip_frac = self.locals.get('clip_fraction', 0)
                value_loss = self.locals.get('value_loss', 0)
                policy_loss = self.locals.get('policy_loss', 0)
                
                logger = {
                    'EpRet': self.episode_rewards,
                    'EpLen': self.episode_lengths,
                    'VVals': [0],
                    'LossPi': [policy_loss],
                    'LossV': [value_loss],
                    'KL': [approx_kl],
                    'ClipFrac': [clip_frac],
                    'StopIter': [0]
                }
                
                self.metrics_logger.log_epoch(
                    logger,
                    self.n_calls // self.training_env.num_envs,
                    self.n_calls,
                    time.time() - self.training_env.start_time
                )
                
                self.episode_rewards = []
                self.episode_lengths = []
        
        return True
    
    def _save_step_data(self, row_data):
        """Save the current and previous step data to a CSV file"""
        # Convert dictionary to DataFrame
        df = pd.DataFrame([row_data])
        
        # Create filename
        filename = "selected_steps.csv"
        filepath = os.path.join(self.metrics_logger.save_dir, filename)
        
        # Append to CSV file
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)





def train(
        env_name,
        seed=0,
        steps_per_epoch=4096,
        epochs=50,
        gamma=0.99,
        clip_ratio=0.2,
        pi_lr=2.5e-4,
        hidden_sizes=[1024, 1024],
        max_ep_len=1000,
        save_freq=10,
        exp_name='ppo'
        ):
    """
    Train a PPO agent using Stable Baselines 3.
    
    Args:
        env_name (str): Gymnasium environment name
        seed (int): Random seed
        steps_per_epoch (int): Number of steps per epoch
        epochs (int): Number of epochs
        gamma (float): Discount factor
        clip_ratio (float): PPO clip ratio
        pi_lr (float): Learning rate
        hidden_sizes (list): Sizes of hidden layers
        max_ep_len (int): Maximum episode length
        save_freq (int): How often to save the model
        exp_name (str): Experiment name for logging
    """
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env = gymnasium.make(env_name)
    env = DummyVecEnv([lambda: env])  # SB3 requires vectorized environments
    env.start_time = time.time()  # Add start time to env for logging
    
    # Initialize metrics logger
    metrics_logger = MetricsLogger(exp_name, env_name)
    
    # Create eval environment for callbacks
    eval_env = gymnasium.make(env_name)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=metrics_logger.save_dir,
        log_path=metrics_logger.save_dir,
        eval_freq=save_freq * steps_per_epoch,
        deterministic=True,
        render=False
    )
    
    reward_callback = RewardCallback(metrics_logger)

    # Create PPO model
    policy_kwargs = dict(
        net_arch=dict(pi=hidden_sizes, vf=hidden_sizes)
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=pi_lr,
        n_steps=steps_per_epoch,
        batch_size=64,
        n_epochs=10,
        gamma=gamma,
        clip_range=clip_ratio,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(metrics_logger.save_dir, "tensorboard"),
        seed=seed,
        verbose=1
    )

    # Train the agent
    total_timesteps = steps_per_epoch * epochs
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, reward_callback],
        progress_bar=True,
        log_interval=1 
    )

    # Save the final model
    model.save(os.path.join(metrics_logger.save_dir, f"PPO_{env_name}_final"))
    
    # Save final metrics
    metrics_logger.save_metrics()
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(
        model, 
        eval_env, 
        n_eval_episodes=10,
        deterministic=True
    )
    
    print(f"\nTraining completed in {time.time() - env.start_time:.2f}s")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    eval_env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--hid', type=int, default=512)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    train(
        args.env,
        hidden_sizes=[args.hid]*args.l,
        exp_name=args.exp_name,
        gamma=args.gamma, 
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs
    )