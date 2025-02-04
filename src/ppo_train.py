import gymnasium
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.logger import configure
from torch.utils.tensorboard import SummaryWriter
from metrics_logger import MetricsLogger
from reward_callback import RewardCallback


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Callback to log rewards
# class RewardCallback(BaseCallback):
#     def __init__(self, metrics_logger, save_every_n_steps=1000):
#         super().__init__()
#         self.metrics_logger = metrics_logger
#         self.episode_rewards = []
#         self.current_episode_reward = 0
        
#     def _on_step(self) -> bool:
#         reward = self.locals['rewards'][0]
#         done = self.locals['dones'][0]
        
#         self.current_episode_reward += reward
#         if done:
#             self.episode_rewards.append(self.current_episode_reward)
#             self.current_episode_reward = 0
            
#             if len(self.episode_rewards) % 5 == 0:
#                 self.metrics_logger.log_metrics({
#                     'EpRet': np.mean(self.episode_rewards)
#                 }, self.num_timesteps)
#                 self.episode_rewards = []
        
#         return True

# PPO training function
def train_ppo(env_name, seed=0, epochs=20, save_freq=1, exp_name='ppo'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = gymnasium.make(env_name)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_directory = f"{exp_name}_{current_time}"
    metrics_logger = MetricsLogger(exp_directory, env_name)
    reward_callback = RewardCallback(metrics_logger)
    
    new_logger = configure(folder=metrics_logger.save_dir, format_strings=["stdout", "csv", "tensorboard"])
    
    ppo_policy = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=2.0633e-05,
        n_steps=512,
        n_epochs=20,
        gamma=0.98,
        gae_lambda=0.92,
        clip_range=0.1,
        ent_coef=0.000401762,
        vf_coef=0.58096,
        max_grad_norm=0.8,
        tensorboard_log=metrics_logger.tensorboard_log,
        seed=seed,
        policy_kwargs=dict(
            log_std_init=-2,
            ortho_init=False,
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[512, 512], vf=[512, 512])]
        ),
        verbose=1
    )
    
    ppo_policy.set_logger(new_logger)
    
    total_timesteps = 2000000
    total_steps_per_epoch = total_timesteps // epochs
    
    eval_env = DummyVecEnv([lambda: gymnasium.make(env_name)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=metrics_logger.save_dir,
        log_path=metrics_logger.tensorboard_log,
        eval_freq=save_freq * (total_steps_per_epoch),
        deterministic=True,
        render=False
    )
    
    for iteration in range(epochs):
        print(f"Iteration {iteration + 1}/{epochs}")
        ppo_policy.learn(
            total_timesteps=total_steps_per_epoch,
            callback=[eval_callback, reward_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )
    
    ppo_policy.save(os.path.join(metrics_logger.save_dir, f"PPO_{env_name}_final"))
    
    mean_reward, std_reward = evaluate_policy(
        ppo_policy, eval_env, n_eval_episodes=10, deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    env.close()
    eval_env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v5')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    train_ppo(
        args.env,
        exp_name=args.exp_name,
        seed=args.seed,
        epochs=args.epochs
    )
