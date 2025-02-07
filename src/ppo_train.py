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
import json


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PPO training function
def train_ppo(env_name, total_timesteps, seed=0, epochs=20, save_freq=1, exp_name='ppo'):
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
    # Inside train_ppo function, before initializing PPO

    hyperparameters = {
        "env_name": env_name,
        "learning_rate": 3e-5,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "log_std_init": -2,
            "ortho_init": True,
            "activation_fn": "Tanh",
            "net_arch": [{"pi": [1024, 1024, 512], "vf": [1024, 1024, 512]}],
        },
        "total_timesteps": total_timesteps,
        "epochs": epochs,
        "seed": seed,
    }

    # Save hyperparameters as JSON
    hyperparam_path = os.path.join(metrics_logger.save_dir, "hyperparameters.json")
    with open(hyperparam_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)

    print(f"Saved hyperparameters to {hyperparam_path}")
    
    # Load hyperparameters from JSON
    with open(hyperparam_path, "r") as f:
        loaded_hyperparameters = json.load(f)
        
    # Convert activation function from string to actual function
    if "policy_kwargs" in loaded_hyperparameters and "activation_fn" in loaded_hyperparameters["policy_kwargs"]:
        loaded_hyperparameters["policy_kwargs"]["activation_fn"] = getattr(torch.nn, loaded_hyperparameters["policy_kwargs"]["activation_fn"].split('.')[-1])
        
    if "policy_kwargs" in loaded_hyperparameters and "net_arch" in loaded_hyperparameters["policy_kwargs"]:
        loaded_hyperparameters["policy_kwargs"]["net_arch"] = [
            dict(pi=loaded_hyperparameters["policy_kwargs"]["net_arch"][0]["pi"],
            vf=loaded_hyperparameters["policy_kwargs"]["net_arch"][0]["vf"])
        ]

    # Remove unnecessary keys before passing into PPO
    ppo_params = {k: v for k, v in loaded_hyperparameters.items() if k not in ["env_name", "total_timesteps", "epochs", "seed"]}
    ppo_policy = PPO(
        policy="MlpPolicy",
        env=env,
        **ppo_params,  # Load hyperparameters dynamically
        seed=seed,
        verbose=1
    )
    
    ppo_policy.set_logger(new_logger)
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--total_timesteps', type=int, default=10000000)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    train_ppo(
        args.env,
        exp_name=args.exp_name,
        seed=args.seed,
        epochs=args.epochs,
        total_timesteps=args.total_timesteps
    )
