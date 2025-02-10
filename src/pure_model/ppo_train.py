import os
import sys
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import make_vec_env, setup_logger, save_hyperparameters, load_hyperparameters, evaluate_model
from metrics_logger import MetricsLogger
from reward_callback import RewardCallback



def train_ppo(env_name, total_timesteps, seed=0, epochs=20, save_freq=1, exp_name='ppo'):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environments
    env = make_vec_env(env_name)
    eval_env = make_vec_env(env_name)

    # Setup logger
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    exp_directory = os.path.join(parent_path, f"results/{env_name}/{exp_name}_{current_time}")
    logger, exp_directory = setup_logger(exp_directory)
    metrics_logger = MetricsLogger(exp_directory, env_name)
    reward_callback = RewardCallback(metrics_logger)

    # Define hyperparameters
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

    # Save hyperparameters
    save_hyperparameters(hyperparameters, exp_directory)

    loaded_hyperparameters = load_hyperparameters(exp_directory)
        
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

    # Set up logger and evaluation callback
    ppo_policy.set_logger(logger)
    total_steps_per_epoch = total_timesteps // epochs
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=metrics_logger.save_dir,
        log_path=metrics_logger.tensorboard_log,
        eval_freq=save_freq * total_steps_per_epoch,
        deterministic=True,
        render=False,
    )

    # Training loop
    for iteration in range(epochs):
        print(f"Iteration {iteration + 1}/{epochs}")
        ppo_policy.learn(
            total_timesteps=total_steps_per_epoch,
            callback=[eval_callback, reward_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )

    # Save the final model
    ppo_policy.save(f"{metrics_logger.save_dir}/PPO_{env_name}_final")

    # Evaluate the model
    evaluate_model(ppo_policy, eval_env)

    # Cleanup
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
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        epochs=args.epochs,
        exp_name=args.exp_name
    )
