import os
import sys
from datetime import datetime

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import make_vec_env, setup_logger, save_hyperparameters, load_hyperparameters, evaluate_model
from metrics_logger import MetricsLogger
from reward_callback import RewardCallback


def train_sac(env_name, total_timesteps, seed=0, epochs=20, save_freq=1, exp_name='sac'):
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
        "policy": "MlpPolicy",
        "env_name": env_name,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "policy_kwargs": {
            "log_std_init": -3,
            "net_arch": [1024, 1024, 512],
        },
        "total_timesteps": total_timesteps,
        "epochs": epochs,
        "seed": seed,
    }

    # Save hyperparameters
    save_hyperparameters(hyperparameters, exp_directory)

    # Load hyperparameters
    loaded_hyperparameters = load_hyperparameters(exp_directory)

    sac_params = {k: v for k, v in loaded_hyperparameters.items() if k not in ["env_name", "total_timesteps", "epochs", "seed"]}
    # Initialize SAC model
    sac_policy = SAC(
        env=env,
        **sac_params,
        seed=seed,
        verbose=1,
    )

    # Set up logger and evaluation callback
    sac_policy.set_logger(logger)
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
        sac_policy.learn(
            total_timesteps=total_steps_per_epoch,
            callback=[eval_callback, reward_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )

    # Save the final model
    sac_policy.save(f"{metrics_logger.save_dir}/SAC_{env_name}_final")

    # Evaluate the model
    evaluate_model(sac_policy, eval_env)

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
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    train_sac(
        args.env,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        epochs=args.epochs,
        exp_name=args.exp_name
    )
