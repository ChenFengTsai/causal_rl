import os
import re
import json

import gymnasium
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def get_next_experiment_number(base_path, prefix):
    """
    Check a directory for experiment folders and return the next available number.
    
    Args:
        base_path (str): The base directory to check
        prefix (str): The prefix for experiment folders (default: "experiment")
    
    Returns:
        str: The next experiment directory name (e.g., "experiment_001")
        int: The next experiment number
    
    Example:
        >>> get_next_experiment_number("/path/to/experiments")
        'experiment_004', 4
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Get all directory names in the base path
    existing_dirs = [d for d in os.listdir(base_path) 
                    if os.path.isdir(os.path.join(base_path, d))]
    
    # Pattern to match experiment folders (e.g., "experiment_001")
    pattern = f"{prefix}_(\d+)"
    
    # Find all experiment numbers
    experiment_numbers = []
    for dirname in existing_dirs:
        match = re.match(pattern, dirname)
        if match:
            try:
                num = int(match.group(1))
                experiment_numbers.append(num)
            except ValueError:
                continue
    
    # Get the next number
    if experiment_numbers:
        next_number = max(experiment_numbers) + 1
    else:
        next_number = 1
    
    # Format the new directory name with leading zeros
    new_dirname = f"{prefix}_{next_number:03d}"
    
    return new_dirname, next_number

def save_config_file(ppo_config, env, file_path):
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    unwrapper_env = env.env.env
    task_config = unwrapper_env._task.get_task_params()
    for task_param in task_config:
        if not isinstance(task_config[task_param], str):
            task_config[task_param] = str(task_config[task_param])
    env_config = unwrapper_env.get_world_params()
    env.close()
    configs_to_save = [task_config, env_config, ppo_config]
    with open(file_path, 'w') as fout:
        json.dump(configs_to_save, fout)

def setup_logger(exp_directory):
    """Set up the logger and experiment directory."""
    os.makedirs(exp_directory, exist_ok=True)
    logger = configure(folder=exp_directory, format_strings=["stdout", "csv", "tensorboard"])
    return logger, exp_directory

def save_hyperparameters(hyperparameters, save_dir):
    """Save hyperparameters to a JSON file."""
    hyperparam_path = os.path.join(save_dir, "hyperparameters.json")
    with open(hyperparam_path, "w") as f:
        json.dump(hyperparameters, f, indent=4)
    print(f"Saved hyperparameters to {hyperparam_path}")

def load_hyperparameters(load_dir):
    """Load hyperparameters from a JSON file."""
    hyperparam_path = os.path.join(load_dir, "hyperparameters.json")
    with open(hyperparam_path, "r") as f:
        return json.load(f)
    
def evaluate_model(model, eval_env, n_episodes=10):
    """Evaluates a trained model and returns the mean and standard deviation of rewards."""
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=n_episodes, deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

def make_vec_env(env_name, n_envs=1):
    """
    Creates a vectorized environment manually if `make_vec_env` is unavailable.
    """
    env = gymnasium.make(env_name)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    return env



