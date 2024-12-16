import os
import re
import json

from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import Env, spaces

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

