from causal_world.task_generators.task import generate_task
from causal_world.envs.causalworld import CausalWorld
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import Env, spaces

from utils import *
import os

import argparse
from torch import nn
import torch
import numpy as np
# Patch deprecated alias
if not hasattr(np, 'bool'):
    np.bool = bool
    


class CausalWorldGymWrapper(Env):
    """Wrapper to make CausalWorld compatible with Gymnasium."""
    
    def __init__(self, env):
        self.env = env
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=env.action_space.low,
            high=env.action_space.high,
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs = self.env.reset()
        return obs, {}  # Gymnasium expects (obs, info)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, False, info  # Gymnasium expects (obs, reward, terminated, truncated, info)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
    
# Check if CUDA is available
if torch.cuda.is_available():
    # Get the name of the current CUDA device
    print(f"Using CUDA device 0: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
else:
    print("CUDA is not available.")
    
# Set the device to CUDA 0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
def train_policy(num_of_envs, log_relative_path, maximum_episode_length,
                 skip_frame, seed_num, ppo_config, total_time_steps,
                 validate_every_timesteps, task_name, tb_log_name):

    def _make_env(rank):
        def _init():
            task = generate_task(task_generator_id=task_name)
            env = CausalWorld(task=task,
                              skip_frame=skip_frame,
                              enable_visualization=False,
                              seed=seed_num + rank,
                              max_episode_length=maximum_episode_length)
            env = CausalWorldGymWrapper(env)
            env = Monitor(env, os.path.join(log_relative_path, "monitor", tb_log_name, f"monitor_{rank}"))
            return env
        return _init

    os.makedirs(log_relative_path, exist_ok=True)
    env = SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])
    set_random_seed(seed_num)

    # Define the policy_kwargs
    policy_kwargs = dict(activation_fn=nn.Tanh, net_arch=[dict(pi=[256, 128], vf=[256, 128])])
    
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                **ppo_config)
    
    save_config_file(ppo_config,
                _make_env(0)(),
                os.path.join(log_relative_path, tb_log_name+'_config', 'config.json'))

    for i in range(int(total_time_steps / validate_every_timesteps)):
        model.learn(total_timesteps=validate_every_timesteps,
                    progress_bar=True,
                    reset_num_timesteps=False,
                    tb_log_name=tb_log_name)
        
        model.save(os.path.join(log_relative_path, 'models', tb_log_name, f'saved_model_{i+1}'))
        print('Model saved at', os.path.join(log_relative_path, 'models', tb_log_name, f'saved_model_{i+1}'))


    env.close()
    return





if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_num", required=False, default=0, type=int, help="Seed number")
    ap.add_argument("--skip_frame", required=False, default=10, type=int, help="Skip frame")
    ap.add_argument("--max_episode_length", required=False, default=2500, type=int, help="Maximum episode length")
    ap.add_argument("--total_time_steps_per_update", required=False, default=100000, type=int, help="Total time steps per update")
    ap.add_argument("--num_of_envs", required=False, default=20, type=int, help="Number of parallel environments")
    ap.add_argument("--task_name", required=False, default="reaching", help="Task name for training")
    ap.add_argument("--fixed_position", required=False, default=True, type=bool, help="Define the reset intervention wrapper")
    ap.add_argument("--log_relative_path", required=True, help="Log folder")
    args = vars(ap.parse_args())

    total_time_steps_per_update = args['total_time_steps_per_update']
    num_of_envs = args['num_of_envs']
    log_relative_path = args['log_relative_path']
    maximum_episode_length = args['max_episode_length']
    skip_frame = args['skip_frame']
    seed_num = args['seed_num']
    task_name = args['task_name']
    fixed_position = args['fixed_position']


    assert (((float(total_time_steps_per_update) / num_of_envs) / 5).is_integer())
    
    new_dir, num = get_next_experiment_number(log_relative_path, 'PPO')
    task_log_relative_path = os.path.join(task_name, log_relative_path)
    os.makedirs(task_log_relative_path, exist_ok=True)
    
    ppo_config = {
        "gamma": 0.9995,
        "n_steps": 5000,
        "ent_coef": 0,
        "learning_rate": 0.00025,
        "vf_coef": 0.5,
        "max_grad_norm": 10,
        "batch_size": 1000,  # Equivalent to nminibatches in SB2
        "n_epochs": 4,
        "tensorboard_log": task_log_relative_path
    }

    
    
    train_policy(num_of_envs=num_of_envs,
                 log_relative_path=task_log_relative_path,
                 maximum_episode_length=maximum_episode_length,
                 skip_frame=skip_frame,
                 seed_num=seed_num,
                 ppo_config=ppo_config,
                 total_time_steps=6000000,
                 validate_every_timesteps=1000000,
                 task_name=task_name,
                 tb_log_name=new_dir
                )
