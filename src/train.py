import os
import gymnasium
import numpy as np
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.buffers import ReplayBuffer
from metrics_logger import MetricsLogger
from reward_callback import RewardCallback
from dynamics_model import DynamicsModel, train_dynamics_model
from environment import ModifiedEnv
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_dyna_ppo(env_name, total_timesteps, seed=0, epochs=20, save_freq=1, exp_name='dyna_ppo'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    original_env = gymnasium.make(env_name)
    
    dynamics_model = DynamicsModel(
        state_dim=original_env.observation_space.shape[0],
        action_dim=original_env.action_space.shape[0]
    ).to(device)

    total_steps_per_epoch = total_timesteps // epochs
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_directory = f"{exp_name}_{current_time}"
    metrics_logger = MetricsLogger(exp_directory, env_name)
    reward_callback = RewardCallback(metrics_logger)

    new_logger = configure(folder=metrics_logger.save_dir, format_strings=["stdout", "csv", "tensorboard"])

    # decide how many extra feature 
    reduce_feature = False
    if not reduce_feature:
        extra_feature_dim = original_env.observation_space.shape[0]
    else:
        extra_feature_dim = original_env.observation_space.shape[0] // 3 # modified this if you want
        
    env = ModifiedEnv(original_env, 
                      extra_feature_dim=extra_feature_dim, 
                      dynamics_model=dynamics_model, 
                      policy=None, 
                      reduce_feature=reduce_feature)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    hyperparameters = {
        "env_name": env_name,
        "learning_rate": 2.0633e-05,
        "n_steps": 512,
        "batch_size": 256,
        "n_epochs": 20,
        "gamma": 0.98,
        "gae_lambda": 0.92,
        "clip_range": 0.1,
        "ent_coef": 0.000401762,
        "vf_coef": 0.58096,
        "max_grad_norm": 0.5,
        "policy_kwargs": {
            "log_std_init": -2,
            "ortho_init": True,
            "activation_fn": "Tanh",
            "net_arch": [{"pi": [512, 512], "vf": [512, 512]}],
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

    env.envs[0].policy = ppo_policy
    ppo_policy.set_logger(new_logger)

    eval_env = DummyVecEnv([lambda: ModifiedEnv(
        gymnasium.make(env_name),
        extra_feature_dim=extra_feature_dim,
        dynamics_model=dynamics_model,
        policy=ppo_policy,
        reduce_feature=reduce_feature 
    )])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=metrics_logger.save_dir,
        log_path=metrics_logger.tensorboard_log,
        eval_freq=save_freq * (total_steps_per_epoch),
        deterministic=True,
        render=False
    )

    real_buffer = ReplayBuffer(
        buffer_size=10000,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device="cpu"
    )

    for iteration in range(epochs):
        print(f"Iteration {iteration + 1}/{epochs}")

        ppo_policy.learn(
            total_timesteps=total_steps_per_epoch,
            callback=[eval_callback, reward_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )

        obs = env.reset()
        dynamics_training_data = int(total_steps_per_epoch * 0.1)
        for _ in range(dynamics_training_data):
            action, _ = ppo_policy.predict(obs)
            next_obs, reward, done, infos = env.step(action)
            real_buffer.add(obs[0], next_obs[0], action[0], reward, done, infos)
            obs = next_obs

        real_data = (
            real_buffer.observations,
            real_buffer.actions,
            real_buffer.next_observations
        )
        
        real_dim = original_env.observation_space.shape[0]
        train_dynamics_model(dynamics_model, real_data, metrics_logger, iteration, real_dim)

    ppo_policy.save(os.path.join(metrics_logger.save_dir, f"PPO_{env_name}_final"))
    torch.save(dynamics_model.state_dict(), os.path.join(metrics_logger.save_dir, f"Dyna_{env_name}_final.pth"))

    mean_reward, std_reward = evaluate_policy(
        ppo_policy,
        eval_env,
        n_eval_episodes=10,
        deterministic=True
    )

    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    eval_env.close()
