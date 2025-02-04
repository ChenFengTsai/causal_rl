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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_dyna_ppo(env_name, seed=0, epochs=20, save_freq=1, exp_name='dyna_ppo'):
    torch.manual_seed(seed)
    np.random.seed(seed)

    original_env = gymnasium.make(env_name)
    
    dynamics_model = DynamicsModel(
        state_dim=original_env.observation_space.shape[0],
        action_dim=original_env.action_space.shape[0]
    ).to(device)

    total_timesteps = 1000000
    total_steps_per_epoch = total_timesteps // epochs
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    exp_directory = f"{exp_name}_{current_time}"
    metrics_logger = MetricsLogger(exp_directory, env_name)
    reward_callback = RewardCallback(metrics_logger)

    new_logger = configure(folder=metrics_logger.save_dir, format_strings=["stdout", "csv", "tensorboard"])

    # decide how many extra feature 
    extra_same_dim = False
    if extra_same_dim:
        extra_feature_dim = original_env.observation_space.shape[0]
    else:
        extra_feature_dim = original_env.observation_space.shape[0] // 10 # modified this if you want
        
    reduce_feature = True
    env = ModifiedEnv(original_env, 
                      extra_feature_dim=extra_feature_dim, 
                      dynamics_model=dynamics_model, 
                      policy=None, 
                      reduce_feature=reduce_feature)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

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
            activation_fn=torch.nn.ReLU,
            net_arch=[dict(pi=[512, 512], vf=[512, 512])]
        ),
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
