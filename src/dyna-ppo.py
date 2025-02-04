import gymnasium
import numpy as np
import torch
import torch.nn as nn
import os
import time
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
from metrics_logger import MetricsLogger
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecNormalize

from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MetricsLogger:
    def __init__(self, exp_name, env_name, tensorboard_log=None):
        self.exp_name = exp_name
        self.env_name = env_name
        self.save_dir = os.path.join('/home/richtsai1103/CRL/src/results/HalfCheetah-v5', exp_name)
        os.makedirs(self.save_dir, exist_ok=True)
        
        if tensorboard_log:
            self.tensorboard_log = tensorboard_log
        else:
            self.tensorboard_log = os.path.join(self.save_dir, "tensorboard")
        
        self.writer = SummaryWriter(log_dir=self.tensorboard_log)
    
    def log_metrics(self, metrics, step):
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def log_loss(self, loss, step, tag="loss"):
        self.writer.add_scalar(tag, loss, step)
        
    def close(self):
        self.writer.close()



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

    def _save_step_data(self, row_data):
        """Save the step data as needed."""
        # Placeholder for saving logic
        print("Saving step data:", row_data)

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
            'done': done,
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
                'prev_reward': self.prev_step_info['reward'],
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

            # Log metrics every 5 episodes
            if len(self.episode_rewards) % 5 == 0:
                self.metrics_logger.log_metrics({
                    'EpRet': np.mean(self.episode_rewards),
                    'EpLen': np.mean(self.episode_lengths),
                }, self.step_counter)

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

class CustomSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')  # Use 'none' to get per-element losses
        self.max_loss = 0.0  # Track maximum loss

    def forward(self, pred, target):
        # Calculate element-wise losses
        element_wise_losses = self.smooth_l1(pred, target)
        
        # Update max loss
        current_max = torch.max(element_wise_losses).item()
        self.max_loss = max(self.max_loss, current_max)
        
        # Return mean loss (same as original SmoothL1Loss)
        return torch.mean(element_wise_losses)
    
    

class ModifiedEnv(gymnasium.Env):
    def __init__(self, original_env, extra_feature_dim, dynamics_model, policy):
        self.original_env = original_env
        self.extra_feature_dim = extra_feature_dim
        self.dynamics_model = dynamics_model
        self.policy = policy  # Add the policy to the environment
        
        # Update observation space
        original_obs_space = original_env.observation_space
        self.observation_space = gymnasium.spaces.Box(
            low=np.hstack((original_obs_space.low, [-np.inf] * extra_feature_dim)),
            high=np.hstack((original_obs_space.high, [np.inf] * extra_feature_dim)),
            dtype=np.float32
        )
        self.action_space = original_env.action_space

    def reset(self, seed=None, options=None):
        obs, info = self.original_env.reset(seed=seed, options=options)
        extra_feature = self._generate_extra_feature(obs, np.zeros(self.action_space.shape))
        extra_feature = extra_feature.detach().cpu().numpy()[0]
        combined_obs = np.hstack((obs, extra_feature))
        self.current_obs = obs
        return combined_obs, info
    
    # def step(self, action):
    #     obs, reward, done, truncated, info = self.original_env.step(action)
    #     extra_feature = self._generate_extra_feature(obs, action) 
    #     extra_feature = extra_feature.detach().cpu().numpy()[0]
    #     combined_obs = np.hstack((obs, extra_feature))
    #     return combined_obs, reward, done, truncated, info
    

    def step(self, action):
        # Step the environment and get the next observation
        next_obs, reward, done, truncated, info = self.original_env.step(action)

        # Generate the extra feature based on the current observation and action
        peek_feature = self._generate_extra_feature(self.current_obs, action)
        peek_feature = peek_feature.detach().cpu().numpy()[0]

        # Combine the current observation with the extra feature to form the augmented observation
        augmented_obs = np.hstack((self.current_obs, peek_feature))

        # Use the augmented observation to predict the next action
        predicted_action, _ = self.policy.predict(augmented_obs, deterministic=True)

        # Generate the extra feature for the next observation
        next_peek_feature = self._generate_extra_feature(next_obs, predicted_action)
        next_peek_feature = next_peek_feature.detach().cpu().numpy()[0]

        # Combine the next observation with the next extra feature
        combined_obs = np.hstack((next_obs, next_peek_feature))

        # Update the current observation for the next step
        self.current_obs = next_obs

        return combined_obs, reward, done, truncated, info


    def _generate_extra_feature(self, obs, action):
        # Convert state and action to tensors
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
        next_state_pred = self.dynamics_model(state_tensor, action_tensor)
        return next_state_pred




# Dynamics Model
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super(DynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)  # Predict next state
        )
    
    def forward(self, state, action):
        state_reshaped = state.squeeze(1)
        action_reshaped = action.squeeze(1)
        x = torch.cat([state_reshaped, action_reshaped], dim=1)
        return self.model(x)


def train_dynamics_model(dynamics_model, real_data, metrics_logger, iteration, real_dim, batch_size=256, lr=1e-3):
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=lr)
    # loss_fn = nn.MSELoss()
    loss_fn = CustomSmoothL1Loss()
    # loss_fn = nn.SmoothL1Loss()

    states, actions, next_states = real_data
    states = torch.tensor(states.squeeze(1)[:, :real_dim], dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states.squeeze(1)[:, :real_dim], dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(states, actions, next_states)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


    for i, (batch_states, batch_actions, batch_next_states) in enumerate(dataloader):
        # Move the batch to the correct device
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)
        batch_next_states = batch_next_states.to(device)
        
        predictions = dynamics_model(batch_states, batch_actions)
        loss = loss_fn(predictions, batch_next_states.squeeze(1))
        
        
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Log loss every N batches or at the end of each epoch
        if i % 5000 == 0:  # Adjust logging frequency as needed
            # Log loss every few batches
            print("dynamics model loss:",loss.item())
            print("dynamics model max loss:", loss_fn.max_loss)
            metrics_logger.log_metrics({"DynamicsModel/Loss": loss.item()}, iteration)


# def generate_synthetic_rollouts(dynamics_model, buffer, policy, rollout_horizon=5):
#     synthetic_states, synthetic_actions, synthetic_next_states = [], [], []
#     state = buffer.sample(1).observations[0]
#     for _ in range(rollout_horizon):
#         # # Random action for simplicity
#         # action = np.random.uniform(-1, 1, size=env.action_space.shape)  
#         # Get the most probable action from the policy
        
#         action, _ = policy.predict(state, deterministic=True)
#         action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#         next_state = dynamics_model(state_tensor, action_tensor).detach().numpy()[0]
#         synthetic_states.append(state)
#         synthetic_actions.append(action)
#         synthetic_next_states.append(next_state)
#         state = next_state
    
#     return np.array(synthetic_states), np.array(synthetic_actions), np.array(synthetic_next_states)


# def linear_schedule(progress_remaining, log_interval = 1000):
#     if int(progress_remaining * 1e6) % log_interval == 0:
#         print(f"Progress remaining: {progress_remaining:.6f}")
#     return progress_remaining * 0.0003

# class LearningRateDecayCallback(BaseCallback):
#     def __init__(self, schedule, total_timesteps, verbose=0):
#         super().__init__(verbose)
#         self.schedule = schedule
#         self.total_timesteps = total_timesteps

#     def _on_step(self) -> bool:
#         # Calculate progress over the entire training process
#         progress = 1 - (self.num_timesteps / self.total_timesteps)
#         new_lr = self.schedule(progress)
        
#         # Update the optimizer's learning rate
#         for param_group in self.model.policy.optimizer.param_groups:
#             param_group['lr'] = new_lr
            
#         # Log the new learning rate to TensorBoard
#         self.logger.record('train/learning_rate', new_lr)
        
#         if self.verbose > 0 and self.n_calls % 1000 == 0:
#             print(f"[Callback] Updated learning rate to {new_lr}")
        
#         return True


# Modified train function
def train_dyna_ppo(
        env_name,
        seed=0,
        epochs=20,
        save_freq=1,
        exp_name='dyna_ppo',
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Initialize dynamics model
    original_env = gymnasium.make(env_name)
    
    dynamics_model = DynamicsModel(
        state_dim=original_env.observation_space.shape[0],
        action_dim=original_env.action_space.shape[0]
    ).to(device)
    
    # Total timesteps for all epochs
    total_timesteps = 1000000
    total_steps_per_epoch = total_timesteps//epochs
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Logging setup
    exp_directory = f"{exp_name}_{current_time}"
    metrics_logger = MetricsLogger(exp_directory, env_name)
    reward_callback = RewardCallback(metrics_logger)

    # lr_callback = LearningRateDecayCallback(linear_schedule, total_timesteps=total_timesteps, verbose=1)
    
    # Set up custom logger
    new_logger = configure(folder=metrics_logger.save_dir, format_strings=["stdout", "csv", "tensorboard"])

    # Initialize the environment
    env = ModifiedEnv(original_env, extra_feature_dim=17, dynamics_model=dynamics_model, policy=None)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    # Initialize the PPO policy with the environment
    ppo_policy = PPO(
        policy="MlpPolicy",
        env=env,  # Pass the environment here
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

    # Pass the policy to the ModifiedEnv
    env.envs[0].policy = ppo_policy

    # Assign custom logger to PPO
    ppo_policy.set_logger(new_logger)

    
    # Create the evaluation environment
    eval_env = DummyVecEnv([lambda: ModifiedEnv(
        gymnasium.make(env_name),
        extra_feature_dim=17,
        dynamics_model=dynamics_model,
        policy=ppo_policy
    )])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=metrics_logger.save_dir,
        log_path=metrics_logger.tensorboard_log,
        eval_freq=save_freq * (1000000 // epochs),
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
        
        # Train PPO on real data
        # ppo_policy.learn(
        #     total_timesteps=total_steps_per_epoch,
        #     callback=[eval_callback, reward_callback, lr_callback],
        #     progress_bar=True,
        #     reset_num_timesteps=False,
        # )
        ppo_policy.learn(
            total_timesteps=total_steps_per_epoch,
            callback=[eval_callback, reward_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )
        
        # Collect real data
        obs = env.reset()
        dynamics_training_data = int(total_steps_per_epoch*0.1)
        for _ in range(dynamics_training_data):
            action, _ = ppo_policy.predict(obs)
            next_obs, reward, done, infos = env.step(action)

            
            real_buffer.add(obs[0], next_obs[0], action[0], reward, done, infos)
            # obs = next_obs if not done else env.reset()
            obs = next_obs
        
        # Train dynamics model
        real_data = (
            real_buffer.observations,
            real_buffer.actions,
            real_buffer.next_observations
        )
        
        # print(real_buffer.observations)
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

    # print(f"\nTraining completed in {time.time() - env.start_time:.2f}s")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    env.close()
    eval_env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v5')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='dyna_ppo')
    args = parser.parse_args()

    train_dyna_ppo(
        args.env,
        exp_name=args.exp_name,
        seed=args.seed,
        epochs=args.epochs
    )
