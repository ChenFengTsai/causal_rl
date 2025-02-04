import os
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, metrics_logger, save_every_n_steps=1000):
        super().__init__()
        self.metrics_logger = metrics_logger
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.step_counter = 0
        self.save_every_n_steps = save_every_n_steps
        self.prev_step_info = None

    def _save_step_data(self, row_data):
        df = pd.DataFrame([row_data])
        filename = "selected_steps.csv"
        filepath = os.path.join(self.metrics_logger.save_dir, filename)
        if os.path.exists(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False)

    def _on_step(self) -> bool:
        self.step_counter += 1

        try:
            current_state = self.locals['new_obs'][0]
        except KeyError:
            current_state = self.locals['obs'][0]

        current_action = self.locals['actions'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        current_step_info = {
            'global_step': self.step_counter,
            'episode': len(self.episode_rewards),
            'step': self.current_episode_length,
            'current_state': current_state.tolist(),
            'current_action': current_action.tolist(),
            'reward': reward,
            'done': done,
        }

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

        self.prev_step_info = current_step_info
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0

            if len(self.episode_rewards) % 5 == 0:
                self.metrics_logger.log_metrics({
                    'EpRet': np.mean(self.episode_rewards),
                    'EpLen': np.mean(self.episode_lengths),
                }, self.step_counter)

                self.episode_rewards = []
                self.episode_lengths = []

        return True
