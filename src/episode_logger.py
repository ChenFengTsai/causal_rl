import numpy as np
import os
import json

class EpisodeLogger:
    def __init__(self, num_episodes_to_log=5, save_dir='episode_logs'):
        self.num_episodes_to_log = num_episodes_to_log
        self.episodes_logged = 0
        self.save_dir = save_dir
        self.current_episode = {
            'states': [],
            'actions': [],
            'rewards': [],
            'total_reward': 0
        }
        self.all_episodes = []
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
    
    def reset(self):
        if self.current_episode['states']:  # If episode has data
            self.save_episode()
            self.current_episode = {
                'states': [],
                'actions': [],
                'rewards': [],
                'total_reward': 0
            }
    
    def log_step(self, state, action, reward):
        if self.episodes_logged < self.num_episodes_to_log:
            self.current_episode['states'].append(state.tolist())
            # Convert action to list if it's a numpy array
            if isinstance(action, np.ndarray):
                action = action.tolist()
            self.current_episode['actions'].append(action)
            self.current_episode['rewards'].append(float(reward))
            self.current_episode['total_reward'] += float(reward)
    
    def save_episode(self):
        if self.episodes_logged < self.num_episodes_to_log:
            self.episodes_logged += 1
            self.all_episodes.append(self.current_episode)
            
            # Save the episode data
            episode_data = {
                'episode_number': self.episodes_logged,
                'data': self.current_episode
            }
            
            filename = os.path.join(self.save_dir, f'episode_{self.episodes_logged}.json')
            with open(filename, 'w') as f:
                json.dump(episode_data, f, indent=4)
            
            # Save summary statistics
            if self.episodes_logged == self.num_episodes_to_log:
                self.save_summary()
    
    def save_summary(self):
        summary = {
            'num_episodes': self.episodes_logged,
            'episode_rewards': [ep['total_reward'] for ep in self.all_episodes],
            'episode_lengths': [len(ep['rewards']) for ep in self.all_episodes],
            'state_dim': len(self.all_episodes[0]['states'][0]),
            'action_dim': len(self.all_episodes[0]['actions'][0]) if isinstance(self.all_episodes[0]['actions'][0], list) else 1
        }
        
        filename = os.path.join(self.save_dir, 'episodes_summary.json')
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=4)