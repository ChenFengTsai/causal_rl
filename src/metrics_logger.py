import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

class MetricsLogger:
    def __init__(self, exp_name, env_name):
        """Initialize the metrics logger.
        
        Args:
            exp_name (str): Name of the experiment
            env_name (str): Name of the environment
        """
        self.metrics = {
            'VVals': [],
            'EpRet': [],
            'EpLen': [],
            'StopIter': [],
            'LossPi': [],
            'LossV': [],
            'KL': [],
            'Entropy': [],
            'ClipFrac': [],
            'DeltaLossPi': [],
            'DeltaLossV': []
        }
        
        self.exp_name = exp_name
        self.env_name = env_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories for saving data
        self.save_dir = f"results/{env_name}/{exp_name}_{self.timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        
    def update(self, epoch_metrics):
        """Update metrics with new epoch data."""
        for key in self.metrics:
            if key in epoch_metrics:
                self.metrics[key].append({
                    'mean': float(np.mean(epoch_metrics[key])),
                    'std': float(np.std(epoch_metrics[key])),
                    'min': float(np.min(epoch_metrics[key])),
                    'max': float(np.max(epoch_metrics[key]))
                })
    
    def save_metrics(self):
        """Save metrics to a JSON file."""
        metrics_path = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=4)
    
    def plot_metrics(self):
        """Plot and save all metrics."""
        metrics_to_plot = [
            ('Training Progress', ['EpRet']),
            ('Policy Loss', ['LossPi']),
            ('Value Loss', ['LossV']),
            ('Policy Metrics', ['KL', 'Entropy', 'ClipFrac']),
            ('Value and Stop Iteration', ['VVals', 'StopIter'])
        ]
        
        for title, metric_group in metrics_to_plot:
            plt.figure(figsize=(12, 6))
            for metric in metric_group:
                means = [m['mean'] for m in self.metrics[metric]]
                stds = [m['std'] for m in self.metrics[metric]]
                epochs = range(len(means))
                
                plt.plot(epochs, means, label=metric)
                plt.fill_between(epochs, 
                               [m - s for m, s in zip(means, stds)],
                               [m + s for m, s in zip(means, stds)],
                               alpha=0.2)
            
            plt.title(f'{title} - {self.env_name}')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plt.savefig(os.path.join(self.save_dir, f'{title.lower().replace(" ", "_")}.png'))
            plt.close()
    
    def log_epoch(self, epoch_metrics, epoch, total_steps, time_elapsed):
        """Log metrics for current epoch and display summary."""
        self.update(epoch_metrics)
        
        print('-' * 40)
        print(f'Epoch: {epoch}')
        print(f'TotalEnvInteracts: {total_steps}')
        
        # Print metrics
        for key, values in epoch_metrics.items():
            if len(values) > 0:
                mean = np.mean(values)
                if key in ['EpRet', 'VVals']:
                    min_val = np.min(values)
                    max_val = np.max(values)
                    std = np.std(values)
                    print(f'{key+":":13s} {mean:.4f}\t{min_val:.4f}(min) {max_val:.4f}(max) {std:.4f}(std)')
                else:
                    print(f'{key+":":13s} {mean:.4f}')
        
        print(f'Time: {time_elapsed:.4f}s')
        print('-' * 40 + '\n')