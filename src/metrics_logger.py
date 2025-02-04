import os
from torch.utils.tensorboard import SummaryWriter

class MetricsLogger:
    def __init__(self, exp_name, env_name, tensorboard_log=None):
        self.exp_name = exp_name
        self.env_name = env_name
        self.save_dir = os.path.join(f'/home/richtsai1103/CRL/src/results/{env_name}', exp_name)
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
