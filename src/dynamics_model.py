import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=1024):
        super(DynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, state, action):
        state_reshaped = state.squeeze(1)
        action_reshaped = action.squeeze(1)
        x = torch.cat([state_reshaped, action_reshaped], dim=1)
        return self.model(x)

class CustomSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.max_loss = 0.0

    def forward(self, pred, target):
        element_wise_losses = self.smooth_l1(pred, target)
        current_max = torch.max(element_wise_losses).item()
        self.max_loss = max(self.max_loss, current_max)
        return torch.mean(element_wise_losses)

def train_dynamics_model(dynamics_model, real_data, metrics_logger, iteration, real_dim, batch_size=256, lr=1e-3):
    optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=lr)
    loss_fn = CustomSmoothL1Loss()

    states, actions, next_states = real_data
    states = torch.tensor(states.squeeze(1)[:, :real_dim], dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states.squeeze(1)[:, :real_dim], dtype=torch.float32).to(device)

    dataset = torch.utils.data.TensorDataset(states, actions, next_states)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, (batch_states, batch_actions, batch_next_states) in enumerate(dataloader):
        batch_states, batch_actions, batch_next_states = batch_states.to(device), batch_actions.to(device), batch_next_states.to(device)

        predictions = dynamics_model(batch_states, batch_actions)
        loss = loss_fn(predictions, batch_next_states.squeeze(1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), max_norm=1.0)
        optimizer.step()

        if i % 5000 == 0:
            print("Dynamics model loss:", loss.item())
            print("Dynamics model max loss:", loss_fn.max_loss)
            metrics_logger.log_metrics({"DynamicsModel/Loss": loss.item()}, iteration)
