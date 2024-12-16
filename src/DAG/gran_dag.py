import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from notears_dag import *

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class GraNDAG(nn.Module):
    def __init__(self, num_vars, hidden_dim=512):
        super(GraNDAG, self).__init__()
        self.num_vars = num_vars
        
        # Create neural networks for each variable
        self.networks = nn.ModuleList([
            MLP(num_vars - 1, hidden_dim, 1) for _ in range(num_vars)
        ])
        
        # Initialize adjacency matrix weights
        self.adj_weights = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)
        
    def get_adjacency_matrix(self):
        # Apply sigmoid to get values between 0 and 1
        return torch.sigmoid(self.adj_weights)
    
    def forward(self, x):
        batch_size = x.size(0)
        adj_matrix = self.get_adjacency_matrix()
        
        # Mask diagonal to ensure no self-loops
        mask = torch.eye(self.num_vars, device=x.device)
        adj_matrix = adj_matrix * (1 - mask)
        
        outputs = []
        for i in range(self.num_vars):
            # Remove the i-th variable from inputs
            mask = torch.ones(self.num_vars, dtype=torch.bool)
            mask[i] = False
            inputs = x[:, mask]
            
            # Weight inputs by adjacency matrix
            weights = adj_matrix[i, mask]
            weighted_inputs = inputs * weights.view(1, -1)
            
            # Pass through neural network
            output = self.networks[i](weighted_inputs)
            outputs.append(output)
        
        return torch.cat(outputs, dim=1)

def h_func(x):
    """Compute acyclicity constraint h(W)"""
    E = torch.matrix_exp(x * x)  # element-wise square
    h = torch.trace(E) - x.size(0)
    return h

def train_gran_dag(model, data, column_name, num_epochs=3000, lr=1e-3, lambda_dag=35, threshold=0.05):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Reconstruction loss
        loss_recon = mse_loss(outputs, data)
        
        # DAG constraint
        adj_matrix = model.get_adjacency_matrix()
        h_val = h_func(adj_matrix)
        loss_dag = lambda_dag * h_val
        
        # Total loss
        loss = loss_recon + loss_dag
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Reconstruction Loss: {loss_recon.item():.4f}")
            print(f"DAG Loss: {loss_dag.item():.4f}")
            print(f"h(W): {h_val.item():.4f}")
            print("-------------------")
            
    # Get learned adjacency matrix
    
    adjacency_matrix = model.get_adjacency_matrix().detach().numpy()
    print("Learned DAG structure:")
    print(adjacency_matrix)
    binary_adjacency = (np.abs(adjacency_matrix) > threshold).astype(np.float32)
    adjacency_matrix = adjacency_matrix * binary_adjacency
    
    print("DAG structure after threshold:")
    print(adjacency_matrix)
    
    # Create a graph from the adjacency matrix
    graph = nx.DiGraph(adjacency_matrix)
    graph = nx.relabel_nodes(graph, {i: col for i, col in enumerate(column_name)})

    return graph

# Example usage
# def generate_synthetic_data(n_samples=1000, n_vars=5, seed=42):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     # Generate random DAG
#     W = np.tril(np.random.uniform(low=-1, high=1, size=(n_vars, n_vars)), k=-1)
    
#     # Generate samples
#     X = np.random.randn(n_samples, n_vars)
#     for i in range(n_vars):
#         X[:, i] += X @ W[:, i]
    
#     return torch.FloatTensor(X)

