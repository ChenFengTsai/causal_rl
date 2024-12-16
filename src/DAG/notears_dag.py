import pandas as pd

from dagma.nonlinear import DagmaMLP, DagmaNonlinear
import networkx as nx
import matplotlib.pyplot as plt
import torch

# Load your dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# Discover relationships using Dagma
def find_relationships(data):
    try:
        # Convert data to a numpy array
        X = data.to_numpy()
        n, d = X.shape

        eq_model = DagmaMLP(dims=[d, 10, 1], bias=True, dtype=torch.double) # create the model for the structural equations, in this case MLPs
        model = DagmaNonlinear(eq_model, dtype=torch.double) # create the model for DAG learning
        adjacency_matrix = model.fit(X, lambda1=0.02, lambda2=0.005)

        # Create a graph from the adjacency matrix
        graph = nx.DiGraph(adjacency_matrix)
        graph = nx.relabel_nodes(graph, {i: col for i, col in enumerate(data.columns)})

        return graph
    except Exception as e:
        print(f"Error in finding relationships: {e}")
        return None

# Visualize the graph
def visualize_graph(graph):
    try:
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
        plt.title("Causal Relationships")
        plt.show()
    except Exception as e:
        print(f"Error visualizing the graph: {e}")


