import pandas as pd
import numpy as np
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
import networkx as nx
import matplotlib.pyplot as plt
# import torch
# from sklearn.metrics import mean_squared_error
# from scipy.stats import spearmanr
from notears_dag import *

def evaluate_dag_structure(graph, data, adjacency_matrix):
    """
    Comprehensive evaluation of the DAG structure using multiple metrics.
    
    Args:
        graph (nx.DiGraph): The discovered DAG
        data (pd.DataFrame): Original dataset
        adjacency_matrix (np.ndarray): Adjacency matrix from DAGMA
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    metrics = {}
    
    # 1. Basic Graph Properties
    metrics['num_nodes'] = graph.number_of_nodes()
    metrics['num_edges'] = graph.number_of_edges()
    metrics['avg_degree'] = sum(dict(graph.degree()).values()) / graph.number_of_nodes()
    metrics['density'] = nx.density(graph)
    
    # 2. Topological Properties
    metrics['is_dag'] = nx.is_directed_acyclic_graph(graph)
    try:
        metrics['longest_path'] = len(nx.dag_longest_path(graph))
    except:
        metrics['longest_path'] = None
    
    # 3. Structural Analysis
    metrics['strongly_connected_components'] = nx.number_strongly_connected_components(graph)
    metrics['weakly_connected_components'] = nx.number_weakly_connected_components(graph)
    
    # 4. Node Importance Metrics
    metrics['betweenness_centrality'] = nx.betweenness_centrality(graph)
    metrics['in_degree_centrality'] = nx.in_degree_centrality(graph)
    metrics['out_degree_centrality'] = nx.out_degree_centrality(graph)
    
    # 5. Sparsity Analysis
    metrics['sparsity'] = 1.0 - (np.count_nonzero(adjacency_matrix) / 
                                (adjacency_matrix.shape[0] * adjacency_matrix.shape[1]))
    
    # 6. Correlation Analysis
    correlation_matrix = data.corr()
    edge_correlations = []
    for edge in graph.edges():
        source, target = edge
        edge_correlations.append(abs(correlation_matrix.loc[source, target]))
    
    if edge_correlations:
        metrics['avg_edge_correlation'] = np.mean(edge_correlations)
        metrics['max_edge_correlation'] = np.max(edge_correlations)
        metrics['min_edge_correlation'] = np.min(edge_correlations)
    
    return metrics

def print_evaluation_results(metrics):
    """
    Print the evaluation results in a formatted way.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
    """
    print("\n=== DAG Structure Evaluation ===")
    
    print("\n1. Basic Properties:")
    print(f"Number of nodes: {metrics['num_nodes']}")
    print(f"Number of edges: {metrics['num_edges']}")
    print(f"Average degree: {metrics['avg_degree']:.2f}")
    print(f"Graph density: {metrics['density']:.2f}")
    
    print("\n2. Topological Properties:")
    print(f"Is DAG: {metrics['is_dag']}")
    print(f"Longest path length: {metrics['longest_path']}")
    
    print("\n3. Connectivity:")
    print(f"Strongly connected components: {metrics['strongly_connected_components']}")
    print(f"Weakly connected components: {metrics['weakly_connected_components']}")
    
    print("\n4. Sparsity:")
    print(f"Sparsity ratio: {metrics['sparsity']:.2f}")
    
    print("\n5. Edge Correlation Analysis:")
    if 'avg_edge_correlation' in metrics:
        print(f"Average edge correlation: {metrics['avg_edge_correlation']:.2f}")
        print(f"Max edge correlation: {metrics['max_edge_correlation']:.2f}")
        print(f"Min edge correlation: {metrics['min_edge_correlation']:.2f}")
    
    print("\n6. Node Centrality Measures:")
    print("\nTop 3 nodes by betweenness centrality:")
    sorted_bc = sorted(metrics['betweenness_centrality'].items(), 
                      key=lambda x: x[1], reverse=True)[:3]
    for node, score in sorted_bc:
        print(f"{node}: {score:.3f}")

def visualize_graph_with_weights(graph, metrics, adjacency_matrix):
    """
    Enhanced visualization of the graph with node sizes based on centrality
    and edge weights from adjacency matrix shown as labels.
    
    Args:
        graph (nx.DiGraph): The discovered DAG
        metrics (dict): Dictionary containing evaluation metrics
        adjacency_matrix (np.ndarray): Adjacency matrix from DAGMA
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, k=1, iterations=50)
    
    # Node sizes based on betweenness centrality
    node_sizes = [v * 5000 for v in metrics['betweenness_centrality'].values()]
    
    # Calculate edge weights from adjacency matrix
    edge_weights = {}
    edge_labels = {}
    for edge in graph.edges():
        source, target = edge
        # Get indices for adjacency matrix
        source_idx = list(graph.nodes()).index(source)
        target_idx = list(graph.nodes()).index(target)
        # Get weight from adjacency matrix
        weight = abs(adjacency_matrix[source_idx, target_idx])
        edge_weights[edge] = weight
        edge_labels[edge] = f'{weight:.2f}'  # Format to 2 decimal places
    
    # Draw the graph
    nx.draw(graph, pos,
            node_color='blue',
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            font_weight='bold',
            arrows=True,
            edge_color='gray',
            arrowsize=20,
            width=[w * 2 for w in edge_weights.values()])  # Edge thickness based on weight
    
    # Add edge labels
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title("Causal Graph with Adjacency Matrix Weights")
    plt.show()

    # Print additional edge information
    print("\nEdge Weights (Adjacency Matrix):")
    for edge, weight in sorted(edge_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"{edge[0]} → {edge[1]}: {weight:.3f}")


