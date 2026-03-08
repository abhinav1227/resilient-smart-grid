import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
import logging
import os

logger = logging.getLogger(__name__)

def explain_attack(model, data, edge_index, edge_attr, device, num_buses, save_path='results/explanation.png'):
    """
    Uses Integrated Gradients to map attack attribution back to physical grid nodes.
    Updated for Spatio-Temporal GATs with physics-informed edge attributes.
    """
    def model_forward(x_batch):
        # THE LOGIC: We must include edge_attr here so the GAT layers don't crash
        # when calculating physical attention weights!
        from torch_geometric.data import Data
        d = Data(
            x=x_batch[0], 
            edge_index=edge_index.to(device), 
            edge_attr=edge_attr.to(device) # INJECTED PHYSICS
        )
        out = model(d.to(device))
        return out.mean().reshape(1)

    baseline = torch.zeros_like(data.x).unsqueeze(0).to(device)
    ig = IntegratedGradients(model_forward)
    
    logger.info("Computing Integrated Gradients...")
    attributions, _ = ig.attribute(
        data.x.unsqueeze(0), 
        baseline,
        return_convergence_delta=True
    )
    
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    
    # THE LOGIC: The STGAT data shape is (14, 5, 2) -> (Nodes, Timesteps, Features).
    # We use .reshape(num_buses, -1) to flatten the Time and Feature dimensions 
    # into a single 1D array per node, ensuring we get exactly 14 scalar values.
    node_importance = np.sum(np.abs(attributions.reshape(num_buses, -1)), axis=1)

    # Build the dynamic graph
    G = nx.Graph()
    edges = edge_index.cpu().t().numpy()
    G.add_edges_from(edges)
    
    # Ensure all nodes are present, even if isolated
    G.add_nodes_from(range(num_buses))

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_importance,
        cmap='hot', 
        node_size=600
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.colorbar(nodes, label='Attribution (Feature Importance)')
    plt.title(f'Attack Detection Heatmap ({num_buses} Buses)')
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Explanation heatmap saved to {save_path}")
    return node_importance

def explain_attack1(model, data, edge_index, device, num_buses, save_path='results/explanation.png'):
    """
    Uses Integrated Gradients to map attack attribution back to physical grid nodes.
    """
    def model_forward(x_batch):
        # Captum passes batched tensors; we must reconstruct the PyG Data object
        from torch_geometric.data import Data
        d = Data(x=x_batch[0], edge_index=edge_index.to(device))
        out = model(d.to(device))
        return out.mean().reshape(1)

    baseline = torch.zeros_like(data.x).unsqueeze(0).to(device)
    ig = IntegratedGradients(model_forward)
    
    logger.info("Computing Integrated Gradients...")
    attributions, _ = ig.attribute(
        data.x.unsqueeze(0), 
        baseline,
        return_convergence_delta=True
    )
    
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    
    # Take absolute sum: large negative or positive changes are both highly important
    node_importance = np.sum(np.abs(attributions), axis=1)

    # Build the dynamic graph
    G = nx.Graph()
    edges = edge_index.cpu().t().numpy()
    G.add_edges_from(edges)
    
    # Ensure all nodes are present, even if isolated
    G.add_nodes_from(range(num_buses))

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_importance,
        cmap='hot', 
        node_size=600
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.colorbar(nodes, label='Attribution (Feature Importance)')
    plt.title(f'Attack Detection Heatmap ({num_buses} Buses)')
    plt.axis('off')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    logger.info(f"Explanation heatmap saved to {save_path}")
    return node_importance