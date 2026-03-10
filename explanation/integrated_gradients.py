import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients
from torch_geometric.data import Data
import os

def explain_attack(model, data, edge_index, edge_attr, device, save_path='results/explanation.png'):
    """
    Logic: Uses Captum IG to attribute model predictions back to specific buses.
    """
    model.eval()
    num_buses = data.x.shape[0]

    def model_forward(x_input):
        # x_input shape: [Batch, Nodes, Window, Feat]
        # Captum adds a batch dimension, so we select x_input[0]
        temp_data = Data(x=x_input[0], edge_index=edge_index.to(device), edge_attr=edge_attr.to(device))
        return model(temp_data).mean().reshape(1)

    ig = IntegratedGradients(model_forward)
    # Attribution shape matches input: (Nodes, Window, Feat)
    attributions = ig.attribute(data.x.unsqueeze(0).to(device), internal_batch_size=1)

    # Flatten Time/Features to get one importance score per Node
    node_importance = attributions.squeeze(0).abs().sum(dim=(1, 2)).cpu().detach().numpy()

    # 2. Build the graph safely
    G = nx.Graph()
    
    # THE FIX: Explicitly add all nodes first so the graph size is consistent
    G.add_nodes_from(range(num_buses))
    
    # Then add the edges
    edges = edge_index.cpu().t().numpy()
    G.add_edges_from(edges)

    # 3. Visualization
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Now len(G.nodes) will be 14, matching len(node_importance)
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_importance,
        cmap='hot', 
        node_size=600
    )
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.colorbar(nodes, label='Attribution (Feature Importance)')
    plt.title(f'Attack Detection Heatmap ({num_buses} Buses)')
    plt.axis('off')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    return node_importance