import torch
import numpy as np

def get_edge_index(net):
    """
    Logic: Converts Pandapower line table into a bidirectional PyG edge_index.
    """
    f = net.line.from_bus.values
    t = net.line.to_bus.values
    
    # Concatenate [from->to] and [to->from] for undirected grid physics
    edge_index = np.vstack([
        np.concatenate([f, t]),
        np.concatenate([t, f])
    ])
    return torch.tensor(edge_index, dtype=torch.long)