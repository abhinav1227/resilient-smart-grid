import torch

def get_edge_index(net):
    """
    Extracts the graph topology from a pandapower network and formats it 
    as an undirected PyTorch Geometric edge_index tensor.
    """
    sources, targets = [], []
    for _, line in net.line.iterrows():
        # Forward connection
        sources.append(int(line.from_bus))
        targets.append(int(line.to_bus))
        
        # Reverse connection (power grids are undirected graphs)
        sources.append(int(line.to_bus))
        targets.append(int(line.from_bus))
        
    return torch.tensor([sources, targets], dtype=torch.long)