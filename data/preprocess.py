import torch
from torch_geometric.data import Data

def create_temporal_pyg_data(features, targets, edge_index, edge_attr, window_size=5):
    """
    Converts raw time-series matrices into Spatio-Temporal Graph objects
    using a sliding window approach. 
    """
    data_list = []
    
    # 1. Convert static graph topology to tensors ONCE.
    # The Logic: The physical cables don't change from minute to minute.
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    
    # 2. Iterate through the simulation, stopping early so the window doesn't fall off the end
    for i in range(len(features) - window_size):
        
        # 3. Extract the sequence of timesteps for the features
        # Shape starts as: (window_size, num_nodes, num_features)
        window_features = features[i : i + window_size]
        
        # The Logic: We transpose the tensor to (num_nodes, window_size, num_features).
        # This groups the entire time history by node, which is how PyTorch Geometric 
        # expects to process node-level temporal data.
        x = torch.tensor(window_features, dtype=torch.float32).transpose(0, 1)
        
        # 4. The Target is the physical state at the immediate next timestep
        y = torch.tensor(targets[i + window_size], dtype=torch.float32)
        
        # 5. Assemble the Spatio-Temporal PyG Data object
        data = Data(
            x=x, 
            edge_index=edge_index_tensor, 
            edge_attr=edge_attr_tensor, 
            y=y
        )
        data_list.append(data)
        
    return data_list

def create_pyg_data(features, targets, edge_index, edge_attr):
    """
    Acts as the bridge between raw Numpy simulations and PyTorch Geometric.
    Packs the physical AC steady-state features, targets, and transmission 
    line attributes (Resistance/Reactance) into GPU-ready Graph objects.
    """
    data_list = []
    
    # 1. Convert static graph topology to tensors ONCE outside the loop.
    # The Logic: The physical cables don't change from minute to minute.
    # edge_index must be 'long' (integers) for indexing, 
    # edge_attr must be 'float32' for mathematical attention weights.
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    
    # 2. Iterate through every single timestep in the simulation
    for i in range(len(features)):
        # Convert dynamic loads (P, Q) and targets (|V|, Angle) to float32 tensors
        x = torch.tensor(features[i], dtype=torch.float32)
        y = torch.tensor(targets[i], dtype=torch.float32)
        
        # 3. Assemble the ultimate PyG Data object
        data = Data(
            x=x, 
            edge_index=edge_index_tensor, 
            edge_attr=edge_attr_tensor, 
            y=y
        )
        data_list.append(data)
        
    return data_list

def train_test_split(data_list, train_ratio=0.8):
    """
    Chronological split to prevent future data from leaking into the training set.
    """
    n_train = int(len(data_list) * train_ratio)
    return data_list[:n_train], data_list[n_train:]