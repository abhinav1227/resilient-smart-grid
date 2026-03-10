import torch
from torch_geometric.data import Data

def create_temporal_pyg_data(features, targets, edge_index, edge_attr, window_size=5):
    """
    Logic: Transforms raw arrays into Spatio-Temporal Graph Objects.
    """
    data_list = []
    ei_tensor = torch.tensor(edge_index, dtype=torch.long)
    ea_tensor = torch.tensor(edge_attr, dtype=torch.float32)
    
    for i in range(len(features) - window_size):
        # Window shape: (Nodes, Window, Features)
        x = torch.tensor(features[i : i + window_size], dtype=torch.float32).transpose(0, 1)
        y = torch.tensor(targets[i + window_size], dtype=torch.float32)
        
        data_list.append(Data(x=x, edge_index=ei_tensor, edge_attr=ea_tensor, y=y))
        
    return data_list

def train_test_split(data_list, train_ratio=0.8):
    n_train = int(len(data_list) * train_ratio)
    return data_list[:n_train], data_list[n_train:]