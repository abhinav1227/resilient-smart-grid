import torch
from torch_geometric.data import Data

def create_pyg_data(features, targets, edge_index):
    """Converts raw numpy arrays into PyTorch Geometric Data objects."""
    data_list = []
    for i in range(len(features)):
        data = Data(
            x=torch.tensor(features[i], dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(targets[i], dtype=torch.float)
        )
        data_list.append(data)
    return data_list

def train_test_split(data_list, train_ratio=0.8):
    """
    Chronological split to prevent future data from leaking into the training set.
    """
    n_train = int(len(data_list) * train_ratio)
    return data_list[:n_train], data_list[n_train:]