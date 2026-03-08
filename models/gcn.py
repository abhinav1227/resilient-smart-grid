import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm

class PowerSTGAT(nn.Module):
    """
    Version 2.1: Spatio-Temporal Graph Attention Network (STGAT).
    Phase 1 (LSTM): Encodes physical inertia over a sliding window of time.
    Phase 2 (GAT): Distributes that inertia across the spatial topology using Ohm's Law.
    """
    def __init__(self, in_channels=2, hidden_dim=32, out_channels=2, dropout=0.2, edge_dim=2):
        super().__init__()
        
        # --- TEMPORAL COMPONENT ---
        # batch_first=True because our data shape is (num_nodes, window_size, features)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_dim, batch_first=True)
        
        # --- SPATIAL COMPONENT ---
        # The GAT now receives the 'hidden_dim' from the LSTM as its input
        self.conv1 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        self.norm1 = LayerNorm(hidden_dim)
        
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        self.norm2 = LayerNorm(hidden_dim)
        
        # Output Layer: Maps to [Magnitude, Angle]
        self.conv3 = GATConv(hidden_dim, out_channels, edge_dim=edge_dim, add_self_loops=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # --- PHASE 1: TEMPORAL INERTIA (LSTM) ---
        # x enters with shape: (total_nodes, window_size=5, features=2)
        lstm_out, (hn, cn) = self.lstm(x)
        
        # We extract the final hidden state 'hn' from the LSTM sequence.
        # This vector now contains the "momentum" of the last 5 timesteps.
        # Shape becomes: (total_nodes, hidden_dim)
        x = hn[-1] 
        
        # --- PHASE 2: SPATIAL PHYSICS (GAT) ---
        # We pass the temporal momentum into the physical grid topology
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        
        return x
    
class PowerGAT(nn.Module):
    """
    Version 2: Physics-Informed Graph Attention Network.
    Uses edge attributes (Resistance and Reactance) to dynamically weigh 
    the electrical influence of neighboring buses.
    """
    def __init__(self, in_channels=2, hidden_dim=32, out_channels=2, dropout=0.2, edge_dim=2):
        super().__init__()
        
        # Layer 1: We tell the GAT to expect 2-dimensional edge features (R and X)
        self.conv1 = GATConv(in_channels, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        self.norm1 = LayerNorm(hidden_dim)
        
        # Layer 2
        self.conv2 = GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False)
        self.norm2 = LayerNorm(hidden_dim)
        
        # Output Layer: Maps to [Magnitude, Angle]
        self.conv3 = GATConv(hidden_dim, out_channels, edge_dim=edge_dim, add_self_loops=False)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # We now extract x, edge_index, AND the new physical edge_attr
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Layer 1
        # Pass the physics features into the attention mechanism via edge_attr
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output Layer
        x = self.conv3(x, edge_index, edge_attr=edge_attr)
        return x