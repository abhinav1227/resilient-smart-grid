import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm

class PowerGCN(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=32, out_channels=1, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.norm1 = LayerNorm(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output Layer
        x = self.conv3(x, edge_index)
        return x