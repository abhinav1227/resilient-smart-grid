import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm, JumpingKnowledge

class DeepPowerSTGAT(nn.Module):
    """
    Logic: Spatio-Temporal GAT with Jumping Knowledge.
    Architecture:
    1. LSTM: Captures temporal inertia from the 5-step window.
    2. GAT Stack: Propagates that inertia through the physical grid.
    3. JK Bridge: Prevents oversmoothing by combining local and global features.
    """
    def __init__(self, in_channels=2, hidden_dim=64, out_channels=2, dropout=0.2, edge_dim=2, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        
        # 1. Temporal Component (LSTM)
        # Input: (Nodes, Window, Feat) -> Output: (Nodes, Hidden)
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_dim, batch_first=True)
        
        # 2. Spatial Component (Stack of GATs)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False))
            self.norms.append(LayerNorm(hidden_dim))
            
        # 3. Jumping Knowledge Bridge
        self.jk = JumpingKnowledge(mode='cat')
        
        # 4. Final Decoder
        self.fc1 = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Phase 1: Temporal Encoding
        # We only care about the final hidden state (hn) representing the full window
        _, (hn, _) = self.lstm(x)
        x = hn[-1] 
        
        # Phase 2: Spatial Propagation
        xs = []
        for i in range(self.num_layers):
            x = F.relu(self.norms[i](self.convs[i](x, edge_index, edge_attr=edge_attr)))
            x = self.dropout(x)
            xs.append(x)
            
        # Phase 3: Aggregation & Prediction
        x = self.jk(xs)
        x = F.relu(self.fc1(x))
        return self.fc2(x)