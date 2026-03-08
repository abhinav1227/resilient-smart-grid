import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LayerNorm, JumpingKnowledge

class DeepPowerSTGAT(nn.Module):
    """
    Version 3: Deep Spatio-Temporal GAT with Jumping Knowledge.
    Designed to conquer massive topologies (IEEE-118) by expanding the 
    receptive field without triggering mathematical oversmoothing.
    """
    # Notice we default to 5 layers and a larger hidden_dim (64)
    def __init__(self, in_channels=2, hidden_dim=64, out_channels=2, dropout=0.2, edge_dim=2, num_layers=5):
        super().__init__()
        self.num_layers = num_layers
        
        # --- TEMPORAL COMPONENT ---
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_dim, batch_first=True)
        
        # --- SPATIAL COMPONENT (DEEP GAT) ---
        # We use ModuleList to dynamically build a deep stack of physics layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim, add_self_loops=False))
            self.norms.append(LayerNorm(hidden_dim))
            
        # --- THE JUMPING KNOWLEDGE BRIDGE ---
        # 'cat' concatenates the outputs from all 5 layers into one massive vector. 
        self.jk = JumpingKnowledge(mode='cat')
        
        # --- DECODER COMPONENT ---
        # Because JK concatenates 5 layers of size 64, the input here is 5 * 64 = 320.
        # We use a Linear layer to compress that massive physics context back down to Voltage and Angle.
        self.fc1 = nn.Linear(hidden_dim * num_layers, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_channels)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # --- PHASE 1: TEMPORAL INERTIA ---
        lstm_out, (hn, cn) = self.lstm(x)
        x = hn[-1] 
        
        # --- PHASE 2: SPATIAL PHYSICS (DEEP GAT) ---
        xs = [] # List to store the output of every single layer
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            xs.append(x) # Save this layer's physical calculation
            
        # --- PHASE 3: JUMPING KNOWLEDGE ---
        # The network "remembers" the local 1-hop physics and the deep 5-hop physics simultaneously
        x_jk = self.jk(xs) 
        
        # --- PHASE 4: PREDICTION ---
        x = F.relu(self.fc1(x_jk))
        x = self.fc2(x)
        
        return x

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
    

