import torch
from torch.nn import Linear, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_layers, num_classes):
        super().__init__()
        torch.manual_seed(1234567)
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_channels))
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        

        self.MLP = torch.nn.Sequential(Linear(hidden_channels, hidden_channels),
                                        ReLU(),
                                        Linear(hidden_channels, num_classes))


        

    def forward(self, x, edge_index, batch):
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        x = self.MLP(x)

        x = global_add_pool(x, batch)          # sum pool

        return x