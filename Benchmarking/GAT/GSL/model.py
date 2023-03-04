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


        

    def forward(self, x1, edge_index1, batch1, x2, edge_index2, batch2):
        
        for i, conv in enumerate(self.convs):
            x1 = conv(x1, edge_index1)
            if i != self.num_layers - 1:
                x1 = F.elu(x1)
                x1 = F.dropout(x1, p=0.5, training=self.training)
        
        for i, conv in enumerate(self.convs):
            x2 = conv(x2, edge_index2)
            if i != self.num_layers - 1:
                x2 = F.elu(x2)
                x2 = F.dropout(x2, p=0.5, training=self.training)
        
        x1 = self.MLP(x1)
        x2 = self.MLP(x2)

        x1 = global_add_pool(x1, batch1)          # sum pool
        x2 = global_add_pool(x2, batch2)          # sum pool

        # dot product ie x1_T * x2 to get scalar
        # x = torch.mm(x1, x2.T)
        # or x1.T * x2
        x = torch.mm(x1.T, x2)          # TODO check if this is correct

        return x