import torch
from torch import nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
from conv.sage_conv import MySAGEConv


class GraphSAGE(nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            s_channels,
            num_layers,
            dropout,
            return_emb,
    ):
        super(GraphSAGE, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(MySAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(MySAGEConv(hidden_channels, hidden_channels))
        self.norms = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        if not return_emb:
            self.norms.append(nn.BatchNorm1d(hidden_channels))
            self.linear = torch.nn.Linear(hidden_channels, s_channels, bias=False)

        self.act = nn.ReLU(hidden_channels)
        self.dropout = dropout
        self.num_layers = num_layers
        self.return_emb = return_emb
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()
        if not self.return_emb:
            self.linear.reset_parameters()

    def forward(self, x, edge_index, edge_weight):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if i == self.num_layers - 1 and self.return_emb:
                return x
            if i == self.num_layers - 1:
                break
            x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.linear(x)
        return x
