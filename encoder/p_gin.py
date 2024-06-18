from torch import nn
import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
import torch.nn.functional as F
from conv.gin_conv import MyGINConv


class PGIN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            s_channels,
            num_layers,
            dropout,
            return_emb=False,
    ):
        super(PGIN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(MyGINConv(nn=Sequential(Linear(in_channels, hidden_channels),
                                                BatchNorm1d(hidden_channels), ReLU(),
                                                Linear(hidden_channels, hidden_channels), ReLU()),
                                  train_eps=True)
                          )
        for _ in range(num_layers - 1):
            self.convs.append(MyGINConv(nn=Sequential(Linear(hidden_channels, hidden_channels),
                                                    BatchNorm1d(hidden_channels), ReLU(),
                                                    Linear(hidden_channels, hidden_channels), ReLU()),
                                      train_eps=True)
                              )
        # self.lin1 = Linear(hidden_channels * num_layers, hidden_channels * num_layers)
        if not return_emb:
            self.lin2 = Linear(hidden_channels * num_layers, s_channels)

        self.dropout = dropout
        self.num_layers = num_layers
        self.return_emb = return_emb
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if not self.return_emb:
            self.lin2.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        h = torch.cat(xs, dim=1)
        if self.return_emb:
            return h
        h = self.lin2(h)
        return h
