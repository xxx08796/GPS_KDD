from typing import Optional
from torch import nn
import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn.functional as F
from torch_sparse import SparseTensor

from conv.gin_conv import MyGINConv
from functional.pooling import my_global_add_pool, center_node_pooling


class SGIN(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            hidden_channels,
            s_channels,
            num_layers,
            dropout,
            return_emb=False,
            center_add_pooling=True,
    ):
        super(SGIN, self).__init__()
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
        if not center_add_pooling:
            self.lin2 = Linear(hidden_channels * num_layers, s_channels)
        else:
            self.lin2 = Linear(hidden_channels * num_layers* 2, s_channels)
        self.center_add_pooling = center_add_pooling

        self.dropout = dropout
        self.num_layers = num_layers
        self.return_emb = return_emb
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, batch_, edge_weight, pooling_mask):
        # x, edge_index, batch = batch_.x, batch_.adj_t, batch_.batch
        x, batch = batch_.x, batch_.batch
        edge_index = batch_.adj_t if hasattr(batch_, 'adj_t') else batch_.edge_index
        try:
            mapping = batch_.mapping
        except:
            pass
        # Node embeddings
        xs = []
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if i < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)

        # Graph-level readout
        if not self.center_add_pooling:
            for i in range(len(xs)):
                xs[i] = my_global_add_pool(xs[i], batch, mask=pooling_mask)
        else:
            for i in range(len(xs)):
                emb_add = my_global_add_pool(xs[i], batch, mask=pooling_mask)
                emb_center = center_node_pooling(xs[i], batch, mapping)
                xs[i] = torch.cat((emb_add, emb_center),dim=-1)

        # Concatenate graph embeddings
        h = None
        for x in xs:
            if h is None:
                h = x
            else:
                h = torch.cat((h, x), dim=1)

        if self.return_emb:
            return h
        h = self.lin2(h)
        return h
