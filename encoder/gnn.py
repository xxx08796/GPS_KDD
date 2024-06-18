from torch import nn
from encoder.p_gin import PGIN
from encoder.s_gin import SGIN
from encoder.sage import GraphSAGE


class GraphEncoder(nn.Module):
    def __init__(
            self,
            gnn_model,
            in_channels,
            hidden_channels,
            s_channels,
            num_layers,
            dropout,
            return_emb,
    ):
        super(GraphEncoder, self).__init__()
        if gnn_model == 'S_GIN':
            self.gnn = SGIN(in_channels, hidden_channels, s_channels, num_layers, dropout, return_emb)
        elif gnn_model == 'P_GIN':
            self.gnn = PGIN(in_channels, hidden_channels, s_channels, num_layers, dropout, return_emb)
        elif gnn_model == "SAGE":
            self.gnn = GraphSAGE(in_channels, hidden_channels, s_channels, num_layers, dropout, return_emb)
        else:
            raise NotImplementedError
        self.gnn_model = gnn_model

    def forward(self, batch, edge_weight=None, pooling_mask=None):
        if self.gnn_model == 'S_GIN':
            x = self.gnn(batch, edge_weight, pooling_mask)
        elif self.gnn_model in ['P_GIN', 'SAGE']:
            if hasattr(batch, 'adj_t'):
                x = self.gnn(batch.x, batch.adj_t, edge_weight)
            else:
                x = self.gnn(batch.x, batch.edge_index, edge_weight)
        else:
            raise NotImplementedError
        return x
