import torch
import torch.nn as nn
from encoder.gnn import GraphEncoder
from encoder.linear_encoder import LinearEncoder
from functional.ghratio import node_level_homo, node_level_struc_ratio


class AttrAtk(nn.Module):
    def __init__(
            self,
            gnn,
            in_channels,
            hidden_channels,
            gnn_layers,
            num_classes,
            dropout,
            pe_types,
            num_nodes,
            torr,
    ):
        super(AttrAtk, self).__init__()
        feature_encoders = []
        if 'trans_x' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['x_in'], hidden_channels['x_hidden'], 'x'))
            gnn_in = hidden_channels['x_hidden']
        else:
            gnn_in = in_channels['x_in']
        if 'mix' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['mix_in'], hidden_channels['mix_hidden'], 'mix'))
            gnn_in += hidden_channels['mix_hidden']
        if 'rwse' in pe_types:
            feature_encoders.append(LinearEncoder(in_channels['rwse_in'], hidden_channels['rwse_hidden'], 'rwse'))
            gnn_in += hidden_channels['rwse_hidden']
        self.feature_encoders = nn.Sequential(*feature_encoders)

        p_gnn, s_gnn = gnn[0], gnn[1]
        self.p_encoder = GraphEncoder(
            gnn_model=p_gnn,
            in_channels=in_channels['x_in'],
            hidden_channels=hidden_channels['gnn_hidden'] * 4,
            s_channels=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            return_emb=False,
        )
        self.s_encoder = GraphEncoder(
            gnn_model=s_gnn,
            in_channels=gnn_in,
            hidden_channels=hidden_channels['gnn_hidden'],
            s_channels=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            return_emb=False,
        )
        self.p_gnn, self.s_gnn = p_gnn, s_gnn
        self.norms = nn.ModuleList()
        self.norms.append(nn.BatchNorm1d(num_classes))
        self.norms.append(nn.BatchNorm1d(num_classes))
        self.p_weight = torch.ones(num_nodes).unsqueeze(1) * 0.5
        self.s_weight = torch.ones(num_nodes).unsqueeze(1) * 0.5
        self.torr = torr

    def forward(self, batch_p, batch_s, batch_idx):
        x_p = self.p_encoder(batch_p)[batch_idx]
        x_p = self.norms[0](x_p)
        batch_s = self.feature_encoders(batch_s)
        x_s = self.s_encoder(batch_s)
        x_s = self.norms[1](x_s)
        x = self.p_weight[batch_idx] * x_p + self.s_weight[batch_idx] * x_s
        return x

    def update_weights(self, edge_index, label):
        p_weight = node_level_homo(edge_index, label).unsqueeze(1)
        s_weight = node_level_struc_ratio(edge_index, label, self.torr).to(p_weight.device)
        s_weight = torch.masked_fill(s_weight, torch.isnan(s_weight), 0)

        tol = p_weight + s_weight
        p_weight = torch.where(tol != 0, torch.div(p_weight, tol), 0.5)
        s_weight = torch.where(tol != 0, torch.div(s_weight, tol), 0.5)
        self.p_weight = p_weight
        self.s_weight = s_weight