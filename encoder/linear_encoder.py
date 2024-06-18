import torch
import torch.nn as nn


class LinearEncoder(torch.nn.Module):
    kernel_type = None
    def __init__(
            self,
            in_channel,
            hidden_channel,
            var_name,
            model_type='linear',
            num_layers=None,
            norm_type=None,
    ):
        super().__init__()
        self.var_name = var_name
        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(in_channel)
        else:
            self.raw_norm = None

        activation = nn.ReLU
        if model_type == 'mlp':
            layers = []
            if num_layers == 1:
                layers.append(nn.Linear(in_channel, hidden_channel))
                layers.append(activation())
            else:
                layers.append(nn.Linear(in_channel, 2 * hidden_channel))
                layers.append(activation())
                for _ in range(num_layers - 2):
                    layers.append(nn.Linear(2 * hidden_channel, 2 * hidden_channel))
                    layers.append(activation())
                layers.append(nn.Linear(2 * hidden_channel, hidden_channel))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(in_channel, hidden_channel)

    def forward(self, batch):
        var = getattr(batch, self.var_name)
        if self.raw_norm:
            var = self.raw_norm(var)
        var = self.pe_encoder(var)
        if self.var_name == 'x':
            batch.x = var
        else:
            h = batch.x
            batch.x = torch.cat((h, var), 1)

        return batch

