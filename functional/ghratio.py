import torch
from torch_geometric.utils import subgraph, degree
from torch_scatter import scatter


def node_level_homo(edge_index, label):
    row, col = edge_index
    out = torch.zeros(row.size(0), device=row.device)
    out[label[row] == label[col]] = 1.
    out = scatter(out, col, 0, dim_size=label.size(0), reduce='mean')
    return out


def node_level_struc_ratio(edge_index, label, torr):
    non_missing = torch.where(label != -1)[0]
    new_edge_index = subgraph(non_missing, edge_index, relabel_nodes=True, num_nodes=label.shape[0])[0]
    new_label = label[non_missing]
    temp = struc_ratio(degree(new_edge_index[0], num_nodes=non_missing.shape[0]), new_label, torr)
    s_weight = torch.zeros(label.shape[0])
    s_weight[non_missing] = temp
    return s_weight.unsqueeze(1)


def struc_ratio(properties, label, tolerance=0):
    properties = properties.cpu()
    label = label.cpu()
    mask = ~torch.eye(properties.size(0)).bool()
    properties_same = torch.abs(properties.unsqueeze(1) - properties.unsqueeze(0)) <= tolerance
    # properties_same = torch.eq(properties.unsqueeze(1), properties.unsqueeze(0))
    properties_same = properties_same[mask].reshape(properties_same.size(0), properties_same.size(0) - 1)

    label_same = torch.eq(label.unsqueeze(1), label.unsqueeze(0))
    label_same = label_same[mask].reshape(label_same.size(0), label_same.size(0) - 1)

    properties_label_same = properties_same & label_same
    same_count = torch.sum(properties_label_same.float(), dim=-1)
    total_count = torch.sum(properties_same.float(), dim=-1)

    ratios = same_count / total_count
    return ratios