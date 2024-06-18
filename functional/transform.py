import copy
import networkx
import torch
from torch_geometric.transforms import AddRandomWalkPE
from torch_geometric.utils import k_hop_subgraph, to_networkx
from tqdm import tqdm

from functional.tools import calculate_embed


def transform_in_memory(sub_graph, whole_graph, opt):
    whole_graph = copy.deepcopy(whole_graph).cpu()
    walk_length = opt.in_channels["rwse_in"] if "rwse_in" in opt.in_channels else None
    if not opt.pe_types: return sub_graph
    metric, rwse = None, None
    netx_graph = to_networkx(whole_graph, to_undirected=True)
    if "mix" in opt.pe_types:
        properties = opt.pe_types["mix"]
        metric = []
        if 'degree' in properties:
            deg = networkx.degree(netx_graph)
            metric.append(torch.tensor(list(dict(deg).values())).reshape(-1, 1))
        if 'embed' in properties:
            embed = calculate_embed(netx_graph)
            metric.append(torch.tensor(embed).reshape(-1, 1))
        if 'ave_nei_deg' in properties:
            ave_nei_deg = networkx.average_neighbor_degree(netx_graph, source='out', target='out')
            metric.append(torch.tensor(list(ave_nei_deg.values())).reshape(-1, 1))
        if 'local_cc' in properties:
            local_cc = networkx.clustering(netx_graph)
            metric.append(torch.tensor(list(local_cc.values())).reshape(-1, 1))
        metric = torch.cat(metric, dim=-1)
        mean_f = torch.mean(metric, dim=0)
        std_f = torch.std(metric, dim=0)
        metric = (metric - mean_f) / std_f
        metric = metric.to(opt.device)

    if 'rwse' in opt.pe_types:
        whole_graph = AddRandomWalkPE(walk_length=walk_length, attr_name='rwse')(whole_graph.cpu())
        rwse = whole_graph.rwse

    data_list = []
    whole_graph.to(opt.device)
    for k in tqdm(range(0, whole_graph.num_nodes)):
        ego_net = sub_graph.get(k)
        p = k_hop_subgraph(k, opt.hops, whole_graph.edge_index, relabel_nodes=True, num_nodes=whole_graph.num_nodes)
        ego_glo_nodes = p[0]
        assert ego_glo_nodes.shape[0]==ego_net.num_nodes
        if 'mix' in opt.pe_types:
            ego_net.mix = metric[ego_glo_nodes].to(torch.float32).cpu()
        if 'rwse' in opt.pe_types:
            ego_net.rwse = rwse[ego_glo_nodes.cpu()]

        data_list.append(ego_net)

    data_list = list(filter(None, data_list))
    sub_graph._indices = None
    sub_graph._data_list = data_list
    sub_graph.data, sub_graph.slices = sub_graph.collate(data_list)
