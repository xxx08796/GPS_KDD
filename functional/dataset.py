import os
from torch import nn, Tensor
from sklearn.model_selection import train_test_split as sk_split
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T
from torch_geometric.io.planetoid import index_to_mask
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


class LoadSubgraph(InMemoryDataset):
    def __init__(
            self,
            root,
            data_name,
            sens_attr,
            hops=2,
            device=None,
            transform=None,
            pre_transform=None,
            pre_filter=None
    ):
        self.root = root
        self.data_name = data_name
        self.sens_attr = sens_attr
        self.hops = hops
        self.device = device
        super(LoadSubgraph, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        if self.data_name.startswith('pokec'):
            return ['region_job_2.csv', 'region_job_2_relationship.txt', 'region_job.csv',
                    'region_job_relationship.txt']
        elif self.data_name.lower() in ['nba']:
            return ['nba.csv', 'nba_relationship.txt']
        else:
            raise NotImplementedError

    @property
    def processed_file_names(self):
        return "{}_{}_{}hops.pt".format(self.data_name, self.sens_attr, self.hops)

    def download(self):
        raise NotImplementedError('No download allowed')

    def process(self):
        whole_graph = load_data(data_name=self.data_name, target=self.sens_attr, path=self.raw_dir, split=False)
        if self.device is not None: whole_graph = whole_graph.to(self.device)
        data_list = []
        for k in tqdm(range(0, whole_graph.num_nodes)):
            p = k_hop_subgraph(k, self.hops, whole_graph.edge_index, relabel_nodes=True,
                               num_nodes=whole_graph.num_nodes)
            ego_edge, nodes_involved, ego_map = p[1], p[0], p[2]
            ego_x = whole_graph.x[nodes_involved, :]
            ego_y = whole_graph.y[k]
            ego_net = Data(x=ego_x, edge_index=ego_edge, y=ego_y, mapping=ego_map)
            data_list.append(ego_net.cpu())

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx):
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data, idx

        else:
            return self.index_select(idx)


def load_data(data_name, target, train_ratio=0.1, path=None, split=True):
    if data_name.startswith('pokec'):
        data = load_pokec(dataset=data_name, target_attr=target, train_ratio=train_ratio, path=path, split=split)
    elif data_name.lower() == 'nba':
        data = load_nba(data_name=data_name, target_attr=target, train_ratio=train_ratio, path=path, split=split)
    else:
        raise NotImplementedError
    return data


def load_nba(data_name, target_attr, train_ratio, path, split):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(data_name)))
    # sensitive attribute
    sens = idx_features_labels[target_attr].values
    sens = torch.LongTensor(sens)
    # train test split
    non_missing = np.where(sens.numpy() >= 0)[0]
    non_missing_sens = sens.numpy()[non_missing]
    res = sk_split(non_missing, non_missing_sens, train_size=train_ratio, stratify=non_missing_sens, random_state=0)
    idx_train, idx_test = res[0], res[1]
    idx_train = index_to_mask(torch.LongTensor(idx_train), size=sens.shape[0])
    idx_test = index_to_mask(torch.LongTensor(idx_test), size=sens.shape[0])
    # feature matrix
    header = list(idx_features_labels.columns)
    for attr in ['user_id', 'country', 'SALARY']: header.remove(attr)
    features = np.array(idx_features_labels[header])
    features = torch.FloatTensor(features)
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    features = 2 * (features - min_values).div(max_values - min_values) - 1
    # edge index
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(data_name)), dtype=int)
    edges = torch.tensor(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape).t()
    # num of cls
    classes = set(sens.numpy())
    classes.discard(-1)
    num_classes = len(classes)
    # pyg graph
    data = Data(x=features, edge_index=edges, y=sens, train_mask=idx_train, test_mask=idx_test, num_classes=num_classes)
    data = T.ToUndirected()(data)
    return data


def load_pokec(dataset, target_attr, train_ratio, path, split):
    if dataset == 'pokec_n':
        dataset = 'region_job_2'
    elif dataset == 'pokec_z':
        dataset = 'region_job'
    else:
        raise NotImplementedError
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    label = idx_features_labels['I_am_working_in_field'].values
    non_missing_label = np.where(label != -1)[0]
    # private attribute or downstream label
    if target_attr is not None:
        target = idx_features_labels[target_attr].values
        target = torch.LongTensor(target)
        if target_attr == "AGE":
            young = torch.where((target >= 18) & (target < 25))[0]
            young_adult = torch.where((target >= 25) & (target < 35))[0]
            middle = torch.where((target >= 35) & (target < 50))[0]
            senior = torch.where((target >= 50) & (target <= 80))[0]
            target[:] = -1
            target[young] = 0
            target[young_adult] = 1
            target[middle] = 2
            target[senior] = 3
        elif target_attr == "I_am_working_in_field":
            target[target > 1] = 1

        classes = set(target.numpy())
        classes.discard(-1)
        num_classes = len(classes)
    else:
        target, num_classes = None, None

    # node attribute
    header = list(idx_features_labels.columns)
    for attr in ['user_id', 'I_am_working_in_field', 'AGE', 'gender', 'region', 'completion_percentage']: header.remove(
        attr)
    features = np.array(idx_features_labels[header])
    features = torch.FloatTensor(features)
    # edge index
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)
    edges = torch.tensor(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape).t()

    data = Data(x=features, edge_index=edges, y=target, num_classes=num_classes)
    data = T.ToUndirected()(data)
    data = data.subgraph(torch.tensor(non_missing_label))
    # split train and test
    if split:
        non_missing_idx = torch.where(data.y >= 0)[0]
        non_missing_sens = data.y[non_missing_idx]
        res = sk_split(non_missing_idx, non_missing_sens, train_size=train_ratio, stratify=non_missing_sens)
        idx_train = res[0]
        idx_test = res[1]
        idx_train = index_to_mask(torch.LongTensor(idx_train), size=data.num_nodes)
        idx_test = index_to_mask(torch.LongTensor(idx_test), size=data.num_nodes)
        data.train_mask, data.test_mask = idx_train, idx_test
    return data
