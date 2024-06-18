from torch_geometric.utils import degree, k_hop_subgraph
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from encoder.gnn import GraphEncoder
from functional.dataset import LoadSubgraph, load_data
from functional.ghratio import node_level_homo, struc_ratio
from model.view_learner import ViewLearner


class AtkModel(nn.Module):
    def __init__(
            self,
            gnn,
            in_channels,
            hidden_channels,
            gnn_layers,
            num_classes,
            dropout,
            return_em,
            num_nodes,
    ):
        super(AtkModel, self).__init__()
        gnn_in = in_channels['x_in']
        p_gnn, s_gnn = gnn[0], gnn[1]
        self.p_encoder = GraphEncoder(
            gnn_model=p_gnn,
            in_channels=in_channels['x_in'],
            hidden_channels=hidden_channels['gnn_hidden'] * 4,
            s_channels=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            return_emb=return_em,
        )
        self.s_encoder = GraphEncoder(
            gnn_model=s_gnn,
            in_channels=gnn_in,
            hidden_channels=hidden_channels['gnn_hidden'],
            s_channels=num_classes,
            num_layers=gnn_layers,
            dropout=dropout,
            return_emb=return_em,
        )
        self.p_gnn, self.s_gnn = p_gnn, s_gnn
        if return_em:
            self.cls_head = torch.nn.Linear(4 * hidden_channels['gnn_hidden'], num_classes, bias=False)
        self.return_em = return_em

        self.norms = nn.ModuleList()
        if return_em:
            self.norms.append(nn.BatchNorm1d(4 * hidden_channels['gnn_hidden']))
            self.norms.append(nn.BatchNorm1d(4 * hidden_channels['gnn_hidden']))
        else:
            self.norms.append(nn.BatchNorm1d(num_classes))
            self.norms.append(nn.BatchNorm1d(num_classes))

        p_weight = torch.ones(num_nodes).unsqueeze(1) * 0.5
        s_weight = torch.ones(num_nodes).unsqueeze(1) * 0.5
        tol = p_weight + s_weight
        p_weight = p_weight / tol
        s_weight = s_weight / tol
        self.p_weight = p_weight
        self.s_weight = s_weight

    def forward(self, batch_p, batch_s, batch_idx):
        x_p = self.p_encoder(batch_p)[batch_idx]
        x_p = self.norms[0](x_p)
        x_s = self.s_encoder(batch_s)
        x_s = self.norms[1](x_s)
        x = self.p_weight[batch_idx] * x_p + self.s_weight[batch_idx] * x_s
        if self.return_em:
            x = self.cls_head(x)
        return x

    def update_weights(self, edge_index, label):
        p_weight = node_level_homo(edge_index, label).unsqueeze(1)
        aug_deg = degree(edge_index[0], num_nodes=label.shape[0])
        s_weight = struc_ratio(aug_deg, label).unsqueeze(1).to(p_weight.device)
        s_weight = torch.masked_fill(s_weight, torch.isnan(s_weight), 0)
        tol = p_weight + s_weight
        p_weight = torch.where(tol != 0, torch.div(p_weight, tol), 0.5)
        s_weight = torch.where(tol != 0, torch.div(s_weight, tol), 0.5)
        self.p_weight = p_weight
        self.s_weight = s_weight


def train(model, device, optimizer, train_loader, whole_graph, train_idx):
    model.train()
    train_loss_accum = 0
    for step, (batch_s, batch_idx) in enumerate(train_loader):
        glo_batch_idx = train_idx[batch_idx]
        batch_s.to(device)
        optimizer.zero_grad()
        batch_p = whole_graph
        x = model.forward(batch_p, batch_s, glo_batch_idx)
        batch_loss = F.cross_entropy(x, batch_s.y)
        batch_loss.backward()
        optimizer.step()
        train_loss_accum += batch_loss.item()
    train_loss = train_loss_accum / (step + 1)
    return train_loss


@torch.no_grad()
def test(model, device, train_loader, test_loader, num_classes, whole_graph, glo_idx):
    model.eval()
    train_idx = glo_idx[0]
    test_idx = glo_idx[1]
    y_pred = torch.ones(whole_graph.num_nodes, dtype=torch.long) * -1
    y_pred = y_pred.to(device)
    train_res_accum = 0
    for step, (batch_s, batch_idx) in enumerate(train_loader):
        glo_batch_idx = train_idx[batch_idx]
        batch_s.to(device)
        batch_p = whole_graph
        x = model.forward(batch_p, batch_s, glo_batch_idx)
        pred = F.softmax(x, dim=-1)
        true = batch_s.y
        if num_classes == 2:
            train_res = roc_auc_score(true.cpu(), pred[:, 1].cpu())
        else:
            pred = torch.argmax(pred, dim=-1)
            train_res = accuracy_score(true.cpu(), pred.cpu())
        train_res_accum += train_res
        y_pred[glo_batch_idx] = torch.argmax(x, dim=-1)
    train_res = train_res_accum / (step + 1)

    test_res_accum = 0
    for step, (batch_s, batch_idx) in enumerate(test_loader):
        glo_batch_idx = test_idx[batch_idx]
        batch_s.to(device)
        batch_p = whole_graph
        x = model.forward(batch_p, batch_s, glo_batch_idx)
        pred = F.softmax(x, dim=-1)
        true = batch_s.y
        if num_classes == 2:
            test_res = roc_auc_score(true.cpu(), pred[:, 1].cpu())
        else:
            pred = torch.argmax(pred, dim=-1)
            test_res = accuracy_score(true.cpu(), pred.cpu())
        test_res_accum += test_res
        y_pred[glo_batch_idx] = torch.argmax(x, dim=-1)
    test_res = test_res_accum / (step + 1)
    return train_res, test_res, y_pred


def measure_privacy(opt, new_edge=None):
    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")
    whole_graph = load_data(data_name=opt.dataset,target=opt.sens_attr,train_ratio=opt.train_ratio,path=opt.root + '/raw/')
    opt.in_channels['x_in'] = whole_graph.num_features
    df = ViewLearner(
        encoder=GraphEncoder(
            gnn_model='SAGE',
            in_channels=opt.in_channels['x_in'],
            hidden_channels=64,
            s_channels=whole_graph.num_classes,
            num_layers=2,
            dropout=0.7,
            return_emb=True,
        ),
        hidden_c=64,
        mlp_edge_model_dim=32
    ).to(device)
    aug_edge_index = new_edge
    whole_graph.edge_index = aug_edge_index
    whole_graph.to(device)
    sub_graph = LoadSubgraph(root=opt.root, data_name=opt.dataset, sens_attr=opt.sens_attr, hops=opt.hops)
    data_list = []
    for idx in range(len(sub_graph)):
        subset, ego_edge, mapping, edge_mask = k_hop_subgraph(
            node_idx=idx,
            num_hops=opt.hops,
            edge_index=whole_graph.edge_index,
            relabel_nodes=True,
            num_nodes=whole_graph.num_nodes
        )
        data = sub_graph.get(idx)
        data.x = whole_graph.x[subset, :]
        data.edge_index = ego_edge
        data.mapping = mapping
        data_list.append(data.cpu())
    data_list = list(filter(None, data_list))
    sub_graph._indices = None
    sub_graph._data_list = data_list
    sub_graph.data, sub_graph.slices = sub_graph.collate(data_list)
    whole_graph.to(device)
    train_idx, test_idx = whole_graph.train_mask, whole_graph.test_mask
    train_idx = torch.where(train_idx)[0]
    test_idx = torch.where(test_idx)[0]
    train_loader = DataLoader(sub_graph[train_idx], batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(sub_graph[test_idx], batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    model = AtkModel(
        gnn=(opt.p_encoder, opt.s_encoder),
        in_channels=opt.in_channels,
        hidden_channels=opt.hidden_channels,
        gnn_layers=opt.num_layers,
        num_classes=sub_graph.num_classes,
        dropout=0.1,
        return_em=False,
        num_nodes=whole_graph.num_nodes,
    ).to(device)
    model.p_weight = model.p_weight.to(device)
    model.s_weight = model.s_weight.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    for epoch in tqdm(range(1, opt.atk_epochs + 1)):
        train(model, device, optimizer, train_loader, whole_graph, train_idx)
        if epoch % 10 == 0 or epoch == 1:
            train_res, test_res, y_pred = test(model, device, train_loader, test_loader, sub_graph.num_classes,
                                               whole_graph, (train_idx, test_idx))
            if epoch > 0: model.update_weights(whole_graph.edge_index, y_pred)
    print(f'privacy preserving res: {test_res:.4f}')
    return test_res