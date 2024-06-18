import copy
import torch
from torch_geometric.utils import degree, k_hop_subgraph, to_dense_adj
from torch_scatter import scatter
from torch_sparse import SparseTensor
from encoder.gnn import GraphEncoder
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from functional.pooling import get_pooling_mask
from model.view_learner import ViewLearner
EPS = 1e-10
flag = True


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

    def forward(self, batch_p, batch_s, batch_idx, edge_weight, batch_edge_weight, pooling_mask):
        x_p = self.p_encoder(batch_p, edge_weight)[batch_idx]
        x_p = self.norms[0](x_p)
        x_s = self.s_encoder(batch_s, batch_edge_weight, pooling_mask)
        x_s = self.norms[1](x_s)
        x = self.p_weight[batch_idx] * x_p + self.s_weight[batch_idx] * x_s
        if self.return_em:
            x = self.cls_head(x)
        return x

    def update_weights(self, edge_index, edge_prob, deg_prob, label):
        p_weight = prox_ratio(edge_index, edge_prob, label).detach().unsqueeze(1)
        p_weight = torch.masked_fill(p_weight, torch.isnan(p_weight), 0)
        s_weight = struc_ratio(deg_prob, label).detach().unsqueeze(1).to(p_weight.device)
        s_weight = torch.masked_fill(s_weight, torch.isnan(s_weight), 0)
        tol = p_weight + s_weight
        p_weight = torch.where(tol != 0, torch.div(p_weight, tol), 0.5)
        s_weight = torch.where(tol != 0, torch.div(s_weight, tol), 0.5)
        self.p_weight = p_weight
        self.s_weight = s_weight


def prox_ratio(edge_index, prob, label):
    row, col = edge_index
    out = torch.zeros(row.size(0), device=row.device)
    out[label[row] == label[col]] = 1.
    prob_label = prob * out
    out_1 = scatter(prob_label, col, 0, dim_size=label.size(0), reduce='sum')
    out_2 = scatter(prob, col, 0, dim_size=label.size(0), reduce='sum')
    return out_1 / (out_2 + EPS)


def struc_ratio(properties, label, tolerance=20):
    properties = properties.cpu()
    label = label.cpu()
    mask = ~torch.eye(properties.size(0)).bool()
    properties_same = torch.abs(properties.unsqueeze(1) - properties.unsqueeze(0))
    properties_same = tolerance - properties_same
    properties_same = properties_same / (torch.abs(properties_same).detach() + EPS)
    properties_same = F.relu(properties_same)
    properties_same = properties_same[mask].reshape(properties_same.size(0), properties_same.size(0) - 1)

    label_same = torch.eq(label.unsqueeze(1), label.unsqueeze(0)).float()
    label_same = label_same[mask].reshape(label_same.size(0), label_same.size(0) - 1)

    properties_label_same = properties_same * label_same
    same_count = torch.sum(properties_label_same.float(), dim=-1)
    total_count = torch.sum(properties_same.float(), dim=-1)

    ratios = same_count / (total_count + EPS)
    return ratios

class DfModel(nn.Module):
    def __init__(self, graph, subgraph, opt):
        super(DfModel, self).__init__()
        self.df = ViewLearner(
            encoder=GraphEncoder(
                gnn_model='SAGE',
                in_channels=opt.in_channels['x_in'],
                hidden_channels=64,
                s_channels=graph.num_classes,
                num_layers=opt.num_layers,
                dropout=opt.dropout,
                return_emb=True,
            ),
            hidden_c=64,
            mlp_edge_model_dim=32
        )
        self.adv = AtkModel(
            gnn=(opt.p_encoder, opt.s_encoder),
            in_channels=opt.in_channels,
            hidden_channels=opt.hidden_channels,
            gnn_layers=opt.num_layers,
            num_classes=graph.num_classes,
            dropout=opt.dropout,
            return_em=False,
            num_nodes=graph.num_nodes,
        )

        self.optimizer_D = torch.optim.AdamW(self.df.parameters(), lr=opt.learning_rate_df, weight_decay=5e-4)
        self.optimizer_A = torch.optim.AdamW(self.adv.parameters(), lr=opt.learning_rate_adv, weight_decay=5e-4)

        self.opt = opt
        self.graph = graph
        self.subgraph = subgraph
        self.ori_edge_index = graph.edge_index
        self.criterion = nn.CrossEntropyLoss()

        self.edge_indices_l = self.get_subgraph_edge_indices()
        self.y_pred = None

    def get_subgraph_edge_indices(self):
        edge_indices_l = []
        for idx in tqdm(range(self.graph.num_nodes)):
            _, _, _, edge_mask = k_hop_subgraph(
                node_idx=idx,
                num_hops=self.opt.hops,
                edge_index=self.graph.edge_index,
                relabel_nodes=True,
                num_nodes=self.graph.num_nodes
            )
            ego_edge_indices = torch.where(edge_mask)[0]
            edge_indices_l.append(ego_edge_indices.cpu())
        return edge_indices_l

    def warmup(self, warmup=10):
        print("warm up")
        for _ in range(warmup):
            self.train()
            self.optimizer_D.zero_grad()
            sim = self.df(self.graph)
            prob = torch.sigmoid(sim).reshape(-1, 1)
            prob_2d = torch.cat([1 - prob, prob], dim=-1)
            reconst_loss = F.cross_entropy(torch.log(prob_2d + 1e-8), torch.ones_like(sim.squeeze(), dtype=torch.long))
            with torch.autograd.set_detect_anomaly(True):
                reconst_loss.backward(retain_graph=False)
            self.optimizer_D.step()
            print(reconst_loss.item())

    def get_ra_loss(self, flag):
        y_pred = copy.deepcopy(self.y_pred).cpu()
        max_value = y_pred.max() + 1
        missing = (y_pred == -1)
        y_pred[missing] = max_value
        class_prob = (torch.bincount(y_pred) / len(self.y_pred)).to(self.y_pred.device)
        sample_prob = class_prob[self.y_pred]
        p_weight = prox_ratio(self.graph.edge_index, self.get_prob(), self.y_pred)
        p_weight = torch.abs(p_weight - sample_prob)[~missing]
        mask = torch.logical_not(torch.isnan(p_weight))
        p_weight = p_weight[mask]
        s_weight = struc_ratio(self.get_deg_prob(), self.y_pred).to(p_weight.device)
        s_weight = torch.abs(s_weight - sample_prob)[~missing]
        mask = torch.logical_not(torch.isnan(s_weight))
        s_weight = s_weight[mask]

        return p_weight.mean() if flag else s_weight.mean()


    def optimize(self, train_df, train_adv, train_idx, train_loader):
        global flag
        self.train()
        df_loss_all = 0
        adv_loss_all = 0
        reconst_loss_all = 0
        df_total_loss_all = 0
        edge_rate = 0
        for step, (batch_s, batch_idx) in enumerate(train_loader):
            glo_batch_idx = train_idx[batch_idx]
            batch_s.to(self.opt.device)
            if train_df:
                self.optimizer_D.zero_grad()
                sim = self.df(self.graph)
                prob = torch.sigmoid(sim).reshape(-1, 1)
                prob_2d = torch.cat([1 - prob, prob], dim=-1)
                samples = F.gumbel_softmax(torch.log(prob_2d + 1e-8), tau=self.opt.tau, hard=True)
                edge_weight = samples[:, 1]
                edge_rate += prob.mean()
                batch_edge_indices = [self.edge_indices_l[idx].to(prob.device) for idx in glo_batch_idx]
                batch_edge_indices = torch.cat(batch_edge_indices, dim=0)
                batch_edge_weight = edge_weight[batch_edge_indices]
                batch_edge = batch_s.edge_index[:, batch_edge_weight.bool()]
                batch_adj = SparseTensor(
                    row=batch_edge[0],
                    col=batch_edge[1],
                    value=torch.ones(batch_edge.shape[1], device=batch_edge.device),
                    sparse_sizes=(batch_s.num_nodes, batch_s.num_nodes)
                ).coalesce()
                _, counts = torch.unique_consecutive(batch_s.batch, return_counts=True)
                offset = counts.cumsum(0)
                offset = torch.cat((torch.tensor([0], device=offset.device), offset[:-1]))
                center_node_idx = offset + batch_s.mapping
                pooling_mask = get_pooling_mask(batch_adj, center_node_idx, self.opt.hops)
                x = self.adv.forward(self.graph, batch_s, glo_batch_idx, edge_weight,
                                     batch_edge_weight, pooling_mask)
                batch_edge_prob = prob_2d[batch_edge_indices.unique(), :]
                reconst_loss = F.cross_entropy(
                    torch.log(batch_edge_prob + 1e-8),
                    torch.ones(batch_edge_prob.size(0), dtype=torch.long, device=prob_2d.device)
                ) * (glo_batch_idx.size(0) / self.graph.num_nodes)
                df_loss = F.cross_entropy(x, batch_s.y)
                if self.y_pred is not None:
                    flag = not flag
                    ratio_loss = self.get_ra_loss(flag)
                else:
                    ratio_loss = 0
                total_loss = -df_loss * self.opt.gamma + self.opt.lam * reconst_loss + self.opt.eta * ratio_loss
                total_loss.backward()
                self.optimizer_D.step()
                df_loss_all += df_loss.item()
                reconst_loss_all += reconst_loss.item()
                df_total_loss_all += total_loss.item()
            if train_adv:
                self.optimizer_A.zero_grad()
                sim = self.df(self.graph)
                prob = torch.sigmoid(sim).reshape(-1, 1)
                prob_2d = torch.cat([1 - prob, prob], dim=-1)
                samples = F.gumbel_softmax(torch.log(prob_2d + 1e-8), tau=self.opt.tau, hard=True)
                edge_weight = samples[:, 1]
                edge_rate += prob.mean()
                batch_edge_indices = [self.edge_indices_l[idx].to(prob.device) for idx in glo_batch_idx]
                batch_edge_indices = torch.cat(batch_edge_indices, dim=0)
                batch_edge_weight = edge_weight[batch_edge_indices]
                batch_edge = batch_s.edge_index[:, batch_edge_weight.bool()]
                batch_adj = SparseTensor(
                    row=batch_edge[0],
                    col=batch_edge[1],
                    value=torch.ones(batch_edge.shape[1], device=batch_edge.device),
                    sparse_sizes=(batch_s.num_nodes, batch_s.num_nodes)
                ).coalesce()
                _, counts = torch.unique_consecutive(batch_s.batch, return_counts=True)
                offset = counts.cumsum(0)
                offset = torch.cat((torch.tensor([0], device=offset.device), offset[:-1]))
                center_node_idx = offset + batch_s.mapping
                pooling_mask = get_pooling_mask(batch_adj, center_node_idx, self.opt.hops)
                x = self.adv.forward(self.graph, batch_s, glo_batch_idx, edge_weight.detach(),
                                     batch_edge_weight.detach(), pooling_mask)
                adv_loss = F.cross_entropy(x, batch_s.y)
                adv_loss.backward()
                adv_loss_all += adv_loss.item()
                self.optimizer_A.step()
        df_total_loss_all = df_total_loss_all / (step + 1)
        df_loss_all = df_loss_all / (step + 1)
        adv_loss_all = adv_loss_all / (step + 1)
        reconst_loss_all = reconst_loss_all / (step + 1)
        edge_rate = edge_rate / (step + 1)
        return df_loss_all, adv_loss_all, reconst_loss_all, df_total_loss_all, edge_rate

    @torch.no_grad()
    def test(self, train_idx, train_loader, test_iter, test_loader):
        self.eval()
        y_pred = torch.ones(self.graph.num_nodes, dtype=torch.long) * -1
        y_pred = y_pred.to(self.opt.device)
        sim = self.df(self.graph)
        prob = torch.sigmoid(sim).reshape(-1, 1)
        prob_2d = torch.cat([1 - prob, prob], dim=-1)
        samples = F.gumbel_softmax(torch.log(prob_2d + 1e-8), tau=self.opt.tau, hard=True)
        edge_weight = samples[:, 1]

        train_res_accum = 0
        for step, (batch_s, batch_idx) in enumerate(train_loader):
            glo_batch_idx = train_idx[batch_idx]
            batch_s.to(self.opt.device)
            batch_edge_indices = [self.edge_indices_l[idx].to(prob.device) for idx in glo_batch_idx]
            batch_edge_indices = torch.cat(batch_edge_indices, dim=0)
            batch_edge_weight = edge_weight[batch_edge_indices]

            batch_edge = batch_s.edge_index[:, batch_edge_weight.bool()]
            batch_adj = SparseTensor(
                row=batch_edge[0],
                col=batch_edge[1],
                value=torch.ones(batch_edge.shape[1], device=batch_edge.device),
                sparse_sizes=(batch_s.num_nodes, batch_s.num_nodes)
            ).coalesce()
            _, counts = torch.unique_consecutive(batch_s.batch, return_counts=True)
            offset = counts.cumsum(0)
            offset = torch.cat((torch.tensor([0], device=offset.device), offset[:-1]))
            center_node_idx = offset + batch_s.mapping
            pooling_mask = get_pooling_mask(batch_adj, center_node_idx, self.opt.hops)
            x = self.adv.forward(self.graph, batch_s, glo_batch_idx, edge_weight, batch_edge_weight, pooling_mask)
            pred = F.softmax(x, dim=-1)
            true = batch_s.y
            if self.graph.num_classes == 2:
                train_res = roc_auc_score(true.cpu(), pred[:, 1].cpu())
            else:
                pred = torch.argmax(pred, dim=-1)
                train_res = accuracy_score(true.cpu(), pred.cpu())
            train_res_accum += train_res
            y_pred[glo_batch_idx] = torch.argmax(x, dim=-1)
        train_res = train_res_accum / (step + 1)

        test_res_accum = 0
        for step, (batch_s, batch_idx) in enumerate(test_loader):
            glo_batch_idx = test_iter[batch_idx]
            batch_s.to(self.opt.device)
            batch_edge_indices = [self.edge_indices_l[idx].to(prob.device) for idx in glo_batch_idx]
            batch_edge_indices = torch.cat(batch_edge_indices, dim=0)
            batch_edge_weight = edge_weight[batch_edge_indices]

            batch_edge = batch_s.edge_index[:, batch_edge_weight.bool()]
            batch_adj = SparseTensor(
                row=batch_edge[0],
                col=batch_edge[1],
                value=torch.ones(batch_edge.shape[1], device=batch_edge.device),
                sparse_sizes=(batch_s.num_nodes, batch_s.num_nodes)
            ).coalesce()
            _, counts = torch.unique_consecutive(batch_s.batch, return_counts=True)
            offset = counts.cumsum(0)
            offset = torch.cat((torch.tensor([0], device=offset.device), offset[:-1]))
            center_node_idx = offset + batch_s.mapping
            pooling_mask = get_pooling_mask(batch_adj, center_node_idx, self.opt.hops)
            x = self.adv.forward(self.graph, batch_s, glo_batch_idx, edge_weight, batch_edge_weight, pooling_mask)
            pred = F.softmax(x, dim=-1)
            true = batch_s.y
            if self.graph.num_classes == 2:
                test_res = roc_auc_score(true.cpu(), pred[:, 1].cpu())
            else:
                pred = torch.argmax(pred, dim=-1)
                test_res = accuracy_score(true.cpu(), pred.cpu())
            test_res_accum += test_res
            y_pred[glo_batch_idx] = torch.argmax(x, dim=-1)
        test_res = test_res_accum / (step + 1)

        return train_res, test_res, y_pred

    def get_deg_prob(self):
        prob = self.get_prob()
        _, col = self.graph.edge_index
        out = scatter(prob, col, 0, dim_size=self.graph.num_nodes, reduce='sum')
        return out

    def get_prob(self):
        # self.eval()
        sim = self.df(self.graph)
        prob = torch.sigmoid(sim).squeeze()
        return prob