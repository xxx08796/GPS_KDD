import argparse
import copy
import os
from functional.config import load_config
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from functional.dataset import LoadSubgraph, load_data
from functional.seed import set_seed
from functional.transform import transform_in_memory
from model.atk_model import AttrAtk

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
torch.use_deterministic_algorithms(True)

def train(model, device, optimizer, train_loader, whole_graph, train_idx):
    model.train()
    train_loss_accum = 0
    for step, (batch_s, batch_idx) in enumerate(train_loader):
        glo_batch_idx = train_idx[batch_idx]
        batch_s = ToSparseTensor()(batch_s.to(device))
        batch_p = ToSparseTensor()(whole_graph)
        optimizer.zero_grad()
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
    # train eval
    train_res_accum = 0
    for step, (batch_s, batch_idx) in enumerate(train_loader):
        glo_batch_idx = train_idx[batch_idx]
        batch_s = ToSparseTensor()(batch_s.to(device))
        batch_p = ToSparseTensor()(whole_graph)
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

    # test eval
    test_res_accum = 0
    for step, (batch_s, batch_idx) in enumerate(test_loader):
        glo_batch_idx = test_idx[batch_idx]
        batch_s = ToSparseTensor()(batch_s.to(device))
        batch_p = ToSparseTensor()(whole_graph)
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


def main(exp_num, opt):
    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")
    # set up dataset
    print('load dataset')
    sub_graph = LoadSubgraph(root=opt.root, data_name=opt.dataset, sens_attr=opt.sens_attr, hops=opt.hops,
                             device=opt.device)
    whole_graph = load_data(data_name=opt.dataset, target=opt.sens_attr, train_ratio=opt.train_ratio,
                            path=opt.root + '/raw/').to(device)
    const_edge_index = whole_graph.edge_index
    print(sub_graph, whole_graph)

    transform_in_memory(sub_graph=sub_graph, whole_graph=copy.deepcopy(whole_graph), opt=opt)

    # split train and test sets
    train_idx, test_idx = whole_graph.train_mask, whole_graph.test_mask
    train_idx = torch.where(train_idx)[0]
    test_idx = torch.where(test_idx)[0]

    train_loader = DataLoader(sub_graph[train_idx], batch_size=opt.batch_size, num_workers=opt.num_workers,
                              shuffle=True)
    test_loader = DataLoader(sub_graph[test_idx], batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)

    # set up model
    opt.in_channels['x_in'] = sub_graph.num_features
    model = AttrAtk(
        gnn=(opt.p_encoder, opt.s_encoder),
        in_channels=opt.in_channels,
        hidden_channels=opt.hidden_channels,
        gnn_layers=opt.num_layers,
        num_classes=sub_graph.num_classes,
        dropout=opt.dropout,
        pe_types=opt.pe_types,
        num_nodes=whole_graph.num_nodes,
        torr=opt.torr,
    ).to(device)
    model.p_weight, model.s_weight = model.p_weight.to(device), model.s_weight.to(device)
    print(model)
    # set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    step_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.scheduler)
    print(optimizer)

    for epoch in range(1, opt.num_epochs + 1):
        train_loss = train(model, device, optimizer, train_loader, whole_graph, train_idx)
        step_scheduler.step()

        if epoch % 10 == 1 or epoch == 1:
            train_res, test_res, y_pred = test(model, device, train_loader, test_loader,
                                               sub_graph.num_classes, whole_graph, (train_idx, test_idx))
            print(
                f'exp {exp_num}, '
                f'epoch {epoch:03d}, '
                f'train loss {train_loss:.4f}, '
                f'train res {train_res:.4f}, '
                f'test res {test_res:.4f}'
            )

        if epoch > opt.init and epoch % opt.upd_gap == 1:
            model.update_weights(const_edge_index, y_pred)

    return test_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('--p_encoder', type=str, default="P_GIN")
    parser.add_argument('--s_encoder', type=str, default="S_GIN")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='nba')
    parser.add_argument('--sens_attr', type=str, default='country')
    args = parser.parse_args()
    param = load_config(source=f'{args.dataset}_{args.sens_attr}.json')
    for key, value in param.items(): setattr(args, key, value)
    print(args)
    set_seed(args.seed)
    test_l = []
    for i in range(5):
        test_res = main(i, args)
        test_l.append(test_res)
    ave, std = np.mean(test_l), np.std(test_l)
    print(f'final result: {ave:.4f}±{std:.4f}')
