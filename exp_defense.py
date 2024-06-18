import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from functional.dataset import load_data, LoadSubgraph
from functional.config import load_config
from functional.seed import set_seed
from model.downstream import downstream
from model.privacy_preserving import measure_privacy
from model.df_model import DfModel
from torch_geometric.loader import DataLoader
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def main(opt):
    device = torch.device(opt.device) if torch.cuda.is_available() else torch.device("cpu")
    sub_graph = LoadSubgraph(root=opt.root, data_name=opt.dataset, sens_attr=opt.sens_attr, hops=opt.hops)
    whole_graph = load_data(data_name=opt.dataset, target=opt.sens_attr, train_ratio=opt.train_ratio, path=opt.root + '/raw/').to(device)
    print(sub_graph, whole_graph)
    train_idx, test_idx = whole_graph.train_mask, whole_graph.test_mask
    train_idx,test_idx = torch.where(train_idx)[0],torch.where(test_idx)[0]
    train_loader = DataLoader(sub_graph[train_idx], batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(sub_graph[test_idx], batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
    opt.in_channels['x_in'] = whole_graph.num_features
    model = DfModel(graph=whole_graph, subgraph=sub_graph, opt=opt).to(device)
    model.adv.p_weight = model.adv.p_weight.to(device)
    model.adv.s_weight = model.adv.s_weight.to(device)
    print(model)
    interval = opt.interval
    train_df, train_adv = True, False
    for epoch in range(1, opt.num_epochs + 1):
        if epoch == 1: model.warmup(warmup=opt.warmup)
        if epoch % interval == 0 and epoch >= opt.adv_train: train_df, train_adv = not train_df, not train_adv
        model.optimize(train_df, train_adv, train_idx, train_loader)
        if epoch % 10 == 0 or epoch == 1:
            train_res, test_res, y_pred = model.test(train_idx, train_loader, test_idx, test_loader)
            model.y_pred = y_pred
            model.adv.update_weights(whole_graph.edge_index, model.get_prob(), model.get_deg_prob(), y_pred)
            print(f'epoch {epoch:03d}, adv train res {train_res:.4f}, adv test res {test_res:.4f}')

    model.train()
    sim = model.df(whole_graph)
    prob = torch.sigmoid(sim).reshape(-1, 1)
    prob_2d = torch.cat([1 - prob, prob], dim=-1)
    samples = F.gumbel_softmax(torch.log(prob_2d + 1e-8), tau=opt.tau, hard=True)
    edge_weight = samples[:, 1]
    mask = torch.where(edge_weight == 1)[0]
    new_edge = whole_graph.edge_index[:, mask]

    ds_list, atk_list = [], []
    for exp in range(5):
        print(f'exp {exp}')
        ds_res = downstream(opt=opt, new_edge=new_edge)
        atk_res = measure_privacy(opt=opt, new_edge=new_edge)
        ds_list.append(ds_res), atk_list.append(atk_res)
    print(f'downstream result:  {np.mean(ds_list):.4f}±{np.std(ds_list):.4f}')
    print(f'attack result:  {np.mean(atk_list):.4f}±{np.std(atk_list):.4f}')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2, )
    parser.add_argument('--train_ratio', type=float, default=0.1)
    parser.add_argument('--p_encoder', type=str, default="P_GIN")
    parser.add_argument('--s_encoder', type=str, default="S_GIN")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='nba')
    parser.add_argument('--sens_attr', type=str, default="country")
    parser.add_argument('--ds_label', type=str, default='SALARY')
    parser.add_argument('--tau', type=float, default=5)
    parser.add_argument('--lam', type=float, default=2.0)
    parser.add_argument('--gamma', type=float, default=3.0)
    parser.add_argument('--eta', type=float, default=2.0)
    parser.add_argument('--atk_epochs', type=int, default=400)
    parser.add_argument('--atk_dropout', type=float, default=0.1)
    args = parser.parse_args()
    param = load_config(source=f'df_{args.dataset}_{args.sens_attr}.json')
    for key, value in param.items():
        setattr(args, key, value)
    print(args)
    set_seed(3000)
    main(args)
