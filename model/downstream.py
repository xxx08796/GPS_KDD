import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
from encoder.gnn import GraphEncoder
from functional.dataset import load_data
from model.view_learner import ViewLearner
import torch.nn.functional as F


def train(data, model, optimizer):
    model.train()
    optimizer.zero_grad()
    x = model(data)
    loss = F.cross_entropy(x[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()
    x = model(data)
    out_s = F.softmax(x, dim=-1)
    if data.num_classes == 2:
        test_res = roc_auc_score(data.y[data.test_mask].cpu(), out_s[data.test_mask, 1].cpu())
        train_res = roc_auc_score(data.y[data.train_mask].cpu(), out_s[data.train_mask, 1].cpu())
    else:
        pred_s = np.argmax(out_s.cpu().numpy(), axis=1)
        test_res = accuracy_score(pred_s[data.test_mask.cpu()], data.y[data.test_mask].cpu().numpy())
        train_res = accuracy_score(pred_s[data.train_mask.cpu()], data.y[data.train_mask].cpu().numpy())
    return test_res, train_res


def downstream(opt, new_edge=None):
    device = torch.device(opt.device)
    data = load_data(data_name=opt.dataset, target=opt.ds_label, train_ratio=opt.train_ratio, path=opt.root + '/raw')
    df = ViewLearner(
        encoder=GraphEncoder(
            gnn_model='SAGE',
            in_channels=data.x.shape[1],
            hidden_channels=64,
            s_channels=data.num_classes,
            num_layers=2,
            dropout=0.7,
            return_emb=True,
        ),
        hidden_c=64,
        mlp_edge_model_dim=32
    ).to(device)
    aug_edge_index = new_edge
    data.edge_index = aug_edge_index
    data.to(device)
    model = GraphEncoder(
        gnn_model='P_GIN',
        in_channels=data.x.shape[1],
        hidden_channels=32,
        s_channels=data.num_classes,
        num_layers=2,
        dropout=0.7,
        return_emb=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    for _ in tqdm(range(1, 300 + 1)):
        train(data, model, optimizer)
        test_r, train_r = test(model, data)
    print(f'downstream res: {test_r:.4f}')
    return test_r
