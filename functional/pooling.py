from typing import Optional
import torch
from torch import Tensor
from torch_geometric.utils import scatter


def my_global_add_pool(
        x: Tensor,
        batch: Optional[Tensor],
        size: Optional[int] = None,
        mask: Optional[Tensor] = None,
) -> Tensor:
    dim = -1 if x.dim() == 1 else -2

    if batch is None:
        return x.sum(dim=dim, keepdim=x.dim() <= 2)
    size = int(batch.max().item() + 1) if size is None else size
    if mask is None:
        return scatter(x, batch, dim=dim, dim_size=size, reduce='sum')
    x = x * mask.to(x.dtype).unsqueeze(1)
    return scatter(x, batch, dim=dim, dim_size=size, reduce='sum')


# tested
def center_node_pooling(
        x: Tensor,
        batch: Tensor,
        mapping: Tensor,
) -> Tensor:
    _, counts = torch.unique_consecutive(batch, return_counts=True)
    offset = counts.cumsum(0)
    offset = torch.cat((torch.tensor([0], device=offset.device), offset[:-1]))
    indices = offset + mapping
    return x[indices, :]


def get_pooling_mask(batch_adj, center_node_idx, hops=2):
    pooling_mask_1hop = batch_adj[center_node_idx, :].to_dense().any(dim=0)
    pooling_mask_0hop = pooling_mask_1hop.new_zeros(pooling_mask_1hop.size())
    pooling_mask_0hop[center_node_idx] = True
    pooling_mask = torch.logical_or(pooling_mask_1hop, pooling_mask_0hop)
    if hops == 2:
        pooling_mask_2hop = batch_adj[center_node_idx, :].matmul(batch_adj).to_dense().any(dim=0)
        pooling_mask = torch.logical_or(pooling_mask, pooling_mask_2hop)
    return pooling_mask