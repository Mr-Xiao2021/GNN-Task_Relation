import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops


def masked_edge_index(edge_index, edge_mask):
    if isinstance(edge_index, torch.Tensor):
        return edge_index[:, edge_mask]


class RGCNEdgeConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        aggr: str = "mean",
        **kwargs,
    ):
        kwargs.setdefault("aggr", aggr)
        super().__init__(**kwargs)  # "Add" aggregation (Step 5).
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        self.weight = Parameter(
            torch.empty(self.num_relations, in_channels, out_channels)
        )

        self.root = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.weight)
        glorot(self.root)
        zeros(self.bias)

    def forward(
        self,
        x: OptTensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
        high_degree_mask: OptTensor = None,
    ):
        r"""
        \(x'_v = \sum_r \left( \operatorname{mean}_{u\in \mathcal N_r(v)} \operatorname{ReLU}(x_u + e_{uv}) \right)W_r + x_vW_{\text{root}} + b\)
        """
        if high_degree_mask is not None:
            return self._mixed_precision_forward(
                x, xe, edge_index, edge_type, high_degree_mask
            )

        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.num_relations):
            edge_mask = edge_type == i
            tmp = masked_edge_index(edge_index, edge_mask) # 只保留当前关系的边索引, shape(tmp)=[2, E1], x_j = x[tmp[0]]为边的源节点特征

            h = self.propagate(tmp, x=x, xe=xe[edge_mask]) # 执行message，，PyG 会按照目标节点 tmp[1] 对收到的消息求aggr="mean"
            out += h @ self.weight[i] # 关系变换并累加

        out += x @ self.root # 加入节点自身信息，self.root可学习自连接
        out += self.bias

        return out

    def _mixed_precision_forward(self, x, xe, edge_index, edge_type, high_degree_mask):
        """Run high-degree targets in FP32 and low-degree targets in FP16."""

        if x.device.type != "cuda":
            raise RuntimeError("degree-aware mixed precision requires CUDA")
        full_precision_parameters = (self.weight, self.root, self.bias)
        if any(
            parameter.dtype != torch.float32 for parameter in full_precision_parameters
        ):
            raise RuntimeError("degree-aware mixed precision requires FP32 parameters")
        if high_degree_mask.shape != (x.size(0),):
            raise ValueError("high_degree_mask must contain one value per node")

        x = x.to(torch.float32)
        xe = xe.to(torch.float32)
        high_degree_mask = high_degree_mask.to(device=x.device, dtype=torch.bool)
        low_degree_mask = torch.logical_not(high_degree_mask)
        target_nodes = edge_index[1]

        with torch.cuda.amp.autocast(enabled=False):
            low_x = x.to(torch.float16)
            low_xe = xe.to(torch.float16)
            low_weight = self.weight.to(torch.float16)
            low_root = self.root.to(torch.float16)
            low_bias = self.bias.to(torch.float16)
            out = torch.zeros(
                x.size(0),
                self.out_channels,
                device=x.device,
                dtype=torch.float32,
            )

            for i in range(self.num_relations):
                relation_mask = edge_type == i
                high_edge_mask = relation_mask & high_degree_mask[target_nodes]
                low_edge_mask = relation_mask & low_degree_mask[target_nodes]

                high_h = self.propagate(
                    masked_edge_index(edge_index, high_edge_mask),
                    x=x,
                    xe=xe[high_edge_mask],
                )
                low_h = self.propagate(
                    masked_edge_index(edge_index, low_edge_mask),
                    x=low_x,
                    xe=low_xe[low_edge_mask],
                )

                out[high_degree_mask] = out[high_degree_mask] + (
                    high_h[high_degree_mask] @ self.weight[i]
                )
                out[low_degree_mask] = out[low_degree_mask] + (
                    low_h[low_degree_mask] @ low_weight[i]
                ).to(torch.float32)

            out[high_degree_mask] = out[high_degree_mask] + (
                x[high_degree_mask] @ self.root
            )
            out[low_degree_mask] = out[low_degree_mask] + (
                low_x[low_degree_mask] @ low_root
            ).to(torch.float32)
            out[high_degree_mask] = out[high_degree_mask] + self.bias
            out[low_degree_mask] = out[low_degree_mask] + low_bias.to(torch.float32)

        return out

    def message(self, x_j, xe):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return (x_j + xe).relu()


class RGATEdgeConv(RGCNEdgeConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_relations: int,
        aggr: str = "sum",
        heads=8,
        add_self_loops=False,
        share_att=False,
        **kwargs,
    ):
        self.heads = heads
        self.add_self_loops = add_self_loops
        self.share_att = share_att
        super().__init__(
            in_channels,
            out_channels,
            num_relations,
            aggr,
            node_dim=0,
            **kwargs,
        )
        self.lin_edge = nn.Linear(self.in_channels, self.out_channels)
        assert self.in_channels % heads == 0
        self.d_model = self.in_channels // heads
        if self.share_att:
            self.att = Parameter(torch.empty(1, heads, self.d_model))
        else:
            self.att = Parameter(
                torch.empty(self.num_relations, heads, self.d_model)
            )

        glorot(self.att)

    def forward(
        self,
        x: OptTensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
    ):
        out = torch.zeros((x.size(0), self.out_channels), device=x.device)

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, xe = add_self_loops(
                edge_index, xe, fill_value="mean", num_nodes=num_nodes
            )

        x_ = x.view(-1, self.heads, self.d_model)
        xe_ = self.lin_edge(xe).view(-1, self.heads, self.d_model)

        for i in range(self.num_relations):
            edge_mask = edge_type == i
            if self.add_self_loops:
                edge_mask = torch.cat(
                    [
                        edge_mask,
                        torch.ones(num_nodes, device=edge_mask.device).bool(),
                    ]
                )

            tmp = masked_edge_index(edge_index, edge_mask)

            h = self.propagate(tmp, x=x_, xe=xe_[edge_mask], rel_index=i)
            h = h.view(-1, self.in_channels)
            out += h @ self.weight[i]

        out += x @ self.root
        out += self.bias

        return out

    def message(self, x_j, xe, rel_index, index, ptr, size_i):
        # x_j has shape [E, out_channels]
        x = F.leaky_relu(x_j + xe)
        if self.share_att:
            att = self.att

        else:
            att = self.att[rel_index : rel_index + 1]

        alpha = (x * att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        return (x_j + xe) * alpha.unsqueeze(-1)
