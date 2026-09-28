import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax, add_self_loops

from gp.nn.degree_quant import (
    mixed_int8_linear,
    mixed_int8_mean_aggregate,
    mixed_fake_quantize,
    mixed_precision_linear,
    symmetric_quantize_int8,
    symmetric_fake_quantize,
)


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
    ):
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)
        for i in range(self.num_relations):
            edge_mask = edge_type == i
            tmp = masked_edge_index(edge_index, edge_mask)

            h = self.propagate(tmp, x=x, xe=xe[edge_mask])
            out += h @ self.weight[i]

        out += x @ self.root
        out += self.bias

        return out

    def message(self, x_j, xe):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return (x_j + xe).relu()


class DegreeQuantRGCNEdgeConv(RGCNEdgeConv):
    """RGCN edge layer with degree-aware FP32/INT8 inference backends.

    The mask marks FP32 nodes. Other nodes use either the legacy QDQ simulation
    or a real INT8 GEMM path, with optional INT32 message aggregation.
    """

    def __init__(
        self, *args, num_bits: int = 8, quantization_backend: str = "qdq", **kwargs
    ):
        self.num_bits = num_bits
        self.quantization_backend = None
        self._quantized_weight = None
        self._quantized_root = None
        self._int8_fused_weight = None
        self._int8_fused_weight_scale = None
        self._fused_weight = None
        self._quantized_parameter_versions = None
        super().__init__(*args, **kwargs)
        self.set_quantization_backend(quantization_backend)

    def reset_parameters(self):
        super().reset_parameters()
        self._quantized_weight = None
        self._quantized_root = None
        self._int8_fused_weight = None
        self._int8_fused_weight_scale = None
        self._fused_weight = None
        self._quantized_parameter_versions = None

    def set_quantization_backend(self, backend):
        if backend not in {"qdq", "int8", "int8_full"}:
            raise ValueError(
                "quantization_backend must be 'qdq', 'int8', or 'int8_full'"
            )
        if backend in {"int8", "int8_full"} and self.num_bits != 8:
            raise ValueError("The real integer kernel currently supports INT8 only")
        self.quantization_backend = backend

    def _quantized_parameters(self):
        versions = (self.weight._version, self.root._version)
        cache_is_current = (
            self._quantized_parameter_versions == versions
            and self._quantized_weight is not None
            and self._quantized_weight.device == self.weight.device
            and self._quantized_weight.dtype == self.weight.dtype
        )
        if not cache_is_current:
            self._quantized_weight = torch.stack(
                [
                    symmetric_fake_quantize(
                        relation_weight,
                        num_bits=self.num_bits,
                        channel_axis=1,
                    )
                    for relation_weight in self.weight
                ]
            )
            self._quantized_root = symmetric_fake_quantize(
                self.root, num_bits=self.num_bits, channel_axis=1
            )
            self._quantized_parameter_versions = versions
        return self._quantized_weight, self._quantized_root

    def _int8_quantized_parameters(self):
        versions = (self.weight._version, self.root._version)
        cache_is_current = (
            self._quantized_parameter_versions == versions
            and self._int8_fused_weight is not None
            and self._int8_fused_weight.device == self.weight.device
        )
        if not cache_is_current:
            self._fused_weight = torch.cat(
                [*[relation_weight for relation_weight in self.weight], self.root],
                dim=0,
            )
            (
                self._int8_fused_weight,
                self._int8_fused_weight_scale,
            ) = symmetric_quantize_int8(
                self._fused_weight, channel_axis=1
            )
            self._quantized_parameter_versions = versions
        return (
            self._fused_weight,
            self._int8_fused_weight,
            self._int8_fused_weight_scale,
        )

    def forward(
        self,
        x: OptTensor,
        xe: OptTensor,
        edge_index: Adj,
        edge_type: OptTensor = None,
        high_precision_mask: OptTensor = None,
    ):
        if high_precision_mask is None:
            return super().forward(x, xe, edge_index, edge_type)
        if high_precision_mask.dtype != torch.bool or len(high_precision_mask) != len(x):
            raise ValueError("high_precision_mask must be boolean with one entry per node")

        if self.quantization_backend in {"int8", "int8_full"}:
            return self._int8_forward(
                x, xe, edge_index, edge_type, high_precision_mask
            )

        low_precision_mask = ~high_precision_mask
        mixed_x = mixed_fake_quantize(x, low_precision_mask, self.num_bits)
        quantized_weight, quantized_root = self._quantized_parameters()
        out = torch.zeros(x.size(0), self.out_channels, device=x.device, dtype=x.dtype)

        for relation in range(self.num_relations):
            relation_mask = edge_type == relation
            relation_edges = masked_edge_index(edge_index, relation_mask)
            edge_low_precision = low_precision_mask[relation_edges[0]]
            relation_edge_attr = mixed_fake_quantize(
                xe[relation_mask], edge_low_precision, self.num_bits
            )
            aggregated = self.propagate(
                relation_edges,
                x=mixed_x,
                xe=relation_edge_attr,
                low_precision_mask=edge_low_precision,
            )
            aggregated = mixed_fake_quantize(
                aggregated, low_precision_mask, self.num_bits
            )
            out += mixed_precision_linear(
                aggregated,
                self.weight[relation],
                low_precision_mask,
                self.num_bits,
                quantized_weight=quantized_weight[relation],
            )

        out += mixed_precision_linear(
            mixed_x,
            self.root,
            low_precision_mask,
            self.num_bits,
            quantized_weight=quantized_root,
        )
        out += self.bias
        return mixed_fake_quantize(out, low_precision_mask, self.num_bits)

    def _int8_forward(self, x, xe, edge_index, edge_type, high_precision_mask):
        low_precision_mask = ~high_precision_mask
        mixed_x = mixed_fake_quantize(x, low_precision_mask, self.num_bits)
        fused_weight, quantized_weight, weight_scale = (
            self._int8_quantized_parameters()
        )
        aggregated_inputs = []

        for relation in range(self.num_relations):
            relation_mask = edge_type == relation
            relation_edges = masked_edge_index(edge_index, relation_mask)
            if relation_edges.shape[1] == 0:
                aggregated_inputs.append(
                    torch.zeros(
                        x.size(0), self.out_channels, device=x.device, dtype=x.dtype
                    )
                )
                continue
            source_low_precision = low_precision_mask[relation_edges[0]]
            relation_edge_attr = mixed_fake_quantize(
                xe[relation_mask], source_low_precision, self.num_bits
            )
            if self.quantization_backend == "int8_full":
                messages = (
                    mixed_x[relation_edges[0]] + relation_edge_attr
                ).relu()
                aggregated = mixed_int8_mean_aggregate(
                    messages,
                    relation_edges[1],
                    source_low_precision,
                    x.size(0),
                )
            else:
                aggregated = self.propagate(
                    relation_edges,
                    x=mixed_x,
                    xe=relation_edge_attr,
                    low_precision_mask=source_low_precision,
                )
            aggregated = mixed_fake_quantize(
                aggregated, low_precision_mask, self.num_bits
            )
            aggregated_inputs.append(aggregated)

        fused_input = torch.cat([*aggregated_inputs, mixed_x], dim=1)
        out = mixed_int8_linear(
            fused_input,
            fused_weight,
            low_precision_mask,
            quantized_weight=quantized_weight,
            weight_scale=weight_scale,
        )
        out += self.bias
        return mixed_fake_quantize(out, low_precision_mask, self.num_bits)

    def message(self, x_j, xe, low_precision_mask=None):
        message = (x_j + xe).relu()
        return mixed_fake_quantize(message, low_precision_mask, self.num_bits)


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
