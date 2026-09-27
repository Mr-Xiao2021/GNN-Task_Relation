"""Inference-only helpers for degree-aware mixed-precision simulation."""

import math
from typing import Optional

import torch


def _validate_num_bits(num_bits: int) -> None:
    if not isinstance(num_bits, int) or not 2 <= num_bits <= 16:
        raise ValueError(f"num_bits must be an integer in [2, 16], got {num_bits!r}")


def symmetric_fake_quantize(
    tensor: torch.Tensor,
    num_bits: int = 8,
    channel_axis: Optional[int] = None,
) -> torch.Tensor:
    """Quantize-dequantize a floating tensor with symmetric integer ranges.

    The returned tensor stays floating point. This intentionally simulates the
    numerical effect of integer inference without claiming integer-kernel speed.
    """
    _validate_num_bits(num_bits)
    if tensor.numel() == 0:
        return tensor.clone()
    if not tensor.is_floating_point():
        raise TypeError("symmetric_fake_quantize expects a floating-point tensor")

    quant_max = (1 << (num_bits - 1)) - 1
    if channel_axis is None:
        max_abs = tensor.detach().abs().amax()
    else:
        channel_axis %= tensor.ndim
        reduce_dims = tuple(dim for dim in range(tensor.ndim) if dim != channel_axis)
        max_abs = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True)

    scale = max_abs / quant_max
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(tensor / safe_scale).clamp(-quant_max, quant_max)
    return quantized * safe_scale


def mixed_fake_quantize(
    tensor: torch.Tensor,
    low_precision_mask: Optional[torch.Tensor],
    num_bits: int = 8,
) -> torch.Tensor:
    """Apply per-tensor fake quantization only to selected first-dimension rows."""
    if low_precision_mask is None:
        return tensor
    if low_precision_mask.dtype != torch.bool:
        raise TypeError("low_precision_mask must be a boolean tensor")
    if low_precision_mask.ndim != 1 or len(low_precision_mask) != len(tensor):
        raise ValueError("low_precision_mask must match tensor's first dimension")
    if not torch.any(low_precision_mask):
        return tensor

    output = tensor.clone()
    output[low_precision_mask] = symmetric_fake_quantize(
        tensor[low_precision_mask], num_bits=num_bits
    )
    return output


def mixed_precision_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    low_precision_mask: Optional[torch.Tensor],
    num_bits: int = 8,
    quantized_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run FP32 rows normally and selected rows through a fake-INT linear path."""
    if low_precision_mask is None or not torch.any(low_precision_mask):
        return inputs @ weight

    # Keep the reference GEMM shape unchanged for bitwise-reproducible FP32 rows.
    # Splitting those rows changes CUDA's reduction path and can alter predictions
    # after several message-passing layers.
    output = inputs @ weight
    quantized_inputs = symmetric_fake_quantize(
        inputs[low_precision_mask], num_bits=num_bits
    )
    if quantized_weight is None:
        quantized_weight = symmetric_fake_quantize(
            weight, num_bits=num_bits, channel_axis=1
        )
    quantized_output = symmetric_fake_quantize(
        quantized_inputs @ quantized_weight, num_bits=num_bits
    )
    output[low_precision_mask] = quantized_output
    return output


def high_degree_mask(
    edge_index: torch.Tensor,
    num_nodes: int,
    high_precision_percent: float,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select the exact top percentage of nodes by in-degree within each graph.

    A positive percentage selects ``ceil(percent * graph_nodes)`` nodes from
    every graph. Equal-degree nodes are resolved by their stable node order.
    """
    if not 0 <= high_precision_percent <= 100:
        raise ValueError(
            "high_precision_percent must be in [0, 100], "
            f"got {high_precision_percent}"
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    if batch.ndim != 1 or len(batch) != num_nodes:
        raise ValueError("batch must contain one graph id per node")

    degree = torch.bincount(edge_index[1], minlength=num_nodes)
    selected = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    if high_precision_percent == 0 or num_nodes == 0:
        return selected

    for graph_id in torch.unique(batch, sorted=True):
        node_indices = torch.nonzero(batch == graph_id, as_tuple=False).flatten()
        high_count = math.ceil(
            node_indices.numel() * float(high_precision_percent) / 100.0
        )
        if high_count == 0:
            continue
        order = torch.argsort(
            degree[node_indices], descending=True, stable=True
        )
        selected[node_indices[order[:high_count]]] = True
    return selected
