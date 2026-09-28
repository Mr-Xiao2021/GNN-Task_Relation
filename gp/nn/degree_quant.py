"""Inference-only helpers for degree-aware mixed-precision execution."""

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
    selected = tensor[low_precision_mask]
    if selected.numel() == 0:
        return tensor

    output = tensor.clone()
    output[low_precision_mask] = symmetric_fake_quantize(
        selected, num_bits=num_bits
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
    if low_precision_mask is None:
        return inputs @ weight
    low_inputs = inputs[low_precision_mask]
    if low_inputs.numel() == 0:
        return inputs @ weight

    # Keep the reference GEMM shape unchanged for bitwise-reproducible FP32 rows.
    # Splitting those rows changes CUDA's reduction path and can alter predictions
    # after several message-passing layers.
    output = inputs @ weight
    quantized_inputs = symmetric_fake_quantize(low_inputs, num_bits=num_bits)
    if quantized_weight is None:
        quantized_weight = symmetric_fake_quantize(
            weight, num_bits=num_bits, channel_axis=1
        )
    quantized_output = symmetric_fake_quantize(
        quantized_inputs @ quantized_weight, num_bits=num_bits
    )
    output[low_precision_mask] = quantized_output
    return output


def symmetric_quantize_int8(
    tensor: torch.Tensor,
    channel_axis: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed INT8 values and their symmetric dequantization scale."""
    if tensor.numel() == 0:
        raise ValueError("Cannot quantize an empty tensor")
    if not tensor.is_floating_point():
        raise TypeError("symmetric_quantize_int8 expects a floating-point tensor")

    if channel_axis is None:
        max_abs = tensor.detach().abs().amax()
    else:
        channel_axis %= tensor.ndim
        reduce_dims = tuple(dim for dim in range(tensor.ndim) if dim != channel_axis)
        max_abs = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True)

    scale = max_abs / 127
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(tensor / safe_scale).clamp(-127, 127).to(torch.int8)
    return quantized, safe_scale


def _int8_mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Execute INT8 x INT8 -> INT32 matrix multiplication.

    CUDA's cuBLASLt INT8 path used by ``torch._int_mm`` requires the row count
    to be a positive multiple of 32 on the target H100, so mixed node groups
    are padded and sliced back. The CPU path is a correctness reference for
    tests; production performance measurements use CUDA.
    """
    if lhs.dtype != torch.int8 or rhs.dtype != torch.int8:
        raise TypeError("_int8_mm expects INT8 inputs")
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[1] != rhs.shape[0]:
        raise ValueError("Incompatible INT8 matrix shapes")

    if lhs.device.type != "cuda":
        return lhs.to(torch.int32) @ rhs.to(torch.int32)

    rows = lhs.shape[0]
    padded_rows = max(32, math.ceil(rows / 32) * 32)
    if padded_rows != rows:
        padded = torch.zeros(
            (padded_rows, lhs.shape[1]), device=lhs.device, dtype=torch.int8
        )
        padded[:rows] = lhs
        lhs = padded
    return torch._int_mm(lhs.contiguous(), rhs.contiguous())[:rows]


def mixed_int8_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    low_precision_mask: torch.Tensor,
    quantized_weight: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run FP32 rows with GEMM and low rows with a real INT8 GEMM kernel."""
    if low_precision_mask.dtype != torch.bool:
        raise TypeError("low_precision_mask must be boolean")
    if low_precision_mask.ndim != 1 or len(low_precision_mask) != len(inputs):
        raise ValueError("low_precision_mask must match inputs' first dimension")
    low_inputs = inputs[low_precision_mask]
    if low_inputs.numel() == 0:
        return inputs @ weight

    output = torch.empty(
        (inputs.shape[0], weight.shape[1]),
        device=inputs.device,
        dtype=inputs.dtype,
    )
    high_precision_mask = ~low_precision_mask
    high_inputs = inputs[high_precision_mask]
    if high_inputs.numel() > 0:
        output[high_precision_mask] = high_inputs @ weight

    q_inputs, input_scale = symmetric_quantize_int8(low_inputs)
    if quantized_weight is None or weight_scale is None:
        quantized_weight, weight_scale = symmetric_quantize_int8(
            weight, channel_axis=1
        )
    accumulator = _int8_mm(q_inputs, quantized_weight)
    low_output = accumulator.to(inputs.dtype) * input_scale * weight_scale
    output[low_precision_mask] = symmetric_fake_quantize(low_output, num_bits=8)
    return output


def mixed_int8_mean_aggregate(
    messages: torch.Tensor,
    target_index: torch.Tensor,
    low_precision_mask: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Mean-aggregate low messages with INT8 values and INT32 accumulators."""
    if len(messages) != len(target_index) or len(messages) != len(low_precision_mask):
        raise ValueError("messages, target_index, and mask must have equal lengths")
    if low_precision_mask.dtype != torch.bool:
        raise TypeError("low_precision_mask must be boolean")

    feature_dim = messages.shape[1]
    output = torch.zeros(
        (num_nodes, feature_dim), device=messages.device, dtype=messages.dtype
    )

    low_messages = messages[low_precision_mask]
    if low_messages.numel() > 0:
        q_messages, message_scale = symmetric_quantize_int8(low_messages)
        int32_sum = torch.zeros(
            (num_nodes, feature_dim), device=messages.device, dtype=torch.int32
        )
        low_targets = target_index[low_precision_mask]
        int32_sum.scatter_add_(
            0,
            low_targets.view(-1, 1).expand(-1, feature_dim),
            q_messages.to(torch.int32),
        )
        output += int32_sum.to(messages.dtype) * message_scale

    high_precision_mask = ~low_precision_mask
    high_messages = messages[high_precision_mask]
    if high_messages.numel() > 0:
        high_targets = target_index[high_precision_mask]
        output.scatter_add_(
            0,
            high_targets.view(-1, 1).expand(-1, feature_dim),
            high_messages,
        )

    counts = torch.bincount(target_index, minlength=num_nodes).clamp_min(1)
    return output / counts.to(messages.dtype).view(-1, 1)


def global_degree_mask(
    global_degree_rank: torch.Tensor,
    global_num_nodes: torch.Tensor,
    real_node_mask: torch.Tensor,
    high_precision_percent: float,
) -> torch.Tensor:
    """Select real nodes by their stable degree rank in the source graph.

    Rank zero is the highest-degree source-graph node. Artificial NOI and class
    prompt nodes have no source-graph identity and always stay in FP32.
    """
    if not 0 <= high_precision_percent <= 100:
        raise ValueError(
            "high_precision_percent must be in [0, 100], "
            f"got {high_precision_percent}"
        )
    if global_degree_rank.ndim != 1:
        raise ValueError("global_degree_rank must be one-dimensional")
    if global_num_nodes.shape != global_degree_rank.shape:
        raise ValueError("global_num_nodes must match global_degree_rank")
    if real_node_mask.shape != global_degree_rank.shape:
        raise ValueError("real_node_mask must match global_degree_rank")
    if real_node_mask.dtype != torch.bool:
        raise TypeError("real_node_mask must be boolean")

    cutoffs = torch.ceil(
        global_num_nodes.to(torch.float64) * float(high_precision_percent) / 100.0
    ).to(torch.long)
    selected_real = real_node_mask & (global_degree_rank >= 0) & (
        global_degree_rank < cutoffs
    )
    return selected_real | ~real_node_mask


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
