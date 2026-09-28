# Degree Quant 量化源码汇编

本文冻结并整理首版 Degree-Aware FP32/INT8 QDQ 推理实验中所有直接参与量化决策、QDQ 计算、混合精度消息传递和量化评测的源码。

> 版本说明：本文对应首次实现提交 `e7278b9`，用于保留可审计的原始 QDQ 来源。后续“源图全局度数 + 真实 INT8 kernel”实现直接演进自这些本仓库函数，未复制 MixQ 源码；其当前结构、差异和结果见 [`GLOBAL_INT8_KERNEL_REPORT.md`](GLOBAL_INT8_KERNEL_REPORT.md)，MixQ 的借鉴点见 [`MIXQ_IMPLEMENTATION_NOTES.md`](MIXQ_IMPLEMENTATION_NOTES.md)。

## 1. 来源与版本

| 项目 | 内容 |
| --- | --- |
| 开发分支 | `quant/dq` |
| 实现提交 | `e7278b95f08415bca1065c269dd9da74f0aa043a` |
| 基础提交 | `31a4cf6` |
| 原始骨干 | `RGCNEdgeConv`、`PyGRGCNEdge` |
| 框架依赖 | PyTorch、PyTorch Geometric、TorchMetrics |
| 外部 DQ 源码 | 未复制外部 GitHub DQ 仓库代码 |

本实现是针对当前仓库的 RGCN edge message-passing 结构新增的 inference-only PTQ/QDQ 实现。核心算法代码在提交 `e7278b9` 中首次加入；原有 RGCN 层和多层模型来自父提交 `31a4cf6`。

可以用以下命令读取该提交中的原始文件，避免后续分支修改造成歧义：

```bash
git show e7278b9:gp/nn/degree_quant.py
git show e7278b9:gp/nn/layer/pyg.py
git show e7278b9:models/model.py
git show e7278b9:exp/degree_quant/evaluate_degree_quant.py
git show e7278b9:exp/degree_quant/test_degree_quant.py
```

## 2. 调用链

```text
evaluate_degree_quant.py
  -> PyGDegreeQuantRGCNEdge.set_high_precision_percent(a)
  -> PyGDegreeQuantRGCNEdge.forward(graph)
       -> high_degree_mask(...)
       -> DegreeQuantRGCNEdgeConv.forward(..., high_precision_mask)
            -> mixed_fake_quantize(node / edge / message / aggregate)
            -> symmetric_fake_quantize(weight)
            -> mixed_precision_linear(...)
       -> mixed_fake_quantize(layer output)
  -> CPU Accuracy / AUROC
```

## 3. 基础量化与度数分组

来源：`gp/nn/degree_quant.py`，提交 `e7278b9`。以下为该文件完整源码。

```python
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
```

### 3.1 五个函数的职责

| 函数 | 量化职责 |
| --- | --- |
| `_validate_num_bits` | 限制 bit-width 为 2 到 16 |
| `symmetric_fake_quantize` | 计算 symmetric scale，执行 round/clamp/dequantize |
| `mixed_fake_quantize` | 只替换低精度 mask 对应的行 |
| `mixed_precision_linear` | FP32 参考 GEMM + 低精度行的 activation/weight/output QDQ |
| `high_degree_mask` | 每张图独立选择前 `ceil(a% * N)` 个入度最高节点 |

## 4. RGCN 层级混合精度

来源：`gp/nn/layer/pyg.py:10` 和 `gp/nn/layer/pyg.py:79`，提交 `e7278b9`。

新增 import：

```python
from gp.nn.degree_quant import (
    mixed_fake_quantize,
    mixed_precision_linear,
    symmetric_fake_quantize,
)
```

新增 `DegreeQuantRGCNEdgeConv` 完整源码：

```python
class DegreeQuantRGCNEdgeConv(RGCNEdgeConv):
    """RGCN edge layer with deterministic degree-aware mixed fake quantization.

    The mask marks FP32 nodes. Other nodes use a quantize-dequantize path for
    inputs, weights, messages, aggregation outputs, and layer updates.
    """

    def __init__(self, *args, num_bits: int = 8, **kwargs):
        self.num_bits = num_bits
        self._quantized_weight = None
        self._quantized_root = None
        self._quantized_parameter_versions = None
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        super().reset_parameters()
        self._quantized_weight = None
        self._quantized_root = None
        self._quantized_parameter_versions = None

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

    def message(self, x_j, xe, low_precision_mask=None):
        message = (x_j + xe).relu()
        return mixed_fake_quantize(message, low_precision_mask, self.num_bits)
```

### 4.1 原始父类来源

量化层继承的 `RGCNEdgeConv` 来自基础提交 `31a4cf6` 的 `gp/nn/layer/pyg.py`。其原始计算骨架如下：

```python
class RGCNEdgeConv(MessagePassing):
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
        return (x_j + xe).relu()
```

量化子类没有改变 `weight`、`root`、`bias` 的参数名或 shape，因此原 checkpoint 可以 strict-load。

## 5. 模型级 Degree Mask 与跨层传播

来源：`models/model.py:39` 和 `models/model.py:672`，提交 `e7278b9`。

新增 import：

```python
from gp.nn.degree_quant import high_degree_mask, mixed_fake_quantize
from gp.nn.layer.pyg import DegreeQuantRGCNEdgeConv, RGCNEdgeConv
```

新增 `PyGDegreeQuantRGCNEdge` 完整源码：

```python
class PyGDegreeQuantRGCNEdge(PyGRGCNEdge):
    """Inference-only FP32/INT mixed-precision variant of ``PyGRGCNEdge``."""

    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
        high_precision_percent=None,
        quant_bits=8,
    ):
        self.quant_bits = quant_bits
        self.high_precision_percent = None
        super().__init__(
            num_layers,
            num_rels,
            inp_dim,
            out_dim,
            drop_ratio=drop_ratio,
            JK=JK,
            batch_norm=batch_norm,
        )
        self.set_high_precision_percent(high_precision_percent)

    def build_input_layer(self):
        return DegreeQuantRGCNEdgeConv(
            self.inp_dim,
            self.out_dim,
            self.num_rels,
            num_bits=self.quant_bits,
        )

    def build_hidden_layer(self):
        return DegreeQuantRGCNEdgeConv(
            self.inp_dim,
            self.out_dim,
            self.num_rels,
            num_bits=self.quant_bits,
        )

    def set_high_precision_percent(self, high_precision_percent):
        if high_precision_percent is not None and not 0 <= high_precision_percent <= 100:
            raise ValueError("high_precision_percent must be None or in [0, 100]")
        self.high_precision_percent = high_precision_percent

    def forward(self, g, drop_mask=None):
        if self.training and self.high_precision_percent is not None:
            raise RuntimeError("Degree-aware quantization is inference-only; call model.eval()")

        if self.high_precision_percent is None:
            high_precision = None
            low_precision = None
        else:
            high_precision = high_degree_mask(
                g.edge_index,
                g.x.size(0),
                self.high_precision_percent,
                getattr(g, "batch", None),
            )
            low_precision = ~high_precision

        h_list = []
        message = self.build_message_from_input(g)
        for layer in range(self.num_layers):
            h = self.conv[layer](
                message["h"],
                message["he"],
                message["g"],
                message["e"],
                high_precision,
            )
            if self.batch_norm:
                h = self.batch_norm[layer](h)
            if layer != self.num_layers - 1:
                h = F.relu(h)
            if self.drop_ratio is not None:
                dropped_h = F.dropout(h, p=self.drop_ratio, training=self.training)
                if drop_mask is not None:
                    h = (
                        drop_mask.view(-1, 1) * dropped_h
                        + torch.logical_not(drop_mask).view(-1, 1) * h
                    )
                else:
                    h = dropped_h
            h = mixed_fake_quantize(h, low_precision, self.quant_bits)
            message = self.build_message_from_output(g, h)
            h_list.append(h)

        if self.JK == "last":
            return h_list[-1]
        if self.JK == "sum":
            return torch.stack(h_list).sum(dim=0)
        if self.JK == "mean":
            return torch.stack(h_list).mean(dim=0)
        return h_list
```

### 5.1 原始父类来源

父类 `PyGRGCNEdge` 来自基础提交 `31a4cf6` 的 `models/model.py`。量化模型复用了它的参数布局和图消息字段：

```python
class PyGRGCNEdge(MultiLayerMessagePassing):
    def __init__(
        self,
        num_layers: int,
        num_rels: int,
        inp_dim: int,
        out_dim: int,
        drop_ratio=0,
        JK="last",
        batch_norm=True,
    ):
        super().__init__(
            num_layers, inp_dim, out_dim, drop_ratio, JK, batch_norm
        )
        self.num_rels = num_rels
        self.build_layers()

    def build_input_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_hidden_layer(self):
        return RGCNEdgeConv(self.inp_dim, self.out_dim, self.num_rels)

    def build_message_from_input(self, g):
        return {
            "g": g.edge_index,
            "h": g.x,
            "e": g.edge_type,
            "he": g.edge_attr,
        }

    def build_message_from_output(self, g, h):
        return {"g": g.edge_index, "h": h, "e": g.edge_type, "he": g.edge_attr}
```

## 6. 评测入口中的量化相关代码

来源：`exp/degree_quant/evaluate_degree_quant.py`，提交 `e7278b9`。

### 6.1 确定性 CUDA 设置

```python
# Required by deterministic CUDA matrix multiplication. Set it before the first
# CUDA context is created so repeated evaluations use the same reduction path.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torchmetrics import AUROC, Accuracy
```

在 `main` 中启用：

```python
torch.use_deterministic_algorithms(True)
torch.set_float32_matmul_precision("high")
```

### 6.2 构建量化兼容模型

```python
backbone = PyGDegreeQuantRGCNEdge(
    num_layers=6,
    num_rels=5,
    inp_dim=768,
    out_dim=768,
    drop_ratio=0.0,
    JK="last",
    high_precision_percent=None,
    quant_bits=8,
)
model = BinGraphModel(
    model=backbone,
    llm_name="ST",
    outdim=768,
    task_dim=1,
    add_rwpe=None,
    dropout=0.0,
)
```

`high_precision_percent=None` 表示 FP32 baseline。载入 checkpoint 后只切换该字段，不重新初始化参数。

### 6.3 CPU 指标累计

```python
def build_metric(test_data):
    if test_data.metric == "acc":
        metric = Accuracy(task="multiclass", num_classes=test_data.classes)
    elif test_data.metric == "auc":
        metric = AUROC(task="binary")
    else:
        raise NotImplementedError(f"Unsupported metric: {test_data.metric}")
    return metric


def update_metric(metric, metric_name, output, batch):
    """Accumulate metrics on CPU to keep deterministic GPU inference enabled."""
    num_classes = int(batch.num_classes[0])
    logits = output.view(-1, num_classes)
    if metric_name == "acc":
        predictions = logits.detach().cpu()
        targets = batch.y.view(-1).to(dtype=torch.long, device="cpu")
    elif metric_name == "auc":
        predictions = torch.softmax(logits, dim=-1)[:, -1].detach().cpu()
        targets = batch.y[:, -1].reshape(-1).detach().cpu()
    else:
        raise NotImplementedError(f"Unsupported metric: {metric_name}")
    metric.update(predictions, targets)
```

### 6.4 单个精度配置的推理

```python
def evaluate(
    model,
    data_module,
    test_data,
    device,
    high_precision_percent,
    seed,
    max_batches,
):
    set_random_seed(seed)
    model.model.set_high_precision_percent(high_precision_percent)
    model.eval()
    metric = build_metric(test_data)
    loader = data_module.test_dataloader()[0]

    batches = 0
    examples = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            update_metric(metric, test_data.metric, output, batch)
            batches += 1
            examples += int(batch.num_graphs)
    synchronize(device)
    elapsed = time.perf_counter() - started
    value = float(metric.compute().detach().cpu())
    return value, batches, examples, elapsed
```

### 6.5 FP32 与四档 Degree Quant 循环

```python
baseline, batches, examples, elapsed = evaluate(
    model,
    data_module,
    test_data,
    device,
    None,
    args.seed,
    args.max_batches,
)

for percent in args.high_precision_percents:
    value, batches, examples, elapsed = evaluate(
        model,
        data_module,
        test_data,
        device,
        percent,
        args.seed,
        args.max_batches,
    )
    delta = value - baseline
    row = {
        "task": task_name,
        "split": test_data.state_name,
        "metric_name": test_data.metric,
        "mode": "degree_mixed_int8",
        "high_precision_percent": float(percent),
        "low_precision_percent": 100.0 - float(percent),
        "metric": value,
        "delta_from_fp32": delta,
        "relative_delta_percent": 100.0 * delta / abs(baseline) if baseline else None,
        "batches": batches,
        "examples": examples,
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
    }
```

评测文件中的 CLI 参数解析、checkpoint 字典和 JSON/CSV 原子写盘不参与量化数值计算，完整源码可通过：

```bash
git show e7278b9:exp/degree_quant/evaluate_degree_quant.py
```

## 7. 量化测试源码

来源：`exp/degree_quant/test_degree_quant.py`，提交 `e7278b9`。

### 7.1 度数 mask 测试

```python
class DegreeMaskTest(unittest.TestCase):
    def test_selects_exact_top_percent_per_graph(self):
        first = Data(
            x=torch.zeros(5, 1),
            edge_index=torch.tensor(
                [[0, 0, 1, 0, 1, 2, 0, 1, 2, 3], [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]]
            ),
        )
        second = Data(
            x=torch.zeros(3, 1),
            edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        )
        batch = Batch.from_data_list([first, second])

        mask = high_degree_mask(batch.edge_index, batch.num_nodes, 40, batch.batch)

        self.assertEqual(mask[:5].nonzero().flatten().tolist(), [3, 4])
        self.assertEqual(mask[5:].nonzero().flatten().tolist(), [1, 2])

    def test_ties_use_stable_node_order(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        mask = high_degree_mask(edge_index, 4, 50)
        self.assertEqual(mask.nonzero().flatten().tolist(), [0, 1])
```

### 7.2 QDQ 与 mixed linear 测试

```python
class FakeQuantizationTest(unittest.TestCase):
    def test_mixed_quantization_preserves_high_precision_rows(self):
        tensor = torch.tensor([[0.12345, -0.9987], [0.3333, 0.7777]])
        low_mask = torch.tensor([False, True])
        output = mixed_fake_quantize(tensor, low_mask, num_bits=8)

        torch.testing.assert_close(output[0], tensor[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(output[1], tensor[1]))

    def test_zero_tensor_remains_zero(self):
        output = symmetric_fake_quantize(torch.zeros(3, 4), num_bits=8)
        self.assertTrue(torch.equal(output, torch.zeros(3, 4)))

    def test_mixed_linear_matches_reference_formulation(self):
        torch.manual_seed(11)
        inputs = torch.randn(5, 4)
        weight = torch.randn(4, 3)
        low_mask = torch.tensor([False, True, False, True, True])

        expected = inputs @ weight
        quantized_inputs = symmetric_fake_quantize(inputs[low_mask], num_bits=8)
        quantized_weight = symmetric_fake_quantize(
            weight, num_bits=8, channel_axis=1
        )
        expected[low_mask] = symmetric_fake_quantize(
            quantized_inputs @ quantized_weight, num_bits=8
        )

        actual = mixed_precision_linear(inputs, weight, low_mask, num_bits=8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
```

### 7.3 checkpoint 兼容与 inference-only 测试

```python
class DegreeQuantModelTest(unittest.TestCase):
    def test_state_dict_is_checkpoint_compatible(self):
        base = PyGRGCNEdge(2, 2, 4, 4, drop_ratio=0.0, batch_norm=False)
        quantized = PyGDegreeQuantRGCNEdge(
            2, 2, 4, 4, drop_ratio=0.0, batch_norm=False
        )
        self.assertEqual(set(base.state_dict()), set(quantized.state_dict()))
        quantized.load_state_dict(base.state_dict(), strict=True)

    def test_all_high_precision_matches_original_model(self):
        torch.manual_seed(7)
        base = PyGRGCNEdge(2, 2, 4, 4, drop_ratio=0.0, batch_norm=False)
        quantized = PyGDegreeQuantRGCNEdge(
            2,
            2,
            4,
            4,
            drop_ratio=0.0,
            batch_norm=False,
            high_precision_percent=100,
        )
        quantized.load_state_dict(base.state_dict(), strict=True)
        base.eval()
        quantized.eval()
        graph = self._graph()

        torch.testing.assert_close(
            quantized(graph.clone()), base(graph.clone()), rtol=0, atol=0
        )

    def test_quantized_mode_rejects_training(self):
        model = PyGDegreeQuantRGCNEdge(
            2, 2, 4, 4, high_precision_percent=20
        )
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            model(self._graph())
```

完整测试文件还包含 import 和 `_graph` fixture，可通过以下命令读取：

```bash
git show e7278b9:exp/degree_quant/test_degree_quant.py
```

## 8. 量化边界总表

| 位置 | mask 依据 | activation | weight | 源码入口 |
| --- | --- | --- | --- | --- |
| 节点输入 `x` | 节点自身 | 低精度行 per-tensor QDQ | - | `DegreeQuantRGCNEdgeConv.forward` |
| relation edge attribute | source node | 低精度边 per-tensor QDQ | - | `DegreeQuantRGCNEdgeConv.forward` |
| message `(x_j + xe).relu()` | source node | 低精度 message QDQ | - | `DegreeQuantRGCNEdgeConv.message` |
| relation aggregate | target node | 低精度行 QDQ | - | `DegreeQuantRGCNEdgeConv.forward` |
| relation linear | target node | input/output QDQ | per-output-channel QDQ | `mixed_precision_linear` |
| root linear | target node | input/output QDQ | per-output-channel QDQ | `mixed_precision_linear` |
| layer update | target node | 低精度行 QDQ | bias 保持 FP32 | `DegreeQuantRGCNEdgeConv.forward` |
| BN/ReLU/dropout 后 | target node | 低精度行 QDQ | - | `PyGDegreeQuantRGCNEdge.forward` |
| 输入 projection | 不量化 | FP32 | FP32 | `BinGraphModel.initial_projection` |
| 最终 prediction MLP | 不量化 | FP32 | FP32 | `BinGraphModel.forward` |

## 9. 与“真实 INT8 推理”的区别

上述源码实现的是 QDQ 数值仿真：张量经过量化、舍入、截断后再反量化为浮点数，后续矩阵乘和 message passing 仍由浮点 kernel 执行。因此它可以用于观察 Accuracy/AUROC 的变化，但不能用于证明真实 INT8 kernel 的延迟、吞吐、能耗或显存收益。

如果下一阶段需要部署性能，需要在保持相同 `high_degree_mask` 语义的前提下，将 `mixed_precision_linear` 和图聚合替换为真实的分组 INT8 GEMM / message-passing kernel。
