# Global degree-aware mixed-precision inference

This experiment measures checkpoint metric sensitivity and inference cost when
globally high-degree real nodes remain FP32 and all other real nodes use an
INT8 path.

## Definition

- Real nodes are ranked once on the full source graph by in-degree, not inside
  each sampled or prompted subgraph.
- The global high-precision set contains exactly
  `ceil(a / 100 * source_graph_num_nodes)` nodes.
- Degree ties are resolved by stable source node ID order.
- Prompt-only NOI/class nodes have no source-graph identity and always remain
  FP32.
- Link-prediction and knowledge-graph ranks use the model-visible training
  graph, so held-out edges cannot leak into the precision assignment.
- `a` is evaluated at 20, 30, 40, and 50 by default.

The realized high-precision occurrence ratio in sampled test subgraphs is not
expected to equal `a`: high-degree source nodes can be sampled more often.

## Backends

- `int8` (default): low-precision node projections use actual INT8 tensors,
  `torch._int_mm` (cuBLASLt on CUDA), INT32 accumulators, and explicit
  dequantization. Relation and root projections are algebraically fused into
  one mixed FP32/INT8 GEMM per layer. Messages are QDQ-quantized and aggregated
  with the regular floating-point PyG scatter path.
- `int8_full`: uses the same true INT8 GEMM and additionally performs the
  low-precision message sum with INT8 values and INT32 `scatter_add_`. This is
  an experimental coverage backend and is currently slower.
- `qdq`: the original floating-point quantize-dequantize simulation, retained
  for numerical comparison.

Bias, input projection, prediction MLP, and all prompt-only nodes stay FP32.
The INT8 paths are inference-only and do not perform quantization-aware
training.

## Metrics

The evaluator records the final task metric, model and end-to-end throughput,
peak CUDA memory, FP32/INT8 MAC counts, and two operation-cost estimates:

- `mixq_bitops_proxy`: operation count multiplied by the selected precision.
  This follows the proxy convention used by MixQ's utility code.
- `matmul_bitops`: MAC count multiplied by activation bits and weight bits.
  This is the more conventional bit-operation estimate for matrix products.

The proxy is analytical and must not be interpreted as measured speedup.
Dynamic row selection, activation quantization, and split FP32/INT8 launches can
make a lower-BitOP implementation slower than the original dense FP32 kernel.

## Run

```bash
CUDA_VISIBLE_DEVICES=1 /data1/xxr_data/new_conda/ofa/bin/python \
  exp/degree_quant/evaluate_degree_quant.py \
  --tasks cora_node cora_link pubmed_node pubmed_link arxiv WN18RR \
  --high-precision-percents 20 30 40 50 \
  --quantization-backend int8 \
  --batch-size 128 --num-workers 4 \
  --output-dir outputs/degree_quant/global_degree_int8_six_tasks_a20_30_40_50
```

Results are written atomically after every condition to `results.json` and
`results.csv`. A short smoke run is:

```bash
CUDA_VISIBLE_DEVICES=1 /data1/xxr_data/new_conda/ofa/bin/python \
  exp/degree_quant/evaluate_degree_quant.py \
  --tasks cora_node --high-precision-percents 20 --max-batches 2 \
  --quantization-backend int8 --num-workers 0
```

Run unit tests with:

```bash
CUDA_VISIBLE_DEVICES=1 /data1/xxr_data/new_conda/ofa/bin/python -m unittest -v \
  exp.degree_quant.test_degree_quant
```

The completed six-task results and implementation discussion are in
[`GLOBAL_INT8_KERNEL_REPORT.md`](GLOBAL_INT8_KERNEL_REPORT.md). The earlier QDQ
experiment is preserved in [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md)
and [`RESULTS.md`](RESULTS.md). MixQ implementation notes and source provenance
are in [`MIXQ_IMPLEMENTATION_NOTES.md`](MIXQ_IMPLEMENTATION_NOTES.md) and
[`QUANTIZATION_SOURCE_CODE.md`](QUANTIZATION_SOURCE_CODE.md).
