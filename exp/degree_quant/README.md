# Degree-aware mixed-precision inference

This experiment measures checkpoint metric sensitivity when high-degree nodes
remain FP32 and all other nodes use a simulated INT8 path.

## Definition

- Nodes are ranked by in-degree independently inside every prompted graph.
- The high-precision set contains exactly `ceil(a / 100 * num_nodes)` nodes.
- Degree ties are resolved by stable node index order.
- `a` is evaluated at 20, 30, 40, and 50 by default.
- Activations use symmetric per-tensor INT8 quantize-dequantize (QDQ).
- RGCN relation and root weights use symmetric per-output-channel INT8 QDQ for
  low-precision target nodes. Bias, input projection, and prediction MLP stay
  FP32.
- A high-precision target still consumes messages from low-precision neighbors;
  this is inherent to mixed-precision message passing.

QDQ stays in floating point and is intended to measure numerical accuracy only.
It does not measure latency, memory reduction, or an integer deployment kernel.
The evaluator enables deterministic PyTorch algorithms and a deterministic
cuBLAS workspace configuration. Its FP32 row is therefore the comparison
baseline for the mixed-precision rows; it can differ slightly from a historical
run that used nondeterministic CUDA reduction paths.

## Run

```bash
CUDA_VISIBLE_DEVICES=1 /data1/xxr_data/new_conda/ofa/bin/python \
  exp/degree_quant/evaluate_degree_quant.py
```

Results are written incrementally to
`outputs/degree_quant/<timestamp>/results.{json,csv}`. A one-batch smoke run is:

```bash
CUDA_VISIBLE_DEVICES=1 /data1/xxr_data/new_conda/ofa/bin/python \
  exp/degree_quant/evaluate_degree_quant.py \
  --tasks cora_node --high-precision-percents 20 --max-batches 1 \
  --num-workers 0
```

Run unit tests with:

```bash
/data1/xxr_data/new_conda/ofa/bin/python -m unittest \
  exp.degree_quant.test_degree_quant
```

The completed six-task result table and observations are in
[`RESULTS.md`](RESULTS.md). A detailed Chinese design and implementation report
is available in [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md).
