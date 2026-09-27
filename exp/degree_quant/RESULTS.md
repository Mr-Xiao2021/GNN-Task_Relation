# Degree-aware FP32/INT8 inference results

Run date: 2026-09-27

The table reports each task's primary test metric. Values in parentheses are
absolute changes from the FP32 row evaluated with the same checkpoint, seed,
batch size, and deterministic execution settings.

| Task | Metric | FP32 | a=20 | a=30 | a=40 | a=50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora node | Accuracy | 0.702611 | 0.701161 (-0.001451) | 0.700677 (-0.001934) | 0.705029 (+0.002418) | 0.703578 (+0.000967) |
| Cora link | AUROC | 0.948873 | 0.948146 (-0.000726) | 0.947926 (-0.000947) | 0.948159 (-0.000714) | 0.948918 (+0.000045) |
| PubMed node | Accuracy | 0.719215 | 0.720102 (+0.000887) | 0.718954 (-0.000261) | 0.719841 (+0.000626) | 0.719528 (+0.000313) |
| PubMed link | AUROC | 0.980520 | 0.980388 (-0.000131) | 0.980456 (-0.000064) | 0.980363 (-0.000157) | 0.980555 (+0.000035) |
| Arxiv | Accuracy | 0.709154 | 0.709504 (+0.000350) | 0.709113 (-0.000041) | 0.708763 (-0.000391) | 0.709586 (+0.000432) |
| WN18RR | Accuracy | 0.973516 | 0.973516 (+0.000000) | 0.973516 (+0.000000) | 0.973835 (+0.000319) | 0.973516 (+0.000000) |

## Observations

- Across all 24 mixed-precision evaluations, the largest degradation was
  -0.001934 (Cora node, a=30), and the largest increase was +0.002418
  (Cora node, a=40).
- The mean absolute metric change across the six tasks was 0.000591, 0.000541,
  0.000771, and 0.000299 for a=20, 30, 40, and 50 respectively. Among these
  four settings, a=50 was the least disruptive on average.
- Metric changes were not monotonic in `a`. This experiment does not show that
  simply assigning FP32 to more high-degree nodes consistently improves the
  final metric.
- Except for Cora node, every task stayed within 0.000947 of its FP32 metric at
  every tested setting.

## Reproduction details

- Branch: `quant/dq`
- Batch size: 128
- DataLoader workers: 4
- Seed: 1
- Precision split: exact per-graph top `ceil(a * num_nodes / 100)` by in-degree
- High-degree path: FP32
- Remaining-node path: symmetric INT8 QDQ simulation
- Raw output: `outputs/degree_quant/six_tasks_a20_30_40_50/results.json`
- Tabular output: `outputs/degree_quant/six_tasks_a20_30_40_50/results.csv`

The raw output records exact checkpoint paths, sample counts, timings, deltas,
and quantization granularity. Historical W&B summaries can report different
baselines because they describe the final training epoch, while this run loads
the explicit checkpoint listed for each task. The paired FP32 row is the valid
baseline for each mixed-precision comparison.
