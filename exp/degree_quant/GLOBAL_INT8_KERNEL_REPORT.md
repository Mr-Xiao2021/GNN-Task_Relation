# 全局度数感知 INT8 推理实现与评测报告

## 1. 本轮目标

本轮修正了首版实验的两个关键边界：

1. 节点精度不再按每个采样子图的局部度数决定，而是按源图中真实节点的全局度数排名决定。
2. 低精度线性投影不再只是 FP32 上的 quantize-dequantize（QDQ）仿真，而是实际构造 `torch.int8` 输入和权重，调用 CUDA INT8 GEMM，并使用 INT32 accumulator。

六个任务仍使用原 checkpoint，不做 QAT 或微调；高精度真实节点使用 FP32，低精度真实节点使用 INT8，`a` 取 20、30、40、50。本文同时报告最终 metric、实测吞吐、MixQ 风格 BitOP proxy 和更常见的 matmul BitOP。

## 2. 全局真实节点度数划分

### 2.1 定义

对每个 source graph 只计算一次入度，并用稳定排序生成全局 rank：

```text
rank(v) = position of v in sort((-degree(v), source_node_id(v)))
cutoff  = ceil(a / 100 * source_graph_num_nodes)
FP32(v) = rank(v) < cutoff
```

采样子图在构建时携带 `global_node_id`、`global_node_degree`、`global_degree_rank`、`global_graph_num_nodes` 和 `real_node_mask`。因此同一个真实节点无论出现在哪个子图、哪个 batch 中，精度归属始终一致。

NOI、class 等 prompt-only 节点没有源图节点 ID，始终保留 FP32。链接预测和知识图谱任务只使用模型可见的训练图计算全局 rank，避免测试边进入精度分组并造成泄漏。

### 2.2 `a` 与实际出现比例

`a` 定义的是源图节点集合中的全局比例，不是测试采样记录中的出现比例。高度数节点更容易进入多个邻域子图，所以评测日志中的 `realized_high_precision_real_percent` 通常显著高于 `a`。这是全局划分和邻域采样共同产生的预期结果，不是分组错误。

## 3. 真实 INT8 kernel

### 3.1 数值路径

低精度 activation 使用 symmetric per-tensor INT8，融合权重使用 per-output-channel INT8：

```text
q_x = clamp(round(x / s_x), -127, 127).to(int8)
q_w = clamp(round(w / s_w), -127, 127).to(int8)
acc = torch._int_mm(q_x, q_w)              # INT8 x INT8 -> INT32
y   = acc.to(float32) * s_x * s_w
```

CUDA 路径调用 `torch._int_mm`，在当前 H100 / PyTorch 2.1 环境中落到 cuBLASLt INT8 kernel。由于该接口要求行数为正的 32 倍数，混合精度低精度行会补零到 32 的倍数，GEMM 后再切回原行数。权重按参数版本缓存；checkpoint 加载或参数变化后自动重建。

### 3.2 融合优化

原始 RGCN 每层执行 5 个 relation projection 和 1 个 root projection。逐 relation 对低精度行分别调用 INT8 GEMM 时，小矩阵启动和动态量化开销过高。本轮利用线性代数等价关系：

```text
sum_r H_r W_r + X W_root
= concat(H_0, ..., H_4, X) @ concat(W_0, ..., W_4, W_root, axis=0)
```

因此每层只执行一次混合 FP32/INT8 projection，并缓存融合后的 INT8 权重。该优化将真实 INT8 路径从每层 6 组 GEMM 降为 1 组 GEMM，但不会改变原 checkpoint 的参数结构。

### 3.3 两种真实整数 backend

| backend | Linear projection | Message aggregation | 用途 |
| --- | --- | --- | --- |
| `int8` | `torch._int_mm`, INT32 accumulator | message 做 INT8 QDQ，使用浮点 PyG scatter | 六任务主评测，当前较快 |
| `int8_full` | `torch._int_mm`, INT32 accumulator | INT8 message + INT32 `scatter_add_` | 覆盖性实验，当前更慢 |
| `qdq` | 浮点 GEMM 上做 QDQ | 浮点 scatter | 首版数值参考 |

主结果使用 `int8`。因此“真实 INT8 kernel”准确地指六层 RGCN 的低精度 linear projection；主 backend 的 message aggregation 仍是浮点 scatter。`int8_full` 已实现真实 INT8/INT32 聚合，但在当前硬件和 PyTorch eager 执行下性能更差，没有作为默认路径。

## 4. 指标口径

### 4.1 精度与吞吐

- Accuracy / AUROC：完整测试集最终指标。
- `steady_model_examples_per_second`：排除首个 warm-up batch 后，仅包围 model forward 的 CUDA Event 吞吐。
- `examples_per_second`：包含 DataLoader、H2D、metric update 等的端到端吞吐。
- `speedup_vs_fp32_model`：同任务 INT8 steady model throughput / FP32 steady model throughput。
- `peak_cuda_memory_mb`：PyTorch allocator 记录的峰值已分配显存。

### 4.2 BitOP

为避免把不同文献口径混为一谈，同时记录两种估算：

```text
MixQ proxy = sum(operation_count * selected_precision_bits)
Matmul BitOP = sum(MAC * activation_bits * weight_bits)
```

MixQ 仓库的 `estimated_bit_operation_precision()` 和硬件绘图代码采用 operation count 乘平均/实际 bit-width 的 proxy；本文保留这一列用于同类比较。Matmul BitOP 则体现 FP32 MAC 的 `32 x 32` 与 INT8 MAC 的 `8 x 8`。两者均是分析值，不包含动态 quantize/dequantize、行索引、padding、kernel launch 和内存流量，不能替代实测吞吐。

## 5. 六任务完整结果

### 5.1 最终 metric

括号内为相对同任务 FP32 的绝对变化。Cora、PubMed 和 Arxiv 使用 Accuracy 或 AUROC 的完整测试集；WN18RR 使用 Accuracy。

| 任务 | 指标 | FP32 | a=20 | a=30 | a=40 | a=50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora node | Accuracy | 0.702611 | 0.704062 (+0.001451) | 0.705996 (+0.003385) | 0.705029 (+0.002418) | 0.702611 (+0.000000) |
| Cora link | AUROC | 0.948873 | 0.948609 (-0.000264) | 0.948277 (-0.000595) | 0.948613 (-0.000260) | 0.948847 (-0.000025) |
| PubMed node | Accuracy | 0.719215 | 0.720050 (+0.000835) | 0.719528 (+0.000313) | 0.719110 (-0.000104) | 0.718380 (-0.000835) |
| PubMed link | AUROC | 0.980520 | 0.980540 (+0.000020) | 0.980538 (+0.000019) | 0.980583 (+0.000063) | 0.980524 (+0.000004) |
| Arxiv | Accuracy | 0.709154 | 0.709380 (+0.000226) | 0.709586 (+0.000432) | 0.709504 (+0.000350) | 0.709545 (+0.000391) |
| WN18RR | Accuracy | 0.973516 | 0.973516 (+0.000000) | 0.973835 (+0.000319) | 0.973197 (-0.000319) | 0.973197 (-0.000319) |

24 组混合精度结果中，最大下降为 `-0.000835`（PubMed node，a=50），最大上升为 `+0.003385`（Cora node，a=30）。a=20、30、40、50 的六任务平均绝对变化分别为 `0.000466`、`0.000844`、`0.000586`、`0.000262`。与首版局部子图 QDQ 结果一致，metric 不随 a 单调变化；当前四档中 a=50 的平均绝对扰动最小。

### 5.2 实际节点比例与理论成本

表内每格为“测试子图中实际 FP32 真实节点出现比例 / MixQ proxy 降幅 / matmul BitOP 降幅”。

| 任务 | a=20 | a=30 | a=40 | a=50 |
| --- | ---: | ---: | ---: | ---: |
| Cora node | 34.70% / 38.91% / 48.67% | 49.22% / 30.26% / 37.85% | 60.27% / 23.67% / 29.61% | 69.80% / 17.99% / 22.51% |
| Cora link | 35.40% / 47.29% / 59.15% | 49.52% / 36.94% / 46.22% | 60.84% / 28.66% / 35.86% | 71.58% / 20.79% / 26.02% |
| PubMed node | 50.58% / 34.01% / 42.56% | 62.77% / 25.62% / 32.06% | 71.67% / 19.49% / 24.40% | 78.78% / 14.60% / 18.28% |
| PubMed link | 59.37% / 29.94% / 37.47% | 70.85% / 21.47% / 26.88% | 78.71% / 15.68% / 19.63% | 85.05% / 11.01% / 13.78% |
| Arxiv | 55.98% / 22.95% / 28.72% | 70.32% / 15.47% / 19.36% | 80.37% / 10.24% / 12.81% | 87.42% / 6.56% / 8.20% |
| WN18RR | 33.79% / 43.25% / 54.09% | 45.01% / 35.92% / 44.93% | 55.86% / 28.83% / 36.06% | 66.17% / 22.10% / 27.64% |

全局 top-a% 在 sampled test subgraph 中被过采样：例如 Arxiv 的全局 a=20 最终占真实节点出现次数的 55.98%。因此可量化 MAC 比例低于直觉上的 `100-a`，尤其在 PubMed/Arxiv 高度数节点重复进入大量邻域时更明显。

## 6. 性能解释

### 6.1 统一吞吐基准

性能表使用 NVIDIA H100 PCIe、PyTorch 2.1.2+cu118、batch size 128、最多 20 个 batch，并排除首个 warm-up batch。Cora node/link 不足 20 个 batch，因此使用其完整 17/9 个 batch。表内 FP32 为 steady model throughput；a=20...50 为“INT8 throughput / 相对 FP32 倍率”。该截断运行只用于性能，PubMed link 的前 20 个 batch 恰好没有负样本，因此其中间 AUC 无意义，最终精度只取 5.1 节的完整测试集。

| 任务 | FP32 ex/s | a=20 | a=30 | a=40 | a=50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora node | 1976.9 | 1133.1 / 0.573x | 1393.1 / 0.705x | 1310.5 / 0.663x | 1387.8 / 0.702x |
| Cora link | 787.1 | 533.0 / 0.677x | 591.0 / 0.751x | 642.9 / 0.817x | 637.8 / 0.810x |
| PubMed node | 1645.4 | 1045.4 / 0.635x | 1145.2 / 0.696x | 1106.0 / 0.672x | 1173.8 / 0.713x |
| PubMed link | 437.8 | 392.9 / 0.898x | 411.8 / 0.941x | 415.7 / 0.949x | 427.1 / 0.976x |
| Arxiv | 746.6 | 585.2 / 0.784x | 606.6 / 0.813x | 607.6 / 0.814x | 621.3 / 0.832x |
| WN18RR | 1334.6 | 886.4 / 0.664x | 912.4 / 0.684x | 900.0 / 0.674x | 947.6 / 0.710x |

当前所有档位都没有超过 FP32，范围为 `0.573x` 到 `0.976x`。最接近持平的是 PubMed link a=50；低精度占比最高的 a=20 反而常更慢，说明动态量化和两路分流成本主导了当前 eager 实现。

峰值显存也没有下降：量化路径同时保留原 FP32 参数、融合 FP32 权重、缓存 INT8 权重、INT32 accumulator 和索引缓冲，六任务量化峰值比 FP32 高约 120--182 MiB。当前实现的价值是建立真实 kernel 和可测量基线，而不是宣称已经获得部署加速。

### 6.2 优化效果与剩余瓶颈

开发过程中的 Cora node 短基准显示：逐 relation 的 INT8 GEMM 约为 FP32 的 `0.21x`；融合 5 个 relation + root 后、但保留 INT32 message aggregation 时约为 `0.32x`；主 backend 再将 message aggregation 恢复为浮点 PyG scatter 后，长一些的统一基准达到 `0.57x--0.70x`。因此 GEMM 融合和避免通用 INT32 scatter 都是有效优化，但仍不足以覆盖节点级动态分流开销。

当前实现证明了低精度 projection 确实执行真实 INT8 kernel，但节点级混合精度还需要额外完成 mask gather/scatter、动态 activation calibration、量化和反量化，并分别发射 FP32 与 INT8 GEMM；当低精度矩阵较小或 FP32 节点出现比例较高时，这些成本会超过 INT8 GEMM 节省的时间。

全量前四任务最初在共享 GPU 上遇到其他进程持续 100% 利用率，因此其全量日志吞吐仅保留为运行记录，不用于表 6.1。Arxiv/WN18RR 全量运行和六任务 20-batch 性能基准切换到无活跃计算负载的另一张 H100；性能结论以统一 20-batch 基准为准。

## 7. 验证

测试覆盖：

1. 同一真实节点跨不同 prompted subgraph 保持相同全局 degree/rank。
2. cutoff 使用 `ceil(a * N / 100)`，并按源节点 ID 稳定打破 degree tie。
3. prompt-only 节点永远保持 FP32。
4. INT8 linear 与 CPU INT32 参考公式一致。
5. CUDA 路径实际调用 `torch._int_mm`。
6. INT8/INT32 message aggregation 与显式参考计算一致。
7. 原始 checkpoint 可 strict-load，`a=100` 与原 FP32 模型等价。
8. 六任务的 PyG collate 都携带完整的全局元数据。

## 8. 局限与后续优化

1. 当前 activation scale 按每次调用动态计算；固定 calibration 和跨层 scale contract 可以减少 reduction 与 QDQ 开销。
2. 节点级 FP32/INT8 分流会产生 gather/scatter 和两个 GEMM；要获得稳定加速，需要 Triton/CUDA grouped kernel 在同一 launch 内处理两种行。
3. `torch._int_mm` 是内部接口，正式部署应固定 PyTorch/CUDA 版本，或换成受支持的 CUTLASS/cuBLASLt 封装。
4. `int8_full` 的通用 INT32 `scatter_add_` 尚未形成高效稀疏聚合 kernel；它用于验证整数覆盖，不代表最终性能实现。
5. BitOP 未建模量化、索引、内存流量和 prompt MLP，因此只能作为理论成本指标。
6. 当前仅评估单 checkpoint、单 seed；若要形成统计结论，应增加多 seed/checkpoint 的均值和置信区间。

## 9. 复现命令

```bash
CUDA_VISIBLE_DEVICES=0 /data1/xxr_data/new_conda/ofa/bin/python \
  exp/degree_quant/evaluate_degree_quant.py \
  --tasks cora_node cora_link pubmed_node pubmed_link arxiv WN18RR \
  --high-precision-percents 20 30 40 50 \
  --quantization-backend int8 \
  --batch-size 128 --num-workers 4 --warmup-batches 1 \
  --output-dir outputs/degree_quant/global_degree_int8_six_tasks_a20_30_40_50
```

```bash
CUDA_VISIBLE_DEVICES=0 /data1/xxr_data/new_conda/ofa/bin/python -m unittest -v \
  exp.degree_quant.test_degree_quant
```

本轮原始结果：

- Cora/PubMed 完整精度：`outputs/degree_quant/global_degree_int8_six_tasks_a20_30_40_50/results.{json,csv}`
- Arxiv/WN18RR 完整精度：`outputs/degree_quant/global_degree_int8_six_tasks_remaining/results.{json,csv}`
- 六任务统一 20-batch 性能基准：`outputs/degree_quant/global_degree_int8_perf_20_batches/results.{json,csv}`
