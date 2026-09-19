# Offline Encode-GNN Time Breakdown

`offline_encode_gnn_time.py` 用于做 HEAT Figure 3 风格的离线推理阶段统计：先将基础数据集的文本全局编码一次，再完整执行所选 test loader 的图推理。

这是一套 **HEAT-style 近似复现**，不是论文原始数值的严格复现。论文没有公开 Figure 3 使用的具体 SentenceBERT checkpoint、batch size 和 RGCN/RGAT 变体；本项目的 `ST` 对应 `sentence-transformers/multi-qa-distilbert-cos-v1`，后端是 `PyGRGCNEdge`。

## 1. 计时口径

脚本输出三个互斥阶段，三者之和为 `profiled_total_seconds`：

```text
profiled_total = encode + gnn + other
```

| 阶段 | 统计范围 |
| --- | --- |
| `encode` | 重放基础数据集 `texts.pkl` 中的全部文本条目，包括 tokenizer、Transformer、pooling 和 embedding 回传 CPU |
| `gnn` | 完整 `model(batch)` 的设备执行时间，包括 `llm_proj`、可选 RWPE/JK、`PyGRGCNEdge` 和 prediction head；CUDA 使用 Event，不在每个 batch 后强制同步 |
| `other` | 完整 loader 墙钟时间减去 `gnn`；包括采样、collate、H2D 和关键路径上的 host/runtime 开销 |

`gnn_core_seconds` 另外记录纯 `model.model`（`PyGRGCNEdge`）的设备时间，用来判断 GNN 内部与 projection/head 的开销；它是诊断字段，不参与上述三阶段占比。

`encode` 使用同步墙钟时间，`gnn` 使用 CUDA Event，GNN 前后的 host、采样、collate 和 H2D 被归入 `other`。这是三阶段关键路径拆分，不要丢掉 `other` 后再把 `encode` 与 `gnn` 归一化成两项比例。

以下内容不计入三阶段占比：模型加载、checkpoint 加载、数据集初始化、`texts.pkl` 磁盘读取、首次缓存构建、warmup，以及 metric 计算。

需要特别注意：

- `texts.pkl` 不只包含原始图节点文本，还可能包含边、类别、NOI 和 prompt-edge 文本。
- Offline encode 不按字符串内容去重；重复条目仍会重复编码。
- benchmark 进程会关闭 `SentenceEncoder` 的 tqdm，避免终端渲染和日志 I/O 污染 encode 时间。
- 被计时生成的 embedding 会被丢弃。GNN 使用已经存在的 offline embedding cache。
- metric 在所有计时完成后单独跑一遍，不会污染 `Other`。

因此，`profiled_total_seconds` 是两段独立实验的代表时间相加，用来近似 HEAT 的阶段占比；它不是一次从新 embedding 写入 cache 到 GNN 输出的连续端到端延迟。脚本没有统计新 embedding 的挂载、cache 写入或相关数据结构重建。

## 2. 论文任务映射

| HEAT 缩写 | 项目任务名 |
| --- | --- |
| CN | `cora_node` |
| CL | `cora_link` |
| PN | `pubmed_node` |
| PL | `pubmed_link` |
| AR | `arxiv` |
| WN | `WN18RR` |

`WN18RR` 大小写敏感。正式统计统一使用：

```text
--split test --loader-index 0 --batch-num -1 --sampling-hops 2
```

`--sampling-hops 2` 只修改当前 profiling 进程中所选 dataset 的 `hop` 属性，不修改训练代码、配置文件或缓存。Cora/PubMed link 在仓库原始构造逻辑中是 3-hop，因此使用该参数后，checkpoint 可能与训练时的采样深度不同。

## 3. 运行前提

从项目根目录执行命令，并确保：

1. 使用包含本项目全部依赖的 Python/Conda 环境。
2. 每次只暴露一张 GPU，并保持机器空闲。
3. 每个任务使用自己的 checkpoint。
4. checkpoint 与 `llm_name`、`emb_dim`、`num_layers`、`JK` 和 `rwpe` 配置一致。
5. Offline embedding cache 由相同的文本模型和 `llm_max_length` 生成。

checkpoint 权重值通常不会明显改变算子耗时，但加载 checkpoint 可以验证模型结构和任务 metric，正式结果必须加载。

当前 cache 路径只包含 `llm_name`，没有记录 `llm_max_length`、模型 revision 和 pooling 指纹，脚本无法自动证明 cache 与本次 encoder 完全一致；JSON 因此固定输出 `embedding_cache_identity_verified=false`。这项一致性需要人工确认。

若 cache 尚不存在，`build_task_data` 会在不计时的数据准备阶段先完成一次全量编码并建 cache，使 encoder 比“已有 cache”路径更热。第一次构建只用于准备数据；结束该进程后，再启动正式计时命令。上面的冒烟测试可以承担这一步。

脚本内始终使用逻辑设备 `cuda:0`。例如要使用物理第 3 张卡，应写 `CUDA_VISIBLE_DEVICES=3 ... --device cuda:0`；不要在多卡均可见时直接传 `--device cuda:3`，因为项目的 `SentenceEncoder` 初始化逻辑固定选择第一张可见卡。

所有 `--...` profiling 参数必须放在 `task_names` 等配置覆盖项之前。`task_names` 之后的参数会由 `argparse.REMAINDER` 交给项目配置系统。

## 4. 单任务冒烟测试

下面的命令只测一个 GNN batch，用于确认环境、数据和模型能够运行。全局 encode 仍会覆盖完整基础数据集，因此该结果不能与 HEAT 比较。

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
python exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
  --split test \
  --loader-index 0 \
  --batch-num 1 \
  --warmup-batches 1 \
  --encode-warmup-batches 1 \
  --repeats 1 \
  --sampling-hops 2 \
  --skip-metric \
  --device cuda:0 \
  task_names cora_node \
  llm_name ST \
  llm_max_length 500 \
  batch_size 64 \
  llm_b_size 100 \
  train_sample_size -1 \
  num_workers 0
```

该报告会包含：

```text
full_loader=false
heat_style_comparable=false
```

这是预期行为。

## 5. 单任务正式运行

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
python exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
  --checkpoint /absolute/path/to/cora_node_best.ckpt \
  --split test \
  --loader-index 0 \
  --batch-num -1 \
  --warmup-batches 5 \
  --encode-warmup-batches 1 \
  --repeats 5 \
  --sampling-hops 2 \
  --device cuda:0 \
  task_names cora_node \
  llm_name ST \
  llm_max_length 500 \
  batch_size 64 \
  llm_b_size 100 \
  train_sample_size -1 \
  num_workers 0
```

`--batch-num -1` 和 `train_sample_size -1` 是正式占比统计的必要条件。前者遍历构造出的完整 loader；后者避免 test loader 使用有限的 replacement sampler。否则分子仍是完整全局 encode，而 GNN 只处理部分样本，会人为放大 encode 占比。

正式归因统计使用 `num_workers 0`，避免每次 repeat 重建 loader 时把 worker 启动和首轮 prefetch 的波动算进 `Other`。若目标是测生产吞吐，可以另跑 `num_workers 4`，但应作为不同实验报告，不能与这里的阶段占比混用。

## 6. 六任务串行运行脚本

先将六个 checkpoint 路径替换为实际路径。所有任务必须串行运行，避免 GPU 资源互相干扰。

```bash
#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
RESULT_DIR="${1:-outputs/offline_heat_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RESULT_DIR"

declare -A CHECKPOINTS=(
  [cora_node]="/absolute/path/to/cora_node_best.ckpt"
  [cora_link]="/absolute/path/to/cora_link_best.ckpt"
  [pubmed_node]="/absolute/path/to/pubmed_node_best.ckpt"
  [pubmed_link]="/absolute/path/to/pubmed_link_best.ckpt"
  [arxiv]="/absolute/path/to/arxiv_best.ckpt"
  [WN18RR]="/absolute/path/to/WN18RR_best.ckpt"
)

tasks=(cora_node cora_link pubmed_node pubmed_link arxiv WN18RR)

for task in "${tasks[@]}"; do
  checkpoint="${CHECKPOINTS[$task]}"
  if [[ ! -f "$checkpoint" && ! -d "$checkpoint" ]]; then
    echo "Missing checkpoint for $task: $checkpoint" >&2
    exit 1
  fi

  echo "[$(date --iso-8601=seconds)] START task=$task"
  CUDA_VISIBLE_DEVICES="$GPU" \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
    --checkpoint "$checkpoint" \
    --split test \
    --loader-index 0 \
    --batch-num -1 \
    --warmup-batches 5 \
    --encode-warmup-batches 1 \
    --repeats 5 \
    --sampling-hops 2 \
    --device cuda:0 \
    task_names "$task" \
    llm_name ST \
    llm_max_length 500 \
    batch_size 64 \
    llm_b_size 100 \
    train_sample_size -1 \
    num_workers 0 \
    2>&1 | tee "$RESULT_DIR/${task}.log"
  echo "[$(date --iso-8601=seconds)] END task=$task"
done

echo "Results: $RESULT_DIR"
```

指定项目 Conda 环境中的 Python，例如：

```bash
PYTHON_BIN=/data1/xxr_data/new_conda/ofa/bin/python \
bash /path/to/the/runner.sh
```

如果显存不足，应降低 `batch_size`，并在最终表格中记录每个任务的实际值。`batch_size` 会改变 GNN 吞吐和最终占比，不能隐藏该差异。`llm_b_size` 同样会影响动态 padding 和 encode 吞吐。

## 7. 主要输出字段

脚本最后输出 JSON。Transformer/GNN/Other 柱状图应使用以下字段：

| 字段 | 含义 |
| --- | --- |
| `encode_seconds` | 多次全局 encode 的中位数 |
| `gnn_seconds` | 多次完整 `model(batch)` 设备执行时间的中位数，是三阶段中的 GNN |
| `gnn_core_seconds` | 多次纯 `PyGRGCNEdge` 时间的中位数，仅用于细粒度诊断 |
| `other_seconds` | 多次下游残差时间的中位数 |
| `profiled_total_seconds` | `encode_seconds + gnn_seconds + other_seconds`，是阶段相加值而非连续端到端延迟 |
| `encode_percent` | `encode_seconds / profiled_total_seconds` |
| `gnn_percent` | `gnn_seconds / profiled_total_seconds` |
| `other_percent` | `other_seconds / profiled_total_seconds` |
| `timing_stats` | 每个阶段的 samples、median、Q1、Q3、IQR、min 和 max |
| `runs` | 每次 repeat 的原始时间和 workload |
| `encoded_text_entries` | `texts.pkl` 中被编码的文本条目数，不是去重文本数 |
| `text_encoder_micro_batches` | 按各 leaf text group 分批后的编码 micro-batch 数 |
| `measured_graphs/nodes/edges` | 完整 loader 中累计处理的采样图、节点和边；重复出现会重复计数 |
| `workload_size_consistent_across_repeats` | 多次采样的 batch/图/节点/边等汇总规模是否一致；不代表样本身份逐项相同 |
| `sampling_hops` | runtime override 前后的采样深度 |
| `max_nodes_per_hop` | 从所选 dataset 读取的实际每跳采样上限；`configured_max_nodes_per_hop` 仅回显全局配置 |
| `embedding_cache_identity_verified` | 当前固定为 `false`，表示仓库 cache 缺少足够元数据，需人工核对 encoder 配置 |
| `metric_value` | 单独未计时 pass 的任务指标 |

顶层代表时间使用各阶段自己的 median，再由三个 median 计算百分比，保证三项百分比相加为 100%。`timing_stats.paired_profiled_total_seconds` 另外保留按 repeat 配对后的阶段相加分布。

`heat_style_comparable=true` 表示当前命令满足：完整 test dataset、完整 loader、2-hop、已加载 checkpoint、`num_workers=0`，且各 repeat 的汇总 workload 规模一致。它仍不代表采样身份逐项相同或 cache 身份已经自动验证；由于硬件、后端模型和论文未公开参数仍有差异，`strict_heat_reproduction` 固定为 `false`。

## 8. 建议报告方式

结果表至少同时给出：

```text
task
encode_seconds / gnn_seconds / other_seconds
gnn_core_seconds（细粒度诊断）
encode_percent / gnn_percent / other_percent
measured_graphs / measured_nodes / measured_edges
encoded_text_entries / text_encoder_micro_batches
batch_size / llm_b_size / sampling_hops / max_nodes_per_hop
median / IQR
GPU 型号 / checkpoint
```

脚本记录 CUDA allocated memory 峰值，不记录系统内存峰值。Arxiv 等大数据集若发生主存不足或换页，时间结果也会被影响，应同时用系统监控记录 RSS 和 swap。

不要把 `benchmark_loaded_g.py` 的单 batch eager 比例与这里的 offline 比例放在同一列直接比较。前者回答“一个 batch 在线重新编码有多慢”，这里回答“全局编码一次，再完成整个 test split 时的系统时间构成”。
