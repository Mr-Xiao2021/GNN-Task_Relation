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

脚本还会在无插桩的 Encode repeats 完成后，额外复放一次相同的
`dataset.text2feature(texts)`，将 SentenceEncoder 内部聚合为 Transformer forward、
全层 QKV projection、Attention 和 FFN。结果位于 `transformer_profile`，该诊断 pass
不参与 `encode_seconds` 或 `profiled_total_seconds`。如需完全跳过，可传入
`--skip-transformer-profile`。该组件诊断只实现单卡 CUDA Event 计时；CPU 运行时
必须使用这个跳过参数，原有 Encode/GNN/Other 逻辑仍可照常执行。

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

推荐每次只暴露一张 GPU。例如要使用物理第 3 张卡，可写 `CUDA_VISIBLE_DEVICES=3 ... --device cuda:0`；`--device` 决定脚本最终放置 SentenceEncoder 和 GNN 的逻辑设备。

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
  num_workers 0
```

`--batch-num -1` 是正式占比统计的必要条件，用于遍历构造出的完整 loader。`train_sample_size` 直接使用 `default_config.yaml` 中的整数 `-1`，避免 test loader 使用有限的 replacement sampler。否则分子仍是完整全局 encode，而 GNN 只处理部分样本，会人为放大 encode 占比。

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
| `transformer_profile` | 独立诊断 replay 的 Transformer 总时间及全层 QKV、Attention、FFN 聚合时间；不提供逐层结果 |
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

`transformer_profile.attention_seconds` 定义为 inclusive Attention 减去 Q/K/V
projection；它还包含 O projection、dropout，以及部分架构的 residual/norm，不能解释为
纯 `QK^T + softmax + AV` kernel 时间。更完整的字段说明见
[`exp/encode/tf_profile/README.md`](../../encode/tf_profile/README.md)。
组件边界在 BERT、DistilBERT 和 Llama 之间不同，这组细分数据只用于同一 Encoder
架构下不同精度或实现策略的对比，不用于跨架构比较 Attention/FFN 百分比。

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

## 9. 六任务实验记录（2026-09-19 至 2026-09-20）

本节记录在 `feat/np-encode` 分支提交 `51b34e7` 上执行的 HEAT 六任务实验。物理设备为 NVIDIA H100 PCIe GPU 1，通过 `CUDA_VISIBLE_DEVICES=1` 映射为脚本内的 `cuda:0`。六个任务串行执行，完整 runner 见 [run_offline_heat_six.sh](./logs/offline_heat_260919234746/run_offline_heat_six.sh)。

### 9.1 实验配置

所有正式任务使用相同的 profiling 参数：

```text
split=test
loader_index=0
batch_num=-1
batch_size=64
llm_b_size=100
llm_max_length=500
repeats=5
warmup_batches=5
encode_warmup_batches=1
sampling_hops=2
train_sample_size=-1
num_workers=0
llm_name=ST
```

Arxiv 和 WN18RR 在正式计时前没有 offline cache，因此先在独立进程中完成 cache-prep，再启动新的正式计时进程。Arxiv cache-prep 用时 1420 秒，WN18RR cache-prep 用时 27 秒；这些时间不进入下表任何阶段。

| 任务 | Checkpoint | `heat_style_comparable` | Metric |
| --- | --- | --- | ---: |
| Cora Node | 已加载 | `true` | acc = 0.7016 |
| Cora Link | 未加载 | `false` | 随机权重，不报告效果 |
| PubMed Node | 已加载 | `true` | acc = 0.7191 |
| PubMed Link | 未加载 | `false` | 随机权重，不报告效果 |
| Arxiv | 未加载 | `false` | 随机权重，不报告效果 |
| WN18RR | 未加载 | `false` | 随机权重，不报告效果 |

未加载 checkpoint 的四项仍能提供算子与系统时间，但不属于完整的 HEAT-style 可比结果。六项的 `strict_heat_reproduction` 和 `embedding_cache_identity_verified` 均为 `false`；后者需要人工核对 cache 的 encoder 身份。

### 9.2 实际运行命令

下面是本次实验实际生效的完整命令。Arxiv 和 WN18RR 因初始 cache 缺失，先各自在独立进程中执行一次 cache-prep；已有 cache 的 Cora 和 PubMed 没有执行该步骤。

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data1/xxr_data/GNN/GNN-Task_Relation
PYTHON_BIN=/data1/xxr_data/new_conda/ofa/bin/python
GPU=1
RESULT_DIR="$PROJECT_ROOT/exp/time_pipe/encode-gnn/results/offline_heat_260919234746"

mkdir -p "$RESULT_DIR"
cd "$PROJECT_ROOT"

declare -A CHECKPOINTS=(
  [cora_node]="$PROJECT_ROOT/saved_exp/2026-09-17 17:25:07.738700/full_cdm/e5t7x7gs/checkpoints/epoch=74-step=150.ckpt"
  [cora_link]=""
  [pubmed_node]="$PROJECT_ROOT/saved_exp/2026-09-17 16:01:00.408337/full_cdm/tn7immv7/checkpoints/epoch=60-step=61.ckpt"
  [pubmed_link]=""
  [arxiv]=""
  [WN18RR]=""
)

declare -A CACHE_DIRS=(
  [cora_node]="$PROJECT_ROOT/cache_data/Cora/ST/processed"
  [cora_link]="$PROJECT_ROOT/cache_data/Cora/ST/processed"
  [pubmed_node]="$PROJECT_ROOT/cache_data/Pubmed/ST/processed"
  [pubmed_link]="$PROJECT_ROOT/cache_data/Pubmed/ST/processed"
  [arxiv]="$PROJECT_ROOT/cache_data/arxiv/ST/processed"
  [WN18RR]="$PROJECT_ROOT/cache_data/WN18RR/ST/processed"
)

TASKS=(cora_node cora_link pubmed_node pubmed_link arxiv WN18RR)

for TASK in "${TASKS[@]}"; do
  CACHE_DIR="${CACHE_DIRS[$TASK]}"
  if [[ ! -f "$CACHE_DIR/texts.pkl" || ! -f "$CACHE_DIR/geometric_data_processed.pt" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
      --split test \
      --loader-index 0 \
      --batch-num 1 \
      --warmup-batches 1 \
      --encode-warmup-batches 1 \
      --repeats 1 \
      --sampling-hops 2 \
      --skip-metric \
      --device cuda:0 \
      task_names "$TASK" \
      llm_name ST \
      llm_max_length 500 \
      batch_size 64 \
      llm_b_size 100 \
      num_workers 0 \
      > "$RESULT_DIR/${TASK}_cache_prep.log" 2>&1
  fi

  CHECKPOINT_ARGS=()
  if [[ -n "${CHECKPOINTS[$TASK]}" ]]; then
    CHECKPOINT_ARGS=(--checkpoint "${CHECKPOINTS[$TASK]}")
  fi

  CUDA_VISIBLE_DEVICES="$GPU" \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
    "${CHECKPOINT_ARGS[@]}" \
    --split test \
    --loader-index 0 \
    --batch-num -1 \
    --warmup-batches 5 \
    --encode-warmup-batches 1 \
    --repeats 5 \
    --sampling-hops 2 \
    --device cuda:0 \
    task_names "$TASK" \
    llm_name ST \
    llm_max_length 500 \
    batch_size 64 \
    llm_b_size 100 \
    num_workers 0 \
    > "$RESULT_DIR/${TASK}.log" 2>&1
done
```

归档的 [run_offline_heat_six.sh](./logs/offline_heat_260919234746/run_offline_heat_six.sh) 在上述有效命令外增加了 cache 存在性检查、逐任务状态记录和单项失败后继续执行，其 profiling 参数完全一致。

### 9.3 三阶段与两阶段结果

以下时间单位均为秒。`Encode`、`GNN` 和 `Other` 是各自 5 次完整 repeat 的中位数，括号内为 IQR。`GNN pipeline` 是本实验报告采用的两阶段派生口径：

```text
GNN pipeline = GNN + Other
Profiled total = Encode + GNN pipeline
```

| 任务 | Encode median (IQR) | GNN median (IQR) | Other median (IQR) | GNN pipeline | Profiled total | Encode | GNN pipeline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora Node | 3.373 (0.012) | 0.321 (0.036) | 2.982 (0.539) | 3.303 | 6.676 | 50.53% | 49.47% |
| Cora Link | 3.363 (0.116) | 0.168 (0.000) | 1.462 (0.007) | 1.629 | 4.993 | 67.36% | 32.64% |
| PubMed Node | 31.608 (0.244) | 3.099 (0.081) | 30.313 (0.549) | 33.411 | 65.020 | 48.61% | 51.39% |
| PubMed Link | 31.757 (0.176) | 1.739 (0.042) | 16.479 (1.026) | 18.218 | 49.976 | 63.55% | 36.45% |
| Arxiv | 230.974 (0.256) | 11.568 (0.004) | 151.256 (0.875) | 162.824 | 393.798 | 58.65% | 41.35% |
| WN18RR | 6.772 (0.100) | 0.581 (0.003) | 6.403 (0.145) | 6.984 | 13.756 | 49.23% | 50.77% |

`GNN pipeline` 表示全局文本编码完成后的完整下游图推理关键路径，既包含 GPU 模型前向，也包含采样、collate、DataLoader 等待、H2D 和 host/runtime 开销。因此它适合与 `Encode` 构成两阶段系统占比，但不能解释为纯 GNN kernel 时间。纯 `PyGRGCNEdge` 的诊断字段是 `gnn_core_seconds`，它已经包含在 `GNN` 中，不能再次加到总时间。

### 9.4 Workload

下表是单次完整 loader repeat 的累计工作量。采样子图中的节点和边会跨 batch 重复出现，因此 `Nodes` 和 `Edges` 不是基础图的去重规模。

| 任务 | Text entries | Encoder micro-batches | Batches | Graphs | Nodes | Edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora Node | 2,821 | 33 | 33 | 2,068 | 80,712 | 234,126 |
| Cora Link | 2,821 | 33 | 17 | 1,056 | 59,353 | 175,302 |
| PubMed Node | 19,726 | 202 | 300 | 19,157 | 940,959 | 3,020,564 |
| PubMed Link | 19,726 | 202 | 139 | 8,866 | 795,617 | 2,670,818 |
| Arxiv | 172,588 | 1,730 | 760 | 48,603 | 6,552,487 | 22,644,838 |
| WN18RR | 40,982 | 414 | 49 | 3,134 | 292,506 | 707,478 |

六项均输出 `workload_size_consistent_across_repeats=true`，表示 5 次 repeat 的 batch、graph、node、edge 和输出规模一致，但不证明每次采样的实体身份逐项相同。

### 9.5 表格列的代码依据

| 展示列 | JSON 来源或公式 | 代码依据 |
| --- | --- | --- |
| `Encode` | `encode_seconds` | [`time_global_offline_encode()`](./offline_encode_gnn_time.py#L290) 在磁盘读取 `texts.pkl` 后同步 GPU，计时 `dataset.text2feature(texts)`；[`make_report()`](./offline_encode_gnn_time.py#L575) 对 5 次结果取中位数。包含 tokenizer、Transformer、pooling 和 embedding 回传 CPU，不含磁盘读取与 cache 写入。 |
| `GNN` | `gnn_seconds` | [`time_model_batches()`](./offline_encode_gnn_time.py#L417) 在 `model(batch)` 前后记录 CUDA Event，最后将完整 loader 所有 batch 的 Event 时间相加；第 495 行赋给 `gnn_seconds`。包含 projection、可选 RWPE/JK、`PyGRGCNEdge` 和 prediction head。 |
| `Other` | `other_seconds` | 同一函数先测完整 loader 墙钟 `pipeline_wall_seconds`，再在第 496 行计算 `max(0, pipeline_wall_seconds - gnn_seconds)`。因此包含采样、collate、loader wait、H2D 和 host/runtime 关键路径开销。 |
| `GNN pipeline` | `gnn_seconds + other_seconds` | 本 README 的派生列，不是当前 JSON 的独立顶层字段。使用两个阶段各自的中位数相加，以便和顶层 `profiled_total_seconds` 保持同一代表值口径。 |
| `Profiled total` | `profiled_total_seconds` | [`make_report()`](./offline_encode_gnn_time.py#L584) 先分别取三个阶段中位数，再执行 `encode_seconds + gnn_seconds + other_seconds`。它不是一次连续 cache-build-to-prediction 请求的实测延迟。 |
| `Encode %` | `encode_percent` | `100 * encode_seconds / profiled_total_seconds`，对应第 588-593、782 行。 |
| `GNN pipeline %` | `100 - encode_percent` | 本 README 的两阶段派生列，等价于 `100 * (gnn_seconds + other_seconds) / profiled_total_seconds`。 |
| IQR | `timing_stats.<phase>.iqr` | `summarize()` 分别对每个阶段的 5 个样本计算 Q1、Q3 和 `Q3 - Q1`。阶段 IQR 不相加。 |
| Batches/Graphs/Nodes/Edges | `measured_*` | [`time_model_batches()`](./offline_encode_gnn_time.py#L463) 在每个 batch 累加 `batch.num_graphs`、`batch.num_nodes` 和 `batch.num_edges`；报告阶段要求多次 workload 一致，否则返回分布并发出 warning。 |

`timing_stats.downstream_pipeline_seconds` 是每次 repeat 中 `pipeline_wall_seconds` 的中位数；而上表 `GNN pipeline` 是 `median(GNN) + median(Other)`。由于中位数一般不满足可加性，两者可能有轻微差异。选择后者是为了严格满足当前顶层报告定义：

```text
Profiled total = median(Encode) + median(GNN) + median(Other)
```

runner 状态表中的进程总历时也不等于 `Profiled total`。进程总历时还包含 5 次重复、模型和 checkpoint 加载、数据初始化、warmup、独立 metric pass、清理等未纳入三阶段的工作。

### 9.6 原始日志

正式结果：

- [Cora Node](./logs/offline_heat_260919234746/cora_node.log)
- [Cora Link](./logs/offline_heat_260919234746/cora_link.log)
- [PubMed Node](./logs/offline_heat_260919234746/pubmed_node.log)
- [PubMed Link](./logs/offline_heat_260919234746/pubmed_link.log)
- [Arxiv](./logs/offline_heat_260919234746/arxiv.log)
- [WN18RR](./logs/offline_heat_260919234746/WN18RR.log)

执行与 cache-prep 记录：

- [Arxiv cache-prep](./logs/offline_heat_260919234746/arxiv_cache_prep.log)
- [WN18RR cache-prep](./logs/offline_heat_260919234746/WN18RR_cache_prep.log)
- [串行时间线](./logs/offline_heat_260919234746/launcher.log)
- [任务状态表](./logs/offline_heat_260919234746/run_status.csv)
- [最终 runner 状态](./logs/offline_heat_260919234746/runner.state)
