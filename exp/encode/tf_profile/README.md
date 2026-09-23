# Offline Transformer 聚合计时

本实验不再构造独立的 eager 模型或数据流，而是直接复用现有离线编码路径：

```text
time_global_offline_encode
  -> dataset.text2feature(texts)
  -> OFAPygDataset.data2vec(data)
  -> SentenceEncoder.encode(data)
  -> LLMModel.encode(text_tokens)
  -> Hugging Face model(...)
```

入口仍然是：

```text
exp/time_pipe/encode-gnn/offline_encode_gnn_time.py
```

`models/transformer_profiler.py` 只负责发现少量组件并聚合 CUDA Event；不会输出
逐层 block、逐模块列表或 operator trace。组件 profiler 仅支持单张 CUDA GPU，
不实现 CPU 计时或多设备兼容。

## 输出

最终 JSON 的 `transformer_profile` 包含：

| 字段 | 含义 |
| --- | --- |
| `model_forward_seconds` | 完整 Hugging Face model call；Llama 路径包含当前代码实际计算的 `lm_head` |
| `transformer_total_seconds` | Transformer backbone 总时间；Llama 路径不包含 `lm_head` |
| `model_wrapper_or_head_seconds` | 完整 model call 与 backbone 之间的差值 |
| `qkv_projection_seconds` | 所有层 Q、K、V projection 的时间总和 |
| `attention_seconds` | inclusive Attention 扣除 Q/K/V 后的剩余时间 |
| `ffn_seconds` | 所有层 FFN/MLP 的时间总和 |
| `other_transformer_seconds` | embedding、其余 norm、pooler 等 backbone 剩余时间 |
| `attention_inclusive_seconds` | 未扣除 Q/K/V 的 Attention 时间，用于审计 |
| `percent_of_transformer` | QKV、Attention、FFN 和 Other 相对 Transformer forward 的比例 |
| `calls` / `expected_calls` | 聚合调用数及按层数推导的期望值 |
| `coverage_complete` | 组件调用数是否完整 |
| `workload_call_count_matches` | model forward 次数是否等于文本 micro-batch 数 |

这里的 `attention_seconds` 不是严格的 `QK^T + softmax + AV` kernel 时间。
普通模块 hook 无法可靠切开这些函数调用；该字段还会包含 O projection、dropout，
并可能包含架构自己的 residual/normalization。定义为：

```text
attention_seconds = attention_inclusive_seconds - qkv_projection_seconds
```

近似分解关系为：

```text
transformer_total ~= QKV + Attention + FFN + Other
```

所有层都聚合到同一个字段，不提供逐层结果。

BERT、DistilBERT 和 Llama 的模块边界并不完全相同。例如 BERT 的 Attention/FFN
模块还包住部分 residual、dropout 和 LayerNorm，而 Llama 的 norm 位于这些模块之外。
因此组件结果用于比较**同一 Encoder 架构的不同精度/实现策略**，不能直接比较不同
架构之间的 Attention 或 FFN 百分比。

## 基线与开销

原来的 `encode_seconds` 仍由无 hook 的 `dataset.text2feature(texts)` 重复运行得到。
完成这些基线 repeat 后，脚本额外复放一次相同的全量文本，用于组件计时。因此：

- hook 和 CUDA Event 不会污染原有 `encode_seconds`。
- `profiled_encode_wall_seconds` 是带监测开销的诊断值。
- 额外 profile replay 不计入 `profiled_total_seconds`。
- 每个 SentenceEncoder micro-batch 的 embedding 回传 CPU 后立即汇总并释放 Event，
  不会为完整 `texts.pkl` 无限保留 CUDA Event。

如果只需要原来的 Encode/GNN/Other 报告，可传入：

```text
--skip-transformer-profile
```

选择 CPU 运行原脚本时也必须传入该参数。

## 支持模型

| `llm_name` | `model_type` | 聚合目标 |
| --- | --- | --- |
| `BERT`、`e5` | `bert` | Attention、Q/K/V、Intermediate + Output FFN |
| `ST` | `distilbert` | Attention、Q/K/V、FFN |
| `llama2_7b`、`llama2_13b` | `llama` | Self-attention、Q/K/V、MLP |

发现的层数或组件数量不符合模型配置时会直接报错，避免输出不完整结果。

## 运行

本轮实验使用 ST encoder，在物理 GPU 1 上串行执行六个任务。图推理使用完整 test
loader 和 2-hop sampling；图 batch size 为 128，文本 batch size 为 100，文本最大
长度为 500。Encode 和 GNN 分别重复 5 次，另执行一次带 Transformer hooks 的全量
文本 replay。

在仓库根目录执行的核心命令如下。Transformer 聚合计时默认开启，无需额外开关；
每个任务的标准输出和错误输出均写入对应日志。

```bash
PROJECT_ROOT=/data1/xxr_data/GNN/GNN-Task_Relation
PYTHON_BIN=/data1/xxr_data/new_conda/ofa/bin/python
GPU=1
RUN_ID=offline_tf_profile_260923005943
RESULT_DIR="$PROJECT_ROOT/exp/encode/tf_profile/results/$RUN_ID"

declare -A CHECKPOINTS=(
  [cora_node]="$PROJECT_ROOT/saved_exp/2026-09-17 17:25:07.738700/full_cdm/e5t7x7gs/checkpoints/epoch=74-step=150.ckpt"
  [cora_link]="$PROJECT_ROOT/saved_exp/2026-09-21 18:06:44.432551/full_cdm/8x52rwr9/checkpoints/epoch=48-step=3479.ckpt"
  [pubmed_node]="$PROJECT_ROOT/saved_exp/2026-09-17 16:01:00.408337/full_cdm/tn7immv7/checkpoints/epoch=60-step=61.ckpt"
  [pubmed_link]="$PROJECT_ROOT/saved_exp/2026-09-21 19:05:58.361032/full_cdm/b9rejg4r/checkpoints/epoch=10-step=6479.ckpt"
  [arxiv]="$PROJECT_ROOT/saved_exp/2026-09-21 20:44:10.956571/full_cdm/hk1d9kki/checkpoints/epoch=6-step=4977.ckpt"
  [WN18RR]="$PROJECT_ROOT/saved_exp/2026-09-22 12:45:33.439358/full_cdm/pyb5uw7f/checkpoints/epoch=32-step=22407.ckpt"
)

mkdir -p "$RESULT_DIR"
cd "$PROJECT_ROOT"

for task in cora_node cora_link pubmed_node pubmed_link arxiv WN18RR; do
  CUDA_VISIBLE_DEVICES="$GPU" \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  "$PYTHON_BIN" exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
    --checkpoint "${CHECKPOINTS[$task]}" \
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
    batch_size 128 \
    llm_b_size 100 \
    num_workers 0 \
    > "$RESULT_DIR/${task}.log" 2>&1
done
```

包含预检、逐任务状态记录和失败后继续执行逻辑的完整 runner 为
[`run_offline_tf_profile_six.sh`](./results/offline_tf_profile_260923005943/run_offline_tf_profile_six.sh)。

## 六任务结果

六个任务均成功加载 checkpoint 并完成 profile，且
`coverage_complete=true`、`workload_call_count_matches=true`。下表时间单位均为秒，
括号内为相对 `transformer_total_seconds` 的占比。

| 任务 | Transformer 总时间 | QKV | Attention | FFN | Other |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cora_node` | 1.893 | 0.133 (7.00%) | 1.164 (61.47%) | 0.423 (22.34%) | 0.174 (9.18%) |
| `cora_link` | 1.893 | 0.133 (7.03%) | 1.163 (61.45%) | 0.423 (22.36%) | 0.173 (9.16%) |
| `pubmed_node` | 14.075 | 0.968 (6.88%) | 8.722 (61.97%) | 3.121 (22.18%) | 1.263 (8.97%) |
| `pubmed_link` | 14.042 | 0.961 (6.84%) | 8.709 (62.02%) | 3.110 (22.15%) | 1.262 (8.99%) |
| `arxiv` | 117.482 | 8.285 (7.05%) | 71.971 (61.26%) | 26.362 (22.44%) | 10.864 (9.25%) |
| `WN18RR` | 2.906 | 0.412 (14.17%) | 1.052 (36.19%) | 1.015 (34.91%) | 0.428 (14.73%) |

## 与原 Offline 结果对比

原 Offline Encode 与本轮无 hook Encode 的计时边界相同，结果基本一致。
`profiled_encode_wall_seconds` 是带 hooks 的完整 Encode replay 墙钟时间，而
`transformer_total_seconds` 只统计 Hugging Face Transformer forward 的 CUDA 时间。
“Encode 内其余时间”为同一次 replay 的墙钟时间减去 Transformer 时间，主要包含
tokenizer、H2D、pooling、D2H 和 micro-batch 调度开销。

| 任务 | 原 Offline Encode | 本轮无 hook Encode | Profile replay 墙钟 | Transformer 总时间 | Encode 内其余时间 | Transformer / replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cora_node` | 3.373 | 3.375 | 3.392 | 1.893 | 1.500 | 55.80% |
| `cora_link` | 3.363 | 3.371 | 3.377 | 1.893 | 1.484 | 56.05% |
| `pubmed_node` | 31.608 | 31.664 | 31.776 | 14.075 | 17.702 | 44.29% |
| `pubmed_link` | 31.757 | 31.644 | 31.746 | 14.042 | 17.704 | 44.23% |
| `arxiv` | 230.974 | 231.092 | 232.062 | 117.482 | 114.580 | 50.63% |
| `WN18RR` | 6.772 | 6.740 | 6.895 | 2.906 | 3.989 | 42.15% |

原 Offline 表中的 `Other` 是 GNN pipeline 的墙钟时间减去完整 GNN device forward；
本实验的 `other_transformer_seconds` 则是 Transformer backbone 内未归入 Attention、
QKV 和 FFN 的部分，两者计时范围不同，不能直接比较。

## 日志

本轮日志位于
[`results/offline_tf_profile_260923005943/`](./results/offline_tf_profile_260923005943/)：

- [`cora_node.log`](./results/offline_tf_profile_260923005943/cora_node.log)
- [`cora_link.log`](./results/offline_tf_profile_260923005943/cora_link.log)
- [`pubmed_node.log`](./results/offline_tf_profile_260923005943/pubmed_node.log)
- [`pubmed_link.log`](./results/offline_tf_profile_260923005943/pubmed_link.log)
- [`arxiv.log`](./results/offline_tf_profile_260923005943/arxiv.log)
- [`WN18RR.log`](./results/offline_tf_profile_260923005943/WN18RR.log)
- [`launcher.log`](./results/offline_tf_profile_260923005943/launcher.log)
- [`run_status.csv`](./results/offline_tf_profile_260923005943/run_status.csv)

## 测试

测试不下载 Hugging Face 权重，使用小型 BERT、DistilBERT 和 Llama 结构验证聚合
调用数、输出不变、hook 清理、CUDA Event 释放，以及 offline replay 仍然调用
`dataset.text2feature(texts)`：

```powershell
python -m unittest exp.encode.tf_profile.test_transformer_profiler -v
```

静态编译：

```powershell
python -m py_compile models/transformer_profiler.py `
  exp/encode/tf_profile/test_transformer_profiler.py `
  exp/time_pipe/encode-gnn/offline_encode_gnn_time.py `
  utils.py
```
