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

在仓库根目录执行原来的 offline 命令即可，Transformer 聚合计时默认开启：

```powershell
python exp/time_pipe/encode-gnn/offline_encode_gnn_time.py `
  --split test `
  --loader-index 0 `
  --batch-num 1 `
  --warmup-batches 1 `
  --encode-warmup-batches 1 `
  --repeats 1 `
  --skip-metric `
  --device cuda:0 `
  task_names cora_node `
  llm_name ST `
  llm_b_size 100
```

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
