# NumPy 文本准备与 eager encode 验证

该实验只验证 PyG `g` 已经由 DataLoader 加载出来之后的文本处理，不运行 GNN、prediction head 或 metric，也不需要训练 checkpoint。

脚本在同一个 batch、同一个 Hugging Face 编码器上比较三层实现：

- `fixed_width_numpy`：把文本重建为 `<Umax>` 固定宽度数组，再执行 `np.concatenate + np.unique`，作为最基础旧实现 baseline；
- `object_numpy`：保持 `dtype=object`，仍执行 `np.concatenate + np.unique`，只验证消除固定宽度对齐的收益；
- `object_hash`：使用变长字符串列表和 Python 字典去重，继续验证替换 NumPy 排序去重的收益。

两条路径分别报告：

- 文本拼接、去重和 mapping 构造时间；
- tokenizer CPU 时间；
- token tensor 从 CPU 搬到 GPU 的时间；
- LLM forward + pooling 时间；
- encoder micro-batch 输出拼接时间。

脚本还会验证三条路径的唯一文本集合和 mapping 还原结果，并检查 `object_numpy`、`object_hash` 按文本对齐后的 embedding 是否满足 `torch.allclose`。固定宽度路径和 object NumPy 路径得到相同的排序唯一文本时，二者后续 tokenizer/LLM 输入完全一致，因此固定宽度 baseline 复用 object NumPy 的 encode 计时，不重复运行一次相同的编码。

## WikiCS 单 batch

从项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python exp/encode/benchmark_loaded_g.py \
  --split test \
  --loader-index 0 \
  --batch-num 1 \
  --batch-size 128 \
  --fixed-width-mode auto \
  --fixed-width-chars 116022 \
  --fixed-width-limit-gib 8 \
  --warmup-batches 1 \
  --device cuda:0 \
  task_names wikics \
  llm_name ST \
  llm_b_size 100 \
  llm_max_length 500 \
  num_workers 4
```

`--batch-num` 控制测量多少个已加载 batch。配置覆盖参数必须放在脚本参数之后，因为最后一段由 `argparse.REMAINDER` 传给项目配置系统。
使用 `--batch-num -1` 可以测量所选 loader 的全部 batch。

`--batch-size` 控制一个 PyG DataLoader batch 中的图样本数，并在构造 DataLoader 前覆盖 YAML 和末尾配置参数。例如 `--batch-size 1 --batch-num 1` 表示只测量一个仅含一个图样本的 batch。

固定宽度 baseline 有三种执行模式：

| 参数 | 行为 |
|---|---|
| `--fixed-width-mode estimate` | 只报告 `<Umax>` 数组大小和预计最低峰值，不实际分配 |
| `--fixed-width-mode auto` | 预计最低峰值不超过 `--fixed-width-limit-gib` 时才实测，否则只报告估算 |
| `--fixed-width-mode force` | 忽略内存阈值，强制构造固定宽度数组并运行 `np.unique` |

WikiCS 的已知全局最大文本长度是 116,022 字符，因此使用 `--fixed-width-chars 116022` 可复现旧缓存的宽度。`force` 可能造成上百 GB 内存占用甚至被系统 OOM killer 终止，只应在确认服务器可用内存后运行：

```bash
--fixed-width-mode force --fixed-width-chars 116022
```

不传 `--fixed-width-chars` 时，脚本使用当前 batch 内最长文本长度，这更适合 Cora/PubMed 的常规对照，但不一定等于旧全局缓存的 dtype 宽度。

## Cora/PubMed 对照

只需要替换任务名：

```bash
task_names cora_node
task_names pubmed_node
```

## 结果解释

所有 `*_over_*` speedup 大于 `1` 都表示分母路径更快。例如 `fixed_width_over_object_numpy_prepare=2.0` 表示固定宽度准备耗时是 object NumPy 的两倍。

重点检查：

```text
correctness.same_unique_texts
correctness.object_numpy_mapping_restores_inputs
correctness.object_hash_mapping_restores_inputs
correctness.embeddings_allclose
```

这些字段都应为 `true`。如果 `embeddings_allclose=false`，需要结合 `embedding_max_abs_diff` 判断是浮点执行顺序差异还是功能错误。

计时从 DataLoader 返回 `g` 后开始，因此不包含数据集初始化、磁盘读取、PyG collate 和模型加载。`node_array_shallow_bytes`/`edge_array_shallow_bytes` 对 object array 只统计引用数组本身，不包含 Python 字符串对象占用。

固定宽度 baseline 从已经加载的 object 文本重新构造 `<Umax>`，因此能够比较数组物化、拼接和去重，但仍不包含磁盘读取旧 `.pt` 缓存和 PyG collate 的耗时。完整评估缓存加载阶段时，还应分别在 `main` 和 `feat/np-encode` 上记录缓存文件大小、DataLoader 首 batch 延迟和进程峰值 RSS。

单 batch 中两条路径的 LLM 执行顺序可能影响首轮 CUDA 开销。脚本会先 warmup，并在多 batch 测量时交替执行顺序。正式比较建议至少运行 5 个 batch，并重复执行整条命令。
