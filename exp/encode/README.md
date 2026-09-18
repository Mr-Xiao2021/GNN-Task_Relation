# NumPy 文本准备与 eager encode 验证

该实验只验证 PyG `g` 已经由 DataLoader 加载出来之后的文本处理，不运行 GNN、prediction head 或 metric，也不需要训练 checkpoint。

脚本在同一个 batch、同一个 Hugging Face 编码器上比较：

- `numpy`：原实现的 `np.concatenate + np.unique(return_inverse=True)`；
- `hash`：变长字符串列表和 Python 字典去重。

两条路径分别报告：

- 文本拼接、去重和 mapping 构造时间；
- tokenizer CPU 时间；
- token tensor 从 CPU 搬到 GPU 的时间；
- LLM forward + pooling 时间；
- encoder micro-batch 输出拼接时间。

脚本还会验证两条路径的唯一文本集合、mapping 还原结果，以及按文本对齐后的 embedding 是否满足 `torch.allclose`。

## WikiCS 单 batch

从项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0 python exp/encode/benchmark_loaded_g.py \
  --split test \
  --loader-index 0 \
  --batch-num 1 \
  --warmup-batches 1 \
  --device cuda:0 \
  task_names wikics \
  llm_name ST \
  llm_b_size 100 \
  llm_max_length 500 \
  batch_size 128 \
  num_workers 4
```

`--batch-num` 控制测量多少个已加载 batch。配置覆盖参数必须放在脚本参数之后，因为最后一段由 `argparse.REMAINDER` 传给项目配置系统。
使用 `--batch-num -1` 可以测量所选 loader 的全部 batch。

## Cora/PubMed 对照

只需要替换任务名：

```bash
task_names cora_node
task_names pubmed_node
```

## 结果解释

`speedup_numpy_over_hash` 大于 `1` 表示 hash 路径更快。例如 `prepare=2.0` 表示原 NumPy 准备耗时是 hash 路径的两倍。

重点检查：

```text
correctness.same_unique_texts
correctness.numpy_mapping_restores_inputs
correctness.hash_mapping_restores_inputs
correctness.embeddings_allclose
```

这些字段都应为 `true`。如果 `embeddings_allclose=false`，需要结合 `embedding_max_abs_diff` 判断是浮点执行顺序差异还是功能错误。

计时从 DataLoader 返回 `g` 后开始，因此不包含数据集初始化、磁盘读取、PyG collate 和模型加载。`node_array_shallow_bytes`/`edge_array_shallow_bytes` 对 object array 只统计引用数组本身，不包含 Python 字符串对象占用。

脚本不会为了复现旧缓存而主动把 object array 转成 `<Umax>` 固定宽度数组，因为 WikiCS 上这一操作本身可能导致数百 GB 峰值内存。要评估缓存和 collate 阶段的改进，应分别在 `main` 和 `feat/np-encode` 上运行，并同时记录缓存文件大小、进程峰值 RSS 以及报告中的输入 dtype。本脚本内的两条路径专门比较 `g` 已经加载后的去重、tokenize 和 encode。

单 batch 中两条路径的 LLM 执行顺序可能影响首轮 CUDA 开销。脚本会先 warmup，并在多 batch 测量时交替执行顺序。正式比较建议至少运行 5 个 batch，并重复执行整条命令。
