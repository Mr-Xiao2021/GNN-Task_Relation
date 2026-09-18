# eager推理encode-gnn占比

本实验针对 `cora_node`、`pubmed_node`、`wikics` 等 e2e_node 任务，统计 eager 推理过程中：

- 文本 encode 时间；
- GNN forward 时间；
- encode 和 GNN 的时间占比；
- 所选验证集或测试集上的任务 metric。e2e_node 对应的 metric 是 accuracy。

当前实验不考虑 `llm_model` 的 PEFT、quantization 或训练。计时脚本会固定设置：

```text
llm_peft=False
llm_quantization=False
llm_trainable=False
llm_adapter_path=None
```

以下命令均从项目根目录执行。建议只暴露一张 GPU，避免 `run_cdm.py` 自动进入多 GPU DeepSpeed 模式。

## 1. 训练并保存 GNN 权重

推荐使用 `load_texts=false` 完成训练。此时文本由同一个 `llm_name` 离线编码并缓存，训练期间不需要在每个 batch 重复运行文本编码器。训练得到的 checkpoint 可以直接加载到 eager 模型；eager 模型缺少的 `llm_model.*` 权重会从 `llm_name` 指定的基础模型加载。

分别训练 Cora、PubMed 和 WikiCS：

```bash
for TASK in cora_node pubmed_node wikics; do
  CUDA_VISIBLE_DEVICES=0 python run_cdm.py \
    task_names "$TASK" \
    exp_name "${TASK}_gnn" \
    load_texts false \
    llm_name ST \
    llm_b_size 100 \
    llm_max_length 500 \
    batch_size 128 \
    num_epochs 50 \
    save_model true \
    load_best true \
    llm_peft false \
    llm_quantization false \
    llm_trainable false
done
```

`save_model=true` 用于启用 checkpoint 保存；`load_best=true` 用于在训练结束后恢复验证集 metric 最优的 checkpoint。

查找训练生成的 checkpoint：

```bash
find saved_exp -type f -path "*/checkpoints/*.ckpt"
```

通常每次实验会生成：

```text
saved_exp/<实验时间>/full_cdm/<run-id>/checkpoints/
├── epoch=<N>-step=<M>.ckpt
└── last.ckpt
```

后续推理优先使用非 `last.ckpt` 的 `epoch=...ckpt`，它是验证集 metric 最优的 checkpoint。

## 2. 执行 eager 推理计时

先填写三个数据集各自的最佳 checkpoint，然后顺序执行：

```bash
declare -A CHECKPOINTS=(
  [cora_node]="/absolute/path/to/cora/best.ckpt"
  [pubmed_node]="/absolute/path/to/pubmed/best.ckpt"
  [wikics]="/absolute/path/to/wikics/best.ckpt"
)
# 注意实际运行时，需要串行，防止显存不足或内存泄漏
for TASK in cora_node pubmed_node wikics; do
  CUDA_VISIBLE_DEVICES=0 python exp/time_pipe/encode-gnn/eager_encode_gnn_time.py \
    --checkpoint "${CHECKPOINTS[$TASK]}" \
    --split test \
    --loader-index 0 \
    --batch-num -1 \
    --warmup-batches 1 \
    --device cuda:0 \
    task_names "$TASK" \
    llm_name ST \
    llm_max_length 500 \
    batch_size 128 \
    llm_b_size 100 \
    num_workers 4
done
```

脚本使用了 `argparse.REMAINDER`。因此 `--checkpoint`、`--split`、`--loader-index`、`--batch-num`、`--warmup-batches` 和 `--device` 必须放在 `task_names ...` 等配置覆盖参数之前。

训练和推理必须保持以下模型结构配置一致，否则 checkpoint 无法完整加载：

```text
llm_name
emb_dim
num_layers
JK
rwpe
```

`batch_size` 是图样本 batch size，`llm_b_size` 是 eager 文本编码的 micro-batch size。二者都会影响时间结果；跨数据集比较时应保持一致。如果显存不足，可以同时降低各实验的 `batch_size` 或 `llm_b_size`。

## 3. 只统计一个 batch

快速检查单个 batch：

```bash
CUDA_VISIBLE_DEVICES=0 python exp/time_pipe/encode-gnn/eager_encode_gnn_time.py \
  --checkpoint "/absolute/path/to/cora/best.ckpt" \
  --split test \
  --loader-index 0 \
  --batch-num 1 \
  --warmup-batches 1 \
  --device cuda:0 \
  task_names cora_node \
  llm_name ST \
  llm_max_length 500 \
  batch_size 128 \
  llm_b_size 100 \
  num_workers 4
```

此时 metric 也只基于这一个 batch，不代表完整测试集 accuracy。需要完整测试集 metric 时使用：

```text
--batch-num -1
```

## 4. 数据划分和 loader_index

这些 e2e_node 任务的 DataLoader 顺序为：

| 参数 | 实际数据 |
|---|---|
| `--split val --loader-index 0` | validation split |
| `--split test --loader-index 0` | test split |
| `--split test --loader-index 1` | train split，以测试方式推理 |

正式测试使用：

```text
--split test --loader-index 0
```

即使选择 `--split train`，脚本仍然处于 `model.eval()` 和 `torch.inference_mode()`，不会进行反向传播或更新参数。由于训练 DataLoader 可能混合多个任务，它没有唯一的评估 metric，因此报告中的 metric 字段为 `null`。

## 5. 输出字段

脚本向标准输出打印 JSON，例如：

```json
{
  "mode": "eager",
  "task_names": [
    "cora_node"
  ],
  "checkpoint_loaded": true,
  "measured_batches": 8,
  "text_prepare_seconds": 0.12,
  "encode_seconds": 12.4,
  "text_restore_seconds": 0.03,
  "gnn_seconds": 0.8,
  "eager_pipeline_seconds": 13.35,
  "encode_percent": 93.94,
  "gnn_percent": 6.06,
  "metric_key": "test_cora_node/acc",
  "metric_name": "acc",
  "metric_value": 0.812
}
```

字段含义：

| 字段 | 含义 |
|---|---|
| `text_prepare_seconds` | CPU 端整理节点/边文本、哈希去重并建立位置映射的时间 |
| `encode_seconds` | tokenizer 和文本编码器 forward 的时间；只有 tokenizer 产出的张量会被送到 GPU |
| `text_restore_seconds` | 按位置映射恢复节点/边 embedding 并写回 `g.x/g.edge_attr` 的时间 |
| `gnn_seconds` | 相同 batch 的 GNN forward 总时间 |
| `encode_percent` | encode 在 `encode + GNN` 中的时间占比 |
| `gnn_percent` | GNN 在 `encode + GNN` 中的时间占比 |
| `eager_pipeline_seconds` | `prepare + encode + restore + GNN` 的完整 eager 推理流水线时间 |
| `metric_value` | 被统计 batch 对应的业务 metric；e2e_node 为 accuracy |
| `checkpoint_loaded` | 是否成功加载指定 checkpoint |

`encode_percent` 和 `gnn_percent` 有意只比较 tokenizer + 文本编码器与 GNN，不包含 CPU 文本整理和 embedding 位置还原。需要观察端到端代价时查看 `eager_pipeline_seconds` 及三个分项。

模型加载、数据准备、warmup 和 metric 更新不计入上述推理时间。没有传入 `--checkpoint` 时脚本仍可运行，但 GNN 使用随机初始化权重，得到的 metric 没有模型效果评估意义。

## 6. 旧 raw text 缓存迁移

旧缓存可能使用 NumPy 固定宽度 Unicode 数组保存文本。以 WikiCS 为例，数组会按最长文本宽度保存每个元素，即使代码已经修复，现有 `.pt` 缓存也不会自动改变。首次使用新实现前，应移动旧缓存，让数据集重新生成 object array 格式：

```bash
mv cache_data/wikics/raw/processed cache_data/wikics/raw/processed_fixed_width_backup
```

如果 Cora 或 PubMed 也已经生成过 `load_texts=true` 缓存，可分别执行：

```bash
mv cache_data/Cora/raw/processed cache_data/Cora/raw/processed_fixed_width_backup
mv cache_data/Pubmed/raw/processed cache_data/Pubmed/raw/processed_fixed_width_backup
```

目录名以本机 `cache_data` 中的实际大小写为准。下一次运行 eager 脚本会重新生成缓存；确认新缓存和推理正常后，再自行处理备份目录。
