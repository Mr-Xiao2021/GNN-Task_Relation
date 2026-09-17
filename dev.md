# 开发日志

本文记录在原 OFA 数据流程上增加的开发功能、设计决策和使用配置。后续开发记录继续追加到本文件。

## 2026-09-16：Batch 到达时编码文本与 PEFT 支持

### 开发目标

原流程只支持在数据预处理阶段调用 `SentenceEncoder`，把整个数据集的文本提前编码为数值特征。此次开发为 `load_texts=True` 补齐运行时文本编码，使文本编码器可以进入训练计算图，并支持 PEFT/LoRA 微调、4-bit 量化以及 adapter 保存和加载。

此次修改保留了原来的 `load_texts=False` 分支。两个分支的目标语义相同：均把节点文本和边文本编码为向量，再经过 `llm_proj` 后送入 GNN；区别主要是编码发生的时间。

| 配置 | 文本编码时间 | 数据缓存内容 | 编码器能否参与训练 |
| --- | --- | --- | --- |
| `load_texts=False` | 数据预处理阶段 | 已编码的数值特征 | 否，编码结果离线缓存 |
| `load_texts=True` | 每个 PyG batch 进入模型时 | 原始节点/边文本 | 可以，由 `llm_trainable` 控制 |

当两个分支使用相同的 `llm_name`、基础模型权重、tokenizer、`llm_max_length` 和 pooling 方式，并且关闭 PEFT 与量化时，两者具有相同的文本特征语义。由于执行设备和浮点精度可能不同，不保证逐位完全一致。

### `load_texts=False`：原离线编码流程

1. `run_cdm.py` 创建 `SentenceEncoder`。
2. `OFAPygDataset.process()` 调用 `text2feature()`。
3. `SentenceEncoder.encode()` 在 GPU 上分批完成 tokenize、LLM forward 和 mean pooling。
4. embedding 被移动到 CPU，并写入处理后的数据缓存。
5. DataLoader 取 batch 时，`OFA_collater` 将数值 `x` 和 `edge_attr` 转成 Tensor。
6. Lightning 自动将 batch 搬到模型所在设备。
7. `BinGraphModel.forward()` 依次执行 `llm_proj`、GNN 和分类头。

该分支不构建运行时文本编码器，不支持对文本编码器进行 PEFT 微调。现有数值特征加载逻辑保持不变。

### `load_texts=True`：运行时编码流程

1. `OFAPygDataset.process()` 调用 `add_raw_texts()`，缓存原始文本而不是 embedding。
2. 每个数据样本仍然是 `torch_geometric.data.Data`。
3. DataLoader 调用 `OFA_collater.__call__()`，通过 PyG Collater 合并成一个 `Batch`。
4. `OFA_collater` 发现 `x`、`edge_attr` 是字符串数组后保留原始文本，不执行 tokenize。
5. `run_cdm.py` 根据 `JK` 创建 `BinGraphLLMModel` 或 `BinGraphAttLLMModel`。
6. 两个模型都把 `EagerSentenceEncoder` 放在继承顺序首位，因此首先进入 `EagerSentenceEncoder.forward()`。
7. `_encode_graph_texts()` 合并节点文本和边文本，通过 `np.unique()` 去重并记录恢复映射。
8. `_encode_texts()` 使用同一个 `LLMModel` 所属的 tokenizer 和 Transformer，在模型所在设备上分批编码文本。
9. 根据恢复映射重建原顺序，并按 `num_nodes` 拆回 `g.x` 和 `g.edge_attr`。
10. `super().forward(g)` 进入 `BinGraphModel.forward()`，执行 `llm_proj`、GNN 和分类头。

核心调用关系为：

```text
DataLoader
  -> OFA_collater(batch: list[Data])
  -> PyG Batch（x/edge_attr 为原始文本）
  -> BinGraphLLMModel.forward()
  -> EagerSentenceEncoder.forward()
  -> tokenize + Transformer + pooling
  -> 恢复 g.x/g.edge_attr
  -> BinGraphModel.forward()
  -> llm_proj -> GNN -> MLP
```

`llm_b_size` 是一次送入文本编码器的唯一文本数量，即文本 micro-batch 大小；它不是 PyG 图 batch 的样本数。PyG 图 batch 大小仍由 `batch_size` 控制。

### `EagerSentenceEncoder`

该类负责把运行时文本编码集中在模型侧，主要职责如下：

- 创建并持有 `LLMModel` 及其 tokenizer。
- 对当前图 batch 内的节点文本、边文本去重。
- 按 `llm_b_size` tokenize 和编码。
- 把编码结果恢复到原文本顺序，并拆回节点特征和边特征。
- 根据 `llm_trainable` 决定编码器是否构建梯度。
- 通过 `save_peft_adapter()` 保存 PEFT adapter 和 tokenizer。

`EagerSentenceEncoder` 本身不是另一套文本模型；实际 Transformer 与 tokenizer 都由内部的同一个 `LLMModel` 提供，从而避免 tokenizer 与模型 checkpoint 不匹配。

### PEFT、量化与训练开关

| 配置 | 含义 |
| --- | --- |
| `llm_peft` | 是否给文本编码器挂载 PEFT/LoRA adapter |
| `llm_trainable` | 文本编码器参数是否参与反向传播 |
| `llm_quantization` | 是否以 bitsandbytes 4-bit 方式加载基础文本模型 |
| `llm_adapter_path` | 加载已有 PEFT adapter 的目录；为空表示创建新 adapter 或不使用 adapter |
| `llm_adapter_save_path` | adapter 保存目录；默认 `auto` |
| `llm_max_length` | tokenizer 最大序列长度 |
| `llm_b_size` | 文本编码 micro-batch 大小 |

推荐组合：

| 使用场景 | `llm_peft` | `llm_trainable` | `llm_adapter_path` |
| --- | --- | --- | --- |
| 冻结基础编码器 | `False` | `False` | 空 |
| 新建 LoRA 并训练 | `True` | `True` | 空 |
| 加载 LoRA 继续训练 | `True` | `True` | 已有 adapter 目录 |
| 加载 LoRA 推理 | `True` | `False` | 已有 adapter 目录 |

量化基础模型并继续训练时必须启用 PEFT，即：

```yaml
llm_quantization: True
llm_peft: True
llm_trainable: True
```

代码会拒绝 `llm_quantization=True + llm_trainable=True + llm_peft=False`，因为当前实现不支持直接训练量化后的完整基础模型。

### Adapter 保存与加载

PEFT 微调期间，LoRA 参数位于 `model.llm_model.model`，并包含在 `run_cdm.py` 创建的 Adam 优化器参数中。冻结参数虽然也可能出现在优化器参数列表里，但没有梯度，因此不会更新。

默认配置为：

```yaml
llm_adapter_path:
llm_adapter_save_path: "auto"
```

当以下条件同时成立时：

```yaml
load_texts: True
llm_peft: True
llm_trainable: True
```

`auto` 会解析为：

```text
<当前 exp_dir>/peft_adapter/
```

训练结束后，`save_peft_adapter()` 使用 PEFT 的 `save_pretrained()` 保存 LoRA adapter，并同时保存 tokenizer。程序会打印最终绝对路径。

可以显式指定保存目录：

```yaml
llm_adapter_save_path: "saved_adapters/cora_node"
```

设置为空或 `"none"` 可以关闭 adapter 保存。

下次加载时，`LLMModel` 先根据同一个 `llm_name` 加载基础模型，再调用 `PeftModel.from_pretrained()` 挂载 adapter。推理配置示例：

```yaml
load_texts: True
llm_name: "ST"
llm_peft: True
llm_trainable: False
llm_quantization: False
llm_adapter_path: "saved_exp/<experiment>/peft_adapter"
llm_adapter_save_path:
```

继续训练已有 adapter 时，将 `llm_trainable` 改为 `True`。如果同时使用 4-bit 量化，加载 adapter 前会执行 `prepare_model_for_kbit_training()`。

### Adapter 与完整任务 checkpoint 的区别

PEFT adapter 只表示文本编码器的 LoRA 增量参数，不包含完整任务所需的全部权重。完整推理还需要：

- `llm_proj` 投影层；
- GNN；
- 分类 MLP/attention；
- 其他任务模型状态。

这些参数由 Lightning checkpoint 管理：

```yaml
save_model: True
```

如果需要按验证指标保存并恢复最佳模型，应同时配置：

```yaml
save_model: True
load_best: True
```

执行顺序是：

```text
训练 -> 保存验证指标最佳 checkpoint -> 训练结束加载最佳 checkpoint
     -> validate/test -> 保存该最佳状态对应的 PEFT adapter
```

如果二者都是 `False`，默认保存的是最后一个 epoch 结束后的 adapter。`load_best=True + save_model=False` 是无效组合，因为没有 checkpoint 可以恢复。

### 推荐训练配置

```yaml
load_texts: True
llm_name: "ST"
llm_b_size: 1
llm_max_length: 500

llm_peft: True
llm_trainable: True
llm_quantization: False
llm_adapter_path:
llm_adapter_save_path: "auto"

save_model: True
load_best: True
```

如果显存有限，可以开启 `llm_quantization: True`。`llm_b_size` 与图 `batch_size` 都会影响峰值显存，需要分别调整。

### 依赖与验证

PEFT/量化路径使用以下项目依赖：

```text
transformers==4.36.2
peft==0.7.1
bitsandbytes==0.42.0
```

本次开发已完成 Python 静态编译、diff 格式检查，以及原数值特征 collate 分支的等价性检查。当前本地 `pytorch` 环境未安装 `peft`，因此尚未在本机完成真实 adapter 的保存再加载回归测试；在 Linux 训练环境运行前应确认上述依赖可导入。

### 主要涉及文件

- `data/ofa_data.py`：根据 `load_texts` 选择缓存原始文本或离线 embedding。
- `ofa_datasets.py`：collate 后区分字符串与数值特征。
- `utils.py`：离线 `SentenceEncoder`。
- `models/model.py`：`LLMModel`、`EagerSentenceEncoder`、PEFT adapter 加载和保存。
- `run_cdm.py`：根据配置选择模型分支、创建优化器并保存 adapter。
- `configs/default_config.yaml`：文本编码、PEFT、adapter 和 checkpoint 配置。

## 2026-09-16：Encode/GNN 推理耗时统计

新增两个单设备推理计时脚本：

- `exp/time_pipe/encode-gnn/offline_encode_gnn_time.py`
- `exp/time_pipe/encode-gnn/eager_encode_gnn_time.py`

两种模式共用的参数解析、数据构造、模型构造、checkpoint 加载和结果汇总位于
`exp/time_pipe/encode-gnn/timing_utils.py`。其中 `build_offline_model()` 只创建
下游任务模型，`build_eager_model()` 额外持有冻结的文本编码器。

两个脚本复用 `run_cdm.py` 的默认配置、YAML override、任务构造器、DataModule 和模型结构，并强制关闭 PEFT、量化和文本编码器训练。可以通过 `--checkpoint` 加载 Lightning `.ckpt` 或 DeepSpeed checkpoint 目录；不提供 checkpoint 时仍可测量算子耗时，但任务模型权重是随机初始化的。

离线脚本从每个已构造数据集的 `texts.pkl` 重新执行一次全局文本编码，只用于计时，不修改 processed cache。其比例口径是“一次全局 encode 时间”对“所选 batch 的 GNN 推理时间”。因此 `batch_num=1` 时，encode 仍覆盖全局文本，这正是离线编码模式的真实一次性成本。

Eager 脚本针对每个被测 batch 分别同步计时：

```text
_encode_graph_texts()
BinGraphModel/BinGraphAttModel.forward()
```

第二段包括 `llm_proj + GNN + prediction head`。模型加载、数据准备、DataLoader 取数和 batch 搬运不计入 encode/GNN 比例；CUDA 计时前后均执行同步。`--warmup-batches` 控制预热 batch 数，`--batch-num -1` 表示测量整个 loader。

以 Cora node 的 test loader 第 0 项、一个 batch 为例：

```bash
python exp/time_pipe/encode-gnn/offline_encode_gnn_time.py \
  --split test --loader-index 0 --batch-num 1 \
  task_names cora_node batch_size 128 num_workers 0

python exp/time_pipe/encode-gnn/eager_encode_gnn_time.py \
  --split test --loader-index 0 --batch-num 1 \
  task_names cora_node batch_size 128 num_workers 0
```

加载训练权重时，将 checkpoint 参数放在尾部键值覆盖之前：

```bash
python exp/time_pipe/encode-gnn/eager_encode_gnn_time.py \
  --checkpoint saved_exp/<experiment>/full_cdm/<run-id>/checkpoints/last.ckpt \
  --batch-num 10 \
  task_names cora_node
```

输出为 JSON，包含 encode/GNN 秒数、占比、每 batch 时间、实际 batch/图数量、计时范围以及被排除的模型加载和数据准备时间。
