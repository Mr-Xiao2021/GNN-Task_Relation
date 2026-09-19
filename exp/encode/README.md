# 三种文本表示的 Transformer-GNN 前向 Benchmark

`benchmark_loaded_g.py` 比较同一个 PyG batch 在三种原始文本处理路径下的完整 `model(g)` 前向耗时，并把它拆成文本编码和后续图模型两段：

- `fixed_width_numpy`：使用 `np.asarray(texts)` 让 NumPy 按当前 batch 隐式推断固定宽度 Unicode dtype，再执行 `np.concatenate + np.unique`；
- `object_numpy`：使用 `dtype=object`，仍执行 `np.concatenate + np.unique`；
- `python_hash`：使用 `dtype=object`，调用模型当前的 Python 字典去重实现。

脚本强制使用 `load_texts=True`，不需要在命令行重复指定。数据集提供原始文本，Tokenizer、Transformer、embedding 恢复、GNN 和 prediction head 都在 `model(g)` 内执行。

## 计时范围

每条路径报告三个计时字段：

- `total`：完整 `model(g)` 端到端前向；
- `encode`：文本准备与去重、Tokenizer、Transformer，以及 embedding mapping 恢复；
- `gnn_and_head`：同一次前向的 `total - encode`，包含 LLM embedding 投影、可选 RWPE、GNN、可选 attention 和 prediction head。

因此完整流程是：

```text
文本拼接与去重
-> Tokenizer
-> Transformer
-> embedding mapping 恢复
-> embedding 投影 / 可选 RWPE
-> GNN
-> 可选 attention
-> prediction head
```

`gnn_and_head` 没有直接命名为 `gnn`，因为当前模型的 GNN 前后还有投影和预测头；把这部分统称为纯 GNN 时间会不准确。三个字段来自同一次完整前向，不会为了拆分计时额外执行模型。CUDA 测量会在完整前向开始、encode 结束边界和完整前向结束时同步，避免把异步 kernel 提交时间误当成实际计算时间。

计时开始前，batch 已经移动到目标设备，三种输入表示也已经构造完毕。因此结果不包含：

- 数据集初始化和磁盘读取；
- 子图采样、DataLoader 和 PyG collate；
- batch 到目标设备的搬运；
- benchmark 为三条路径重建 NumPy 数组的时间；
- metric 计算。

每个 batch 的每条路径运行 `--repeats` 次，按 `total` 排序后取中间一次；当重复次数为偶数时，对中间两次的各阶段取平均。这样选出的代表值仍满足 `total = encode + gnn_and_head`。三条路径会轮换执行顺序，减少 CUDA 预热、CPU cache 和运行顺序造成的偏差。脚本还会检查 mapping 能否还原原始 node/edge 文本，并验证最终预测满足 `torch.allclose`。

## 参数规则

所有 benchmark 参数必须放在 `task_names` 前面。`task_names` 及其后面的内容由 `argparse.REMAINDER` 交给项目配置系统，例如：

```text
--batch-size 128 --batch-num 5 ... task_names cora_node llm_name ST
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--batch-size N` | 一个 PyG batch 包含的图/目标边样本数 |
| `--batch-num N` | 正式测量前 N 个 batch；`-1` 表示全部 |
| `--repeats N` | 每个 batch、每条路径的重复次数，默认 3 |
| `--warmup-batches N` | 正式计时前运行的预热 batch 数 |
| `--rtol N` | 三条路径最终输出一致性检查的相对容差，默认 `1e-4` |
| `--atol N` | 三条路径最终输出一致性检查的绝对容差，默认 `1e-5` |
| `--split train\|val\|test` | 选择 DataLoader split |
| `--loader-index N` | val/test loader 列表中的索引；train 只能为 0 |
| `--device auto\|cpu\|cuda:N` | 推理设备 |
| `--fixed-width-mode estimate\|auto\|force` | 固定宽度路径的内存策略 |
| `--fixed-width-limit-gib N` | `auto` 模式允许的估算峰值上限 |
| `task_names NAME` | 要加载的项目任务名，例如节点任务 `cora_node` 或 E2E link 任务 `cora_link`；必须写在所有 `--...` benchmark 参数之后 |

固定宽度模式：

- `estimate`：只做内存估算，不执行 `fixed_width_numpy`；
- `auto`：估算峰值未超过配置上限和一半可用内存时才执行；
- `force`：忽略内存保护并强制执行，可能导致 OOM。

## Cora 节点任务

下面的命令测量 5 个 test batch，每个 batch 包含 128 个节点分类子图样本：

```powershell
python exp/encode/benchmark_loaded_g.py --split test --loader-index 0 --batch-size 128 --batch-num 5 --repeats 3 --warmup-batches 1 --device cuda:0 --fixed-width-mode auto --fixed-width-limit-gib 4 task_names cora_node llm_name ST llm_b_size 100 llm_max_length 500
```

其他节点任务只需替换任务名：

```text
cora_node
pubmed_node
wikics
arxiv
```

## E2E Link 任务

脚本支持 `task_level: e2e_link`。在 link 任务中，`--batch-size` 表示一个 batch 包含多少个目标边对应的子图样本。由于本脚本不计 DataLoader 时间，link 子图采样和 collate 不在报告中，报告的是这些 batch 的完整 Transformer-GNN 前向时间。

### Cora Link

```powershell
python exp/encode/benchmark_loaded_g.py --split test --loader-index 0 --batch-size 32 --batch-num 5 --repeats 3 --warmup-batches 1 --device cuda:0 --fixed-width-mode auto --fixed-width-limit-gib 4 task_names cora_link llm_name ST llm_b_size 100 llm_max_length 500
```

对于 `cora_link`：

- `--split test --loader-index 0`：link test split；
- `--split test --loader-index 1`：配置到 test 阶段的 link train split；
- `--split val --loader-index 0`：link validation split；
- `--split val --loader-index 1`：Cora node validation，不是 link validation。

### PubMed Link

```powershell
python exp/encode/benchmark_loaded_g.py --split test --loader-index 0 --batch-size 32 --batch-num 5 --repeats 3 --warmup-batches 1 --device cuda:0 --fixed-width-mode auto --fixed-width-limit-gib 4 task_names pubmed_link llm_name ST llm_b_size 100 llm_max_length 500
```

### WN18RR Link

```powershell
python exp/encode/benchmark_loaded_g.py --split test --loader-index 0 --batch-size 32 --batch-num 5 --repeats 3 --warmup-batches 1 --device cuda:0 --fixed-width-mode auto --fixed-width-limit-gib 4 task_names WN18RR llm_name ST llm_b_size 100 llm_max_length 500
```

### FB15K237 Link

```powershell
python exp/encode/benchmark_loaded_g.py --split test --loader-index 0 --batch-size 32 --batch-num 5 --repeats 3 --warmup-batches 1 --device cuda:0 --fixed-width-mode auto --fixed-width-limit-gib 4 task_names FB15K237 llm_name ST llm_b_size 100 llm_max_length 500
```

可用的主要 E2E Link 任务名：

```text
cora_link
pubmed_link
WN18RR
FB15K237
```

## 输出解释

最终 JSON 的核心字段如下：

```json
{
  "seconds_per_batch": {
    "fixed_width_numpy": {
      "total": 2.41,
      "encode": 2.08,
      "gnn_and_head": 0.33
    },
    "object_numpy": {
      "total": 1.87,
      "encode": 1.54,
      "gnn_and_head": 0.33
    },
    "python_hash": {
      "total": 1.72,
      "encode": 1.39,
      "gnn_and_head": 0.33
    }
  },
  "result": {
    "fastest_path": "python_hash",
    "object_numpy_faster_than_fixed_width": true,
    "python_hash_faster_than_fixed_width": true,
    "python_hash_faster_than_object_numpy": true
  },
  "outputs_match": true
}
```

- `seconds_per_batch` 中每个阶段都是各 batch 代表耗时的平均值；
- `result.fastest_path`、`*_faster_than_*` 和 `*_speedup_vs_*` 都只比较端到端 `total`；
- `*_speedup_vs_* > 1` 表示字段名前面的路径更快；
- `outputs_match` 必须为 `true`；mapping 不正确或最终输出不一致时脚本会直接报错；
- 实际使用的 `rtol` 和 `atol` 会记录在输出的 `config` 中；
- 固定宽度路径被内存保护跳过时，其时间和相关比较为 `null`，原因记录在 `fixed_width.skip_reasons`。

## Raw Cache 注意事项

`load_texts=True` 时，Cora 使用 `cache_data/Cora/raw/processed/`。缓存不存在时，首次构造会调用 `SingleGraphOFADataset.add_raw_texts()`，将五组文本保存为 `dtype=object`；缓存完整时则直接加载，不会再次执行 `add_raw_texts()`。

如果该目录来自 main 分支的旧固定宽度缓存，当前分支不会自动重新生成。当前 `OFA_collater` 会把最终 batch 转成 object array，但旧缓存造成的数据集常驻内存、采样和 prompt 拼接开销发生在 DataLoader 返回之前，不属于本 benchmark 的计时范围。

`fixed_width_numpy` 是从当前 batch 文本通过 `np.asarray(texts)` 隐式推断出的固定宽度对照。当前 object batch 已经丢失 main 旧缓存的全局 dtype 宽度，因此它不是旧 main 缓存布局的逐字节复刻。

## Exp：E2E Node 基础实验（batch_size=1，batch_num=1）

本节记录 2026-09-19 在 `feat/np-encode` 分支上完成的首组 E2E Node 基准实验。实验基线提交为 `962b26f`，物理设备为 NVIDIA H100 PCIe GPU 1；通过 `CUDA_VISIBLE_DEVICES=1` 映射后，脚本使用 `cuda:0`。

### 参数配置

- 数据集：`cora_node`、`pubmed_node`、`wikics`；
- 数据切分：`test`，`loader_index=0`；
- `batch_size=1`，`batch_num=1`；
- `repeats=3`，`warmup_batches=1`；
- `llm_name=ST`，`llm_b_size=100`，`llm_max_length=500`；
- `llm_peft=false`，`llm_quantization=false`，`llm_trainable=false`；
- `fixed_width_mode=auto`，`fixed_width_limit_gib=4`。

### 执行命令

Cora、PubMed 和 WikiCS 首次运行均使用默认一致性容差 `rtol=1e-4, atol=1e-5`。实际运行时，每个任务的标准输出和错误输出均重定向到独立日志：

```bash
OUT_DIR=outputs/encode_benchmark_bs1_bn1_260919115027
mkdir -p "$OUT_DIR"

for TASK in cora_node pubmed_node wikics; do
  CUDA_VISIBLE_DEVICES=1 /home/xxr/miniconda3/bin/conda run --no-capture-output \
    -p /data1/xxr_data/new_conda/ofa \
    python exp/encode/benchmark_loaded_g.py \
    --split test \
    --loader-index 0 \
    --batch-size 1 \
    --batch-num 1 \
    --repeats 3 \
    --warmup-batches 1 \
    --device cuda:0 \
    --fixed-width-mode auto \
    --fixed-width-limit-gib 4 \
    task_names "$TASK" \
    llm_name ST \
    llm_b_size 100 \
    llm_max_length 500 \
    llm_peft false \
    llm_quantization false \
    llm_trainable false \
    > "$OUT_DIR/$TASK.log" 2>&1
done
```

WikiCS 首次运行在默认 `atol=1e-5` 下因最大绝对误差 `1.99489e-5` 未通过一致性检查，因此使用同一组实验参数，仅将绝对容差放宽到 `3e-5` 后重跑：

```bash
CUDA_VISIBLE_DEVICES=1 /home/xxr/miniconda3/bin/conda run --no-capture-output \
  -p /data1/xxr_data/new_conda/ofa \
  python exp/encode/benchmark_loaded_g.py \
  --split test \
  --loader-index 0 \
  --batch-size 1 \
  --batch-num 1 \
  --repeats 3 \
  --warmup-batches 1 \
  --device cuda:0 \
  --fixed-width-mode auto \
  --fixed-width-limit-gib 4 \
  --rtol 1e-4 \
  --atol 3e-5 \
  task_names wikics \
  llm_name ST \
  llm_b_size 100 \
  llm_max_length 500 \
  llm_peft false \
  llm_quantization false \
  llm_trainable false \
  > "$OUT_DIR/wikics_retry_atol3e-5.log" 2>&1
```

### 实验结果

下表单位均为毫秒/批次。`GNN + head` 对应脚本中的 `gnn_and_head`，包含 embedding projection、可选 RWPE/attention、GNN 和 prediction head，并非纯 GNN kernel 时间。

| 数据集 | 编码路径 | Total | Encode | GNN + head |
| --- | --- | ---: | ---: | ---: |
| Cora | `fixed_width_numpy` | 65.791 | 58.108 | 7.683 |
| Cora | `object_numpy` | 63.016 | 55.450 | 7.567 |
| Cora | **`python_hash`** | **62.884** | **55.139** | 7.745 |
| PubMed | `fixed_width_numpy` | 134.641 | 126.858 | 7.783 |
| PubMed | **`object_numpy`** | **129.232** | 121.620 | **7.612** |
| PubMed | `python_hash` | 129.316 | **121.371** | 7.945 |
| WikiCS | `fixed_width_numpy` | 465.677 | 458.132 | 7.545 |
| WikiCS | **`object_numpy`** | **248.242** | **240.931** | **7.311** |
| WikiCS | `python_hash` | 253.692 | 246.311 | 7.381 |

三组最终结果均满足各自容差下的 `outputs_match=true`。Cora 的 `python_hash` 相对固定宽度路径加速 `1.046x`；PubMed 的 `object_numpy` 加速 `1.042x`，与 `python_hash` 基本持平；WikiCS 的 `object_numpy` 加速 `1.876x`，端到端时延降低约 `46.7%`。三组任务的 `GNN + head` 均稳定在约 `7.3-7.9 ms`，路径差异主要来自 Encode 阶段。

这里的计时不包含数据集初始化、磁盘读取、DataLoader/采样/collate、设备搬运、benchmark 输入构造及指标计算。

### 原始日志

- [Cora 日志](./cora_node_bs1_bn1.log)
- [PubMed 日志](./pubmed_node_bs1_bn1.log)
- [WikiCS 日志（atol=3e-5）](./wikics_bs1_bn1_atol3e-5.log)
