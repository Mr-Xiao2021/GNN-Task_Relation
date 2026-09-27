# Degree-Aware Quant GNN 实现报告

## 1. 目标与范围

本次开发用于观察“按节点度数分配推理精度”对六类 GNN 任务最终指标的影响：

- Cora 节点分类
- Cora 链接预测
- PubMed 节点分类
- PubMed 链接预测
- Arxiv 节点分类
- WN18RR 知识图谱任务

实验将每个输入图中度数排名前 `a%` 的节点定义为高精度节点，其余节点定义为低精度节点。高精度路径使用 FP32，低精度路径使用 INT8 quantize-dequantize（QDQ）数值仿真，`a` 分别取 20、30、40、50。

实现只修改推理路径，不进行量化感知训练（QAT），也不改变已有 checkpoint 的参数结构。目标是先回答 metric sensitivity 问题，不把 QDQ 运行时间解释为真实 INT8 kernel 性能。

## 2. 节点分组定义

### 2.1 度数口径

对 PyG batch 中的每张 prompted graph 独立统计入度：

```text
degree(v) = count(edge_index[1] == v)
```

选择数量为：

```text
high_count = ceil(a / 100 * graph_node_count)
```

每张图内部按入度降序排序，前 `high_count` 个节点进入 FP32 集合，其余节点进入 INT8 集合。相同度数使用稳定排序，按节点原始索引先后决定，保证边界节点选择可复现。

该定义有三个重要性质：

1. 每张图都严格获得指定比例的高精度节点，不会被同 batch 中更大的图挤占。
2. 正数比例下，小图至少有一个高精度节点。
3. 度数并列时结果稳定，不依赖 GPU 排序的任意顺序。

### 2.2 混合精度的传播语义

节点精度并不是完全隔离的。低精度源节点产生的消息可以传给高精度目标节点，因此高精度节点仍可能受到低精度邻居的量化误差影响。这符合消息传递 GNN 的数据依赖关系，也避免将图人为切成互不连通的两个子图。

## 3. INT8 QDQ 设计

### 3.1 对称量化

低精度路径使用 signed INT8 对称量化，整数范围为 `[-127, 127]`：

```text
scale = max(abs(x)) / 127
q     = clamp(round(x / scale), -127, 127)
x_qdq = q * scale
```

全零张量使用安全 scale，并保持输出全零。QDQ 的返回值仍是浮点张量，因此可以直接复用现有 PyTorch/PyG 算子，同时模拟 INT8 舍入与截断造成的数值误差。

### 3.2 量化粒度

| 对象 | 粒度 | 说明 |
| --- | --- | --- |
| 节点 activation | per-tensor | 仅在低精度节点行上标定与 QDQ |
| 边属性与 message | per-tensor | 按源节点是否为低精度节点决定 |
| 聚合输出 | per-tensor | 按目标节点是否为低精度节点决定 |
| RGCN relation weight | per-output-channel | 每个 relation 独立量化 |
| RGCN root weight | per-output-channel | 低精度目标节点使用量化权重 |
| bias | FP32 | 不量化 |

### 3.3 模型中的量化边界

| 阶段 | 高度数节点 | 低度数节点 |
| --- | --- | --- |
| ST embedding 输入投影 | FP32 | FP32 |
| RGCN 节点输入 | FP32 | INT8 QDQ |
| edge attribute / message | FP32 源节点 | INT8 QDQ 源节点 |
| relation aggregation | FP32 目标节点 | INT8 QDQ 目标节点 |
| relation/root linear | FP32 weight | INT8 QDQ weight |
| layer update | FP32 | INT8 QDQ |
| BatchNorm/ReLU 后输出 | FP32 | INT8 QDQ |
| 最终 prediction MLP | FP32 | FP32 |

当前范围保留输入投影和最终 MLP 为 FP32，重点隔离六层 RGCN message-passing backbone 的 Degree Quant 影响。

### 3.4 权重量化缓存

relation weight 和 root weight 在推理期间不会变化，因此量化结果按参数 `_version`、device 和 dtype 缓存。加载 checkpoint、修改参数或迁移设备后缓存会自动失效并重建。

缓存不注册为 parameter 或 buffer，所以不会进入 `state_dict`，也不会改变 checkpoint key。测试确认原始 `PyGRGCNEdge` state dict 可以 strict-load 到量化模型。

## 4. 代码结构

### `gp/nn/degree_quant.py`

提供与模型无关的基础函数：

- `symmetric_fake_quantize`：对称 QDQ，支持 per-tensor 和指定 channel axis。
- `mixed_fake_quantize`：仅量化由 mask 选中的张量行。
- `mixed_precision_linear`：FP32 参考矩阵乘与低精度行 QDQ 替换。
- `high_degree_mask`：按每张图入度选择精确比例的高精度节点。

`mixed_precision_linear` 保留完整 FP32 GEMM，再替换低精度行。这样 FP32 行的矩阵形状和归约路径与原模型一致，避免因拆分子矩阵而产生额外浮点漂移。

### `gp/nn/layer/pyg.py`

新增 `DegreeQuantRGCNEdgeConv`，继承现有 `RGCNEdgeConv`，保持原有 relation/root/bias 参数布局。该层负责：

1. 对低精度节点输入做 QDQ。
2. 根据源节点 mask 量化 relation edge attribute 和 message。
3. 根据目标节点 mask 量化聚合结果。
4. 分别执行 relation linear 与 root linear 的混合精度路径。
5. 对低精度 layer update 再做 QDQ。

未提供 mask 时直接调用原始 FP32 forward，用于同模型、同 checkpoint 的公平基线。

### `models/model.py`

新增 `PyGDegreeQuantRGCNEdge`，在进入六层 message passing 前计算一次 degree mask，并在所有层复用。它保持原模型的 BatchNorm、ReLU、dropout 和 JK 行为，在每层后处理结束后重新建立低精度 activation 边界。

量化模式被显式限制为 inference-only；若模型仍处于 training mode，会直接报错，避免误把当前实现当作 QAT 使用。

### `exp/degree_quant/evaluate_degree_quant.py`

提供六任务统一评测入口，主要能力包括：

- 内置六个 checkpoint 路径。
- 默认运行 FP32、a=20、30、40、50 共 30 行结果。
- 复用本地 ST feature cache，不加载文本编码器。
- 每完成一个条件即原子更新 JSON/CSV，降低长任务中断后的数据损失风险。
- 记录 checkpoint、样本数、batch 数、耗时、绝对 delta 和相对 delta。

### 测试与文档

- `exp/degree_quant/test_degree_quant.py`：8 个 focused unit tests。
- `exp/degree_quant/README.md`：使用说明与实验定义。
- `exp/degree_quant/RESULTS.md`：精简结果表与主要观察。
- `exp/degree_quant/IMPLEMENTATION_REPORT.md`：本实现报告。

## 5. 确定性与评测口径

### 5.1 确定性设置

GNN 的 CUDA 聚合和矩阵乘在默认设置下可能出现微小非确定性，经过六层传播后可能翻转边界样本。评测入口采用以下措施：

- 固定随机种子为 1。
- 设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- 启用 `torch.use_deterministic_algorithms(True)`。
- 固定 batch size 128、DataLoader workers 4。
- 在 CPU 上累计 Accuracy/AUROC，避免 GPU AUC `cumsum` 与确定性模式冲突。

代表条件的独立复跑得到逐值一致的 FP32 与混合精度结果。

### 5.2 FP32 对照

每个任务先用同一个新模型、同一个 checkpoint、同一数据顺序运行一次未启用 quant mask 的 FP32 基线。所有 delta 均相对该行计算。

历史 W&B summary 通常记录训练最后 epoch 的模型，而本实验加载明确列出的 checkpoint。例如 Cora link 使用 epoch 48 checkpoint，历史 summary 对应 epoch 400，因此两者绝对基线不要求一致。配对的 FP32 行才是量化影响的有效对照。

## 6. 六任务结果

括号内为相对同任务 FP32 的绝对变化。

| 任务 | 指标 | FP32 | a=20 | a=30 | a=40 | a=50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora node | Accuracy | 0.702611 | 0.701161 (-0.001451) | 0.700677 (-0.001934) | 0.705029 (+0.002418) | 0.703578 (+0.000967) |
| Cora link | AUROC | 0.948873 | 0.948146 (-0.000726) | 0.947926 (-0.000947) | 0.948159 (-0.000714) | 0.948918 (+0.000045) |
| PubMed node | Accuracy | 0.719215 | 0.720102 (+0.000887) | 0.718954 (-0.000261) | 0.719841 (+0.000626) | 0.719528 (+0.000313) |
| PubMed link | AUROC | 0.980520 | 0.980388 (-0.000131) | 0.980456 (-0.000064) | 0.980363 (-0.000157) | 0.980555 (+0.000035) |
| Arxiv | Accuracy | 0.709154 | 0.709504 (+0.000350) | 0.709113 (-0.000041) | 0.708763 (-0.000391) | 0.709586 (+0.000432) |
| WN18RR | Accuracy | 0.973516 | 0.973516 (+0.000000) | 0.973516 (+0.000000) | 0.973835 (+0.000319) | 0.973516 (+0.000000) |

### 6.1 汇总结论

- 24 组混合精度结果中，最大下降为 `-0.001934`（Cora node, a=30）。
- 最大上升为 `+0.002418`（Cora node, a=40）。
- a=20、30、40、50 的六任务平均绝对变化分别为 `0.000591`、`0.000541`、`0.000771`、`0.000299`。
- 在当前四个候选值中，a=50 的平均绝对扰动最小。
- 除 Cora node 外，其余任务所有档位均保持在 FP32 的 `0.000947` 以内。
- metric 不随 a 单调变化。当前实验不支持“FP32 高度数节点越多，metric 必然越高”的结论。

这些结果表明：对当前 checkpoint 和量化配置，Degree Quant 的整体精度影响较小，但节点比例并不是一个简单的单调控制量。若后续以部署为目标，应同时测量真实 INT8 kernel 的收益，再在 metric 与硬件性能之间选择工作点。

## 7. 验证

测试覆盖以下行为：

1. 每张图精确选择 top percentage。
2. 度数并列时使用稳定节点顺序。
3. 高精度行保持逐元素不变。
4. 全零张量量化后仍为全零。
5. mixed linear 与参考公式逐元素一致。
6. 原始与量化模型 state dict key 完全兼容。
7. a=100 时量化模型与原模型逐元素一致。
8. training mode 下拒绝启用量化路径。

最终验证结果：

```text
Ran 8 tests in 0.013s
OK
```

同时完成 Python 语法检查、`git diff --check` 和完整 30 行结果完整性检查。

## 8. 复现命令

完整六任务评测：

```bash
CUDA_VISIBLE_DEVICES=1 /data1/xxr_data/new_conda/ofa/bin/python \
  exp/degree_quant/evaluate_degree_quant.py \
  --tasks cora_node cora_link pubmed_node pubmed_link arxiv WN18RR \
  --high-precision-percents 20 30 40 50 \
  --num-workers 4 \
  --output-dir outputs/degree_quant/six_tasks_a20_30_40_50
```

单元测试：

```bash
/data1/xxr_data/new_conda/ofa/bin/python -m unittest -v \
  exp.degree_quant.test_degree_quant
```

结果文件：

- `outputs/degree_quant/six_tasks_a20_30_40_50/results.json`
- `outputs/degree_quant/six_tasks_a20_30_40_50/results.csv`

## 9. 局限与后续方向

### 当前局限

1. QDQ 仍由浮点算子执行，不能代表真实 INT8 latency、吞吐或显存占用。
2. 度数在每个 prompted graph 内计算，不是原始完整大图上的全局度数。
3. activation scale 基于当前 batch 中的低精度行，batch size 会影响标定范围。
4. 仅覆盖 INT8、单个 checkpoint 和单个固定 seed，尚未给出多次运行置信区间。
5. 输入 projection、bias 和 prediction MLP 保持 FP32，结果不能外推到全模型 INT8。
6. 高精度节点仍接收低精度邻居消息，因此节点级 precision 并非误差隔离域。

### 建议的下一步

1. 实现真实 mixed INT8 kernel 或分组 GEMM，测量端到端延迟、吞吐和峰值显存。
2. 比较 per-graph、per-batch、EMA calibration 和 percentile calibration。
3. 扩展 INT4/INT6/INT8 与不同 weight/activation bit-width 组合。
4. 比较入度、出度、总度数、PageRank、中心性等节点重要性指标。
5. 对多个 seed 和 checkpoint 重复评测，报告均值、标准差和置信区间。
6. 若 PTQ 精度不能满足目标，再引入 QAT 或轻量 calibration fine-tuning。
