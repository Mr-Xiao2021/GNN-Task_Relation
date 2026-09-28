# MixQ-GNN 实现阅读笔记

## 1. 仓库与阅读范围

本笔记只提取 MixQ-GNN 的量化工程实现，不将其“按算子学习位宽”的算法思想迁移到当前 Degree-Aware FP32/INT8 方案。

| 项目 | 内容 |
| --- | --- |
| 上游仓库 | <https://github.com/SamirMoustafa/MixQ> |
| 论文 | *Efficient Mixed Precision Quantization in Graph Neural Networks*, ICDE 2025 |
| 本地目录 | `MixQ/` |
| 上游分支 | `main` |
| 固定提交 | `f101b11b08b434f68d6ce29a80c68f48ec32d1c8` |
| 提交日期 | 2025-05-19 |
| 阅读日期 | 2026-09-28 |

重点阅读了量化参数、fake quantize、混合位宽搜索、固定量化、GCN/GIN message passing、Degree-Quant 对照实现、训练流程、测试和硬件 benchmark。没有改动 `MixQ/` 的源码，也没有把其代码复制进当前实现。

## 2. MixQ 如何实现混合精度

### 2.1 量化边界按 GNN 组件拆分

MixQ 没有把一层 GNN 视为单个量化单元，而是拆成多个独立边界：

- 节点输入和线性层输入；
- 线性层权重与输出；
- 归一化后的 edge weight；
- message passing 输入与聚合输出；
- 激活函数输入与输出。

以 GCN 为例，`MQGraphConvolution` 由 `MQLinear`、`MQInput` 和 `MQMessagePassing` 组合，并为 `lin_*`、`e_*`、`mp_*` 分别输出位宽选择结果。相关代码：

- [`quantization/mixed_modules/parametric/graph_convolution.py`](../../MixQ/quantization/mixed_modules/parametric/graph_convolution.py)
- [`quantization/mixed_modules/non_parametric/message_passing.py`](../../MixQ/quantization/mixed_modules/non_parametric/message_passing.py)

这种拆分方式值得借鉴，因为它让 calibration、误差定位和测试都能落到明确的数据边界；但当前项目仍应在这些边界上应用既定的“高度数 FP32、低度数 INT8”掩码，而不是改成 MixQ 的位宽搜索。

### 2.2 搜索阶段使用可微的软选择

每个候选位宽对应一套 `QuantizationParameters`。以 `MQLinear` 为例，输入、权重和输出都会执行所有候选位宽的 QDQ，再按可学习系数的 softmax 加权求和：

```text
x_mixed = sum_b softmax(alpha)[b] * fake_quantize_b(x)
```

分类损失之外，训练还加入“期望位宽 x 张量元素数”的代价项，促使模型在精度和量化成本之间权衡。训练结束后，对各边界的 softmax 系数取 top-1，生成固定的离散位宽配置。

实现位置：

- [`quantization/mixed_modules/base_module.py`](../../MixQ/quantization/mixed_modules/base_module.py)
- [`quantization/mixed_modules/parametric/linear.py`](../../MixQ/quantization/mixed_modules/parametric/linear.py)
- [`examples/mix_q_freezable_demo.py`](../../MixQ/examples/mix_q_freezable_demo.py)

这是 MixQ 的核心算法设计，不应迁移到当前实验。当前实验的分组依据仍然是源图真实节点的全局入度排名和固定的 `a=20/30/40/50`。

### 2.3 搜索与部署使用两套模块

代码明确区分：

- `MQ*`：搜索候选位宽的 relaxed mixed-precision 模块；
- `Q*`：位宽确定后的固定量化模块。

完整示例采用三阶段流程：

1. 训练 `MQGCN`，同时优化任务损失和位宽代价；
2. 取每个边界的获胜位宽，实例化 `QGCN`；
3. 先训练 FP32 路径，再用 fake quantization 微调，最后 `freeze()` 并切换到 quantized inference。

对应实现见 [`examples/mix_q_freezable_demo.py`](../../MixQ/examples/mix_q_freezable_demo.py)。这种“模式分离”是可借鉴的工程结构，但当前项目是 inference-only PTQ 数值实验，不应因此引入搜索训练或 QAT。

## 3. 可借鉴的实现细节

### 3.1 独立的量化参数对象

`QuantizationParameters` 将 `scale`、`zero_point`、观测到的最小值和最大值封装为模块状态，集中提供：

- calibration；
- quantize / dequantize / fake quantize；
- state dict 加载与复制；
- per-tensor 和 per-column 粒度。

相关代码：[`quantization/base_parameter.py`](../../MixQ/quantization/base_parameter.py)。

当前 `symmetric_fake_quantize()` 每次从输入即时计算 scale。后续若要比较静态 calibration、EMA calibration 或 percentile calibration，可以借鉴这种状态对象，但应继续使用当前对称 INT8 定义，避免同时改变实验变量。

### 3.2 自定义 STE fake quantize

`FakeQuantize` 的 forward 执行 affine quantize-dequantize，backward 对量化范围内的输入使用 STE，并为 `scale`、`zero_point` 提供梯度。代码还显式检查直接嵌套的重复 fake quantization。

相关代码：[`quantization/functional.py`](../../MixQ/quantization/functional.py)。

当前实验没有反向传播，因此不需要引入 STE；如果未来做 QAT，可借鉴其 autograd 边界和重复量化保护，而不是直接复制其 affine 参数化。真实 INT8 路径采用对称 scale、`torch.int8` 张量和 INT32 accumulator，不依赖 STE。

### 3.3 freeze 阶段传播量化参数

固定量化路径在 `freeze()` 时计算相邻边界的比例因子，并把上一模块的输出量化参数传给下一模块的输入。例如 GCN 中：

```text
linear output qparams -> message-passing input qparams
edge input qparams   -> message multiplier qparams
```

`QLinear` 使用 `M = scale_w * scale_in / scale_out`，message passing 使用 `M = scale_in * scale_edge / scale_out`。这样可以减少模块之间不必要的反量化和重新标定。

实现位置：

- [`quantization/fixed_modules/parametric/linear.py`](../../MixQ/quantization/fixed_modules/parametric/linear.py)
- [`quantization/fixed_modules/parametric/graph_convolution.py`](../../MixQ/quantization/fixed_modules/parametric/graph_convolution.py)
- [`quantization/fixed_modules/non_parametric/message_passing.py`](../../MixQ/quantization/fixed_modules/non_parametric/message_passing.py)

这是将来做真实 INT8 kernel 时最值得吸收的部分：明确记录每个边界的 scale contract，并在层间复用，而不是每个函数各自临时求 scale。

### 3.4 量化消息传递保持 message/aggregate 可注入

MixQ 重新实现 PyG 的参数收集和分发，使量化包装器仍可复用不同卷积层自己的 `message()` 和 `aggregate()`。GCN、GIN、GraphSAGE 和 TAGConv 因此能够共享相同的量化 message-passing 骨架。

实现位置：

- [`quantization/message_passing_base.py`](../../MixQ/quantization/message_passing_base.py)
- [`quantization/fixed_modules/non_parametric/message_passing.py`](../../MixQ/quantization/fixed_modules/non_parametric/message_passing.py)
- [`sage_and_tagconv_extention/`](../../MixQ/sage_and_tagconv_extention/)

当前项目只有一套 RGCN edge layer，暂时不需要抽象到同等复杂度。当前真实 INT8 实现提供 `int8`（INT8 GEMM + 浮点聚合）和 `int8_full`（额外启用 INT32 message aggregation）两个显式 backend。若六任务后续扩展到多种 GNN backbone，可借鉴“传播机制与量化策略解耦”的接口。

### 3.5 显式区分三种 forward

固定模型提供三条明确路径：

- `full_precision_forward`；
- `simulated_quantize_forward`；
- `quantize_inference`。

这种模式比在普通 `forward()` 内隐式判断训练状态更容易审计。当前实现已通过 `high_precision_percent=None` 表示 FP32、非空表示 QDQ，并拒绝训练模式下启用量化；后续可进一步改成显式枚举模式，减少状态组合。

### 3.6 对消息聚合单独做一致性测试

上游测试不仅检查 shape，还比较 simulated quantization 与 frozen inference 的数值接近性，并验证自定义 message passing 与 PyG 参考聚合一致：

- [`test/test_message_passing.py`](../../MixQ/test/test_message_passing.py)
- [`test/test_graph_conv_module.py`](../../MixQ/test/test_graph_conv_module.py)
- [`test/test_graph_iso_module.py`](../../MixQ/test/test_graph_iso_module.py)

当前测试已覆盖全局 rank、prompt 节点 FP32、QDQ、checkpoint 兼容、`a=100` 等价性、CPU INT8 参考计算，以及 CUDA 路径确实调用 `torch._int_mm`。后续仍应增加 layer-scale contract 和 source-node message mask / target-node aggregate mask 的独立数值对照。

### 3.7 把数值效果与硬件性能分开

仓库的 `hardware_speedup/message_passing_with_diff_precision.py` 会把张量实际转换成 `int8/int16/int32/float32`，再对 scatter-add message passing 做独立 benchmark。这和主量化模块的 QDQ / 逻辑整数推理是两个层次。

这个实验组织方式值得保留：metric sensitivity 只回答数值影响；只有接入真实 integer dtype 和对应 kernel 后，才能报告延迟、吞吐、能耗或显存收益。

## 4. 与当前 Degree-Aware 实现的边界

| 维度 | 当前项目 | MixQ-GNN | 决策 |
| --- | --- | --- | --- |
| 分配粒度 | 节点行 | 算子/张量边界 | 保留节点行分配 |
| 分配依据 | 源图全局入度 top `a%` | 学习 softmax 位宽系数 | 不迁移搜索思想 |
| 高/低精度 | FP32 / INT8 | 候选位宽通常为 2/4/8/16/32 | 保留 FP32 / INT8 |
| 训练方式 | inference-only PTQ/QDQ | 搜索训练 + QAT + freeze | 不引入重训练 |
| 量化形式 | symmetric INT8、zero-point=0、INT32 累加 | affine、可学习 scale/zero-point | 保持当前量化口径 |
| message mask | 按源节点决定消息精度 | 按组件选择统一位宽 | 保持当前 mask 语义 |
| aggregate mask | 按目标节点决定输出精度 | 按组件选择统一位宽 | 保持当前 mask 语义 |
| 目标 | 观察六任务 metric 波动 | 学习精度/成本折中配置 | 不改变实验目标 |

MixQ 仓库还包含一个 `MaskQuantMessagePassing`，引用 Degree-Quant：训练时随机保护高概率节点，只量化未保护节点；但 eval 时会量化所有消息和输出。它不是“推理时按源图全局 rank 长期保留高度数节点为 FP32”的实现，因此不能替代当前确定性的 `global_degree_mask()`。

## 5. 阅读中发现的实现风险

这些问题不影响我们借鉴架构，但不建议原样复制代码：

1. **搜索开销大。** relaxed forward 会计算所有候选位宽的 QDQ，再加权求和；它适合搜索，不等同于部署时真正的混合 kernel。
2. **固定推理不一定是硬件 INT kernel。** 主量化函数返回的是带整数数值的普通张量，没有显式转换到 `torch.int8`；`torch.nn.functional.linear` 能否获得真实整数加速不能由该路径本身证明。
3. **`MQLinear` 的 bias 路径可疑。** 模块创建了 bias 并同步多个候选模块，但 forward 调用 `linear(x, weight)` 时没有传 bias。当前项目不应复制这种实现。
4. **候选权重用 `.data` 共享。** `MQLinear` 通过覆盖 `.weight.data` 同步候选层，优化器状态和参数别名关系需要谨慎验证。
5. **calibration 默认只执行一次。** 上游允许后续用梯度更新 scale/zero-point；这与稳定的离线 PTQ calibration 语义不同。
6. **Degree-Quant 对照的 train/eval 语义不同。** `MaskQuantMessagePassing` 在训练时只量化未保护节点，eval 时量化全部节点，不能作为当前推理分组的参考实现。
7. **许可证文件缺失。** README 显示 MIT badge，但当前提交树中没有 `LICENSE`/`COPYING` 文件。可以借鉴工程细节；如需复制源码，应先向上游确认许可文本和适用范围。

## 6. 对当前项目的具体建议

按收益和改动风险排序：

1. **近期不改算法。** 继续以当前 `global_degree_mask()`、FP32/INT8 和四个 `a` 值为实验基线。
2. **下一轮 calibration 对照中引入状态对象。** 将 scale 的观测、冻结和复用从 QDQ 函数中拆出，分别比较当前 per-batch、固定 calibration、EMA 和 percentile；每次只改变一个变量。
3. **为量化边界增加可观测性。** 分别记录 input、message、aggregate、weight、layer output 的 scale、饱和率和 QDQ 误差，定位 metric 波动来源。
4. **继续完善 `prepare/calibrate/freeze/infer` 生命周期。** 当前实现已经缓存融合 INT8 权重；下一步应固定 activation calibration，并复用层间 scale contract，减少每层动态量化和反量化。
5. **增加 BOP/字节流量估算，但与实测性能分栏。** 理论成本只能用于比较配置；最终性能结论必须来自目标硬件 kernel benchmark。
6. **保持源码独立。** 如确需复用某段上游代码，先解决许可证问题，再以 `third_party` 或清晰的 attribution 方式引入；不要把上游模块静默混入当前核心实现。

## 7. 验证记录

- 上游仓库已完整克隆到项目根目录下的 `MixQ/`，checkout 为 `main@f101b11`。
- `MixQ/` 源码未修改。
- 尝试运行上游 `python -m unittest discover ./test`；当前项目的 Python 3.9 环境缺少上游声明的 `torch_operation_counter`，测试在 collection 阶段停止，未安装额外依赖，也未执行训练或 benchmark。
- 上游仓库约 48 MB，包含实验日志和少量数据文件。
