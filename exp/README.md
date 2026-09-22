# Experiments

## 原权重说明

以下六个任务的权重均为本机训练生成的实验产物。训练时使用
`load_texts=False` 和 `llm_trainable=False`，因此 Lightning checkpoint 包含下游任务模型
（输入投影、GNN 和预测头），不包含可训练的 LLM 文本编码器。

| 任务 | 任务类型 | 训练轮数 | W&B run | 最佳 checkpoint（相对项目根目录） |
| --- | --- | ---: | --- | --- |
| `cora_node` | 节点分类 | 300 | `cora_node_gnn_nodrop_ST_ofa1_260917172507` | `saved_exp/2026-09-17 17:25:07.738700/full_cdm/e5t7x7gs/checkpoints/epoch=74-step=150.ckpt` |
| `cora_link` | 链路预测 | 400 | `cora_link_gnn_ST_ofa1_260921180644` | `saved_exp/2026-09-21 18:06:44.432551/full_cdm/8x52rwr9/checkpoints/epoch=48-step=3479.ckpt` |
| `pubmed_node` | 节点分类 | 300 | `pubmed_node_gnn_ST_ofa1_260917160100` | `saved_exp/2026-09-17 16:01:00.408337/full_cdm/tn7immv7/checkpoints/epoch=60-step=61.ckpt` |
| `pubmed_link` | 链路预测 | 300 | `pubmed_link_gnn_ST_ofa1_260921190558` | `saved_exp/2026-09-21 19:05:58.361032/full_cdm/b9rejg4r/checkpoints/epoch=10-step=6479.ckpt` |
| `arxiv` | 节点分类 | 200 | `arxiv_gnn_ST_ofa1_260921204410` | `saved_exp/2026-09-21 20:44:10.956571/full_cdm/hk1d9kki/checkpoints/epoch=6-step=4977.ckpt` |
| `WN18RR` | 知识图谱链路预测 | 200 | `WN18RR_gnn_ST_ofa1_260922124533` | `saved_exp/2026-09-22 12:45:33.439358/full_cdm/pyb5uw7f/checkpoints/epoch=32-step=22407.ckpt` |

每个 checkpoint 目录还包含 `last.ckpt`。推理时默认使用表中由验证指标选出的
`epoch=...-step=...` 最佳权重；仅在明确需要训练最后一轮状态时使用 `last.ckpt`。

`saved_exp/` 已被 `.gitignore` 排除，因此上述权重只保存在当前机器，不会随 Git
提交或推送。迁移仓库或清理磁盘前需单独备份这些文件。
