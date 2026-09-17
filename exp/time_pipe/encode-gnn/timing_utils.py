"""Offline/Eager encode-GNN 计时脚本共用函数。"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torchmetrics import AUROC, Accuracy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
while str(PROJECT_ROOT) in sys.path:
    sys.path.remove(str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import utils as project_utils
from gp.lightning.data_template import DataModule
from gp.utils.io import load_yaml
from gp.utils.utils import combine_dict, load_pretrained_state, merge_mod, set_random_seed
from models.model import (
    BinGraphAttLLMModel,
    BinGraphAttModel,
    BinGraphLLMModel,
    BinGraphModel,
    PyGRGCNEdge,
)
from task_constructor import UnifiedTaskConstructor


DEFAULT_BATCH_NUM = 10


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--override", type=str, help="YAML override, as in run_cdm.py")
    parser.add_argument("--checkpoint", type=str, help="Optional Lightning .ckpt or DeepSpeed directory")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--loader-index", type=int, default=0, help="Index for val/test loader lists")
    parser.add_argument("--batch-num", type=int, default=None, help="Measured batches; -1 means all")
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Config key/value overrides, matching run_cdm.py",
    )
    return parser.parse_args()


def load_params(args, load_texts):
    configs = [load_yaml(PROJECT_ROOT / "configs" / "default_config.yaml")]
    if args.override is not None:
        configs.append(load_yaml(args.override))
    values = merge_mod(combine_dict(*configs), args.opts)
    values["load_texts"] = load_texts
    values["llm_peft"] = False
    values["llm_quantization"] = False
    values["llm_trainable"] = False
    values["llm_adapter_path"] = None
    values["batch_num"] = args.batch_num if args.batch_num is not None else values.get(
        "batch_num", DEFAULT_BATCH_NUM
    )
    if values["batch_num"] == 0 or values["batch_num"] < -1:
        raise ValueError("batch_num must be a positive integer or -1")
    if args.warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")
    set_random_seed(values["seed"])
    torch.set_float32_matmul_precision("high")
    return SimpleNamespace(**values)


def resolve_device(name):
    if name == "auto":
        device, _ = project_utils.get_available_devices()
        return device
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
        torch.cuda.set_device(device)
    return device


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def normalize_task_names(task_names):
    if isinstance(task_names, str):
        return [name.strip() for name in task_names.split(",")]
    return task_names


def normalize_multipliers(params):
    if hasattr(params, "d_multiple"):
        multiple = params.d_multiple
        if isinstance(multiple, str):
            multiple = [float(value) for value in multiple.split(",")]
    else:
        multiple = [1]

    if hasattr(params, "d_min_ratio"):
        min_ratio = params.d_min_ratio
        if isinstance(min_ratio, str):
            min_ratio = [float(value) for value in min_ratio.split(",")]
    else:
        min_ratio = [1]
    return multiple, min_ratio


def build_task_data(params, encoder):
    """复用 run_cdm.py 的任务构造流程创建数据。

    Offline 模式传入 SentenceEncoder，数据集提供已经离线编码的数值特征；
    Eager 模式传入 None，数据集保留原始文本，等 batch 到达模型后再编码。
    """
    task_configs = load_yaml(PROJECT_ROOT / "configs" / "task_config.yaml")
    data_configs = load_yaml(PROJECT_ROOT / "configs" / "data_config.yaml")
    tasks = UnifiedTaskConstructor(
        normalize_task_names(params.task_names),
        params.load_texts,
        encoder,
        task_configs,
        data_configs,
        batch_size=params.batch_size,
        sample_size=params.train_sample_size,
    )
    val_indices, _ = tasks.construct_exp()
    multiple, min_ratio = normalize_multipliers(params)
    train_data = tasks.make_train_data(multiple, min_ratio, data_val_index=val_indices)
    datasets = tasks.make_full_dm_list(multiple, min_ratio, train_data)
    # Timing is intentionally single-process even on a multi-GPU server.
    data_module = DataModule(datasets, gpu_size=1, num_workers=params.num_workers)
    return tasks, data_module


def select_loader(data_module, split, loader_index):
    if split == "train":
        if loader_index != 0:
            raise ValueError("loader_index must be 0 for the train split")
        return data_module.train_dataloader()

    loaders = data_module.val_dataloader() if split == "val" else data_module.test_dataloader()
    if loader_index < 0 or loader_index >= len(loaders):
        raise IndexError(
            f"loader_index={loader_index} is out of range for {split}; "
            f"available loaders: 0..{len(loaders) - 1}"
        )
    return loaders[loader_index]


def _select_data_meta(data_module, split, loader_index):
    if split == "train":
        if loader_index != 0:
            raise ValueError("loader_index must be 0 for the train split")
        return data_module.datasets["train"]

    data_entries = data_module.datasets["val" if split == "val" else "test"]
    if not isinstance(data_entries, list):
        data_entries = [data_entries]
    if loader_index < 0 or loader_index >= len(data_entries):
        raise IndexError(
            f"loader_index={loader_index} is out of range for {split}; "
            f"available loaders: 0..{len(data_entries) - 1}"
        )
    return data_entries[loader_index]


def build_eval_metric(data_module, split, loader_index, device):
    """Build the evaluator configured for the selected run_cdm validation/test dataset."""
    data_meta = _select_data_meta(data_module, split, loader_index)
    metric_name = data_meta.metric
    if metric_name is None:
        return None

    if metric_name == "acc":
        evaluator = Accuracy(task="multiclass", num_classes=data_meta.classes)
    elif metric_name == "auc":
        evaluator = AUROC(task="binary")
    elif metric_name == "apr":
        evaluator = project_utils.MultiApr(num_labels=data_meta.classes)
    elif metric_name == "aucmulti":
        evaluator = project_utils.MultiAuc(num_labels=data_meta.classes)
    else:
        raise ValueError(f"Unsupported evaluation metric: {metric_name}")

    return SimpleNamespace(
        name=metric_name,
        state_name=data_meta.state_name,
        evaluator=evaluator.to(device),
        eval_func=data_meta.meta_data["eval_func"],
    )


def _build_gnn(params):
    out_dim = params.emb_dim + (params.rwpe if params.rwpe is not None else 0)
    gnn = PyGRGCNEdge(
        params.num_layers,
        5,
        out_dim,
        out_dim,
        drop_ratio=params.dropout,
        JK=params.JK,
    )
    return gnn, out_dim


def build_offline_model(params):
    """创建使用全局离线 embedding 的下游任务模型。

    返回的 BinGraphModel/BinGraphAttModel 只包含 llm_proj、GNN 和预测头，
    不包含文本编码器。SentenceEncoder 位于模型外，因此 Offline checkpoint
    只会保存这些下游任务模型权重。
    """
    gnn, out_dim = _build_gnn(params)
    model_class = BinGraphAttModel if params.JK == "none" else BinGraphModel
    return model_class(
        model=gnn,
        llm_name=params.llm_name,
        outdim=out_dim,
        task_dim=1,
        add_rwpe=params.rwpe,
        dropout=params.dropout,
    )


def build_eager_model(params):
    """创建在 batch 到达时编码原始文本的完整模型。

    与 Offline 模型不同，BinGraphLLMModel/BinGraphAttLLMModel 除了 llm_proj、
    GNN 和预测头，还持有 LLMModel。当前计时实验固定冻结文本编码器，并关闭
    PEFT 和量化。
    """
    gnn, out_dim = _build_gnn(params)
    model_class = BinGraphAttLLMModel if params.JK == "none" else BinGraphLLMModel
    return model_class(
        model=gnn,
        llm_name=params.llm_name,
        peft=False,
        quantization=False,
        train_text_encoder=False,
        adapter_path=None,
        max_length=params.llm_max_length,
        text_batch_size=params.llm_b_size,
        outdim=out_dim,
        task_dim=1,
        add_rwpe=params.rwpe,
        dropout=params.dropout,
    )


def load_model_checkpoint(model, checkpoint):
    """加载形状匹配的权重，并要求所有下游任务权重完整存在。

    Offline 模型没有 ``llm_model.*``，因此只加载 llm_proj、GNN 和预测头。
    Eager 模型在 checkpoint 包含 encoder 权重时也会加载它；如果传入 Offline
    checkpoint，则允许缺少 ``llm_model.*``，继续使用 ``llm_name`` 加载的基础
    encoder。无论哪种模式，缺少下游任务权重都会报错。
    """
    if checkpoint is None:
        return False
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if checkpoint_path.is_dir():
        state_dict = load_pretrained_state(str(checkpoint_path), deepspeed=True)
    else:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload)

    # Lightning wraps the task model under GraphPredLightning.model.
    wrapped = any(
        key.startswith("model.llm_proj.") or key.startswith("model.llm_model.")
        for key in state_dict
    )
    if wrapped:
        state_dict = {
            key[len("model."):]: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }

    target_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    missing_downstream = [
        key
        for key in target_state
        if key not in compatible_state and not key.startswith("llm_model.")
    ]
    if missing_downstream:
        preview = ", ".join(missing_downstream[:5])
        raise RuntimeError(
            "Checkpoint is missing downstream task-model weights "
            f"({preview}). Check llm_name, JK, emb_dim, num_layers, and rwpe."
        )
    model.load_state_dict(compatible_state, strict=False)
    return True


def move_batch(batch, device):
    return batch.to(device, non_blocking=device.type == "cuda")


def update_eval_metric(metric, output, batch):
    if metric is not None:
        metric.eval_func(metric.evaluator, output, batch)


def eval_metric_report(metric):
    if metric is None:
        return {
            "metric_key": None,
            "metric_name": None,
            "metric_value": None,
            "metric_state": None,
        }

    value = metric.evaluator.compute()
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError(f"Expected one scalar value from metric {metric.name}, got {value}")
    return {
        "metric_key": f"{metric.state_name}/{metric.name}",
        "metric_name": metric.name,
        "metric_value": value.detach().cpu().item(),
        "metric_state": metric.state_name,
    }


def make_report(mode, params, args, device, checkpoint_loaded, encode_seconds, gnn_seconds,
                measured_batches, measured_graphs, output_values, extra=None):
    measured_total = encode_seconds + gnn_seconds
    encode_ratio = encode_seconds / measured_total if measured_total else 0.0
    gnn_ratio = gnn_seconds / measured_total if measured_total else 0.0
    report = {
        "mode": mode,
        "task_names": normalize_task_names(params.task_names),
        "split": args.split,
        "loader_index": args.loader_index,
        "device": str(device),
        "graph_batch_size": params.batch_size,
        "text_batch_size": params.llm_b_size,
        "num_workers": params.num_workers,
        "warmup_batches": args.warmup_batches,
        "checkpoint": args.checkpoint,
        "checkpoint_loaded": checkpoint_loaded,
        "weights": "checkpoint" if checkpoint_loaded else "random task-model initialization",
        "requested_batch_num": params.batch_num,
        "measured_batches": measured_batches,
        "measured_graphs": measured_graphs,
        "output_values": output_values,
        "encode_seconds": encode_seconds,
        "gnn_seconds": gnn_seconds,
        "measured_total_seconds": measured_total,
        "encode_ratio": encode_ratio,
        "gnn_ratio": gnn_ratio,
        "encode_percent": 100.0 * encode_ratio,
        "gnn_percent": 100.0 * gnn_ratio,
        "gnn_seconds_per_batch": gnn_seconds / measured_batches,
    }
    if extra:
        report.update(extra)
    print(json.dumps(report, indent=2, ensure_ascii=False))
