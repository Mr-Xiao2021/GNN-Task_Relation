#!/usr/bin/env python3
"""Evaluate degree-aware FP32/INT8 inference on the six local checkpoints."""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Required by deterministic CUDA matrix multiplication. Set it before the first
# CUDA context is created so repeated evaluations use the same reduction path.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torchmetrics import AUROC, Accuracy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gp.lightning.data_template import DataModule
from gp.utils.utils import load_yaml, set_random_seed
from models.model import BinGraphModel, PyGDegreeQuantRGCNEdge
from task_constructor import UnifiedTaskConstructor


TASK_CHECKPOINTS = {
    "cora_node": "saved_exp/2026-09-17 17:25:07.738700/full_cdm/e5t7x7gs/checkpoints/epoch=74-step=150.ckpt",
    "cora_link": "saved_exp/2026-09-21 18:06:44.432551/full_cdm/8x52rwr9/checkpoints/epoch=48-step=3479.ckpt",
    "pubmed_node": "saved_exp/2026-09-17 16:01:00.408337/full_cdm/tn7immv7/checkpoints/epoch=60-step=61.ckpt",
    "pubmed_link": "saved_exp/2026-09-21 19:05:58.361032/full_cdm/b9rejg4r/checkpoints/epoch=10-step=6479.ckpt",
    "arxiv": "saved_exp/2026-09-21 20:44:10.956571/full_cdm/hk1d9kki/checkpoints/epoch=6-step=4977.ckpt",
    "WN18RR": "saved_exp/2026-09-22 12:45:33.439358/full_cdm/pyb5uw7f/checkpoints/epoch=32-step=22407.ckpt",
}


class CachedFeatureEncoder:
    """Minimal encoder identity used when all ST features already exist on disk."""

    llm_name = "ST"
    model = None

    def encode(self, _texts, _to_tensor=True):
        raise RuntimeError("A required ST feature cache is missing; regenerate it first")

    def get_model(self):
        raise RuntimeError("A required ST feature cache is missing; regenerate it first")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_CHECKPOINTS),
        default=list(TASK_CHECKPOINTS),
    )
    parser.add_argument(
        "--high-precision-percents",
        nargs="+",
        type=float,
        default=[20, 30, 40, 50],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    lightning_state = checkpoint["state_dict"]
    model_state = {
        key[len("model.") :]: value
        for key, value in lightning_state.items()
        if key.startswith("model.")
    }
    model.load_state_dict(model_state, strict=True)


def build_task(task_name: str, batch_size: int, num_workers: int):
    task_config = load_yaml(PROJECT_ROOT / "configs" / "task_config.yaml")
    data_config = load_yaml(PROJECT_ROOT / "configs" / "data_config.yaml")
    constructor = UnifiedTaskConstructor(
        [task_name],
        False,
        CachedFeatureEncoder(),
        task_config,
        data_config,
        root=str(PROJECT_ROOT / "cache_data"),
        batch_size=batch_size,
        eval_batch_size=batch_size,
        sample_size=-1,
    )
    constructor.construct_exp()
    primary_test = constructor.datasets["test"][0]
    data_module = DataModule(
        {"test": [primary_test]},
        gpu_size=1,
        num_workers=num_workers,
    )

    backbone = PyGDegreeQuantRGCNEdge(
        num_layers=6,
        num_rels=5,
        inp_dim=768,
        out_dim=768,
        drop_ratio=0.0,
        JK="last",
        high_precision_percent=None,
        quant_bits=8,
    )
    model = BinGraphModel(
        model=backbone,
        llm_name="ST",
        outdim=768,
        task_dim=1,
        add_rwpe=None,
        dropout=0.0,
    )
    checkpoint_path = PROJECT_ROOT / TASK_CHECKPOINTS[task_name]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    load_checkpoint(model, checkpoint_path)
    return model, data_module, primary_test, checkpoint_path


def build_metric(test_data):
    if test_data.metric == "acc":
        metric = Accuracy(task="multiclass", num_classes=test_data.classes)
    elif test_data.metric == "auc":
        metric = AUROC(task="binary")
    else:
        raise NotImplementedError(f"Unsupported metric: {test_data.metric}")
    return metric


def update_metric(metric, metric_name, output, batch):
    """Accumulate metrics on CPU to keep deterministic GPU inference enabled."""
    num_classes = int(batch.num_classes[0])
    logits = output.view(-1, num_classes)
    if metric_name == "acc":
        predictions = logits.detach().cpu()
        targets = batch.y.view(-1).to(dtype=torch.long, device="cpu")
    elif metric_name == "auc":
        predictions = torch.softmax(logits, dim=-1)[:, -1].detach().cpu()
        targets = batch.y[:, -1].reshape(-1).detach().cpu()
    else:
        raise NotImplementedError(f"Unsupported metric: {metric_name}")
    metric.update(predictions, targets)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate(
    model,
    data_module,
    test_data,
    device,
    high_precision_percent,
    seed,
    max_batches,
):
    set_random_seed(seed)
    model.model.set_high_precision_percent(high_precision_percent)
    model.eval()
    metric = build_metric(test_data)
    loader = data_module.test_dataloader()[0]

    batches = 0
    examples = 0
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            update_metric(metric, test_data.metric, output, batch)
            batches += 1
            examples += int(batch.num_graphs)
    synchronize(device)
    elapsed = time.perf_counter() - started
    value = float(metric.compute().detach().cpu())
    return value, batches, examples, elapsed


def write_results(output_dir: Path, metadata, rows):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"

    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps({**metadata, "results": rows}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    json_tmp.replace(json_path)

    if rows:
        csv_tmp = csv_path.with_suffix(".csv.tmp")
        with csv_tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        csv_tmp.replace(csv_path)
    return json_path, csv_path


def main():
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    for percent in args.high_precision_percents:
        if not 0 <= percent <= 100:
            raise ValueError("All high-precision percentages must be in [0, 100]")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or PROJECT_ROOT / "outputs" / "degree_quant" / timestamp
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tasks": args.tasks,
        "high_precision_percents": args.high_precision_percents,
        "quantization": {
            "kind": "inference-only symmetric QDQ simulation",
            "bits": 8,
            "node_partition": "per-graph top in-degree with stable node-index tie break",
            "activation_granularity": "per-tensor over low-precision rows",
            "weight_granularity": "per-output-channel",
        },
    }
    rows = []
    write_results(output_dir, metadata, rows)

    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    for task_name in args.tasks:
        print(json.dumps({"event": "task_start", "task": task_name}), flush=True)
        model, data_module, test_data, checkpoint_path = build_task(
            task_name, args.batch_size, args.num_workers
        )
        model.to(device)

        baseline, batches, examples, elapsed = evaluate(
            model,
            data_module,
            test_data,
            device,
            None,
            args.seed,
            args.max_batches,
        )
        baseline_row = {
            "task": task_name,
            "split": test_data.state_name,
            "metric_name": test_data.metric,
            "mode": "fp32",
            "high_precision_percent": 100.0,
            "low_precision_percent": 0.0,
            "metric": baseline,
            "delta_from_fp32": 0.0,
            "relative_delta_percent": 0.0,
            "batches": batches,
            "examples": examples,
            "elapsed_seconds": elapsed,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        }
        rows.append(baseline_row)
        write_results(output_dir, metadata, rows)
        print(json.dumps({"event": "result", **baseline_row}), flush=True)

        for percent in args.high_precision_percents:
            value, batches, examples, elapsed = evaluate(
                model,
                data_module,
                test_data,
                device,
                percent,
                args.seed,
                args.max_batches,
            )
            delta = value - baseline
            row = {
                "task": task_name,
                "split": test_data.state_name,
                "metric_name": test_data.metric,
                "mode": "degree_mixed_int8",
                "high_precision_percent": float(percent),
                "low_precision_percent": 100.0 - float(percent),
                "metric": value,
                "delta_from_fp32": delta,
                "relative_delta_percent": 100.0 * delta / abs(baseline) if baseline else None,
                "batches": batches,
                "examples": examples,
                "elapsed_seconds": elapsed,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            }
            rows.append(row)
            write_results(output_dir, metadata, rows)
            print(json.dumps({"event": "result", **row}), flush=True)

        model.cpu()
        del model, data_module, test_data
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_path, csv_path = write_results(output_dir, metadata, rows)
    print(
        json.dumps(
            {
                "event": "complete",
                "json": str(json_path),
                "csv": str(csv_path),
                "result_count": len(rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
