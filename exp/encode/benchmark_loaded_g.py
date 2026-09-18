"""Compare NumPy and hash-based text preparation after a PyG batch is loaded."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMING_UTILS_DIR = PROJECT_ROOT / "exp" / "time_pipe" / "encode-gnn"
sys.path.insert(0, str(TIMING_UTILS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import timing_utils as project_timing


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare text preparation and encoder inference paths after a PyG "
            "batch has already been loaded"
        )
    )
    parser.add_argument("--override", type=str, help="YAML override, as in run_cdm.py")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--batch-num", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Config key/value overrides, matching run_cdm.py",
    )
    return parser.parse_args()


def _as_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected raw text, got {type(value).__name__}")


def _as_text_list(values):
    array = np.asarray(values)
    return [_as_text(value) for value in array.reshape(-1).tolist()]


def prepare_numpy(g):
    """Reproduce the original concatenate + sorted unique implementation."""
    node_texts = np.asarray(g.x).reshape(-1)
    edge_texts = np.asarray(g.edge_attr).reshape(-1)
    text_inputs = np.concatenate((node_texts, edge_texts), axis=0)
    unique_texts, text_mapping = np.unique(text_inputs, return_inverse=True)
    return unique_texts, text_mapping, text_inputs


def prepare_hash(g):
    """Deduplicate variable-length strings in first-occurrence order."""
    node_texts = _as_text_list(g.x)
    edge_texts = _as_text_list(g.edge_attr)
    unique_texts = []
    text_mapping = []
    text_to_index = {}

    for texts in (node_texts, edge_texts):
        for text in texts:
            text_index = text_to_index.get(text)
            if text_index is None:
                text_index = len(unique_texts)
                text_to_index[text] = text_index
                unique_texts.append(text)
            text_mapping.append(text_index)
    return unique_texts, text_mapping, node_texts, edge_texts


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def encode_unique_texts(model, texts, device):
    """Run the production tokenizer/encoder operations with separate timers."""
    text_batch_size = model.text_batch_size if model.text_batch_size > 0 else len(texts)
    if not texts:
        raise ValueError("Cannot encode an empty text batch")

    outputs = []
    tokenizer_seconds = 0.0
    host_to_device_seconds = 0.0
    llm_forward_seconds = 0.0

    for start in range(0, len(texts), text_batch_size):
        text_batch = texts[start:start + text_batch_size]

        started = time.perf_counter()
        if not isinstance(text_batch, list):
            text_batch = text_batch.tolist()
        token_batch = model.llm_model.tokenizer(
            text_batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=model.llm_model.max_length,
        )
        tokenizer_seconds += time.perf_counter() - started

        synchronize(device)
        started = time.perf_counter()
        token_batch = {key: value.to(device) for key, value in token_batch.items()}
        synchronize(device)
        host_to_device_seconds += time.perf_counter() - started

        started = time.perf_counter()
        output, _ = model.llm_model.encode(token_batch, pooling=True)
        synchronize(device)
        llm_forward_seconds += time.perf_counter() - started
        outputs.append(output)

    started = time.perf_counter()
    features = torch.cat(outputs, dim=0)
    synchronize(device)
    output_concat_seconds = time.perf_counter() - started
    return features, {
        "tokenizer_seconds": tokenizer_seconds,
        "host_to_device_seconds": host_to_device_seconds,
        "llm_forward_seconds": llm_forward_seconds,
        "output_concat_seconds": output_concat_seconds,
        "encode_total_seconds": (
            tokenizer_seconds
            + host_to_device_seconds
            + llm_forward_seconds
            + output_concat_seconds
        ),
        "encoder_micro_batches": (len(texts) + text_batch_size - 1) // text_batch_size,
    }


def mapping_restores_inputs(unique_texts, text_mapping, text_inputs):
    if len(text_mapping) != len(text_inputs):
        return False
    return all(
        _as_text(unique_texts[int(text_index)]) == _as_text(text)
        for text_index, text in zip(text_mapping, text_inputs)
    )


def compare_features(
    numpy_texts,
    numpy_features,
    hash_texts,
    hash_features,
    rtol,
    atol,
):
    numpy_texts = _as_text_list(numpy_texts)
    hash_texts = _as_text_list(hash_texts)
    hash_indices = {text: index for index, text in enumerate(hash_texts)}
    if set(numpy_texts) != set(hash_texts):
        return {"same_unique_texts": False, "embeddings_allclose": False}

    alignment = torch.as_tensor(
        [hash_indices[text] for text in numpy_texts],
        dtype=torch.long,
        device=hash_features.device,
    )
    aligned_hash_features = hash_features.index_select(0, alignment)
    difference = (numpy_features - aligned_hash_features).abs()
    return {
        "same_unique_texts": True,
        "embeddings_allclose": bool(
            torch.allclose(
                numpy_features,
                aligned_hash_features,
                rtol=rtol,
                atol=atol,
            )
        ),
        "embedding_max_abs_diff": float(difference.max().item()),
        "embedding_mean_abs_diff": float(difference.mean().item()),
    }


def warmup(loader, model, device, warmup_batches):
    if warmup_batches == 0:
        return
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= warmup_batches:
                break
            unique_texts, _, _, _ = prepare_hash(batch)
            sample_size = model.text_batch_size if model.text_batch_size > 0 else len(unique_texts)
            model._encode_texts(unique_texts[:sample_size])
    synchronize(device)


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def main():
    args = parse_args()
    params = project_timing.load_params(args, load_texts=True)
    params.batch_num = args.batch_num
    device = project_timing.resolve_device(args.device)

    # Dataset/model construction and DataLoader work are deliberately excluded.
    _, data_module = project_timing.build_task_data(params, encoder=None)
    loader = project_timing.select_loader(data_module, args.split, args.loader_index)
    model = project_timing.build_eager_model(params).to(device).eval()
    warmup(loader, model, device, args.warmup_batches)

    batch_reports = []
    totals = {
        "numpy_prepare_seconds": 0.0,
        "hash_prepare_seconds": 0.0,
        "numpy_encode_seconds": 0.0,
        "hash_encode_seconds": 0.0,
    }

    with torch.inference_mode():
        for batch_index, g in enumerate(loader):
            if args.batch_num != -1 and batch_index >= args.batch_num:
                break

            started = time.perf_counter()
            numpy_texts, numpy_mapping, numpy_inputs = prepare_numpy(g)
            numpy_prepare_seconds = time.perf_counter() - started

            started = time.perf_counter()
            hash_texts, hash_mapping, hash_node_inputs, hash_edge_inputs = prepare_hash(g)
            hash_prepare_seconds = time.perf_counter() - started
            hash_inputs = hash_node_inputs + hash_edge_inputs

            # Alternate execution order so multi-batch runs do not always favor one path.
            if batch_index % 2 == 0:
                numpy_features, numpy_encode = encode_unique_texts(
                    model, numpy_texts, device
                )
                hash_features, hash_encode = encode_unique_texts(model, hash_texts, device)
                encode_order = ["numpy", "hash"]
            else:
                hash_features, hash_encode = encode_unique_texts(model, hash_texts, device)
                numpy_features, numpy_encode = encode_unique_texts(
                    model, numpy_texts, device
                )
                encode_order = ["hash", "numpy"]

            correctness = compare_features(
                numpy_texts,
                numpy_features,
                hash_texts,
                hash_features,
                args.rtol,
                args.atol,
            )
            correctness.update(
                {
                    "numpy_mapping_restores_inputs": mapping_restores_inputs(
                        numpy_texts, numpy_mapping, numpy_inputs
                    ),
                    "hash_mapping_restores_inputs": mapping_restores_inputs(
                        hash_texts, hash_mapping, hash_inputs
                    ),
                }
            )

            node_array = np.asarray(g.x)
            edge_array = np.asarray(g.edge_attr)
            batch_report = {
                "batch_index": batch_index,
                "encode_order": encode_order,
                "node_text_count": int(node_array.size),
                "edge_text_count": int(edge_array.size),
                "node_dtype": str(node_array.dtype),
                "edge_dtype": str(edge_array.dtype),
                "node_array_shallow_bytes": int(node_array.nbytes),
                "edge_array_shallow_bytes": int(edge_array.nbytes),
                "unique_text_count": len(hash_texts),
                "max_text_characters": max(map(len, hash_inputs)),
                "numpy": {
                    "prepare_seconds": numpy_prepare_seconds,
                    **numpy_encode,
                },
                "hash": {
                    "prepare_seconds": hash_prepare_seconds,
                    **hash_encode,
                },
                "speedup_numpy_over_hash": {
                    "prepare": _ratio(numpy_prepare_seconds, hash_prepare_seconds),
                    "tokenizer": _ratio(
                        numpy_encode["tokenizer_seconds"],
                        hash_encode["tokenizer_seconds"],
                    ),
                    "llm_forward": _ratio(
                        numpy_encode["llm_forward_seconds"],
                        hash_encode["llm_forward_seconds"],
                    ),
                    "prepare_plus_encode": _ratio(
                        numpy_prepare_seconds + numpy_encode["encode_total_seconds"],
                        hash_prepare_seconds + hash_encode["encode_total_seconds"],
                    ),
                },
                "correctness": correctness,
            }
            batch_reports.append(batch_report)

            totals["numpy_prepare_seconds"] += numpy_prepare_seconds
            totals["hash_prepare_seconds"] += hash_prepare_seconds
            totals["numpy_encode_seconds"] += numpy_encode["encode_total_seconds"]
            totals["hash_encode_seconds"] += hash_encode["encode_total_seconds"]

            del numpy_features, hash_features

    if not batch_reports:
        raise RuntimeError("No batches were measured")

    totals["prepare_speedup_numpy_over_hash"] = _ratio(
        totals["numpy_prepare_seconds"], totals["hash_prepare_seconds"]
    )
    totals["prepare_plus_encode_speedup_numpy_over_hash"] = _ratio(
        totals["numpy_prepare_seconds"] + totals["numpy_encode_seconds"],
        totals["hash_prepare_seconds"] + totals["hash_encode_seconds"],
    )
    report = {
        "scope": "after DataLoader returned g; GNN and metric excluded",
        "task_names": project_timing.normalize_task_names(params.task_names),
        "split": args.split,
        "loader_index": args.loader_index,
        "device": str(device),
        "llm_name": params.llm_name,
        "llm_batch_size": params.llm_b_size,
        "llm_max_length": params.llm_max_length,
        "measured_batches": len(batch_reports),
        "totals": totals,
        "batches": batch_reports,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
