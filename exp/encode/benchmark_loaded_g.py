"""Compare NumPy and hash-based text preparation after a PyG batch is loaded."""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

"""
DataLoader -> batch(g)
    |-> fixed_width_numpy: reconstruct <Umax>, concatenate, unique
    |-> object_numpy: existing object array, concatenate, unique
    `-> object_hash: variable-length strings, hash deduplication

The two executable encode paths are aligned by text and checked for numerical
equivalence. fixed_width_numpy reuses object_numpy's encode timing only when
their sorted unique-text digests match.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMING_UTILS_DIR = PROJECT_ROOT / "exp" / "time_pipe" / "encode-gnn"
sys.path.insert(0, str(TIMING_UTILS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import timing_utils as utils


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
    parser.add_argument(
        "--batch-size",
        type=int,
        help="PyG graphs per DataLoader batch; overrides YAML/config opts",
    )
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--fixed-width-mode",
        choices=("estimate", "auto", "force"),
        default="auto",
        help="Estimate, safely run, or force the fixed-width NumPy baseline",
    )
    parser.add_argument(
        "--fixed-width-chars",
        type=int,
        help="Unicode width for the baseline; defaults to the longest text in the batch",
    )
    parser.add_argument(
        "--fixed-width-limit-gib",
        type=float,
        default=8.0,
        help="auto mode skips allocation above this estimated minimum peak memory",
    )
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
    """Run concatenate + sorted unique on the batch's existing array dtype."""
    node_texts = np.asarray(g.x).reshape(-1)
    edge_texts = np.asarray(g.edge_attr).reshape(-1)
    text_inputs = np.concatenate((node_texts, edge_texts), axis=0)
    unique_texts, text_mapping = np.unique(text_inputs, return_inverse=True)
    return unique_texts, text_mapping, text_inputs


def _text_digest(texts):
    digest = hashlib.sha256()
    for value in texts:
        encoded = _as_text(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little"))
        digest.update(encoded)
    return digest.hexdigest()


def run_fixed_width_baseline(g, mode, width_override, limit_gib):
    """Materialize the old <Umax> representation when the memory policy permits."""
    node_texts = _as_text_list(g.x)
    edge_texts = _as_text_list(g.edge_attr)
    text_count = len(node_texts) + len(edge_texts)
    max_text_chars = max(map(len, node_texts + edge_texts))
    width_chars = width_override if width_override is not None else max_text_chars
    if width_chars <= 0:
        raise ValueError("fixed_width_chars must be a positive integer")
    if width_chars < max_text_chars:
        raise ValueError(
            f"fixed_width_chars={width_chars} would truncate a {max_text_chars}-character text"
        )
    if limit_gib <= 0:
        raise ValueError("fixed_width_limit_gib must be positive")

    unicode_dtype = np.dtype(f"<U{width_chars}")
    combined_bytes = text_count * unicode_dtype.itemsize
    # node/edge arrays + concatenated input + worst-case unique output + inverse mapping.
    minimum_peak_bytes = 3 * combined_bytes + text_count * np.dtype(np.int64).itemsize
    limit_bytes = int(limit_gib * 1024 ** 3)
    report = {
        "status": "estimated_only",
        "unicode_dtype": str(unicode_dtype),
        "unicode_width_characters": width_chars,
        "text_count": text_count,
        "combined_array_bytes": combined_bytes,
        "combined_array_gib": combined_bytes / 1024 ** 3,
        "estimated_minimum_peak_bytes": minimum_peak_bytes,
        "estimated_minimum_peak_gib": minimum_peak_bytes / 1024 ** 3,
        "memory_limit_gib": limit_gib,
        "prepare_seconds": None,
        "unique_text_count": None,
        "mapping_restores_inputs": None,
    }

    if mode == "estimate":
        report["skip_reason"] = "fixed_width_mode=estimate"
        return report
    if mode == "auto" and minimum_peak_bytes > limit_bytes:
        report["skip_reason"] = "estimated minimum peak exceeds fixed_width_limit_gib"
        return report

    started = time.perf_counter()
    node_array = np.asarray(node_texts, dtype=unicode_dtype)
    edge_array = np.asarray(edge_texts, dtype=unicode_dtype)
    text_inputs = np.concatenate((node_array, edge_array), axis=0)
    unique_texts, text_mapping = np.unique(text_inputs, return_inverse=True)
    prepare_seconds = time.perf_counter() - started

    report.update(
        {
            "status": "measured",
            "prepare_seconds": prepare_seconds,
            "unique_text_count": len(unique_texts),
            "unique_text_digest": _text_digest(unique_texts),
            "mapping_restores_inputs": mapping_restores_inputs(
                unique_texts, text_mapping, text_inputs
            ),
        }
    )
    del node_array, edge_array, text_inputs, unique_texts, text_mapping
    gc.collect()
    return report


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

"""
python exp/encode/benchmark_loaded_g.py \
  --batch-num 1 \
  --batch-size 10 \
  --device cuda:0 \
  task_names wikics \
  llm_name ST \
  llm_b_size 100

"""
def main():
    args = parse_args()
    params = utils.load_params(args, load_texts=True)
    params.batch_num = args.batch_num
    if args.batch_size is not None: # override
        if args.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        params.batch_size = args.batch_size
    device = utils.resolve_device(args.device)

    # Dataset/model construction and DataLoader work are deliberately excluded.
    _, data_module = utils.build_task_data(params, encoder=None)
    loader = utils.select_loader(data_module, args.split, args.loader_index)
    model = utils.build_eager_model(params).to(device).eval()
    warmup(loader, model, device, args.warmup_batches)

    batch_reports = []
    totals = {
        "fixed_width_measured_batches": 0,
        "fixed_width_prepare_seconds": 0.0,
        "object_numpy_prepare_seconds": 0.0,
        "object_hash_prepare_seconds": 0.0,
        "object_numpy_encode_seconds": 0.0,
        "object_hash_encode_seconds": 0.0,
    }

    with torch.inference_mode():
        for batch_index, g in enumerate(loader):
            if args.batch_num != -1 and batch_index >= args.batch_num:
                break

            fixed_width = run_fixed_width_baseline(
                g,
                args.fixed_width_mode,
                args.fixed_width_chars,
                args.fixed_width_limit_gib,
            )

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
                encode_order = ["object_numpy", "object_hash"]
            else:
                hash_features, hash_encode = encode_unique_texts(model, hash_texts, device)
                numpy_features, numpy_encode = encode_unique_texts(
                    model, numpy_texts, device
                )
                encode_order = ["object_hash", "object_numpy"]

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
                    "object_numpy_mapping_restores_inputs": mapping_restores_inputs(
                        numpy_texts, numpy_mapping, numpy_inputs
                    ),
                    "object_hash_mapping_restores_inputs": mapping_restores_inputs(
                        hash_texts, hash_mapping, hash_inputs
                    ),
                }
            )
            numpy_unique_digest = _text_digest(numpy_texts)
            fixed_width["unique_texts_match_object_numpy"] = (
                fixed_width.get("unique_text_digest") == numpy_unique_digest
                if fixed_width["status"] == "measured"
                else None
            )
            if fixed_width["status"] == "measured":
                fixed_width["encode_reused_from_object_numpy"] = fixed_width[
                    "unique_texts_match_object_numpy"
                ]
                fixed_width["prepare_plus_encode_seconds"] = (
                    fixed_width["prepare_seconds"]
                    + numpy_encode["encode_total_seconds"]
                    if fixed_width["unique_texts_match_object_numpy"]
                    else None
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
                "fixed_width_numpy": fixed_width,
                "object_numpy": {
                    "prepare_seconds": numpy_prepare_seconds,
                    **numpy_encode,
                },
                "object_hash": {
                    "prepare_seconds": hash_prepare_seconds,
                    **hash_encode,
                },
                "speedup": {
                    "object_numpy_over_object_hash_prepare": _ratio(
                        numpy_prepare_seconds, hash_prepare_seconds
                    ),
                    "tokenizer": _ratio(
                        numpy_encode["tokenizer_seconds"],
                        hash_encode["tokenizer_seconds"],
                    ),
                    "llm_forward": _ratio(
                        numpy_encode["llm_forward_seconds"],
                        hash_encode["llm_forward_seconds"],
                    ),
                    "object_numpy_over_object_hash_prepare_plus_encode": _ratio(
                        numpy_prepare_seconds + numpy_encode["encode_total_seconds"],
                        hash_prepare_seconds + hash_encode["encode_total_seconds"],
                    ),
                    "fixed_width_over_object_numpy_prepare": _ratio(
                        fixed_width["prepare_seconds"], numpy_prepare_seconds
                    ) if fixed_width["status"] == "measured" else None,
                    "fixed_width_over_object_hash_prepare_plus_encode": _ratio(
                        fixed_width["prepare_plus_encode_seconds"],
                        hash_prepare_seconds + hash_encode["encode_total_seconds"],
                    ) if fixed_width.get("prepare_plus_encode_seconds") is not None else None,
                },
                "correctness": correctness,
            }
            batch_reports.append(batch_report)

            if fixed_width["status"] == "measured":
                totals["fixed_width_measured_batches"] += 1
                totals["fixed_width_prepare_seconds"] += fixed_width["prepare_seconds"]
            totals["object_numpy_prepare_seconds"] += numpy_prepare_seconds
            totals["object_hash_prepare_seconds"] += hash_prepare_seconds
            totals["object_numpy_encode_seconds"] += numpy_encode["encode_total_seconds"]
            totals["object_hash_encode_seconds"] += hash_encode["encode_total_seconds"]

            del numpy_features, hash_features

    if not batch_reports:
        raise RuntimeError("No batches were measured")

    totals["object_numpy_over_object_hash_prepare_speedup"] = _ratio(
        totals["object_numpy_prepare_seconds"],
        totals["object_hash_prepare_seconds"],
    )
    totals["object_numpy_over_object_hash_prepare_plus_encode_speedup"] = _ratio(
        totals["object_numpy_prepare_seconds"]
        + totals["object_numpy_encode_seconds"],
        totals["object_hash_prepare_seconds"]
        + totals["object_hash_encode_seconds"],
    )
    if totals["fixed_width_measured_batches"] == len(batch_reports):
        totals["fixed_width_over_object_numpy_prepare_speedup"] = _ratio(
            totals["fixed_width_prepare_seconds"],
            totals["object_numpy_prepare_seconds"],
        )
    else:
        totals["fixed_width_over_object_numpy_prepare_speedup"] = None
    report = {
        "scope": "after DataLoader returned g; GNN and metric excluded",
        "task_names": utils.normalize_task_names(params.task_names),
        "split": args.split,
        "loader_index": args.loader_index,
        "device": str(device),
        "llm_name": params.llm_name,
        "batch_size": params.batch_size,
        "llm_batch_size": params.llm_b_size,
        "llm_max_length": params.llm_max_length,
        "fixed_width_mode": args.fixed_width_mode,
        "fixed_width_chars": args.fixed_width_chars,
        "fixed_width_limit_gib": args.fixed_width_limit_gib,
        "measured_batches": len(batch_reports),
        "totals": totals,
        "batches": batch_reports,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
