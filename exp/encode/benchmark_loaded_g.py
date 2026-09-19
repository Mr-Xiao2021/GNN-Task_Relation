"""Compare model-forward timing phases for three raw-text representations."""

import argparse
import copy
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMING_UTILS_DIR = PROJECT_ROOT / "exp" / "time_pipe" / "encode-gnn"
GIB = 1024**3
RTOL = 1e-4
ATOL = 1e-5
TIMING_PHASES = ("total", "encode", "gnn_and_head")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed-width NumPy, object NumPy, and Python-hash paths "
            "using total, text-encoding, and downstream model-forward time"
        )
    )
    parser.add_argument("--override", type=str, help="YAML override, as in run_cdm.py")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--loader-index", type=int, default=0)
    parser.add_argument("--batch-num", type=int, default=1)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--fixed-width-mode",
        choices=("estimate", "auto", "force"),
        default="auto",
    )
    parser.add_argument(
        "--fixed-width-limit-gib",
        type=float,
        default=4.0,
        help="auto mode skips the fixed-width path above this memory guard",
    )
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Config key/value overrides, matching run_cdm.py",
    )
    args = parser.parse_args()
    if args.batch_num == 0 or args.batch_num < -1:
        parser.error("--batch-num must be positive or -1")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be non-negative")
    if args.fixed_width_limit_gib <= 0:
        parser.error("--fixed-width-limit-gib must be positive")
    return args


def load_timing_utils():
    os.chdir(PROJECT_ROOT)
    for path in (str(PROJECT_ROOT), str(TIMING_UTILS_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import timing_utils

    return timing_utils


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def as_text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def as_text_list(values):
    array = np.asarray(values).reshape(-1)
    return [as_text(value) for value in array.tolist()]


def prepare_numpy(graph):
    node_texts = np.asarray(graph.x).reshape(-1)
    edge_texts = np.asarray(graph.edge_attr).reshape(-1)
    text_inputs = np.concatenate((node_texts, edge_texts), axis=0)
    if text_inputs.size == 0:
        raise ValueError("Cannot benchmark an empty text batch")
    unique_texts, mapping = np.unique(text_inputs, return_inverse=True)
    return unique_texts, mapping, len(node_texts)


def available_memory_bytes():
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.virtual_memory().available)


def plan_fixed_width_inputs(node_texts, edge_texts, args):
    longest_node = max(node_texts, key=len, default="")
    longest_edge = max(edge_texts, key=len, default="")
    node_itemsize = np.asarray([longest_node]).itemsize
    edge_itemsize = np.asarray([longest_edge]).itemsize
    combined_itemsize = np.asarray([longest_node, longest_edge]).itemsize
    text_count = len(node_texts) + len(edge_texts)
    input_bytes = len(node_texts) * node_itemsize + len(edge_texts) * edge_itemsize
    combined_bytes = text_count * combined_itemsize
    inverse_bytes = text_count * np.dtype(np.int64).itemsize
    # Inputs plus concatenate/unique outputs and conservative NumPy work arrays.
    guard_bytes = input_bytes + 4 * combined_bytes + 4 * inverse_bytes
    configured_limit = int(args.fixed_width_limit_gib * GIB)
    available = available_memory_bytes()
    effective_limit = configured_limit
    if available is not None:
        effective_limit = min(effective_limit, int(available * 0.5))

    report = {
        "status": "estimated_only",
        "input_gib": input_bytes / GIB,
        "guard_gib": guard_bytes / GIB,
    }
    if args.fixed_width_mode == "estimate":
        report["skip_reason"] = "fixed_width_mode=estimate"
        return False, report
    if args.fixed_width_mode == "auto" and guard_bytes > effective_limit:
        report["skip_reason"] = "memory guard exceeds configured/available budget"
        return False, report

    report["status"] = "measured"
    return True, report


def materialize_fixed_width_inputs(node_texts, edge_texts, report):
    node_inputs = np.asarray(node_texts) if node_texts else np.asarray([], dtype=str)
    edge_inputs = np.asarray(edge_texts) if edge_texts else np.asarray([], dtype=str)
    for name, values in (("node", node_inputs), ("edge", edge_inputs)):
        if values.size and values.dtype.kind not in {"U", "S"}:
            raise TypeError(f"NumPy did not infer a string dtype for {name} texts")
    report.setdefault("node_dtype", str(node_inputs.dtype))
    report.setdefault("edge_dtype", str(edge_inputs.dtype))
    return node_inputs, edge_inputs


@contextmanager
def model_method_override(model, method_name, callback):
    had_override = method_name in model.__dict__
    previous = model.__dict__.get(method_name)
    setattr(model, method_name, callback)
    try:
        yield
    finally:
        if had_override:
            setattr(model, method_name, previous)
        else:
            delattr(model, method_name)


def isolated_batch(batch, node_values, edge_values):
    candidate = copy.copy(batch)
    candidate.x = node_values
    candidate.edge_attr = edge_values
    return candidate


def validate_text_mapping(prepared, node_texts, edge_texts):
    if prepared is None:
        raise RuntimeError("Model forward did not prepare graph texts")
    unique_texts, mapping, num_nodes = prepared
    expected_count = len(node_texts) + len(edge_texts)
    if num_nodes != len(node_texts) or len(mapping) != expected_count:
        raise RuntimeError("Text mapping has the wrong node or total length")

    position = 0
    for expected_texts in (node_texts, edge_texts):
        for expected in expected_texts:
            actual = as_text(unique_texts[int(mapping[position])])
            if actual != expected:
                raise RuntimeError(
                    f"Text mapping does not restore input at position {position}"
                )
            position += 1


def time_complete_forward(
    model, batch, path, device, node_texts, edge_texts, validate_mapping
):
    candidate = isolated_batch(batch, path["nodes"], path["edges"])
    prepared = None
    encode_finished = None
    prepare = path["prepare"]
    encode_graph_texts = model._encode_graph_texts

    def capture_preparation(graph):
        nonlocal prepared
        prepared = prepare(graph)
        return prepared

    def timed_encode(graph):
        nonlocal encode_finished
        encoded_graph = encode_graph_texts(graph)
        synchronize(device)
        encode_finished = time.perf_counter()
        return encoded_graph

    with model_method_override(
        model, "_prepare_graph_texts", capture_preparation
    ), model_method_override(model, "_encode_graph_texts", timed_encode):
        synchronize(device)
        started = time.perf_counter()
        output = model(candidate)
        synchronize(device)
        finished = time.perf_counter()
    if not torch.is_tensor(output):
        raise TypeError("Expected model(g) to return a Tensor")
    if encode_finished is None:
        raise RuntimeError("Model forward did not execute _encode_graph_texts")
    output_snapshot = output.detach().cpu()
    if validate_mapping:
        validate_text_mapping(prepared, node_texts, edge_texts)
    timings = {
        "total": finished - started,
        "encode": encode_finished - started,
        "gnn_and_head": finished - encode_finished,
    }
    del output, candidate
    return timings, output_snapshot


def compare_outputs(reference, candidate):
    if reference.shape != candidate.shape:
        return {"allclose": False, "max_abs_diff": None}
    difference = (reference - candidate).abs()
    return {
        "allclose": bool(torch.allclose(reference, candidate, rtol=RTOL, atol=ATOL)),
        "max_abs_diff": float(difference.max().item()) if difference.numel() else 0.0,
    }


def representative_timings(samples):
    ordered = sorted(samples, key=lambda sample: sample["total"])
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return {
        phase: (ordered[middle - 1][phase] + ordered[middle][phase]) / 2
        for phase in TIMING_PHASES
    }


def benchmark_batch(batch, model, device, args, batch_index):
    batch = batch.to(device, non_blocking=device.type == "cuda")
    node_texts = as_text_list(batch.x)
    edge_texts = as_text_list(batch.edge_attr)
    if not node_texts and not edge_texts:
        raise ValueError("Cannot benchmark an empty text batch")

    fixed_enabled, fixed_report = plan_fixed_width_inputs(node_texts, edge_texts, args)

    object_paths = {
        "object_numpy": prepare_numpy,
        "python_hash": model._prepare_graph_texts,
    }
    path_names = list(object_paths)
    if fixed_enabled:
        path_names.insert(0, "fixed_width_numpy")
    samples = {name: [] for name in path_names}
    outputs = {}
    for repeat_index in range(args.repeats):
        shift = (batch_index + repeat_index) % len(path_names)
        order = path_names[shift:] + path_names[:shift]
        for name in order:
            gc.collect()
            if name == "fixed_width_numpy":
                path_inputs = materialize_fixed_width_inputs(
                    node_texts, edge_texts, fixed_report
                )
                prepare = prepare_numpy
            else:
                path_inputs = (
                    np.asarray(node_texts, dtype=object),
                    np.asarray(edge_texts, dtype=object),
                )
                prepare = object_paths[name]
            path = {
                "nodes": path_inputs[0],
                "edges": path_inputs[1],
                "prepare": prepare,
            }
            timings, output = time_complete_forward(
                model,
                batch,
                path,
                device,
                node_texts,
                edge_texts,
                validate_mapping=name not in outputs,
            )
            samples[name].append(timings)
            outputs.setdefault(name, output)
            if outputs[name] is not output:
                del output
            del path, path_inputs
            gc.collect()

    medians = {
        name: representative_timings(path_samples)
        for name, path_samples in samples.items()
    }
    correctness = {
        "object_numpy_vs_python_hash": compare_outputs(
            outputs["object_numpy"], outputs["python_hash"]
        )
    }
    if "fixed_width_numpy" in outputs:
        correctness["fixed_width_vs_object_numpy"] = compare_outputs(
            outputs["fixed_width_numpy"], outputs["object_numpy"]
        )
    if not all(check["allclose"] for check in correctness.values()):
        raise RuntimeError(
            f"Forward outputs differ for batch {batch_index}: {correctness}"
        )

    return {
        "medians": medians,
        "correctness": correctness,
        "fixed_width": fixed_report,
    }


def warmup(loader, model, device, warmup_batches):
    if warmup_batches == 0:
        return 0
    completed = 0
    with torch.inference_mode():
        for batch in islice(loader, warmup_batches):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            node_texts = np.asarray(as_text_list(batch.x), dtype=object)
            edge_texts = np.asarray(as_text_list(batch.edge_attr), dtype=object)
            candidate = isolated_batch(batch, node_texts, edge_texts)
            model(candidate)
            del candidate
            completed += 1
    synchronize(device)
    return completed


def ratio(reference, candidate):
    if reference is None or candidate is None or candidate <= 0:
        return None
    return reference / candidate


def aggregate(batch_reports):
    measured_batches = len(batch_reports)
    names = ("fixed_width_numpy", "object_numpy", "python_hash")
    totals = {name: {} for name in names}
    for name in names:
        for phase in TIMING_PHASES:
            values = [
                report["medians"].get(name, {}).get(phase) for report in batch_reports
            ]
            totals[name][phase] = (
                sum(values) if all(value is not None for value in values) else None
            )

    measured = {
        name: phases["total"]
        for name, phases in totals.items()
        if phases["total"] is not None
    }
    fastest = min(measured, key=measured.get)
    fixed_seconds = totals["fixed_width_numpy"]["total"]
    object_seconds = totals["object_numpy"]["total"]
    python_seconds = totals["python_hash"]["total"]
    seconds_per_batch = {
        name: {
            phase: total / measured_batches if total is not None else None
            for phase, total in phases.items()
        }
        for name, phases in totals.items()
    }
    result = {
        "fastest_path": fastest,
        "object_numpy_faster_than_fixed_width": (
            object_seconds < fixed_seconds if fixed_seconds is not None else None
        ),
        "python_hash_faster_than_fixed_width": (
            python_seconds < fixed_seconds if fixed_seconds is not None else None
        ),
        "python_hash_faster_than_object_numpy": python_seconds < object_seconds,
        "object_numpy_speedup_vs_fixed_width": ratio(fixed_seconds, object_seconds),
        "python_hash_speedup_vs_fixed_width": ratio(fixed_seconds, python_seconds),
        "python_hash_speedup_vs_object_numpy": ratio(object_seconds, python_seconds),
    }
    outputs_match = all(
        check["allclose"]
        for report in batch_reports
        for check in report["correctness"].values()
    )
    fixed_reports = [report["fixed_width"] for report in batch_reports]
    fixed_statuses = {report["status"] for report in fixed_reports}
    if fixed_statuses == {"measured"}:
        fixed_status = "measured"
    elif fixed_statuses == {"estimated_only"}:
        fixed_status = "estimated_only"
    else:
        fixed_status = "partially_measured"
    fixed_width = {
        "status": fixed_status,
        "measured_batches": sum(
            report["status"] == "measured" for report in fixed_reports
        ),
        "max_input_gib": max(report["input_gib"] for report in fixed_reports),
        "max_estimated_peak_gib": max(report["guard_gib"] for report in fixed_reports),
    }
    materialized_reports = [
        report for report in fixed_reports if "node_dtype" in report
    ]
    if materialized_reports:
        fixed_width["inferred_node_dtypes"] = sorted(
            {report["node_dtype"] for report in materialized_reports}
        )
        fixed_width["inferred_edge_dtypes"] = sorted(
            {report["edge_dtype"] for report in materialized_reports}
        )
    skip_reasons = sorted(
        {report["skip_reason"] for report in fixed_reports if "skip_reason" in report}
    )
    if skip_reasons:
        fixed_width["skip_reasons"] = skip_reasons
    return seconds_per_batch, result, outputs_match, fixed_width


def main():
    args = parse_args()
    utils = load_timing_utils()
    params = utils.load_params(args, load_texts=True)  # force load_texts=True
    params.batch_num = args.batch_num
    params.num_workers = 0
    if args.batch_size is not None:
        params.batch_size = args.batch_size
    device = utils.resolve_device(args.device)

    _, data_module = utils.build_task_data(params, encoder=None)
    loader = utils.select_loader(data_module, args.split, args.loader_index)
    model = utils.build_eager_model(params).to(device).eval()
    warmup(loader, model, device, args.warmup_batches)
    # 选取batch数，如果是-1默认选取全部batch
    measured_loader = loader if args.batch_num == -1 else islice(loader, args.batch_num)
    reports = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(measured_loader):
            reports.append(benchmark_batch(batch, model, device, args, batch_index))
    if not reports:
        raise RuntimeError("No batches were measured")

    seconds_per_batch, result, outputs_match, fixed_width = aggregate(reports)
    report = {
        "scope": (
            "total=model(g); encode=text preparation + tokenize + Transformer + "
            "feature restoration; gnn_and_head=total - encode and includes projection, "
            "optional RWPE/attention, GNN, and prediction head; DataLoader, device "
            "transfer, and benchmark input construction excluded; CUDA synchronized at "
            "the forward start, encode boundary, and forward end"
        ),
        "config": {
            "task_names": utils.normalize_task_names(params.task_names),
            "split": args.split,
            "device": str(device),
            "llm_name": params.llm_name,
            "graph_batch_size": params.batch_size,
            "text_batch_size": params.llm_b_size,
            "max_text_length": params.llm_max_length,
            "measured_batches": len(reports),
            "repeats": args.repeats,
        },
        "seconds_per_batch": seconds_per_batch,
        "result": result,
        "outputs_match": outputs_match,
        "fixed_width": fixed_width,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
