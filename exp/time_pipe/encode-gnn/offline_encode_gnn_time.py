"""Profile global offline text encoding and full-loader GNN inference."""

import argparse
import gc
import json
import math
import time
from itertools import islice

import torch


DEFAULT_REPEATS = 5
DEFAULT_WARMUP_BATCHES = 5
DEFAULT_ENCODE_WARMUP_BATCHES = 1

# Imported lazily in main so ``--help`` does not require the full project stack.
utils = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Profile one global offline text-encoding pass followed by inference "
            "over a selected graph loader"
        )
    )
    parser.add_argument("--override", type=str, help="YAML override, as in run_cdm.py")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Optional Lightning .ckpt or DeepSpeed directory",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--loader-index",
        type=int,
        default=0,
        help="Index for val/test loader lists",
    )
    parser.add_argument(
        "--batch-num",
        type=int,
        default=-1,
        help="Measured graph batches; -1 (the default) means the complete loader",
    )
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=DEFAULT_WARMUP_BATCHES,
        help="Untimed GNN warmup batches",
    )
    parser.add_argument(
        "--encode-warmup-batches",
        type=int,
        default=DEFAULT_ENCODE_WARMUP_BATCHES,
        help="Untimed text-encoder micro-batches",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Timed repetitions for both global encoding and graph inference",
    )
    parser.add_argument(
        "--sampling-hops",
        type=int,
        help=(
            "Runtime override for the selected graph dataset's sampling hops; "
            "use 2 for the HEAT-style experiment"
        ),
    )
    parser.add_argument(
        "--skip-metric",
        action="store_true",
        help="Skip the separate untimed metric pass",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "auto, cpu, cuda, or cuda:0; select a physical GPU with "
            "CUDA_VISIBLE_DEVICES"
        ),
    )
    parser.add_argument(
        "opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Config key/value overrides, matching run_cdm.py",
    )
    args = parser.parse_args()

    if args.batch_num == 0 or args.batch_num < -1:
        parser.error("--batch-num must be a positive integer or -1")
    if args.warmup_batches < 0:
        parser.error("--warmup-batches must be non-negative")
    if args.encode_warmup_batches < 0:
        parser.error("--encode-warmup-batches must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.sampling_hops is not None and args.sampling_hops <= 0:
        parser.error("--sampling-hops must be positive")
    return args


def percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values, include_samples=True):
    samples = [float(value) for value in values]
    q1 = percentile(samples, 0.25)
    median = percentile(samples, 0.5)
    q3 = percentile(samples, 0.75)
    report = {
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": min(samples),
        "max": max(samples),
    }
    if include_samples:
        report["samples"] = samples
    return report


def normalize_train_sample_size(params):
    value = params.train_sample_size
    if isinstance(value, bool):
        raise ValueError("train_sample_size must be an integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("train_sample_size must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("train_sample_size must be an integer")
    params.train_sample_size = normalized


def build_sentence_encoder(params, device):
    """Construct on the requested device despite SentenceEncoder auto-discovery."""
    project_utils = utils.project_utils
    original_get_devices = project_utils.get_available_devices

    def get_selected_device():
        gpu_ids = [device.index] if device.type == "cuda" else []
        return device, gpu_ids

    project_utils.get_available_devices = get_selected_device
    try:
        return project_utils.SentenceEncoder(
            params.llm_name,
            batch_size=params.llm_b_size,
            max_length=params.llm_max_length,
        )
    finally:
        project_utils.get_available_devices = original_get_devices


def disable_encoder_progress_bar():
    """Remove tqdm rendering from the measured SentenceEncoder path."""
    original_trange = utils.project_utils.trange

    def quiet_trange(*args, **kwargs):
        kwargs["disable"] = True
        return original_trange(*args, **kwargs)

    utils.project_utils.trange = quiet_trange


def load_cached_texts(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def iter_text_groups(value):
    """Yield the leaf string sequences consumed by OFAPygDataset.text2feature."""
    if isinstance(value, str):
        yield [value]
        return
    try:
        length = len(value)
    except TypeError as exc:
        raise TypeError(
            f"Unsupported cached text value: {type(value).__name__}"
        ) from exc
    if length == 0:
        return
    if isinstance(value[0], str):
        yield value
        return
    for item in value:
        yield from iter_text_groups(item)


def build_encode_jobs(tasks, text_batch_size, warmup_batches):
    jobs = []
    manifest = []
    warmup_limit = text_batch_size * warmup_batches
    warmup_texts = []

    for dataset_name, dataset in tasks.dataset.items():
        text_path = dataset.processed_paths[1]
        texts = load_cached_texts(text_path)
        entries = 0
        leaf_groups = 0
        encoder_micro_batches = 0
        for group in iter_text_groups(texts):
            group_size = len(group)
            entries += group_size
            leaf_groups += 1
            encoder_micro_batches += math.ceil(group_size / text_batch_size)
            remaining = warmup_limit - len(warmup_texts)
            if remaining > 0:
                warmup_texts.extend(str(text) for text in group[:remaining])

        jobs.append((dataset_name, dataset, text_path))
        manifest.append(
            {
                "dataset": dataset_name,
                "text_path": str(text_path),
                "text_entries": entries,
                "text_leaf_groups": leaf_groups,
                "text_encoder_micro_batches": encoder_micro_batches,
            }
        )
        del texts
        gc.collect()

    if not jobs:
        raise RuntimeError("No base datasets were constructed for offline encoding")
    if not any(item["text_entries"] for item in manifest):
        raise RuntimeError("The constructed datasets contain no cached text entries")
    return jobs, manifest, warmup_texts


def reset_peak_memory(device):
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats(device)
    return torch.cuda.memory_allocated(device)


def peak_memory_report(device, baseline):
    if device.type != "cuda":
        return {
            "gpu_peak_allocated_bytes": None,
            "gpu_peak_increment_bytes": None,
        }
    peak = torch.cuda.max_memory_allocated(device)
    return {
        "gpu_peak_allocated_bytes": int(peak),
        "gpu_peak_increment_bytes": int(max(0, peak - baseline)),
    }


def record_cuda_event(device):
    event = torch.cuda.Event(enable_timing=True)
    event.record(torch.cuda.current_stream(device))
    return event


def warmup_encoder(encoder, texts):
    if not texts:
        return 0
    utils.synchronize(encoder.device)
    with torch.inference_mode():
        embeddings = encoder.encode(texts)
    utils.synchronize(encoder.device)
    del embeddings
    return len(texts)


def time_global_offline_encode(jobs, device):
    """Replay every texts.pkl while excluding its disk-read time."""
    baseline = reset_peak_memory(device)
    elapsed = 0.0
    with torch.inference_mode():
        for _, dataset, text_path in jobs:
            texts = load_cached_texts(text_path)
            utils.synchronize(device)
            started = time.perf_counter()
            embeddings = dataset.text2feature(texts)
            utils.synchronize(device)
            elapsed += time.perf_counter() - started
            del embeddings, texts
            gc.collect()
    result = {"encode_seconds": elapsed}
    result.update(peak_memory_report(device, baseline))
    return result


def selected_graph_dataset(data_module, split, loader_index):
    if split == "train":
        if loader_index != 0:
            raise ValueError("loader_index must be 0 for the train split")
        return data_module.datasets["train"].data

    entries = data_module.datasets["val" if split == "val" else "test"]
    if not isinstance(entries, list):
        entries = [entries]
    if loader_index < 0 or loader_index >= len(entries):
        raise IndexError(
            f"loader_index={loader_index} is out of range for {split}; "
            f"available loaders: 0..{len(entries) - 1}"
        )
    return entries[loader_index].data


def iter_hop_datasets(dataset, seen=None):
    if seen is None:
        seen = set()
    identity = id(dataset)
    if identity in seen:
        return
    seen.add(identity)
    if hasattr(dataset, "hop"):
        yield dataset
    for child in getattr(dataset, "datas", []):
        yield from iter_hop_datasets(child, seen)


def configure_sampling_hops(data_module, args):
    selected = selected_graph_dataset(data_module, args.split, args.loader_index)
    datasets = list(iter_hop_datasets(selected))
    if args.sampling_hops is not None and not datasets:
        raise ValueError(
            "--sampling-hops was provided, but the selected dataset has no hop setting"
        )

    settings = []
    for dataset in datasets:
        original = int(dataset.hop)
        if args.sampling_hops is not None:
            dataset.hop = args.sampling_hops
        settings.append(
            {
                "dataset_type": type(dataset).__name__,
                "original_hops": original,
                "effective_hops": int(dataset.hop),
                "max_nodes_per_hop": getattr(dataset, "max_nodes_per_hop", None),
            }
        )
    return settings


def run_gnn_warmup(loader, model, device, warmup_batches):
    if warmup_batches == 0:
        return 0
    warmed = 0
    with torch.inference_mode():
        for batch in islice(loader, warmup_batches):
            model(utils.move_batch(batch, device))
            warmed += 1
    utils.synchronize(device)
    return warmed


class GnnCoreTimer:
    """Time only model.model (PyGRGCNEdge) without synchronizing each batch."""

    def __init__(self, module, device):
        self.device = device
        self.event_pairs = []
        self.cpu_seconds = 0.0
        self.current_start = None
        self.pre_handle = module.register_forward_pre_hook(self._before)
        self.post_handle = module.register_forward_hook(self._after)

    def _before(self, _module, _inputs):
        if self.device.type == "cuda":
            self.current_start = record_cuda_event(self.device)
        else:
            self.current_start = time.perf_counter()

    def _after(self, _module, _inputs, _output):
        if self.device.type == "cuda":
            end = record_cuda_event(self.device)
            self.event_pairs.append((self.current_start, end))
        else:
            self.cpu_seconds += time.perf_counter() - self.current_start
        self.current_start = None

    def reset(self):
        self.event_pairs = []
        self.cpu_seconds = 0.0
        self.current_start = None

    def seconds(self):
        if self.device.type == "cuda":
            return (
                sum(start.elapsed_time(end) for start, end in self.event_pairs) / 1000.0
            )
        return self.cpu_seconds

    def close(self):
        self.pre_handle.remove()
        self.post_handle.remove()


def time_model_batches(loader, model, core_timer, device, batch_num):
    baseline = reset_peak_memory(device)
    core_timer.reset()
    measured_batches = 0
    measured_graphs = 0
    measured_nodes = 0
    measured_edges = 0
    output_values = 0
    nodes_per_batch = []
    edges_per_batch = []
    loader_wait_seconds = 0.0
    downstream_event_pairs = []
    device_pipeline_event_pairs = []
    cpu_downstream_seconds = 0.0
    cpu_device_pipeline_seconds = 0.0

    utils.synchronize(device)
    wall_started = time.perf_counter()
    iterator = iter(loader)
    with torch.inference_mode():
        while batch_num == -1 or measured_batches < batch_num:
            fetch_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                loader_wait_seconds += time.perf_counter() - fetch_started
                break
            loader_wait_seconds += time.perf_counter() - fetch_started

            if device.type == "cuda":
                device_started = record_cuda_event(device)
                batch = utils.move_batch(batch, device)
                forward_started = record_cuda_event(device)
                output = model(batch)
                forward_finished = record_cuda_event(device)
                device_pipeline_event_pairs.append((device_started, forward_finished))
                downstream_event_pairs.append((forward_started, forward_finished))
            else:
                device_started = time.perf_counter()
                batch = utils.move_batch(batch, device)
                forward_started = time.perf_counter()
                output = model(batch)
                forward_finished = time.perf_counter()
                cpu_device_pipeline_seconds += forward_finished - device_started
                cpu_downstream_seconds += forward_finished - forward_started

            batch_nodes = int(batch.num_nodes)
            batch_edges = int(batch.num_edges)
            measured_batches += 1
            measured_graphs += int(batch.num_graphs)
            measured_nodes += batch_nodes
            measured_edges += batch_edges
            output_values += int(output.numel())
            nodes_per_batch.append(batch_nodes)
            edges_per_batch.append(batch_edges)
            del output, batch

    utils.synchronize(device)
    pipeline_wall_seconds = time.perf_counter() - wall_started
    if measured_batches == 0:
        raise RuntimeError(
            "No batches were measured; check split, batch_size, and drop_last"
        )

    if device.type == "cuda":
        downstream_device_seconds = (
            sum(start.elapsed_time(end) for start, end in downstream_event_pairs)
            / 1000.0
        )
        device_pipeline_seconds = (
            sum(start.elapsed_time(end) for start, end in device_pipeline_event_pairs)
            / 1000.0
        )
    else:
        downstream_device_seconds = cpu_downstream_seconds
        device_pipeline_seconds = cpu_device_pipeline_seconds

    gnn_core_seconds = core_timer.seconds()
    gnn_seconds = downstream_device_seconds
    other_seconds = max(0.0, pipeline_wall_seconds - gnn_seconds)
    result = {
        "pipeline_wall_seconds": pipeline_wall_seconds,
        "gnn_seconds": gnn_seconds,
        "gnn_core_seconds": gnn_core_seconds,
        "other_seconds": other_seconds,
        "downstream_device_seconds": downstream_device_seconds,
        "downstream_noncore_device_seconds": max(
            0.0, downstream_device_seconds - gnn_core_seconds
        ),
        "device_pipeline_seconds": device_pipeline_seconds,
        "device_transfer_seconds": max(
            0.0, device_pipeline_seconds - downstream_device_seconds
        ),
        "loader_wait_seconds": loader_wait_seconds,
        "measured_batches": measured_batches,
        "measured_graphs": measured_graphs,
        "measured_nodes": measured_nodes,
        "measured_edges": measured_edges,
        "output_values": output_values,
        "nodes_per_batch": summarize(nodes_per_batch, include_samples=False),
        "edges_per_batch": summarize(edges_per_batch, include_samples=False),
        "other_was_clamped": gnn_seconds > pipeline_wall_seconds,
    }
    result.update(peak_memory_report(device, baseline))
    return result


def evaluate_batches(loader, model, metric, device, batch_num):
    if metric is None:
        return
    measured = 0
    source = loader if batch_num == -1 else islice(loader, batch_num)
    with torch.inference_mode():
        for batch in source:
            batch = utils.move_batch(batch, device)
            output = model(batch)
            utils.update_eval_metric(metric, output, batch)
            measured += 1
    utils.synchronize(device)
    if measured == 0:
        raise RuntimeError("No batches were available for the metric pass")


def same_workload_size(runs):
    keys = (
        "measured_batches",
        "measured_graphs",
        "measured_nodes",
        "measured_edges",
        "output_values",
    )
    return all(all(run[key] == runs[0][key] for key in keys) for run in runs[1:])


def representative_count(runs, key):
    values = [run[key] for run in runs]
    if all(value == values[0] for value in values):
        return values[0]
    return summarize(values)


def make_report(
    params,
    args,
    device,
    checkpoint_loaded,
    encoder_load_seconds,
    model_load_seconds,
    data_preparation_seconds,
    text_manifest,
    warmed_texts,
    warmed_graph_batches,
    hop_settings,
    encode_runs,
    graph_runs,
    metric_report,
    model,
):
    encode_stats = summarize(run["encode_seconds"] for run in encode_runs)
    gnn_stats = summarize(run["gnn_seconds"] for run in graph_runs)
    gnn_core_stats = summarize(run["gnn_core_seconds"] for run in graph_runs)
    other_stats = summarize(run["other_seconds"] for run in graph_runs)
    pipeline_stats = summarize(run["pipeline_wall_seconds"] for run in graph_runs)
    downstream_stats = summarize(run["downstream_device_seconds"] for run in graph_runs)
    loader_stats = summarize(run["loader_wait_seconds"] for run in graph_runs)
    transfer_stats = summarize(run["device_transfer_seconds"] for run in graph_runs)

    encode_seconds = encode_stats["median"]
    gnn_seconds = gnn_stats["median"]
    other_seconds = other_stats["median"]
    profiled_total_seconds = encode_seconds + gnn_seconds + other_seconds
    if profiled_total_seconds:
        encode_ratio = encode_seconds / profiled_total_seconds
        gnn_ratio = gnn_seconds / profiled_total_seconds
        other_ratio = other_seconds / profiled_total_seconds
    else:
        encode_ratio = gnn_ratio = other_ratio = 0.0

    paired_profiled_total = [
        encode_run["encode_seconds"]
        + graph_run["gnn_seconds"]
        + graph_run["other_seconds"]
        for encode_run, graph_run in zip(encode_runs, graph_runs)
    ]
    run_reports = []
    for index, (encode_run, graph_run) in enumerate(zip(encode_runs, graph_runs), 1):
        run_total = (
            encode_run["encode_seconds"]
            + graph_run["gnn_seconds"]
            + graph_run["other_seconds"]
        )
        graph_payload = {
            key: value
            for key, value in graph_run.items()
            if key not in ("gpu_peak_allocated_bytes", "gpu_peak_increment_bytes")
        }
        run_reports.append(
            {
                "repeat": index,
                "encode_seconds": encode_run["encode_seconds"],
                **graph_payload,
                "encode_gpu_peak_allocated_bytes": encode_run[
                    "gpu_peak_allocated_bytes"
                ],
                "encode_gpu_peak_increment_bytes": encode_run[
                    "gpu_peak_increment_bytes"
                ],
                "graph_gpu_peak_allocated_bytes": graph_run["gpu_peak_allocated_bytes"],
                "graph_gpu_peak_increment_bytes": graph_run["gpu_peak_increment_bytes"],
                "profiled_total_seconds": run_total,
                "encode_percent": 100.0 * encode_run["encode_seconds"] / run_total,
                "gnn_percent": 100.0 * graph_run["gnn_seconds"] / run_total,
                "other_percent": 100.0 * graph_run["other_seconds"] / run_total,
            }
        )

    task_names = utils.normalize_task_names(params.task_names)
    full_dataset_sampling = params.train_sample_size <= 0
    full_test_loader = (
        args.split == "test"
        and args.loader_index == 0
        and params.batch_num == -1
        and full_dataset_sampling
    )
    effective_hops = [setting["effective_hops"] for setting in hop_settings]
    two_hop = bool(effective_hops) and all(hops == 2 for hops in effective_hops)
    workload_size_consistent = same_workload_size(graph_runs)
    stable_loader_workers = params.num_workers == 0
    warnings = []
    if not checkpoint_loaded:
        warnings.append(
            "No checkpoint was loaded; timing is valid, but the task metric "
            "uses random weights."
        )
    if params.batch_num != -1:
        warnings.append(
            "Global encoding is compared with only part of the graph loader; "
            "do not compare this ratio with HEAT."
        )
    if not full_dataset_sampling:
        warnings.append(
            "train_sample_size is positive, so the test loader uses replacement "
            "sampling rather than traversing the complete test dataset."
        )
    if not stable_loader_workers:
        warnings.append(
            "num_workers is nonzero; worker startup and prefetch behavior can "
            "increase the variance attributed to Other."
        )
    if not workload_size_consistent:
        warnings.append(
            "The aggregate batch/graph/node/edge workload changed across repeats; "
            "compare timing distributions with caution."
        )
    if not full_test_loader:
        warnings.append(
            "HEAT-style reporting should use --split test --loader-index 0 "
            "--batch-num -1."
        )
    if effective_hops and not two_hop:
        warnings.append(
            "The effective sampling depth is not 2-hop; pass --sampling-hops 2 "
            "for the HEAT-style run."
        )
    if any(
        setting["original_hops"] != setting["effective_hops"]
        for setting in hop_settings
    ):
        warnings.append(
            "Sampling hops were overridden only for this profiling process; "
            "the checkpoint may have been trained with another depth."
        )
    warnings.append(
        "The repository cache has no encoder fingerprint metadata, so matching "
        "the offline embedding cache to llm_name, max length, model revision, "
        "and pooling configuration must be verified manually."
    )

    total_text_entries = sum(item["text_entries"] for item in text_manifest)
    total_text_groups = sum(item["text_leaf_groups"] for item in text_manifest)
    total_text_micro_batches = sum(
        item["text_encoder_micro_batches"] for item in text_manifest
    )
    effective_node_limits = [
        setting["max_nodes_per_hop"]
        for setting in hop_settings
        if setting["max_nodes_per_hop"] is not None
    ]
    effective_node_limit = (
        effective_node_limits[0]
        if effective_node_limits
        and all(limit == effective_node_limits[0] for limit in effective_node_limits)
        else effective_node_limits
    )
    first_parameter = next(model.parameters())
    report = {
        "report_version": 3,
        "mode": "offline_heat_style",
        "strict_heat_reproduction": False,
        "heat_style_comparable": (
            full_test_loader
            and two_hop
            and checkpoint_loaded
            and workload_size_consistent
            and stable_loader_workers
        ),
        "embedding_cache_identity_verified": False,
        "task_names": task_names,
        "split": args.split,
        "loader_index": args.loader_index,
        "full_loader": params.batch_num == -1 and full_dataset_sampling,
        "full_dataset_sampling": full_dataset_sampling,
        "requested_batch_num": params.batch_num,
        "configured_sample_size": params.train_sample_size,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "torch_version": torch.__version__,
        "model_dtype": str(first_parameter.dtype),
        "llm_name": params.llm_name,
        "gnn_model": type(model.model).__name__,
        "emb_dim": params.emb_dim,
        "num_layers": params.num_layers,
        "JK": params.JK,
        "rwpe": params.rwpe,
        "dropout": params.dropout,
        "max_nodes_per_hop": effective_node_limit,
        "configured_max_nodes_per_hop": params.max_nodes_per_hop,
        "seed": params.seed,
        "graph_batch_size": params.batch_size,
        "text_batch_size": params.llm_b_size,
        "max_text_length": params.llm_max_length,
        "num_workers": params.num_workers,
        "repeats": args.repeats,
        "warmup_batches_requested": args.warmup_batches,
        "warmup_batches_executed": warmed_graph_batches,
        "encode_warmup_batches_requested": args.encode_warmup_batches,
        "encode_warmup_texts": warmed_texts,
        "checkpoint": args.checkpoint,
        "checkpoint_loaded": checkpoint_loaded,
        "weights": (
            "checkpoint" if checkpoint_loaded else "random task-model initialization"
        ),
        "sampling_hops": hop_settings,
        "text_manifest": text_manifest,
        "encoded_text_entries": total_text_entries,
        "text_leaf_groups": total_text_groups,
        "text_encoder_micro_batches": total_text_micro_batches,
        "measured_batches": representative_count(graph_runs, "measured_batches"),
        "measured_graphs": representative_count(graph_runs, "measured_graphs"),
        "measured_nodes": representative_count(graph_runs, "measured_nodes"),
        "measured_edges": representative_count(graph_runs, "measured_edges"),
        "output_values": representative_count(graph_runs, "output_values"),
        "workload_size_consistent_across_repeats": workload_size_consistent,
        "encode_seconds": encode_seconds,
        "gnn_seconds": gnn_seconds,
        "gnn_core_seconds": gnn_core_stats["median"],
        "other_seconds": other_seconds,
        "downstream_pipeline_seconds": pipeline_stats["median"],
        "downstream_device_seconds": downstream_stats["median"],
        "profiled_total_seconds": profiled_total_seconds,
        "encode_ratio": encode_ratio,
        "gnn_ratio": gnn_ratio,
        "other_ratio": other_ratio,
        "encode_percent": 100.0 * encode_ratio,
        "gnn_percent": 100.0 * gnn_ratio,
        "other_percent": 100.0 * other_ratio,
        "gnn_core_percent_of_profiled_total": (
            100.0 * gnn_core_stats["median"] / profiled_total_seconds
            if profiled_total_seconds
            else 0.0
        ),
        "gnn_seconds_per_batch": (gnn_seconds / graph_runs[0]["measured_batches"]),
        "gnn_core_seconds_per_batch": (
            gnn_core_stats["median"] / graph_runs[0]["measured_batches"]
        ),
        "timing_stats": {
            "encode_seconds": encode_stats,
            "gnn_seconds": gnn_stats,
            "gnn_core_seconds": gnn_core_stats,
            "other_seconds": other_stats,
            "downstream_pipeline_seconds": pipeline_stats,
            "downstream_device_seconds": downstream_stats,
            "loader_wait_seconds": loader_stats,
            "device_transfer_seconds": transfer_stats,
            "paired_profiled_total_seconds": summarize(paired_profiled_total),
        },
        "runs": run_reports,
        "encode_scope": (
            "all text entries in each base dataset texts.pkl: tokenizer + "
            "Transformer + pooling + embedding transfer to CPU; disk I/O and "
            "cache construction are excluded"
        ),
        "gnn_scope": (
            "full model(batch) device execution: input projection, optional "
            "RWPE/JK logic, PyGRGCNEdge, and prediction head"
        ),
        "gnn_core_scope": (
            "model.model (PyGRGCNEdge) device execution; diagnostic only"
        ),
        "other_scope": (
            "full selected-loader wall time minus full GNN device execution; "
            "includes sampling, collate, H2D, and host/runtime overhead on the "
            "critical path"
        ),
        "ratio_scope": (
            "additive phase profile: global encode plus complete selected-loader "
            "inference; not one continuous cache-build-to-prediction run"
        ),
        "representative_value": (
            "each phase uses its median over repeats; percentages use the sum "
            "of those phase medians"
        ),
        "embedding_replay_note": (
            "Timed embeddings are discarded. GNN inference consumes the existing "
            "offline cache built with the configured encoder, so profiled_total "
            "does not include writing or attaching the newly encoded features."
        ),
        **metric_report,
        "metric_scope": (
            "separate untimed loader pass"
            if not args.skip_metric
            else "skipped by --skip-metric"
        ),
        "encoder_model_load_seconds_excluded": encoder_load_seconds,
        "task_model_load_seconds_excluded": model_load_seconds,
        "data_preparation_seconds_excluded": data_preparation_seconds,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


def main():
    args = parse_args()

    global utils
    import timing_utils as timing_utils

    utils = timing_utils
    disable_encoder_progress_bar()
    params = utils.load_params(args, load_texts=False)
    normalize_train_sample_size(params)
    task_names = utils.normalize_task_names(params.task_names)
    if len(task_names) != 1:
        raise ValueError(
            "Offline HEAT-style profiling requires exactly one task name per run"
        )
    if params.llm_b_size <= 0:
        raise ValueError("llm_b_size must be positive for offline SentenceEncoder")
    device = utils.resolve_device(args.device)
    if device.type == "cuda" and device.index not in (None, 0):
        raise ValueError(
            "SentenceEncoder must use logical cuda:0. Select another physical GPU "
            "with CUDA_VISIBLE_DEVICES=<id> and pass --device cuda:0."
        )

    encoder_load_started = time.perf_counter()
    encoder = build_sentence_encoder(params, device)
    if device.type == "cuda":
        # Keep later allocations and implicit CUDA operations on the target device.
        torch.cuda.set_device(device)
    if encoder.device != device:
        encoder.device = device
        encoder.model.to(device)
    utils.synchronize(device)
    encoder_load_seconds = time.perf_counter() - encoder_load_started

    data_started = time.perf_counter()
    tasks, data_module = utils.build_task_data(params, encoder)
    data_preparation_seconds = time.perf_counter() - data_started
    hop_settings = configure_sampling_hops(data_module, args)

    jobs, text_manifest, warmup_texts = build_encode_jobs(
        tasks,
        params.llm_b_size,
        args.encode_warmup_batches,
    )
    warmed_texts = warmup_encoder(encoder, warmup_texts)
    encode_runs = [
        time_global_offline_encode(jobs, device) for _ in range(args.repeats)
    ]
    encoder.flush_model()
    gc.collect()

    model_load_started = time.perf_counter()
    model = utils.build_offline_model(params)
    checkpoint_loaded = utils.load_model_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.set_device(device)
    utils.synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_started

    utils.set_random_seed(params.seed)
    warmup_loader = utils.select_loader(data_module, args.split, args.loader_index)
    warmed_graph_batches = run_gnn_warmup(
        warmup_loader,
        model,
        device,
        args.warmup_batches,
    )
    del warmup_loader
    gc.collect()

    core_timer = GnnCoreTimer(model.model, device)
    graph_runs = []
    try:
        for _ in range(args.repeats):
            utils.set_random_seed(params.seed)
            loader = utils.select_loader(data_module, args.split, args.loader_index)
            graph_runs.append(
                time_model_batches(
                    loader,
                    model,
                    core_timer,
                    device,
                    params.batch_num,
                )
            )
    finally:
        core_timer.close()

    if args.skip_metric:
        metric_report = {
            "metric_key": None,
            "metric_name": None,
            "metric_value": None,
            "metric_state": None,
        }
    else:
        utils.set_random_seed(params.seed)
        metric = utils.build_eval_metric(
            data_module,
            args.split,
            args.loader_index,
            device,
        )
        metric_loader = utils.select_loader(
            data_module,
            args.split,
            args.loader_index,
        )
        evaluate_batches(
            metric_loader,
            model,
            metric,
            device,
            params.batch_num,
        )
        metric_report = utils.eval_metric_report(metric)

    make_report(
        params,
        args,
        device,
        checkpoint_loaded,
        encoder_load_seconds,
        model_load_seconds,
        data_preparation_seconds,
        text_manifest,
        warmed_texts,
        warmed_graph_batches,
        hop_settings,
        encode_runs,
        graph_runs,
        metric_report,
        model,
    )


if __name__ == "__main__":
    main()
