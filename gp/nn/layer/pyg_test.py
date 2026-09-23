"""Benchmark the FP32 and degree-aware branches of RGCNEdgeConv on CUDA."""

import argparse
import statistics

import torch

if __package__:
    from .pyg import RGCNEdgeConv
else:
    from pyg import RGCNEdgeConv


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare accuracy and CUDA execution time for the regular FP32 "
            "and degree-aware mixed-precision RGCNEdgeConv branches."
        )
    )
    parser.add_argument("--num-nodes", type=int, default=4096)
    parser.add_argument("--num-edges", type=int, default=32768)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--num-relations", type=int, default=5)
    parser.add_argument("--degree-threshold", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=5e-3)
    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args, parser):
    positive = (
        "num_nodes",
        "num_edges",
        "channels",
        "num_relations",
        "degree_threshold",
        "runs",
        "repeats",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error("--{} must be at least 1".format(name.replace("_", "-")))
    if args.num_nodes < 2:
        parser.error("--num-nodes must be at least 2")
    if args.num_edges < args.degree_threshold:
        parser.error("--num-edges must be at least --degree-threshold")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.rtol < 0 or args.atol < 0:
        parser.error("--rtol and --atol must be non-negative")


def random_inputs(args, device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    source = torch.randint(
        args.num_nodes,
        (args.num_edges,),
        generator=generator,
        device=device,
    )
    # Reserve the last node as an isolated low-degree node. Force node 0 to
    # reach the threshold so both precision partitions are always exercised.
    target = torch.randint(
        args.num_nodes - 1,
        (args.num_edges,),
        generator=generator,
        device=device,
    )
    target[: args.degree_threshold] = 0
    edge_index = torch.stack((source, target))
    edge_type = torch.randint(
        args.num_relations,
        (args.num_edges,),
        generator=generator,
        device=device,
    )
    x = torch.randn(
        args.num_nodes,
        args.channels,
        generator=generator,
        device=device,
    )
    xe = torch.randn(
        args.num_edges,
        args.channels,
        generator=generator,
        device=device,
    )
    return x, xe, edge_index, edge_type


def error_metrics(reference, candidate, rtol, atol):
    if reference.numel() == 0:
        return {
            "count": 0,
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
            "relative_l2": None,
            "allclose": True,
        }

    difference = candidate.float() - reference.float()
    reference = reference.float()
    denominator = torch.linalg.vector_norm(reference).clamp_min(
        torch.finfo(torch.float32).eps
    )
    return {
        "count": reference.size(0),
        "max_abs": difference.abs().max().item(),
        "mean_abs": difference.abs().mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "relative_l2": (torch.linalg.vector_norm(difference) / denominator).item(),
        "allclose": torch.allclose(candidate, reference, rtol=rtol, atol=atol),
    }


def timed_block(function, runs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / runs


def benchmark_pair(fp32_forward, dq_forward, warmup, runs, repeats):
    for _ in range(warmup):
        fp32_forward()
        dq_forward()
    torch.cuda.synchronize()

    samples = {"fp32": [], "dq": []}
    for repeat in range(repeats):
        order = (
            (("fp32", fp32_forward), ("dq", dq_forward))
            if repeat % 2 == 0
            else (("dq", dq_forward), ("fp32", fp32_forward))
        )
        for name, function in order:
            samples[name].append(timed_block(function, runs))
    return samples


def benchmark_single(function, warmup, runs, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    return [timed_block(function, runs) for _ in range(repeats)]


def timing_summary(samples):
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def print_metrics(name, metrics):
    print(
        "{}: count={} max_abs={:.6e} mean_abs={:.6e} "
        "rmse={:.6e} relative_l2={:.6e} allclose={}".format(
            name,
            metrics["count"],
            metrics["max_abs"],
            metrics["mean_abs"],
            metrics["rmse"],
            metrics["relative_l2"],
            metrics["allclose"],
        )
    )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA device")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")

    x, xe, edge_index, edge_type = random_inputs(args, device)
    layer = RGCNEdgeConv(
        args.channels,
        args.channels,
        args.num_relations,
    ).to(device)
    layer.eval()

    def build_mask():
        return torch.bincount(edge_index[1], minlength=args.num_nodes) >= (
            args.degree_threshold
        )

    high_degree_mask = build_mask()
    low_degree_mask = torch.logical_not(high_degree_mask)
    if not high_degree_mask.any() or not low_degree_mask.any():
        raise RuntimeError("The generated graph must contain both degree partitions")

    def fp32_forward():
        return layer(x, xe, edge_index, edge_type)

    def dq_forward():
        return layer(x, xe, edge_index, edge_type, high_degree_mask)

    with torch.inference_mode():
        fp32_output = fp32_forward()
        dq_output = dq_forward()
        if not torch.isfinite(fp32_output).all() or not torch.isfinite(dq_output).all():
            raise RuntimeError("Non-finite values found in benchmark output")

        overall_metrics = error_metrics(
            fp32_output,
            dq_output,
            args.rtol,
            args.atol,
        )
        high_metrics = error_metrics(
            fp32_output[high_degree_mask],
            dq_output[high_degree_mask],
            1e-5,
            1e-6,
        )
        low_metrics = error_metrics(
            fp32_output[low_degree_mask],
            dq_output[low_degree_mask],
            args.rtol,
            args.atol,
        )
        if not high_metrics["allclose"]:
            raise AssertionError("The FP32 high-degree partition changed unexpectedly")

        timing_samples = benchmark_pair(
            fp32_forward,
            dq_forward,
            args.warmup,
            args.runs,
            args.repeats,
        )
        mask_samples = benchmark_single(
            build_mask,
            args.warmup,
            args.runs,
            args.repeats,
        )

    fp32_timing = timing_summary(timing_samples["fp32"])
    dq_timing = timing_summary(timing_samples["dq"])
    mask_timing = timing_summary(mask_samples)
    layer_speedup = fp32_timing["median_ms"] / dq_timing["median_ms"]
    single_layer_speedup_with_mask = fp32_timing["median_ms"] / (
        dq_timing["median_ms"] + mask_timing["median_ms"]
    )
    saving_percent = (1.0 - dq_timing["median_ms"] / fp32_timing["median_ms"]) * 100
    degree = torch.bincount(edge_index[1], minlength=args.num_nodes)

    print("device: {}".format(torch.cuda.get_device_name(device)))
    print(
        "graph: nodes={} edges={} channels={} relations={} threshold={}".format(
            args.num_nodes,
            args.num_edges,
            args.channels,
            args.num_relations,
            args.degree_threshold,
        )
    )
    print(
        "partition: high={} low={} degree_min={} degree_mean={:.2f} "
        "degree_max={}".format(
            high_degree_mask.sum().item(),
            low_degree_mask.sum().item(),
            degree.min().item(),
            degree.float().mean().item(),
            degree.max().item(),
        )
    )
    print_metrics("accuracy/all", overall_metrics)
    print_metrics("accuracy/high", high_metrics)
    print_metrics("accuracy/low", low_metrics)
    print(
        "output_dtype: fp32={} dq={}".format(
            fp32_output.dtype,
            dq_output.dtype,
        )
    )
    print(
        "timing: fp32={:.3f} ms dq={:.3f} ms mask_once={:.3f} ms".format(
            fp32_timing["median_ms"],
            dq_timing["median_ms"],
            mask_timing["median_ms"],
        )
    )
    print(
        "speedup: layer_only={:.3f}x single_layer_with_mask={:.3f}x "
        "saving={:.2f}%".format(
            layer_speedup,
            single_layer_speedup_with_mask,
            saving_percent,
        )
    )
    print("note: speedup > 1.0 means the DQ branch is faster")


if __name__ == "__main__":
    main()
