"""Aggregate Transformer timing for the offline SentenceEncoder path."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn


MODEL_FORWARD = "model_forward"
TRANSFORMER = "transformer"
ATTENTION = "attention"
QKV_PROJECTION = "qkv_projection"
FFN = "ffn"
PROFILE_GROUPS = (MODEL_FORWARD, TRANSFORMER, ATTENTION, QKV_PROJECTION, FFN)


@dataclass(frozen=True)
class _ProfileTarget:
    module: nn.Module
    group: str


def _append_module(modules, module):
    if isinstance(module, nn.Module) and all(module is not item for item in modules):
        modules.append(module)


def _discover_bert(structural_model):
    backbone = getattr(structural_model, "bert", structural_model)
    layers = list(getattr(getattr(backbone, "encoder", None), "layer", []))
    groups = {ATTENTION: [], QKV_PROJECTION: [], FFN: []}
    for layer in layers:
        attention = getattr(layer, "attention", None)
        attention_core = getattr(attention, "self", None)
        _append_module(groups[ATTENTION], attention)
        for name in ("query", "key", "value"):
            _append_module(groups[QKV_PROJECTION], getattr(attention_core, name, None))
        _append_module(groups[FFN], getattr(layer, "intermediate", None))
        _append_module(groups[FFN], getattr(layer, "output", None))
    return groups, len(layers), 2, backbone


def _discover_distilbert(structural_model):
    backbone = getattr(structural_model, "distilbert", structural_model)
    layers = list(getattr(getattr(backbone, "transformer", None), "layer", []))
    groups = {ATTENTION: [], QKV_PROJECTION: [], FFN: []}
    for layer in layers:
        attention = getattr(layer, "attention", None)
        _append_module(groups[ATTENTION], attention)
        for name in ("q_lin", "k_lin", "v_lin"):
            _append_module(groups[QKV_PROJECTION], getattr(attention, name, None))
        _append_module(groups[FFN], getattr(layer, "ffn", None))
    return groups, len(layers), 1, backbone


def _discover_llama(structural_model):
    candidate = getattr(structural_model, "model", None)
    backbone = candidate if hasattr(candidate, "layers") else structural_model
    layers = list(getattr(backbone, "layers", []))
    groups = {ATTENTION: [], QKV_PROJECTION: [], FFN: []}
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        _append_module(groups[ATTENTION], attention)
        for name in ("q_proj", "k_proj", "v_proj"):
            _append_module(groups[QKV_PROJECTION], getattr(attention, name, None))
        _append_module(groups[FFN], getattr(layer, "mlp", None))
    return groups, len(layers), 1, backbone


def discover_transformer_components(model):
    """Find only the aggregate groups used by the offline timing report."""

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if model_type == "bert":
        groups, layer_count, ffn_modules_per_layer, backbone = _discover_bert(
            model
        )
    elif model_type == "distilbert":
        groups, layer_count, ffn_modules_per_layer, backbone = _discover_distilbert(
            model
        )
    elif model_type == "llama":
        groups, layer_count, ffn_modules_per_layer, backbone = _discover_llama(
            model
        )
    else:
        raise ValueError(
            "Unsupported Transformer model_type for aggregate profiling: "
            f"{model_type!r}. Supported values: bert, distilbert, llama."
        )

    expected_layers = getattr(config, "num_hidden_layers", None)
    if expected_layers is None:
        expected_layers = getattr(config, "n_layers", None)
    if expected_layers is not None and int(expected_layers) != layer_count:
        raise RuntimeError(
            f"Expected {expected_layers} Transformer layers, discovered {layer_count}."
        )

    expected_group_sizes = {
        ATTENTION: layer_count,
        QKV_PROJECTION: layer_count * 3,
        FFN: layer_count * ffn_modules_per_layer,
    }
    actual_group_sizes = {name: len(modules) for name, modules in groups.items()}
    if layer_count == 0 or actual_group_sizes != expected_group_sizes:
        raise RuntimeError(
            "Incomplete Transformer component discovery: "
            f"expected={expected_group_sizes}, actual={actual_group_sizes}."
        )

    targets = [_ProfileTarget(model, MODEL_FORWARD)]
    separate_transformer_target = backbone is not model
    if separate_transformer_target:
        targets.append(_ProfileTarget(backbone, TRANSFORMER))
    for group in (ATTENTION, QKV_PROJECTION, FFN):
        targets.extend(_ProfileTarget(module, group) for module in groups[group])
    return {
        "model_type": model_type,
        "attention_implementation": getattr(config, "_attn_implementation", None),
        "discovered_layers": layer_count,
        "ffn_modules_per_layer": ffn_modules_per_layer,
        "separate_transformer_target": separate_transformer_target,
        "group_module_counts": {
            MODEL_FORWARD: 1,
            TRANSFORMER: 1,
            **actual_group_sizes,
        },
        "targets": targets,
    }


class AggregateTransformerProfiler:
    """Aggregate all layers into model, QKV, attention, and FFN timings."""

    def __init__(self, model, device=None):
        self.model = model
        self.discovery = discover_transformer_components(model)
        actual_device = next(model.parameters()).device
        self.device = torch.device(device) if device is not None else actual_device
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Aggregate Transformer profiling requires CUDA.")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if actual_device != self.device:
            raise ValueError(
                f"Profiler device {self.device} does not match model device "
                f"{actual_device}."
            )
        self._handles = []
        self._stacks = defaultdict(list)
        self._pending_cuda_events = []
        self._seconds = defaultdict(float)
        self._calls = defaultdict(int)

    def _before(self, target_index):
        started = torch.cuda.Event(enable_timing=True)
        started.record(torch.cuda.current_stream(self.device))
        self._stacks[target_index].append(started)

    def _after(self, target_index, group):
        if not self._stacks[target_index]:
            return
        started = self._stacks[target_index].pop()
        self._calls[group] += 1
        finished = torch.cuda.Event(enable_timing=True)
        finished.record(torch.cuda.current_stream(self.device))
        self._pending_cuda_events.append((group, started, finished))

    def start(self):
        if self._handles:
            raise RuntimeError("AggregateTransformerProfiler is already active.")
        try:
            for index, target in enumerate(self.discovery["targets"]):
                self._handles.append(
                    target.module.register_forward_pre_hook(
                        lambda _module, _inputs, target_index=index: self._before(
                            target_index
                        )
                    )
                )
                self._handles.append(
                    target.module.register_forward_hook(
                        lambda _module, _inputs, _output, target_index=index,
                        group=target.group: self._after(target_index, group)
                    )
                )
        except Exception:
            self.close()
            raise
        return self

    def flush_completed(self):
        """Fold completed CUDA Events into scalars after SentenceEncoder's D2H copy."""

        if not self._pending_cuda_events:
            return
        for group, started, finished in self._pending_cuda_events:
            self._seconds[group] += started.elapsed_time(finished) / 1000.0
        self._pending_cuda_events.clear()

    def reset(self):
        if self._pending_cuda_events:
            torch.cuda.synchronize(self.device)
            self.flush_completed()
        self._stacks.clear()
        self._pending_cuda_events.clear()
        self._seconds.clear()
        self._calls.clear()

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._stacks.clear()
        self.discovery["targets"] = ()
        self.model = None

    def summary(self):
        torch.cuda.synchronize(self.device)
        self.flush_completed()

        model_forward_seconds = self._seconds[MODEL_FORWARD]
        if self.discovery["separate_transformer_target"]:
            transformer_seconds = self._seconds[TRANSFORMER]
            transformer_calls = self._calls[TRANSFORMER]
        else:
            transformer_seconds = model_forward_seconds
            transformer_calls = self._calls[MODEL_FORWARD]
        attention_inclusive = self._seconds[ATTENTION]
        qkv_seconds = self._seconds[QKV_PROJECTION]
        ffn_seconds = self._seconds[FFN]
        attention_seconds = max(0.0, attention_inclusive - qkv_seconds)
        other_raw = transformer_seconds - attention_inclusive - ffn_seconds
        other_seconds = max(0.0, other_raw)
        model_calls = self._calls[MODEL_FORWARD]
        layers = self.discovery["discovered_layers"]
        ffn_per_layer = self.discovery["ffn_modules_per_layer"]
        expected_calls = {
            MODEL_FORWARD: model_calls,
            TRANSFORMER: model_calls,
            ATTENTION: model_calls * layers,
            QKV_PROJECTION: model_calls * layers * 3,
            FFN: model_calls * layers * ffn_per_layer,
        }
        actual_calls = {group: self._calls[group] for group in PROFILE_GROUPS}
        actual_calls[TRANSFORMER] = transformer_calls

        def percentage(value):
            return 100.0 * value / transformer_seconds if transformer_seconds else None

        return {
            "backend": "cuda_event",
            "device": str(self.device),
            "model_type": self.discovery["model_type"],
            "attention_implementation": self.discovery["attention_implementation"],
            "discovered_layers": layers,
            "model_forward_seconds": model_forward_seconds,
            "transformer_total_seconds": transformer_seconds,
            "model_wrapper_or_head_seconds": max(
                0.0, model_forward_seconds - transformer_seconds
            ),
            "qkv_projection_seconds": qkv_seconds,
            "attention_seconds": attention_seconds,
            "ffn_seconds": ffn_seconds,
            "other_transformer_seconds": other_seconds,
            "attention_inclusive_seconds": attention_inclusive,
            "partition_residual_seconds": other_raw,
            "percent_of_transformer": {
                "qkv_projection": percentage(qkv_seconds),
                "attention": percentage(attention_seconds),
                "ffn": percentage(ffn_seconds),
                "other": percentage(other_seconds),
            },
            "calls": actual_calls,
            "expected_calls": expected_calls,
            "coverage_complete": model_calls > 0 and actual_calls == expected_calls,
            "cross_architecture_component_comparison_supported": False,
            "semantics": {
                "model_forward": (
                    "Complete Hugging Face model call. For LlamaForCausalLM this includes "
                    "lm_head logits that the current LLMModel discards."
                ),
                "transformer_total": (
                    "Transformer backbone forward. It excludes the Llama lm_head."
                ),
                "qkv_projection": "Sum of Q, K, and V projection modules over all layers.",
                "attention": (
                    "Inclusive attention modules minus Q/K/V projections. This also includes "
                    "O projection, dropout, and architecture-specific residual/norm work; it "
                    "is not pure QK^T + softmax + AV time."
                ),
                "ffn": "Sum of complete FFN/MLP modules over all layers.",
                "other": (
                    "Transformer total minus inclusive attention and FFN; includes embeddings, "
                    "remaining norms, pooler, and backbone framework overhead."
                ),
                "overhead": (
                    "This is a separate diagnostic replay. Hooks and CUDA Events do not affect "
                    "the unprofiled encode_seconds baseline."
                ),
                "comparability": (
                    "Component boundaries differ across BERT, DistilBERT, and Llama. Compare "
                    "precision strategies within the same encoder architecture, not component "
                    "percentages across architectures."
                ),
            },
        }
