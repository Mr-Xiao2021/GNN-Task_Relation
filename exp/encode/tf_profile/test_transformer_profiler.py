"""Tests for aggregate timing on the existing offline encode path."""

from __future__ import annotations

import gc
import importlib.util
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_profiler import AggregateTransformerProfiler  # noqa: E402


class ToyBertSelfAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)

    def forward(self, hidden):
        return (self.query(hidden) + self.key(hidden) + self.value(hidden)) / 3.0


class ToyBertAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self = ToyBertSelfAttention(width)
        self.dense = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden):
        return self.norm(self.dense(self.self(hidden)) + hidden)


class ToyBertIntermediate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.dense = nn.Linear(width, width * 2)

    def forward(self, hidden):
        return torch.nn.functional.gelu(self.dense(hidden))


class ToyBertOutput(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.dense = nn.Linear(width * 2, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden, residual):
        return self.norm(self.dense(hidden) + residual)


class ToyBertLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = ToyBertAttention(width)
        self.intermediate = ToyBertIntermediate(width)
        self.output = ToyBertOutput(width)

    def forward(self, hidden):
        attended = self.attention(hidden)
        return self.output(self.intermediate(attended), attended)


class ToyBertModel(nn.Module):
    def __init__(self, width=8, layers=2):
        super().__init__()
        self.config = SimpleNamespace(model_type="bert", num_hidden_layers=layers)
        self.embeddings = nn.Embedding(32, width)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([ToyBertLayer(width) for _ in range(layers)])

    def forward(self, input_ids):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return hidden


class ToyDistilAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_lin = nn.Linear(width, width)
        self.k_lin = nn.Linear(width, width)
        self.v_lin = nn.Linear(width, width)
        self.out_lin = nn.Linear(width, width)

    def forward(self, hidden):
        mixed = self.q_lin(hidden) + self.k_lin(hidden) + self.v_lin(hidden)
        return self.out_lin(mixed)


class ToyDistilFFN(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lin1 = nn.Linear(width, width * 2)
        self.lin2 = nn.Linear(width * 2, width)

    def forward(self, hidden):
        return self.lin2(torch.nn.functional.gelu(self.lin1(hidden)))


class ToyDistilLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = ToyDistilAttention(width)
        self.ffn = ToyDistilFFN(width)

    def forward(self, hidden):
        hidden = hidden + self.attention(hidden)
        return hidden + self.ffn(hidden)


class ToyDistilBertModel(nn.Module):
    def __init__(self, width=8, layers=2):
        super().__init__()
        self.config = SimpleNamespace(model_type="distilbert", n_layers=layers)
        self.embeddings = nn.Embedding(32, width)
        self.transformer = nn.Module()
        self.transformer.layer = nn.ModuleList(
            [ToyDistilLayer(width) for _ in range(layers)]
        )

    def forward(self, input_ids):
        hidden = self.embeddings(input_ids)
        for layer in self.transformer.layer:
            hidden = layer(hidden)
        return hidden


class ToyLlamaAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden):
        mixed = self.q_proj(hidden) + self.k_proj(hidden) + self.v_proj(hidden)
        return self.o_proj(mixed)


class ToyLlamaMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(width, width * 2, bias=False)
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width, bias=False)

    def forward(self, hidden):
        gated = torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(gated)


class ToyLlamaLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = ToyLlamaAttention(width)
        self.mlp = ToyLlamaMLP(width)

    def forward(self, hidden):
        hidden = hidden + self.self_attn(hidden)
        return hidden + self.mlp(hidden)


class ToyLlamaBackbone(nn.Module):
    def __init__(self, width=8, layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, width)
        self.layers = nn.ModuleList([ToyLlamaLayer(width) for _ in range(layers)])

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class ToyLlamaForCausalLM(nn.Module):
    def __init__(self, width=8, layers=2):
        super().__init__()
        self.config = SimpleNamespace(model_type="llama", num_hidden_layers=layers)
        self.model = ToyLlamaBackbone(width, layers)
        self.lm_head = nn.Linear(width, 32, bias=False)

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))


def load_offline_timing_module():
    path = PROJECT_ROOT / "exp" / "time_pipe" / "encode-gnn" / "offline_encode_gnn_time.py"
    spec = importlib.util.spec_from_file_location("offline_encode_gnn_time_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AggregateTransformerProfilerTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("Aggregate Transformer profiling is CUDA-only")
        self.device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(7)

    def profile_once(self, model, input_ids):
        model.to(self.device).eval()
        input_ids = input_ids.to(self.device)
        reference = model(input_ids)
        profiler = AggregateTransformerProfiler(model, device=self.device).start()
        with torch.inference_mode():
            actual = model(input_ids)
        report = profiler.summary()
        profiler.close()
        torch.testing.assert_close(actual, reference)
        for module in model.modules():
            self.assertEqual(len(module._forward_hooks), 0)
            self.assertEqual(len(module._forward_pre_hooks), 0)
        return report

    def test_bert_aggregates_all_layers_without_per_layer_output(self):
        report = self.profile_once(
            ToyBertModel(layers=2),
            torch.randint(0, 32, (2, 5)),
        )
        self.assertEqual(report["calls"]["model_forward"], 1)
        self.assertEqual(report["calls"]["attention"], 2)
        self.assertEqual(report["calls"]["qkv_projection"], 6)
        self.assertEqual(report["calls"]["ffn"], 4)
        self.assertTrue(report["coverage_complete"])
        self.assertAlmostEqual(
            report["attention_seconds"],
            max(
                0.0,
                report["attention_inclusive_seconds"]
                - report["qkv_projection_seconds"],
            ),
        )
        self.assertNotIn("by_layer", report)
        self.assertNotIn("by_module", report)

    def test_distilbert_uses_one_ffn_target_per_layer(self):
        report = self.profile_once(
            ToyDistilBertModel(layers=2),
            torch.randint(0, 32, (1, 4)),
        )
        self.assertEqual(report["model_type"], "distilbert")
        self.assertEqual(report["calls"]["attention"], 2)
        self.assertEqual(report["calls"]["qkv_projection"], 6)
        self.assertEqual(report["calls"]["ffn"], 2)

    def test_llama_includes_full_model_forward_but_aggregates_layers(self):
        report = self.profile_once(
            ToyLlamaForCausalLM(layers=2),
            torch.randint(0, 32, (1, 4)),
        )
        self.assertEqual(report["model_type"], "llama")
        self.assertEqual(report["discovered_layers"], 2)
        self.assertEqual(report["calls"]["attention"], 2)
        self.assertEqual(report["calls"]["qkv_projection"], 6)
        self.assertEqual(report["calls"]["ffn"], 2)
        self.assertGreaterEqual(
            report["model_forward_seconds"],
            report["transformer_total_seconds"],
        )
        self.assertGreaterEqual(report["model_wrapper_or_head_seconds"], 0.0)

    def test_close_releases_model_references(self):
        model = ToyBertModel(layers=1).to(self.device)
        model_reference = weakref.ref(model)
        profiler = AggregateTransformerProfiler(model, device=self.device).start()
        profiler.close()
        self.assertIsNone(profiler.model)
        self.assertEqual(profiler.discovery["targets"], ())
        del model
        gc.collect()
        self.assertIsNone(model_reference())

    def test_offline_replay_still_calls_dataset_text2feature(self):
        offline = load_offline_timing_module()
        offline.utils = SimpleNamespace(synchronize=lambda _device: None)

        class FakeDataset:
            def __init__(self):
                self.received = []

            def text2feature(self, texts):
                self.received.append(texts)
                return torch.ones(len(texts), 2)

        class FakeProfiler:
            def __init__(self):
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1

            def summary(self):
                return {"transformer_total_seconds": 0.25}

        dataset = FakeDataset()
        profiler = FakeProfiler()
        with tempfile.TemporaryDirectory() as directory:
            text_path = Path(directory) / "texts.pkl"
            torch.save(["first", "second"], text_path)
            result = offline.time_global_offline_encode(
                [("toy", dataset, text_path)],
                torch.device("cpu"),
                transformer_profiler=profiler,
            )

        self.assertEqual(dataset.received, [["first", "second"]])
        self.assertEqual(profiler.reset_calls, 1)
        self.assertIn("encode_seconds", result)
        self.assertEqual(
            result["transformer_profile"]["transformer_total_seconds"],
            0.25,
        )

    def test_cuda_events_can_be_flushed_per_micro_batch(self):
        model = ToyBertModel(layers=1).to(self.device).eval()
        profiler = AggregateTransformerProfiler(model, device=self.device).start()
        with torch.inference_mode():
            output = model(torch.randint(0, 32, (1, 3), device=self.device))
            output.cpu()
        profiler.flush_completed()
        self.assertEqual(profiler._pending_cuda_events, [])
        report = profiler.summary()
        profiler.close()
        self.assertEqual(report["backend"], "cuda_event")
        self.assertTrue(report["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
