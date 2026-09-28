import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data

from gp.nn.degree_quant import (
    global_degree_mask,
    high_degree_mask,
    mixed_int8_linear,
    mixed_int8_mean_aggregate,
    mixed_fake_quantize,
    mixed_precision_linear,
    symmetric_fake_quantize,
    symmetric_quantize_int8,
)
from models.model import PyGDegreeQuantRGCNEdge, PyGRGCNEdge


class DegreeMaskTest(unittest.TestCase):
    def test_global_rank_is_consistent_across_prompt_subgraphs(self):
        ranks = torch.tensor([0, 4, -1, 1, 7, -1])
        global_sizes = torch.full((6,), 10)
        real_nodes = ranks >= 0

        mask = global_degree_mask(ranks, global_sizes, real_nodes, 20)

        self.assertEqual(mask.tolist(), [True, False, True, True, False, True])

    def test_global_rank_uses_exact_ceil_cutoff(self):
        ranks = torch.arange(7)
        mask = global_degree_mask(
            ranks,
            torch.full((7,), 7),
            torch.ones(7, dtype=torch.bool),
            20,
        )
        self.assertEqual(mask.nonzero().flatten().tolist(), [0, 1])

    def test_prompt_nodes_always_remain_high_precision(self):
        mask = global_degree_mask(
            torch.tensor([4, -1, -1]),
            torch.full((3,), 10),
            torch.tensor([True, False, False]),
            0,
        )
        self.assertEqual(mask.tolist(), [False, True, True])

    def test_selects_exact_top_percent_per_graph(self):
        first = Data(
            x=torch.zeros(5, 1),
            edge_index=torch.tensor(
                [[0, 0, 1, 0, 1, 2, 0, 1, 2, 3], [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]]
            ),
        )
        second = Data(
            x=torch.zeros(3, 1),
            edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        )
        batch = Batch.from_data_list([first, second])

        mask = high_degree_mask(batch.edge_index, batch.num_nodes, 40, batch.batch)

        self.assertEqual(mask[:5].nonzero().flatten().tolist(), [3, 4])
        self.assertEqual(mask[5:].nonzero().flatten().tolist(), [1, 2])

    def test_ties_use_stable_node_order(self):
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        mask = high_degree_mask(edge_index, 4, 50)
        self.assertEqual(mask.nonzero().flatten().tolist(), [0, 1])


class FakeQuantizationTest(unittest.TestCase):
    def test_mixed_quantization_preserves_high_precision_rows(self):
        tensor = torch.tensor([[0.12345, -0.9987], [0.3333, 0.7777]])
        low_mask = torch.tensor([False, True])
        output = mixed_fake_quantize(tensor, low_mask, num_bits=8)

        torch.testing.assert_close(output[0], tensor[0], rtol=0, atol=0)
        self.assertFalse(torch.equal(output[1], tensor[1]))

    def test_zero_tensor_remains_zero(self):
        output = symmetric_fake_quantize(torch.zeros(3, 4), num_bits=8)
        self.assertTrue(torch.equal(output, torch.zeros(3, 4)))

    def test_mixed_linear_matches_reference_formulation(self):
        torch.manual_seed(11)
        inputs = torch.randn(5, 4)
        weight = torch.randn(4, 3)
        low_mask = torch.tensor([False, True, False, True, True])

        expected = inputs @ weight
        quantized_inputs = symmetric_fake_quantize(inputs[low_mask], num_bits=8)
        quantized_weight = symmetric_fake_quantize(
            weight, num_bits=8, channel_axis=1
        )
        expected[low_mask] = symmetric_fake_quantize(
            quantized_inputs @ quantized_weight, num_bits=8
        )

        actual = mixed_precision_linear(inputs, weight, low_mask, num_bits=8)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_int8_linear_matches_integer_reference(self):
        torch.manual_seed(19)
        inputs = torch.randn(7, 4)
        weight = torch.randn(4, 3)
        low_mask = torch.tensor([False, True, True, False, True, False, True])
        q_weight, weight_scale = symmetric_quantize_int8(weight, channel_axis=1)
        q_inputs, input_scale = symmetric_quantize_int8(inputs[low_mask])

        expected = torch.empty(7, 3)
        expected[~low_mask] = inputs[~low_mask] @ weight
        integer_output = q_inputs.to(torch.int32) @ q_weight.to(torch.int32)
        expected[low_mask] = symmetric_fake_quantize(
            integer_output.float() * input_scale * weight_scale,
            num_bits=8,
        )

        actual = mixed_int8_linear(
            inputs,
            weight,
            low_mask,
            quantized_weight=q_weight,
            weight_scale=weight_scale,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_int8_message_aggregate_matches_reference(self):
        messages = torch.tensor(
            [[0.13, 0.25], [0.47, -0.32], [0.91, 0.77], [-0.21, 0.63]]
        )
        targets = torch.tensor([0, 0, 1, 1])
        low_mask = torch.tensor([True, False, True, False])
        q_messages, scale = symmetric_quantize_int8(messages[low_mask])
        expected = torch.zeros(2, 2)
        expected.index_add_(0, targets[low_mask], q_messages.float() * scale)
        expected.index_add_(0, targets[~low_mask], messages[~low_mask])
        expected /= 2

        actual = mixed_int8_mean_aggregate(messages, targets, low_mask, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_path_calls_real_int8_kernel(self):
        inputs = torch.randn(40, 32, device="cuda")
        weight = torch.randn(32, 32, device="cuda")
        low_mask = torch.ones(40, dtype=torch.bool, device="cuda")
        with patch("torch._int_mm", wraps=torch._int_mm) as int_mm:
            output = mixed_int8_linear(inputs, weight, low_mask)
        self.assertEqual(output.shape, (40, 32))
        int_mm.assert_called_once()


class DegreeQuantModelTest(unittest.TestCase):
    def _graph(self):
        graph = Data(
            x=torch.randn(5, 4),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 1, 3], [1, 2, 3, 4, 0, 4, 1]]
            ),
            edge_attr=torch.randn(7, 4),
            edge_type=torch.tensor([0, 1, 0, 1, 0, 1, 0]),
        )
        graph.global_node_id = torch.arange(5)
        graph.global_node_degree = torch.tensor([3, 4, 2, 2, 1])
        graph.global_degree_rank = torch.tensor([1, 0, 2, 3, 4])
        graph.global_graph_num_nodes = torch.full((5,), 5)
        graph.real_node_mask = torch.ones(5, dtype=torch.bool)
        return graph

    def test_state_dict_is_checkpoint_compatible(self):
        base = PyGRGCNEdge(2, 2, 4, 4, drop_ratio=0.0, batch_norm=False)
        quantized = PyGDegreeQuantRGCNEdge(
            2, 2, 4, 4, drop_ratio=0.0, batch_norm=False
        )
        self.assertEqual(set(base.state_dict()), set(quantized.state_dict()))
        quantized.load_state_dict(base.state_dict(), strict=True)

    def test_all_high_precision_matches_original_model(self):
        torch.manual_seed(7)
        base = PyGRGCNEdge(2, 2, 4, 4, drop_ratio=0.0, batch_norm=False)
        quantized = PyGDegreeQuantRGCNEdge(
            2,
            2,
            4,
            4,
            drop_ratio=0.0,
            batch_norm=False,
            high_precision_percent=100,
        )
        quantized.load_state_dict(base.state_dict(), strict=True)
        base.eval()
        quantized.eval()
        graph = self._graph()

        torch.testing.assert_close(
            quantized(graph.clone()), base(graph.clone()), rtol=0, atol=0
        )

    def test_quantized_mode_rejects_training(self):
        model = PyGDegreeQuantRGCNEdge(
            2, 2, 4, 4, high_precision_percent=20
        )
        with self.assertRaisesRegex(RuntimeError, "inference-only"):
            model(self._graph())

    def test_quantized_mode_requires_global_metadata(self):
        model = PyGDegreeQuantRGCNEdge(
            2, 2, 4, 4, high_precision_percent=20
        )
        model.eval()
        graph = self._graph()
        del graph.global_degree_rank
        with self.assertRaisesRegex(ValueError, "Global degree metadata"):
            model(graph)


if __name__ == "__main__":
    unittest.main()
