import unittest

import torch
from torch_geometric.data import Batch, Data

from gp.nn.degree_quant import (
    high_degree_mask,
    mixed_fake_quantize,
    mixed_precision_linear,
    symmetric_fake_quantize,
)
from models.model import PyGDegreeQuantRGCNEdge, PyGRGCNEdge


class DegreeMaskTest(unittest.TestCase):
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


class DegreeQuantModelTest(unittest.TestCase):
    def _graph(self):
        return Data(
            x=torch.randn(5, 4),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 1, 3], [1, 2, 3, 4, 0, 4, 1]]
            ),
            edge_attr=torch.randn(7, 4),
            edge_type=torch.tensor([0, 1, 0, 1, 0, 1, 0]),
        )

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


if __name__ == "__main__":
    unittest.main()
