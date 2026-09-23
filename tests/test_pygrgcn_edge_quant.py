import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from models.model import PyGRGCNEdge


class PyGRGCNEdgeDegreeQuantTest(unittest.TestCase):
    def _model(self, **kwargs):
        return PyGRGCNEdge(
            num_layers=1,
            num_rels=1,
            inp_dim=2,
            out_dim=2,
            batch_norm=False,
            **kwargs,
        )

    def _graph(self, device="cpu"):
        return SimpleNamespace(
            x=torch.tensor(
                [
                    [0.12345, -0.3333],
                    [1.0003, -0.2002],
                    [0.7001, -0.7002],
                ],
                device=device,
            ),
            edge_index=torch.tensor(
                [
                    [0, 0, 1],
                    [1, 2, 2],
                ],
                device=device,
            ),
            edge_type=torch.zeros(3, dtype=torch.long, device=device),
            edge_attr=torch.zeros(3, 2, device=device),
        )

    def test_partition_uses_target_in_degree_and_keeps_boundary_full_precision(
        self,
    ):
        model = self._model(dq_enabled=True, dq_degree_threshold=2)
        edge_index = torch.tensor(
            [
                [0, 0, 1, 2],
                [1, 2, 2, 2],
            ]
        )

        mask = model._high_degree_mask(edge_index, num_nodes=3)

        torch.testing.assert_close(mask, torch.tensor([False, False, True]))

    def test_disabled_quantization_preserves_forward_and_state_dict(self):
        baseline = self._model().eval()
        disabled_dq = self._model(dq_enabled=False).eval()
        disabled_dq.load_state_dict(baseline.state_dict())
        graph = self._graph()

        baseline_output = baseline(graph)
        with patch.object(
            disabled_dq,
            "_high_degree_mask",
            wraps=disabled_dq._high_degree_mask,
        ) as select_nodes:
            disabled_dq_output = disabled_dq(graph)

        torch.testing.assert_close(
            disabled_dq_output,
            baseline_output,
            rtol=0,
            atol=0,
        )
        self.assertEqual(
            set(disabled_dq.state_dict()),
            set(baseline.state_dict()),
        )
        select_nodes.assert_not_called()

    def test_quantization_is_disabled_during_training(self):
        baseline = self._model()
        training_dq = self._model(
            dq_enabled=True,
            dq_degree_threshold=2,
        )
        training_dq.load_state_dict(baseline.state_dict())

        with patch.object(
            training_dq,
            "_high_degree_mask",
            wraps=training_dq._high_degree_mask,
        ) as select_nodes:
            torch.testing.assert_close(
                training_dq(self._graph()),
                baseline(self._graph()),
                rtol=0,
                atol=0,
            )
        select_nodes.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for DQ")
    def test_eval_uses_fp32_for_high_degree_and_fp16_for_low_degree(self):
        baseline = self._model().cuda().eval()
        mixed = (
            self._model(
                dq_enabled=True,
                dq_degree_threshold=2,
            )
            .cuda()
            .eval()
        )
        with torch.no_grad():
            baseline.conv[0].weight.copy_(torch.eye(2, device="cuda").unsqueeze(0))
            baseline.conv[0].root.copy_(torch.eye(2, device="cuda"))
            baseline.conv[0].bias.zero_()
        mixed.load_state_dict(baseline.state_dict())
        message_dtypes = []
        original_message = mixed.conv[0].message

        def record_message_dtype(x_j, xe):
            message_dtypes.append((x_j.dtype, xe.dtype))
            return original_message(x_j, xe)

        mixed.conv[0].message = record_message_dtype
        graph = self._graph(device="cuda")

        baseline_output = baseline(graph)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mixed_output = mixed(graph)

        self.assertEqual(mixed_output.dtype, torch.float32)
        torch.testing.assert_close(
            mixed_output[2],
            baseline_output[2],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(mixed_output[:2], baseline_output[:2]))
        self.assertIn((torch.float32, torch.float32), message_dtypes)
        self.assertIn((torch.float16, torch.float16), message_dtypes)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for DQ")
    def test_eval_handles_empty_precision_partitions(self):
        baseline = self._model().cuda().eval()
        all_high = (
            self._model(
                dq_enabled=True,
                dq_degree_threshold=1,
            )
            .cuda()
            .eval()
        )
        all_low = (
            self._model(
                dq_enabled=True,
                dq_degree_threshold=2,
            )
            .cuda()
            .eval()
        )
        all_high.load_state_dict(baseline.state_dict())
        all_low.load_state_dict(baseline.state_dict())
        graph = SimpleNamespace(
            x=torch.tensor([[0.2, -0.3], [0.7, -0.8]], device="cuda"),
            edge_index=torch.tensor([[0, 1], [1, 0]], device="cuda"),
            edge_type=torch.zeros(2, dtype=torch.long, device="cuda"),
            edge_attr=torch.zeros(2, 2, device="cuda"),
        )

        torch.testing.assert_close(
            all_high(graph),
            baseline(graph),
            rtol=0,
            atol=0,
        )
        self.assertTrue(torch.isfinite(all_low(graph)).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for DQ")
    def test_all_high_preserves_root_and_bias_accumulation_order(self):
        baseline = self._model().cuda().eval()
        all_high = (
            self._model(
                dq_enabled=True,
                dq_degree_threshold=1,
            )
            .cuda()
            .eval()
        )
        with torch.no_grad():
            baseline.conv[0].weight.zero_()
            baseline.conv[0].root.zero_()
            baseline.conv[0].bias.zero_()
            baseline.conv[0].weight[0, 0, 0] = 1e20
            baseline.conv[0].root[0, 0] = -1e20
            baseline.conv[0].bias[0] = 3.140625
        all_high.load_state_dict(baseline.state_dict())
        graph = SimpleNamespace(
            x=torch.tensor([[1.0, 0.0]], device="cuda"),
            edge_index=torch.tensor([[0], [0]], device="cuda"),
            edge_type=torch.zeros(1, dtype=torch.long, device="cuda"),
            edge_attr=torch.zeros(1, 2, device="cuda"),
        )

        torch.testing.assert_close(
            all_high(graph),
            baseline(graph),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for DQ")
    def test_all_low_matches_standard_fp16_operation_order(self):
        all_low = (
            self._model(
                dq_enabled=True,
                dq_degree_threshold=2,
            )
            .cuda()
            .eval()
        )
        with torch.no_grad():
            all_low.conv[0].weight.zero_()
            all_low.conv[0].root.zero_()
            all_low.conv[0].bias.zero_()
            all_low.conv[0].root[0, 0] = 1.0
            all_low.conv[0].bias[0] = 0.0006

        fp16_reference = self._model().cuda().half().eval()
        fp16_reference.load_state_dict(all_low.state_dict())
        graph = SimpleNamespace(
            x=torch.tensor([[1.0, 0.0]], device="cuda"),
            edge_index=torch.tensor([[0], [0]], device="cuda"),
            edge_type=torch.zeros(1, dtype=torch.long, device="cuda"),
            edge_attr=torch.zeros(1, 2, device="cuda"),
        )
        fp16_graph = SimpleNamespace(
            x=graph.x.half(),
            edge_index=graph.edge_index,
            edge_type=graph.edge_type,
            edge_attr=graph.edge_attr.half(),
        )

        torch.testing.assert_close(
            all_low(graph),
            fp16_reference(fp16_graph),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for DQ")
    def test_eval_runs_multiple_layers_with_batch_norm(self):
        model = (
            PyGRGCNEdge(
                num_layers=2,
                num_rels=1,
                inp_dim=2,
                out_dim=2,
                dq_enabled=True,
                dq_degree_threshold=2,
            )
            .cuda()
            .eval()
        )

        with patch.object(
            model,
            "_high_degree_mask",
            wraps=model._high_degree_mask,
        ) as select_nodes:
            output = model(self._graph(device="cuda"))

        self.assertEqual(output.shape, (3, 2))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        select_nodes.assert_called_once()

    def test_eval_dq_rejects_cpu_execution(self):
        model = self._model(
            dq_enabled=True,
            dq_degree_threshold=2,
        ).eval()

        with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
            model(self._graph())

    def test_invalid_degree_threshold_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dq_degree_threshold"):
            self._model(dq_degree_threshold=0)


if __name__ == "__main__":
    unittest.main()
