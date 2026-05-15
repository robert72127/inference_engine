import unittest

import torch

from model_processor import ModelProcessor, ServerHandle


class DecodeBatchTests(unittest.TestCase):
    def setUp(self):
        self.processor = ModelProcessor.__new__(ModelProcessor)

    def test_decode_batch_uses_greedy_when_temperature_is_zero(self):
        logits = torch.tensor(
            [[[0.1, 0.9, 0.0]], [[0.8, 0.1, 0.1]]],
            dtype=torch.float32,
        )
        batch = [
            ServerHandle(0, torch.tensor([1]), temperature=0.0, top_p=1.0),
            ServerHandle(1, torch.tensor([2]), temperature=0.0, top_p=1.0),
        ]

        out = self.processor._decode_batch(logits, batch)

        self.assertTrue(torch.equal(out, torch.tensor([1, 0])))

    def test_decode_batch_keeps_batch_shape_for_sampling(self):
        torch.manual_seed(0)
        logits = torch.tensor(
            [[[0.1, 0.9, 0.0]], [[0.8, 0.1, 0.1]]],
            dtype=torch.float32,
        )
        batch = [
            ServerHandle(0, torch.tensor([1]), temperature=1.0, top_p=1.0),
            ServerHandle(1, torch.tensor([2]), temperature=1.0, top_p=1.0),
        ]

        out = self.processor._decode_batch(logits, batch)

        self.assertEqual(tuple(out.shape), (2,))
        self.assertTrue(out.dtype == torch.long)

    def test_decode_batch_supports_mixed_sampling_params(self):
        torch.manual_seed(0)
        logits = torch.tensor(
            [[[0.1, 0.9, 0.0]], [[0.8, 0.1, 0.1]]],
            dtype=torch.float32,
        )
        batch = [
            ServerHandle(0, torch.tensor([1]), temperature=0.0, top_p=1.0),
            ServerHandle(1, torch.tensor([2]), temperature=0.8, top_p=0.9),
        ]

        out = self.processor._decode_batch(logits, batch)

        self.assertEqual(tuple(out.shape), (2,))
        self.assertEqual(out[0].item(), 1)
