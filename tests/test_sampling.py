import unittest

import torch

from sampling import top_k_top_p_sample


class SamplingTests(unittest.TestCase):
    def test_returns_batch_of_token_ids_for_2d_logits(self):
        torch.manual_seed(0)
        logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]], dtype=torch.float32)

        out = top_k_top_p_sample(logits, top_k=None, top_p=1.0, temperature=1.0)

        self.assertEqual(tuple(out.shape), (2,))
        self.assertEqual(out.dtype, torch.long)

    def test_rejects_non_2d_logits(self):
        logits = torch.zeros((1, 1, 3), dtype=torch.float32)

        with self.assertRaises(ValueError) as ctx:
            top_k_top_p_sample(logits)

        self.assertIn("[B,V]", str(ctx.exception))
