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

    def test_top_k_1_always_returns_argmax(self):
        logits = torch.tensor([[0.1, 0.9, 0.2, 0.3]], dtype=torch.float32)

        for _ in range(20):
            out = top_k_top_p_sample(logits, top_k=1, top_p=1.0, temperature=1.0)
            self.assertEqual(out.item(), 1)

    def test_top_k_restricts_sampled_tokens(self):
        # tokens 2 and 3 have real probability but are outside top-2
        logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

        seen = set()
        torch.manual_seed(42)
        for _ in range(100):
            out = top_k_top_p_sample(logits, top_k=2, top_p=1.0, temperature=2.0)
            seen.add(out.item())

        self.assertTrue(seen.issubset({0, 1}))

    def test_top_k_equal_to_vocab_size_allows_all_tokens(self):
        logits = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)

        seen = set()
        torch.manual_seed(0)
        for _ in range(200):
            out = top_k_top_p_sample(logits, top_k=3, top_p=1.0, temperature=1.0)
            seen.add(out.item())

        self.assertEqual(seen, {0, 1, 2})
