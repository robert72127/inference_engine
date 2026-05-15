import asyncio
import unittest

import torch

from model_processor import ModelProcessor, ServerHandle


class FakeConcurrentModel:
    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.vocab_size = 8
        self.generate_calls = 0

    def __call__(self, input_ids, prefill, mask, uuid):
        batch, seq_len = input_ids.shape
        logits = torch.full(
            (batch, seq_len, self.vocab_size),
            -1000.0,
            device=self.device,
        )
        if prefill:
            logits[:, :, 1] = 0.0
        else:
            self.generate_calls += 1
            next_token = 2 if self.generate_calls == 1 else 7
            logits[:, :, next_token] = 0.0
        return logits


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


class ModelProcessorConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_handles_complete_without_crashing(self):
        processor = ModelProcessor(
            FakeConcurrentModel,
            torch.device("cpu"),
            eos_token_id=7,
            max_batch_prefill=8,
            max_batch_generate=8,
        )

        handle_ids = await asyncio.gather(
            processor.prefill([11, 12, 13], temperature=0.0, top_p=1.0),
            processor.prefill([21, 22], temperature=0.0, top_p=1.0),
            processor.prefill([31], temperature=0.0, top_p=1.0),
        )

        await asyncio.sleep(0.1)

        all_tokens = []
        for handle_id in handle_ids:
            handle = processor.handles[handle_id]
            tokens = []
            while not handle.token_q.empty():
                tokens.append(handle.token_q.get_nowait())
            self.assertGreaterEqual(len(tokens), 2)
            self.assertEqual(tokens[0], 1)
            self.assertEqual(tokens[-1], 7)
            self.assertTrue(handle.finished)
            all_tokens.append(tokens)

        self.assertEqual(all_tokens[1:], [all_tokens[0], all_tokens[0]])

        for handle_id in handle_ids:
            await processor.release(handle_id)
