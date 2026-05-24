import asyncio
import unittest

import torch

from model_processor import ModelProcessor, ServerHandle


class FakeConcurrentModel:
    class FakeCache:
        def __init__(self, total_blocks=128, block_size=1):
            self.release_calls = []
            self.total_blocks = total_blocks
            self.block_size = block_size

        def release(self, uuid):
            self.release_calls.extend(uuid)

        def get_occupancy_info(self):
            return {
                "blocks_cnt": self.total_blocks,
                "block_size": self.block_size,
                "free_blocks": self.total_blocks,
                "uuid_cnt": 0,
            }

        def get_blocks_available(self):
            return self.total_blocks

        def blocks_for_tokens(self, token_count):
            return (token_count + self.block_size - 1) // self.block_size

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.vocab_size = 8
        self.generate_calls = 0
        self.kv_caches = [self.FakeCache()]

    def __call__(self, input_ids, tokens, prefill, mask, lengths, uuid, prefill_is_last_chunk=True):
        batch = input_ids.shape[0]
        logits = torch.full(
            (batch, self.vocab_size),
            -1000.0,
            device=self.device,
        )
        if prefill:
            logits[:, 1] = 0.0
        else:
            self.generate_calls += 1
            next_token = 2 if self.generate_calls == 1 else 7
            logits[:, next_token] = 0.0
        return logits

class DecodeBatchTests(unittest.TestCase):
    def setUp(self):
        self.processor = ModelProcessor.__new__(ModelProcessor)

    def test_decode_batch_uses_greedy_when_temperature_is_zero(self):
        logits = torch.tensor(
            [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
            dtype=torch.float32,
        )
        batch = [
            ServerHandle(0, torch.tensor([1]), temperature=0.0, top_p=1.0, max_new_tokens=1),
            ServerHandle(1, torch.tensor([2]), temperature=0.0, top_p=1.0, max_new_tokens=1),
        ]

        out = self.processor._decode_batch(logits, batch)

        self.assertTrue(torch.equal(out, torch.tensor([1, 0])))

    def test_decode_batch_keeps_batch_shape_for_sampling(self):
        torch.manual_seed(0)
        logits = torch.tensor(
            [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
            dtype=torch.float32,
        )
        batch = [
            ServerHandle(0, torch.tensor([1]), temperature=1.0, top_p=1.0, max_new_tokens=1),
            ServerHandle(1, torch.tensor([2]), temperature=1.0, top_p=1.0, max_new_tokens=1),
        ]

        out = self.processor._decode_batch(logits, batch)

        self.assertEqual(tuple(out.shape), (2,))
        self.assertTrue(out.dtype == torch.long)

    def test_decode_batch_supports_mixed_sampling_params(self):
        torch.manual_seed(0)
        logits = torch.tensor(
            [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
            dtype=torch.float32,
        )
        batch = [
            ServerHandle(0, torch.tensor([1]), temperature=0.0, top_p=1.0, max_new_tokens=1),
            ServerHandle(1, torch.tensor([2]), temperature=0.8, top_p=0.9, max_new_tokens=1),
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
            max_request_len=64,
            max_batch_prefill=8,
            max_batch_generate=8,
        )

        handle_ids = await asyncio.gather(
            processor.prefill([11, 12, 13], temperature=0.0, top_p=1.0),
            processor.prefill([21, 22], temperature=0.0, top_p=1.0),
            processor.prefill([31], temperature=0.0, top_p=1.0),
        )
        handles = [processor.handles[handle_id] for handle_id in handle_ids]

        await asyncio.sleep(0.1)

        all_tokens = []
        for handle in handles:
            tokens = []
            while not handle.token_q.empty():
                tokens.append(handle.token_q.get_nowait())
            self.assertGreaterEqual(len(tokens), 2)
            self.assertEqual(tokens[0], 1)
            self.assertEqual(tokens[-1], 7)
            self.assertTrue(handle.finished)
            all_tokens.append(tokens)

        for handle_id in handle_ids:
            await processor.release(handle_id)

        self.assertEqual(sorted(processor.kv_caches[0].release_calls), sorted(handle_ids))

    async def test_stops_at_kv_cache_limit_before_generate_overflow(self):
        processor = ModelProcessor(
            FakeConcurrentModel,
            torch.device("cpu"),
            eos_token_id=7,
            max_request_len=2,
            max_batch_prefill=8,
            max_batch_generate=8,
        )

        handle_id = await processor.prefill([11, 12], temperature=0.0, top_p=1.0)
        handle = processor.handles[handle_id]
        await asyncio.sleep(0.1)

        tokens = []
        while not handle.token_q.empty():
            tokens.append(handle.token_q.get_nowait())

        self.assertEqual(tokens, [1])
        self.assertEqual(processor.model.generate_calls, 0)

    async def test_prefill_rejects_prompt_longer_than_kv_cache_limit(self):
        processor = ModelProcessor(
            FakeConcurrentModel,
            torch.device("cpu"),
            eos_token_id=7,
            max_request_len=2,
            max_batch_prefill=8,
            max_batch_generate=8,
        )

        with self.assertRaises(RuntimeError):
            await processor.prefill([11, 12, 13], temperature=0.0, top_p=1.0)

    async def test_prefill_admission_follows_cache_occupancy(self):
        class OccupancyModel(FakeConcurrentModel):
            def __init__(self, device: torch.device):
                super().__init__(device)
                self.kv_caches = [self.FakeCache(total_blocks=3, block_size=2)]
                self.prefill_batch_sizes = []

            def __call__(self, input_ids, tokens, prefill, mask, lengths, uuid, prefill_is_last_chunk=True):
                if prefill:
                    self.prefill_batch_sizes.append(input_ids.size(0))
                logits = super().__call__(input_ids, tokens, prefill, mask, lengths, uuid, prefill_is_last_chunk)
                if not prefill:
                    logits[:, :] = -1000.0
                    logits[:, 7] = 0.0
                return logits

        processor = ModelProcessor(
            OccupancyModel,
            torch.device("cpu"),
            eos_token_id=7,
            max_request_len=4,
        )

        await asyncio.gather(
            processor.prefill([11, 12], temperature=0.0, top_p=1.0, max_new_tokens=2),
            processor.prefill([21, 22], temperature=0.0, top_p=1.0, max_new_tokens=2),
        )
        await asyncio.sleep(0.1)

        self.assertEqual(processor.model.prefill_batch_sizes, [1, 1])
