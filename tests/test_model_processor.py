import asyncio
import time
import unittest

import torch

from model_processor import ModelProcessor, OP, ServerHandle
from models.model import DecodeStateBuff, PrefillStateBuff


class FakeCache:
    def __init__(self, total_blocks=128, block_size=2):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.release_calls = []
        self.states = {}
        self.next_page = 0

    def init(self, uuid, toks):
        self.states[uuid] = {"token_count": 0, "indexes": []}

    def _reserve(self, uuid, token_count):
        state = self.states[uuid]
        state["token_count"] += token_count
        pages_needed = (state["token_count"] + self.block_size - 1) // self.block_size
        while len(state["indexes"]) < pages_needed:
            state["indexes"].append(self.next_page)
            self.next_page += 1

    def init_prefill(self, uuid, toks):
        self._reserve(uuid, int(toks.numel()))

    def finish_prefill(self, uuid):
        pass

    def append_decode(self, uuid, tok):
        self._reserve(uuid, 1)

    def get_indexes(self, uuid):
        return list(self.states[uuid]["indexes"])

    def get_blocks_available(self):
        return self.total_blocks - self.next_page

    def release(self, uuid):
        self.release_calls.append(uuid)
        self.states.pop(uuid, None)


class FakeConcurrentModel:
    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.vocab_size = 8
        self.kv_caches = [FakeCache()]
        self.prefill_seq_lens = []
        self.decode_calls = 0

    def KV_init(self, handle_id, prompt_tokens):
        for cache in self.kv_caches:
            cache.init(handle_id, prompt_tokens)

    def prefill(self, state: PrefillStateBuff):
        self.prefill_seq_lens.append(state.seq_lens.detach().cpu().tolist())
        logits = torch.full((state.batch_size, self.vocab_size), -1000.0, device=self.device)
        logits[:, 1] = 0.0
        return logits

    def decode(self, state: DecodeStateBuff):
        self.decode_calls += 1
        time.sleep(0.01)
        logits = torch.full((state.batch_size, self.vocab_size), -1000.0, device=self.device)
        logits[:, 7] = 0.0
        return logits


def make_handle(handle_id, tokens, *, cache_pos=0, temperature=0.0):
    handle = ServerHandle(
        handle_id,
        torch.tensor(tokens),
        temperature=temperature,
        top_p=1.0,
        max_new_tokens=4,
    )
    handle.cache_pos = cache_pos
    return handle


class DecodeBatchTests(unittest.TestCase):
    def setUp(self):
        self.processor = ModelProcessor.__new__(ModelProcessor)

    def test_decode_batch_uses_greedy_when_temperature_is_zero(self):
        logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        batch = [make_handle(0, [1]), make_handle(1, [2])]

        out = self.processor._decode_batch(logits, batch)

        self.assertTrue(torch.equal(out, torch.tensor([1, 0])))

    def test_decode_batch_supports_mixed_sampling_params(self):
        torch.manual_seed(0)
        logits = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
        batch = [make_handle(0, [1]), make_handle(1, [2], temperature=0.8)]

        out = self.processor._decode_batch(logits, batch)

        self.assertEqual(tuple(out.shape), (2,))
        self.assertEqual(out.dtype, torch.long)
        self.assertEqual(out[0].item(), 1)


class BatchedInputTests(unittest.TestCase):
    def setUp(self):
        self.processor = ModelProcessor.__new__(ModelProcessor)
        self.processor.device = torch.device("cpu")
        self.processor.prefill_chunk_size = 2
        self.processor.prefill_buffs = {
            1: PrefillStateBuff(1, 2, 4, self.processor.device),
            2: PrefillStateBuff(2, 2, 4, self.processor.device),
        }
        self.processor.decode_buffs = {
            1: DecodeStateBuff(1, 4, self.processor.device),
            2: DecodeStateBuff(2, 4, self.processor.device),
        }
        self.processor.kv_caches = [FakeCache()]

    def init_handles(self, handles):
        for handle in handles:
            self.processor.kv_caches[0].init(handle.id, handle.tokens)

    def test_prefill_uses_fixed_chunk_buffer_and_tracks_real_lengths(self):
        batch = [make_handle(0, [11, 12, 13]), make_handle(1, [21])]
        self.init_handles(batch)

        state, last_chunk = self.processor._make_batched_input(batch, OP.PREFILL, 2)

        self.assertTrue(torch.equal(state.tokens, torch.tensor([[11, 12], [21, 0]])))
        self.assertTrue(torch.equal(state.seq_lens, torch.tensor([2, 1], dtype=torch.int32)))
        self.assertTrue(torch.equal(state.offsets, torch.tensor([0, 0], dtype=torch.int32)))
        self.assertEqual(last_chunk, [False, True])
        self.assertEqual(state.page_indexes[0, 0].item(), 0)
        self.assertEqual(state.page_indexes[1, 0].item(), 1)

    def test_prefill_continues_from_cache_position(self):
        handle = make_handle(0, [11, 12, 13], cache_pos=2)
        self.init_handles([handle])
        self.processor.kv_caches[0]._reserve(handle.id, 2)

        state, last_chunk = self.processor._make_batched_input([handle], OP.PREFILL, 1)

        self.assertTrue(torch.equal(state.tokens, torch.tensor([[13, 0]])))
        self.assertEqual(state.seq_lens.item(), 1)
        self.assertEqual(state.offsets.item(), 2)
        self.assertEqual(last_chunk, [True])

    def test_decode_reserves_token_and_populates_metadata(self):
        handle = make_handle(0, [11], cache_pos=1)
        handle.input_token = 5
        self.init_handles([handle])
        self.processor.kv_caches[0]._reserve(handle.id, 1)

        state = self.processor._make_batched_input([handle], OP.GENERATE, 1)

        self.assertEqual(state.tokens.item(), 5)
        self.assertEqual(state.offsets.item(), 1)
        self.assertEqual(self.processor.kv_caches[0].states[0]["token_count"], 2)


class BatchSizeTests(unittest.TestCase):
    def test_selects_largest_supported_batch_not_exceeding_request_count(self):
        processor = ModelProcessor.__new__(ModelProcessor)
        processor.supported_batch_sizes = [1, 2, 4, 8]

        self.assertEqual(processor._get_batch_size(0), 0)
        self.assertEqual(processor._get_batch_size(1), 1)
        self.assertEqual(processor._get_batch_size(3), 2)
        self.assertEqual(processor._get_batch_size(7), 4)
        self.assertEqual(processor._get_batch_size(20), 8)


class ModelProcessorConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def wait_until(self, predicate, timeout=1.0):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate():
            if loop.time() >= deadline:
                self.fail("timed out waiting for model processor")
            await asyncio.sleep(0.01)

    async def test_three_concurrent_handles_use_supported_batches_and_complete(self):
        processor = ModelProcessor(
            FakeConcurrentModel,
            torch.device("cpu"),
            eos_token_id=7,
            max_request_len=64,
            max_batch_size=8,
            prefill_chunk_size=2,
        )

        handle_ids = await asyncio.gather(
            processor.prefill(torch.tensor([11, 12, 13]), temperature=0.0, top_p=1.0),
            processor.prefill(torch.tensor([21, 22]), temperature=0.0, top_p=1.0),
            processor.prefill(torch.tensor([31]), temperature=0.0, top_p=1.0),
        )

        await self.wait_until(lambda: all(handle_id in processor.kv_caches[0].release_calls for handle_id in handle_ids))

        self.assertEqual(sorted(processor.kv_caches[0].release_calls), sorted(handle_ids))
        self.assertTrue(all(len(lengths) in processor.supported_batch_sizes for lengths in processor.model.prefill_seq_lens))

    async def test_prefill_rejects_prompt_longer_than_cache_limit(self):
        processor = ModelProcessor(
            FakeConcurrentModel,
            torch.device("cpu"),
            eos_token_id=7,
            max_request_len=2,
        )

        with self.assertRaises(RuntimeError):
            await processor.prefill(torch.tensor([11, 12, 13]), temperature=0.0, top_p=1.0)

    async def test_prefill_requeues_until_final_fixed_size_chunk(self):
        processor = ModelProcessor(
            FakeConcurrentModel,
            torch.device("cpu"),
            eos_token_id=7,
            max_request_len=8,
            prefill_chunk_size=2,
        )

        handle_id = await processor.prefill(
            torch.tensor([11, 12, 13]),
            temperature=0.0,
            top_p=1.0,
        )
        await self.wait_until(lambda: handle_id in processor.kv_caches[0].release_calls)

        flattened = [length for call in processor.model.prefill_seq_lens for length in call]
        self.assertEqual(flattened[:2], [2, 1])


if __name__ == "__main__":
    unittest.main()
