import unittest

import torch

from kvcache import RadixKVCache, blocks_for_tokens


class KVCacheTests(unittest.TestCase):
    def make_cache(self, *, d_head=4, head_cnt=1, num_blocks=6, block_size=2):
        return RadixKVCache(
            d_head=d_head,
            head_cnt=head_cnt,
            num_blocks=num_blocks,
            block_size=block_size,
            max_requests_per_uuid=blocks_for_tokens(8, block_size),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_prefill_reserve_commit_and_prefix_reuse(self):
        cache = self.make_cache()
        prompt = torch.tensor([11, 12, 13])

        cache.init(11, prompt.tolist())
        cache.init_prefill(11, prompt)
        indexes_before_commit = cache.get_indexes(11)
        self.assertEqual(len(indexes_before_commit), 2)

        cache.finish_prefill(11)
        first_state = cache.uuids[11]
        first_prefix_page = first_state.commited_blocks[0]
        self.assertEqual(first_state.write_block_occupancy, 1)
        cache.release(11)

        cache.init(22, [11, 12, 21])
        reused_state = cache.uuids[22]
        self.assertEqual(reused_state.commited_blocks, [first_prefix_page])

        cache.init_prefill(22, torch.tensor([21]))
        self.assertEqual(len(cache.get_indexes(22)), 2)
        cache.finish_prefill(22)

        cache.append_decode(22, 31)
        self.assertEqual(cache.uuids[22].write_block_occupancy, 2)
        self.assertEqual(cache.uuids[22].write_block_toks, [21, 31])

    def test_init_only_reuses_full_blocks(self):
        cache = self.make_cache(d_head=2)

        cache.init(7, [1, 2, 3])
        cache.init_prefill(7, torch.tensor([1, 2, 3]))
        cache.finish_prefill(7)
        cache.release(7)

        cache.init(9, [1])
        self.assertEqual(cache.uuids[9].commited_blocks, [])

        cache.init(10, [1, 2, 4])
        self.assertEqual(len(cache.uuids[10].commited_blocks), 1)

    def test_release_after_init_without_prefill_is_safe(self):
        cache = self.make_cache()

        cache.init(123, [1, 2, 3])
        cache.release(123)

        self.assertNotIn(123, cache.uuids)

    def test_prefill_reuses_write_block_across_chunks(self):
        cache = self.make_cache(d_head=2)
        cache.init(5, [10, 11, 12])

        cache.init_prefill(5, torch.tensor([10]))
        cache.finish_prefill(5)
        state = cache.uuids[5]
        first_write_block = state.write_block
        self.assertEqual(state.write_block_occupancy, 1)

        cache.init_prefill(5, torch.tensor([11, 12]))
        self.assertEqual(cache.get_indexes(5)[0], first_write_block)
        cache.finish_prefill(5)

        state = cache.uuids[5]
        self.assertEqual(state.commited_blocks, [first_write_block])
        self.assertEqual(state.write_block_occupancy, 1)
        self.assertEqual(state.write_block_toks, [12])

    def test_blocks_for_tokens_keeps_a_decode_spare_at_page_boundary(self):
        self.assertEqual(blocks_for_tokens(0, 2), 0)
        self.assertEqual(blocks_for_tokens(1, 2), 1)
        self.assertEqual(blocks_for_tokens(2, 2), 2)
        self.assertEqual(blocks_for_tokens(3, 2), 2)


if __name__ == "__main__":
    unittest.main()
