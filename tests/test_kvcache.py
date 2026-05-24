import unittest

import torch

from kvcache import RadixKVCache


class KVCacheTests(unittest.TestCase):
    def make_cache(self, *, d_head=4, head_cnt=1, num_blocks=4, block_size=2):
        return RadixKVCache(
            d_head=d_head,
            head_cnt=head_cnt,
            num_blocks=num_blocks,
            block_size=block_size,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    def test_init_prefill_append_fetch_and_release_reuses_prefix_block(self):
        cache = self.make_cache()

        prefill_tokens = torch.tensor([[11, 12, 13]])
        prefill_k = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]])
        prefill_v = prefill_k + 10.0
        mask = torch.tensor([[True, True, True]])

        cache.init((11,), prefill_tokens)

        cache.prefill((11,), prefill_tokens, prefill_k, prefill_v, mask)
        self.assertTrue(torch.equal(cache.get_q_position((11,)), torch.tensor([[3]])))

        cache.release((11,))

        reuse_tokens = torch.tensor([[11, 12, 21]])
        reuse_chunk_tokens = torch.tensor([[21]])
        reuse_k = torch.tensor([[[[109.0, 110.0, 111.0, 112.0]]]])
        reuse_v = reuse_k + 10.0

        cache.init((22,), reuse_tokens)
        self.assertEqual(len(cache.uuids[22].commited_blocks), 1)

        cache.prefill((22,), reuse_chunk_tokens, reuse_k, reuse_v, torch.tensor([[True]]))
        self.assertTrue(torch.equal(cache.get_q_position((22,)), torch.tensor([[3]])))

        append_k = torch.tensor([[[[201.0, 202.0, 203.0, 204.0]]]])
        append_v = append_k + 10.0
        pages_info = cache.append_and_fetch((22,), append_k, append_v, torch.tensor([31]))

        self.assertEqual(len(pages_info), 1)
        state = pages_info[0]
        self.assertEqual(state["token_count"], 4)
        self.assertEqual(len(state["indexes"]), 2)

        pages_k, pages_v = cache.get_pages(state["indexes"])
        self.assertTrue(torch.equal(pages_k[0][0, 0], prefill_k[0, 0, 0]))
        self.assertTrue(torch.equal(pages_k[0][0, 1], prefill_k[0, 0, 1]))
        self.assertTrue(torch.equal(pages_v[0][0, 0], prefill_v[0, 0, 0]))
        self.assertTrue(torch.equal(pages_k[1][0, 0], reuse_k[0, 0, 0]))
        self.assertTrue(torch.equal(pages_k[1][0, 1], append_k[0, 0, 0]))

    def test_init_only_reuses_full_blocks(self):
        cache = self.make_cache(d_head=2, num_blocks=4, block_size=2)

        full_tokens = torch.tensor([[1, 2, 3]])
        K = torch.tensor([[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]])
        V = K + 10.0
        full_mask = torch.tensor([[True, True, True]])

        cache.init((7,), full_tokens)
        cache.prefill((7,), full_tokens, K, V, full_mask)
        cache.release((7,))

        shorter_tokens = torch.tensor([[1]])
        cache.init((9,), shorter_tokens)
        self.assertEqual(cache.uuids[9].commited_blocks, [])

    def test_release_after_init_without_prefill_is_safe(self):
        cache = self.make_cache()

        cache.init((123,), torch.tensor([[1, 2, 3]]))
        cache.release((123,))

        self.assertNotIn(123, cache.uuids)

    def test_prefill_reuses_existing_write_block_across_chunks(self):
        cache = self.make_cache(d_head=2, num_blocks=6, block_size=2)

        full_tokens = torch.tensor([[10, 11, 12]])
        cache.init((5,), full_tokens)

        chunk_1_toks = torch.tensor([[10]])
        chunk_1_k = torch.tensor([[[[1.0, 2.0]]]])
        chunk_1_v = chunk_1_k + 10.0
        cache.prefill((5,), chunk_1_toks, chunk_1_k, chunk_1_v, torch.tensor([[True]]))

        state = cache.uuids[5]
        self.assertEqual(state.commited_blocks, [])
        self.assertEqual(state.write_block_occupancy, 1)
        first_write_block = state.write_block

        chunk_2_toks = torch.tensor([[11, 12]])
        chunk_2_k = torch.tensor([[[[3.0, 4.0], [5.0, 6.0]]]])
        chunk_2_v = chunk_2_k + 10.0
        cache.prefill((5,), chunk_2_toks, chunk_2_k, chunk_2_v, torch.tensor([[True, True]]))

        state = cache.uuids[5]
        self.assertEqual(len(state.commited_blocks), 1)
        self.assertEqual(state.commited_blocks[0], first_write_block)
        self.assertEqual(state.write_block_occupancy, 1)
        self.assertEqual(state.write_block_toks, [12])
        self.assertTrue(torch.equal(cache.get_q_position((5,)), torch.tensor([[3]])))

        pages_k, pages_v = cache.get_pages([state.commited_blocks[0], state.write_block])
        self.assertTrue(torch.equal(pages_k[0][0, 0], chunk_1_k[0, 0, 0]))
        self.assertTrue(torch.equal(pages_k[0][0, 1], chunk_2_k[0, 0, 0]))
        self.assertTrue(torch.equal(pages_v[0][0, 1], chunk_2_v[0, 0, 0]))
        self.assertTrue(torch.equal(pages_k[1][0, 0], chunk_2_k[0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
