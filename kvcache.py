from collections import deque
import torch

def blocks_for_tokens(token_count: int, block_size: int):
    blocks_cnt = (token_count + block_size - 1) // block_size
    if token_count > 0 and token_count % block_size == 0:
        blocks_cnt += 1
    return blocks_cnt

class PagedKVCache:
    class UUIDState:
        def __init__(self, 
                    kv_cache,
                    block_size,
                    blocks_count,
                    commited_blocks,
                    write_block,
                    write_block_occupancy):
            
            self.kv_cache = kv_cache
            self.block_size = block_size

            self.commited_blocks = commited_blocks
            self.write_block = write_block
            self.blocks_count = blocks_count
            self.write_block_occupancy = write_block_occupancy

        def get_pages_and_meta(self):
            K = [self.kv_cache.K[idx] for idx in self.commited_blocks] + [self.kv_cache.K[self.write_block]]
            V = [self.kv_cache.V[idx] for idx in self.commited_blocks] + [self.kv_cache.V[self.write_block]]
            seq_len = self.block_size * len(self.commited_blocks) + self.write_block_occupancy
            return {
                "K":K,
                "V":V,
                "token_count":seq_len
            }

        def insert_single(self, K,V):
            if self.write_block_occupancy == self.kv_cache.block_size:
               self.commited_blocks += [self.write_block]
               self.write_block = self.kv_cache.get_free_blocks()[0]
               self.write_block_occupancy = 0 
               self.blocks_count +=1

            self.kv_cache.K[self.write_block][:, self.write_block_occupancy, :] = K 
            self.kv_cache.V[self.write_block][:, self.write_block_occupancy, :] = V 
            self.write_block_occupancy+=1

        def fill(self, K, V):
            block_offset = 0
            for block in self.commited_blocks:
                start = block_offset * self.block_size
                end = start + self.block_size
                K[:, start:end, :] = self.kv_cache.K[block]
                V[:, start:end, :] = self.kv_cache.V[block]
                block_offset += 1

            start = block_offset * self.block_size
            for j in range(self.write_block_occupancy):
                K[:, start + j, :] = self.kv_cache.K[self.write_block][:, j, :]
                V[:, start + j, :] = self.kv_cache.V[self.write_block][:, j, :]

    def __init__(self, d_head, head_cnt, num_blocks, block_size, device, dtype):
        self.d_head = d_head
        self.head_cnt = head_cnt
        self.blocks_cnt = num_blocks
        self.block_size = block_size

        self.uuids = {}

        self.free_slots = deque([i for i in range (num_blocks)])
        self.free_slots_cnt = num_blocks

        self.device = device
        self.dtype = dtype
        self.K = torch.zeros((num_blocks,self.head_cnt, block_size, self.d_head), device=device, dtype=dtype)
        self.V = torch.zeros((num_blocks,self.head_cnt, block_size, self.d_head), device=device, dtype=dtype)

    # todo extract info from this in the engine
    def get_occupancy_info(self):
        return {
            "blocks_cnt":  self.blocks_cnt,
            "block_size":  self.block_size, 
            "free_blocks": self.free_slots_cnt, 
            "uuid_cnt":    len(self.uuids)}

    def get_free_blocks(self, block_cnt=1):
        blocks = [self.free_slots.popleft() for _ in range(block_cnt)]
        self.free_slots_cnt -= block_cnt
        return blocks

    def _free_blocks(self, uuid):
        if uuid in self.uuids:
            uuid_state =self.uuids[uuid]
            self.free_slots += uuid_state.commited_blocks + [uuid_state.write_block]
            self.free_slots_cnt += uuid_state.blocks_count
            self.uuids.pop(uuid)

    def _get_pages(self, indexes):
        K_ = [self.K[ind] for ind in indexes]
        V_ = [self.V[ind] for ind in indexes]
        return K_, V_

    def prefill(self, uuids, K, V, mask):
        for i, uuid in enumerate(uuids):
            seq_len = int(mask[i].sum().item())
            blocks_cnt = blocks_for_tokens(seq_len, self.block_size)
            indexes = self.get_free_blocks(blocks_cnt)
            pages_K, pages_V = self._get_pages(indexes)
            for j in range(seq_len):
                block_idx = j // self.block_size
                in_block_idx = j % self.block_size
                pages_K[block_idx][:, in_block_idx, :] = K[i, :, j, :]
                pages_V[block_idx][:, in_block_idx, :] = V[i, :, j, :]

            uuid_state = self.UUIDState(self, self.block_size, blocks_cnt, indexes[:-1], indexes[-1], seq_len % self.block_size)
            self.uuids[uuid] = uuid_state

    def get_q_position(self, uuids):
        positions = []
        for uuid in uuids:
            uuid_state = self.uuids[uuid]
            positions.append((uuid_state.blocks_count -1) * self.block_size + uuid_state.write_block_occupancy)
        return torch.tensor(positions, device=self.K.device).unsqueeze(1)

    def append_and_fetch(self, uuids, K, V):
        max_len = 0
        batch_size = 0
        lengths = []
        pages_and_meta = []
        for i, uuid in enumerate(uuids):
            uuid_state = self.uuids[uuid]
            uuid_state.insert_single(K[i, :, 0, :], V[i, :, 0, :])
            length = (uuid_state.blocks_count - 1) * self.block_size + uuid_state.write_block_occupancy
            max_len = max(max_len, length)
            lengths.append(length)
            batch_size += 1
            pages_and_meta += [uuid_state.get_pages_and_meta()]
        return pages_and_meta

    def release(self, uuids):
        for uuid in uuids:
            self._free_blocks(uuid)
