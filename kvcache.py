from collections import deque
import torch

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

        def insert_single(self, K,V):
            if self.write_block_occupancy == self.kv_cache.block_size:
               self.commited_blocks += [self.write_block]
               self.write_block = self.kv_cache.get_free_blocks()[0]
               self.write_block_occupancy = 0 
               self.blocks_count +=1

            self.kv_cache.K[self.write_block][self.write_block_occupancy] = K 
            self.kv_cache.V[self.write_block][self.write_block_occupancy] = V 
            self.write_block_occupancy+=1

        def fill(self, K, V):
            i = 0
            for blck in self.commited_blocks:
                for j in range(self.block_size):
                    K[i * self.block_size + j] = self.kv_cache.K[blck][j]
                    V[i * self.block_size + j] = self.kv_cache.V[blck][j]
                    i+= 1
            for j in range(self.write_block_occupancy):
                K[i * self.block_size + j] = self.kv_cache.K[self.write_block][j]
                V[i * self.block_size + j] = self.kv_cache.V[self.write_block][j]

    def __init__(self, d_model, num_blocks, block_size, device, dtype):
        self.d_model = d_model
        self.blocks_cnt = num_blocks
        self.block_size = block_size

        self.uuids = {}

        self.free_slots = deque([i for i in range (num_blocks)])
        self.free_slots_cnt = num_blocks

        self.device = device
        self.dtype = dtype
        self.K = torch.zeros((num_blocks, block_size, d_model), device=device, dtype=dtype)
        self.V = torch.zeros((num_blocks, block_size, d_model), device=device, dtype=dtype)

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
            self.free_slots_cnt += uuid_state.blocks_cnt
            self.uuids.pop(uuid)

    def _get_pages(self, indexes):
        K_ = [self.K[ind] for ind in indexes]
        V_ = [self.V[ind] for ind in indexes]
        return K_, V_

    def prefill(self, uuids, K, V, mask):
        for i, uuid in enumerate(uuids):
            seq_len = int(mask[i].sum().item())
            blocks_cnt = (seq_len + self.block_size - 1) // self.block_size
            # preallocate new block for write if we hit exact multiply of block size
            if seq_len == blocks_cnt * self.block_size:
                blocks_cnt +=1
            indexes = self.get_free_blocks(blocks_cnt)
            pages_K, pages_V = self._get_pages(indexes)
            for j in range(seq_len):
                block_idx = j // self.block_size
                in_block_idx = j % self.block_size
                pages_K[block_idx][in_block_idx] = K[i][j]
                pages_V[block_idx][in_block_idx] = V[i][j]

            uuid_state = self.UUIDState(self, self.block_size, blocks_cnt, indexes[:-1], indexes[-1], seq_len % self.block_size)
            self.uuids[uuid] = uuid_state

    def append_and_fetch(self, uuids, K, V):
        max_len = 0
        batch_size = 0
        lengths = []
        for i, uuid in enumerate(uuids):
            uuid_state = self.uuids[uuid]
            uuid_state.insert_single(K[i], V[i])
            len = (uuid_state.blocks_count - 1) * self.block_size + uuid_state.write_block_occupancy
            max_len = max(max_len, len)
            lengths += [len]
            batch_size +=1
        
        K = torch.zeros((batch_size, max_len, self.d_model), device=self.device, dtype=self.dtype)
        V = torch.zeros((batch_size, max_len, self.d_model), device=self.device, dtype=self.dtype)
        for i, uuid in enumerate(uuids):
            uuid_state = self.uuids[uuid]
            uuid_state.fill(K[i], V[i])

        lengths_t = torch.tensor(lengths, device=self.device, dtype=torch.long)

        kv_mask = (
            torch.arange(max_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, max_len) < lengths_t.unsqueeze(1))

        k_positions = (
            torch.arange(max_len, device=self.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, max_len)
        )

        q_positions = torch.tensor(
            q_positions,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(1)
        
        return K, V, q_positions, k_positions, kv_mask

    def release(self, uuids):
        for uuid in uuids:
            self._free_blocks(uuid)