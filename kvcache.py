from collections import deque
import torch

def blocks_for_tokens(token_count: int, block_size: int):
    blocks_cnt = (token_count + block_size - 1) // block_size
    if token_count > 0 and token_count % block_size == 0:
        blocks_cnt += 1
    return blocks_cnt

class RadixKVCache:
    class UUIDState:
        def __init__(self, 
                    kv_cache,
                    block_size,
                    blocks_count,
                    commited_blocks):
            
            self.kv_cache = kv_cache
            self.block_size = block_size
            self.commited_blocks = commited_blocks

            self.write_block = None
            self.write_block_toks = []

            self.blocks_count = blocks_count
            self.write_block_occupancy = 0

        def get_pages_info(self):
            K = [self.kv_cache.K[idx] for idx in self.commited_blocks] + [self.kv_cache.K[self.write_block]]
            V = [self.kv_cache.V[idx] for idx in self.commited_blocks] + [self.kv_cache.V[self.write_block]]
            seq_len = self.block_size * len(self.commited_blocks) + self.write_block_occupancy
            return {
                "indexes": self.commited_blocks + [self.write_block],
                "token_count":seq_len
            }

        def insert_prefill(self, commited_blocks, write_block, write_block_occupancy, write_block_toks):
            self.commited_blocks +=  [commited_blocks]
            self.write_block = write_block
            self.write_block_occupancy = write_block_occupancy
            self.write_block_toks = write_block_toks
            self.blocks_count += len(commited_blocks) + 1

        def insert_single(self, tok, K,V):
            if self.write_block_occupancy == self.kv_cache.block_size:
               self.write_block = self.kv_cache.radix_insert(self, self.write_block, self.write_block_toks)
               self.commited_blocks += [self.write_block]
               self.write_block = self.kv_cache.get_free_blocks()[0]
               self.write_block_occupancy = 0 
               self.blocks_count +=1
               self.write_block_toks = []

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

    class RadixNode:
        def __init__(self, page_id, key):
            self.key=key
            self.page_id = page_id
            self.children = []

        def search(self, key_lst, nodes_list):
            if not key_lst:
                return nodes_list
            for child in self.children:
                if child.key == key_lst[1]:
                    self.kv_cache.locked_slots[child.page_id] += 1
                    return child.search(key_lst[1:], nodes_list + [child])
            return nodes_list

    class RadixTree:
        def __init__(self, kv_cache):
            self.root = RadixKVCache.RadixNode(page_id=None, key=None)
            self.uuid_to_last_node = {}
            self.kv_cache = kv_cache

        def get_last_node(self, uuid):
            return self.uuid_to_last_node[uuid]

        def add_node(self, uuid, node):
            self.uuid_to_last_node[uuid] = node
            
        def insert(self, uuid, block_idx, key):
            node = self.get_last_node(uuid)
            for child in node.children:
                if child.key == key:
                    self.kv_cache.locked_slots[child.page_id] += 1
                    self.uuid_to_last_node[uuid] = child
                    self.kv_cache.free_slots.append(block_idx)
                    self.kv_cache.free_slots_cnt += 1
                    del self.kv_cache.locked_slots[block_idx]
                    return child.page_id
            new_node = RadixKVCache.RadixNode(page_id=block_idx, key=key, kv_cache=self.kv_cache, radix_tree=self)
            node.children.append(new_node)
            self.uuid_to_last_node[uuid] = new_node
            return block_idx

        def search(self, toks):
            return self.root.search(toks, [])
        
    def __init__(self, d_head, head_cnt, num_blocks, block_size, max_requests_per_uuid, device, dtype):
        self.d_head = d_head
        self.head_cnt = head_cnt
        self.blocks_cnt = num_blocks
        self.block_size = block_size

        self.max_requests_per_uuid = max_requests_per_uuid
        self.uuids = {}
        
        # blocks are put there when they are used by at least one active uuid
        self.locked_slots = {}

        self.free_slots = deque([i for i in range (num_blocks)])
        self.free_slots_cnt = num_blocks

        self.device = device
        self.dtype = dtype
        self.K = torch.zeros((num_blocks,self.head_cnt, block_size, self.d_head), device=device, dtype=dtype)
        self.V = torch.zeros((num_blocks,self.head_cnt, block_size, self.d_head), device=device, dtype=dtype)

        self.radix_tree = RadixTree(self)

    # might return different index if block is already present, then this block will be auto freed, and found block lock_count will be incremented
    def radix_insert(self, uuid, block_idx, toks):
        return self.radix_tree.insert(uuid, block_idx, toks) 

    # returns longest list of pages that match prefix, update locked_slots if found pages
    def radix_search(self, uuid, toks):
        nodes = self.radix_tree.search(toks)
        self.radix_tree.insert(uuid, nodes[-1] if nodes else self.radix_tree.root)
        return [node.page_id for node in nodes], len(nodes)

    # todo return amount of non locked pages - amount current uuids could still consume
    def get_blocks_available(self):
        reserved=0
        for uuid in self.uuids:
            uuid_state = self.uuids[uuid]
            reserved += self.max_requests_per_uuid - uuid_state.blocks_count
        
        return self.blocks_cnt - len(self.locked_slots) - reserved

    # todo needs to be updated, ie decide on eviction policy
    def get_free_blocks(self, block_cnt=1):
        blocks = [self.free_slots.popleft() for _ in range(block_cnt)]
        self.free_slots_cnt -= block_cnt
        self.locked_slots.update({block: 1 for block in blocks})
        return blocks

    def _free_blocks(self, uuid):
        if uuid in self.uuids:
            uuid_state =self.uuids[uuid]
            for block in uuid_state.commited_blocks + [uuid_state.write_block]:
                self.locked_slots[block] -= 1
                if self.locked_slots[block] == 0:
                    self.free_slots.append(block)
                    self.free_slots_cnt += 1
                    del self.locked_slots[block]
            self.uuids.pop(uuid)

    def _get_pages(self, indexes):
        K_ = [self.K[ind] for ind in indexes]
        V_ = [self.V[ind] for ind in indexes]
        return K_, V_

    def get_q_position(self, uuids):
        positions = []
        for uuid in uuids:
            uuid_state = self.uuids[uuid]
            positions.append((uuid_state.blocks_count -1) * self.block_size + uuid_state.write_block_occupancy)
        return torch.tensor(positions, device=self.K.device).unsqueeze(1)

    # first step in prefill, serach for pages matching longest prefix of toks
    def init(self, uuids, toks):
        for i, uuid in enumerate(uuids):
            blocks, blocks_cnt = self.radix_search(toks[i])
            self.uuids[uuid] = self.UUIDState(self, self.block_size, blocks_cnt=blocks_cnt, commited_blocks=blocks)
            return blocks, blocks_cnt

    def prefill(self, uuids, toks, K, V, mask):
        for i, uuid in enumerate(uuids):
            uuid_state = self.uuids(uuid)
            seq_len = int(mask[i].sum().item())
            blocks_cnt = blocks_for_tokens(seq_len, self.block_size)
            indexes = self.get_free_blocks(blocks_cnt)
            pages_K, pages_V = self._get_pages(indexes)
            for j in range(seq_len):
                block_idx = j // self.block_size
                in_block_idx = j % self.block_size
                pages_K[block_idx][:, in_block_idx, :] = K[i, :, j, :]
                pages_V[block_idx][:, in_block_idx, :] = V[i, :, j, :]
            
            for b in range(blocks_cnt):
                indexes[b] = self.radix_insert(uuid, indexes[b], toks[i, b*self.block_size:(b+1)*self.block_size].tolist())

            uuid_state.insert_prefill(
                self,
                commited_blocks=indexes[:-1], 
                write_block=indexes[-1], 
                write_block_occupancy=seq_len % self.block_size,
                write_block_toks=toks[i][seq_len % self.block_size])

    def append_and_fetch(self, uuids, K, V, toks):
        max_len = 0
        batch_size = 0
        lengths = []
        pages_info = []
        for i, uuid in enumerate(uuids):
            uuid_state = self.uuids[uuid]
            uuid_state.insert_single(toks[i], K[i, :, 0, :], V[i, :, 0, :])
            length = (uuid_state.blocks_count - 1) * self.block_size + uuid_state.write_block_occupancy
            max_len = max(max_len, length)
            lengths.append(length)
            batch_size += 1
            pages_info += [uuid_state.get_pages_info()]
        return pages_info

    def release(self, uuids):
        for uuid in uuids:
            self._free_blocks(uuid)
