from collections import deque
from collections import OrderedDict
import torch

def blocks_for_tokens(token_count: int, block_size: int):
    blocks_cnt = (token_count + block_size - 1) // block_size
    if token_count > 0 and token_count % block_size == 0:
        blocks_cnt += 1
    return blocks_cnt

class RadixTree:
    def __init__(self, kv_cache):
        self.root = self.RadixNode(page_id=None, key=None)
        self.uuid_to_last_node = {}

    def get_last_node(self, uuid):
        return self.uuid_to_last_node[uuid]

    def add_node(self, uuid, node):
        self.uuid_to_last_node[uuid] = node

    def remove_uuid(self, uuid):
        self.uuid_to_last_node.pop(uuid, None)

    def remove_node(self, node):
        if node == self.root:
            return
        for child in node.children:
            self.remove_node(child)
        if node.parent:
            node.parent.children.remove(node)
        del node

    def insert(self, uuid, block_idx, key):
        node = self.get_last_node(uuid)
        for child in node.children:
            if child.key == key:
                self.uuid_to_last_node[uuid] = child
                return child.page_id
        new_node = self.RadixNode(page_id=block_idx, key=key)
        new_node.parent = node
        node.children.append(new_node)
        self.uuid_to_last_node[uuid] = new_node
        return new_node
        
    def search(self, toks):
        return self.root.search(toks, [])
    
    class RadixNode:
        def __init__(self, page_id, key):
            self.key=key
            self.page_id = page_id
            self.parent = None
            self.children = []

        def search(self, key_lst, nodes_list):
            if not key_lst:
                return nodes_list
            for child in self.children:
                if child.key == key_lst[0]:
                    return child.search(key_lst[1:], nodes_list + [child])
            return nodes_list

class BlockAllocator:
    def __init__(self, num_blocks):
        self.locked_slots = {}
        
        self.evictable_blocks = deque()
        
        self.free_slots = deque([i for i in range (num_blocks)])
        self.free_slots_cnt = num_blocks

    def incr_lock_count(self, blocks):
        for block in blocks:
            if block in self.evictable_blocks:
                self.evictable_blocks.remove(block)
                self.locked_slots[block] =  1
            else:
                self.locked_slots[block] +=  1

    def free_blocks(self, blocks):
        # by traversing in revese order lower nodes from radix will be evicted first
        for block in reversed(blocks):
            self.locked_slots[block] -= 1
            if self.locked_slots[block] == 0:
                del self.locked_slots[block]
                self.evictable_blocks.add(block)

    def get_blocks_available(self):
        reserved=0
        for uuid in self.uuids:
            uuid_state = self.uuids[uuid]
            reserved += self.max_requests_per_uuid - uuid_state.blocks_count
        
        return self.free_slots_cnt + len(self.evictable_blocks) - reserved

    def get_free_blocks(self, block_cnt=1):
        blocks = []
        if self.free_slots:
           block = self.free_slots.popleft()
           self.free_slots_cnt -= 1
           blocks.append(block)
        block_cnt -= len(blocks)
        for _ in range(block_cnt):
            blocks.append(self.evictable_blocks.popleft())
        return blocks

class RadixKVCache:
    class UUIDState:
        def __init__(self, 
                    uuid,
                    kv_cache,
                    block_size,
                    blocks_count,
                    commited_blocks):
            
            self.uuid = uuid
            self.kv_cache = kv_cache
            self.block_size = block_size
            self.commited_blocks = commited_blocks

            self.write_block = None
            self.write_block_toks = []

            self.blocks_count = blocks_count
            self.write_block_occupancy = 0

        def get_pages_info(self):
            seq_len = self.block_size * len(self.commited_blocks) + self.write_block_occupancy
            return {
                "indexes": self.commited_blocks + [self.write_block],
                "token_count":seq_len
            }

        def insert_prefill(self, commited_blocks, write_block, write_block_occupancy, write_block_toks):
            self.commited_blocks += commited_blocks
            self.write_block = write_block
            self.write_block_occupancy = write_block_occupancy
            self.write_block_toks = write_block_toks
            self.blocks_count += len(commited_blocks) + 1

        def insert_single(self, tok, K,V):
            if self.write_block_occupancy == self.kv_cache.block_size:
               self.write_block = self.kv_cache.radix_insert(self.uuid, self.write_block, self.write_block_toks)
               self.commited_blocks += [self.write_block]
               self.write_block = self.kv_cache.allocate_blocks()[0]
               self.write_block_occupancy = 0 
               self.blocks_count +=1
               self.write_block_toks = []

            self.kv_cache.K[self.write_block][:, self.write_block_occupancy, :] = K 
            self.kv_cache.V[self.write_block][:, self.write_block_occupancy, :] = V 
            self.write_block_toks.append(int(tok.item()) if torch.is_tensor(tok) else int(tok))
            self.write_block_occupancy+=1

    def __init__(self, d_head, head_cnt, num_blocks, block_size, max_requests_per_uuid=None, device=None, dtype=None):
        self.d_head = d_head
        self.head_cnt = head_cnt
        self.blocks_cnt = num_blocks
        self.block_size = block_size

        self.max_requests_per_uuid = num_blocks if max_requests_per_uuid is None else max_requests_per_uuid
        self.uuids = {}
        
        # blocks are put there when they are used by at least one active uuid
        self.block_to_node = {}

        self.block_allocator = BlockAllocator(num_blocks)

        self.device = device
        self.dtype = dtype
        self.K = torch.zeros((num_blocks,self.head_cnt, block_size, self.d_head), device=device, dtype=dtype)
        self.V = torch.zeros((num_blocks,self.head_cnt, block_size, self.d_head), device=device, dtype=dtype)

        self.radix_tree = RadixTree(self)

    # might return different index if block is already present, then this block will be auto freed, and found block lock_count will be incremented
    def radix_insert(self, uuid, block_idx, toks):
        new_node = self.radix_tree.insert(uuid, block_idx, toks) 
        self.block_to_node[block_idx] = new_node
        new_block_idx = new_node.page_id
        if new_block_idx != block_idx:
            self.block_allocator.free_blocks([block_idx])
        return new_block_idx

    # returns longest list of pages that match prefix, update locked_slots if found pages
    def radix_search(self, uuid, toks):
        nodes = self.radix_tree.search(toks)
        pages = [node.page_id for node in nodes]
        self.block_allocator.incr_lock_count(pages)
        self.radix_tree.add_node(uuid, nodes[-1] if nodes else self.radix_tree.root)
        return pages, len(nodes)

    def _release(self, uuid):
        uuid_state = self.uuids.pop(uuid, None)
        
        self.radix_tree.remove_uuid(uuid)

        blocks = uuid_state.commited_blocks + [uuid_state.write_block]
        self.block_allocator.free_blocks(blocks)

    def release(self, uuids):
        for uuid in uuids:
            self._release(uuid)

    def allocate_blocks(self, block_cnt=1):
        blocks = self.block_allocator.get_free_blocks(block_cnt)
        # if they were in tree remove from tree
        for block in blocks:
            if block in self.block_to_node:
                node = self.block_to_node.pop(block)
                self.radix_tree.remove_node(node)
        self.block_allocator.incr_lock_count(blocks)
        return blocks

    def get_pages(self, indexes):
        K_ = [self.K[ind] for ind in indexes]
        V_ = [self.V[ind] for ind in indexes]
        return K_, V_

    def _block_keys_for_tokens(self, toks):
        if torch.is_tensor(toks):
            toks = toks.tolist()
        return [
            toks[i:i + self.block_size]
            for i in range(0, len(toks) - len(toks) % self.block_size, self.block_size)
        ]

    def get_q_position(self, uuids):
        positions = []
        for uuid in uuids:
            uuid_state = self.uuids[uuid]
            positions.append((uuid_state.blocks_count -1) * self.block_size + uuid_state.write_block_occupancy)
        return torch.tensor(positions, device=self.K.device).unsqueeze(1)

    # first step in prefill, serach for pages matching longest prefix of toks
    def init(self, uuids, toks):
        out = []
        for i, uuid in enumerate(uuids):
            blocks, blocks_cnt = self.radix_search(uuid, self._block_keys_for_tokens(toks[i]))
            self.uuids[uuid] = self.UUIDState(
                uuid,
                self,
                self.block_size,
                blocks_count=blocks_cnt,
                commited_blocks=blocks,
            )
            out.append((blocks, blocks_cnt))
        return out

    def prefill(self, uuids, toks, K, V, mask):
        for i, uuid in enumerate(uuids):
            uuid_state = self.uuids[uuid]
            seq_len = int(mask[i].sum().item())
            cached_tokens = len(uuid_state.commited_blocks) * self.block_size
            if cached_tokens > seq_len:
                raise ValueError("Cached prefix is longer than provided sequence")

            remaining_tokens = seq_len - cached_tokens
            blocks_cnt = blocks_for_tokens(remaining_tokens, self.block_size)
            if blocks_cnt == 0:
                blocks_cnt = 1

            indexes = self.get_free_blocks(blocks_cnt)
            pages_K, pages_V = self.get_pages(indexes)

            for j in range(remaining_tokens):
                src_idx = cached_tokens + j
                block_idx = j // self.block_size
                in_block_idx = j % self.block_size
                pages_K[block_idx][:, in_block_idx, :] = K[i, :, src_idx, :]
                pages_V[block_idx][:, in_block_idx, :] = V[i, :, src_idx, :]

            commited_blocks = indexes[:-1]
            for b, block_idx in enumerate(commited_blocks):
                start = cached_tokens + b * self.block_size
                end = start + self.block_size
                block_toks = toks[i, start:end].tolist()
                commited_blocks[b] = self.radix_insert(uuid, block_idx, block_toks)

            write_block_occupancy = remaining_tokens % self.block_size
            if remaining_tokens > 0 and write_block_occupancy == 0:
                write_block_occupancy = 0
                write_block_toks = []
            else:
                write_start = seq_len - write_block_occupancy
                write_block_toks = toks[i, write_start:seq_len].tolist()

            uuid_state.insert_prefill(
                commited_blocks=commited_blocks, 
                write_block=indexes[-1], 
                write_block_occupancy=write_block_occupancy,
                write_block_toks=write_block_toks,
            )

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

