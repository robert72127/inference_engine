from collections import deque
import torch

def blocks_for_tokens(token_count: int, block_size: int):
    blocks_cnt = (token_count + block_size - 1) // block_size
    if token_count > 0 and token_count % block_size == 0:
        blocks_cnt += 1
    return blocks_cnt

class PrefixBlockTrie:
    class Node:
        def __init__(self, page_id=None, key=None, parent=None):
            self.page_id = page_id
            self.key = key
            self.parent = parent
            self.children = {}

    def __init__(self):
        self.root = self.Node()
        self.uuid_to_last_node = {}

    def set_last_node(self, uuid, node):
        self.uuid_to_last_node[uuid] = node

    def get_last_node(self, uuid):
        return self.uuid_to_last_node.get(uuid, self.root)

    def remove_uuid(self, uuid):
        self.uuid_to_last_node.pop(uuid, None)

    def search(self, block_keys):
        node = self.root
        nodes = []

        for key in block_keys:
            key = tuple(key)
            child = node.children.get(key)
            if child is None:
                break
            nodes.append(child)
            node = child

        return nodes

    def insert_after_uuid(self, uuid, page_id, key):
        parent = self.get_last_node(uuid)
        key = tuple(key)

        child = parent.children.get(key)
        if child is None:
            child = self.Node(page_id=page_id, key=key, parent=parent)
            parent.children[key] = child

        self.uuid_to_last_node[uuid] = child
        return child

    def remove_subtree(self, node):
        if node is self.root:
            return []

        removed = []
        stack = [node]
        removed_set = set()

        while stack:
            n = stack.pop()
            removed.append(n)
            removed_set.add(n)
            stack.extend(n.children.values())

        fallback = node.parent or self.root

        for uuid, last in list(self.uuid_to_last_node.items()):
            cur = last
            while cur is not None:
                if cur in removed_set:
                    self.uuid_to_last_node[uuid] = fallback
                    break
                cur = cur.parent

        if node.parent is not None:
            node.parent.children.pop(node.key, None)

        return removed


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
                self.locked_slots[block] = 1
            else:
                self.locked_slots[block] = self.locked_slots.get(block, 0) + 1

    def free_blocks(self, blocks):
        # by traversing in revese order lower nodes from radix will be evicted first
        for block in reversed(blocks):
            if block is None or block not in self.locked_slots:
                continue
            self.locked_slots[block] -= 1
            if self.locked_slots[block] == 0:
                del self.locked_slots[block]
                self.evictable_blocks.append(block)

    def get_free_blocks(self, block_cnt=1):
        blocks = []
        while len(blocks) < block_cnt and self.free_slots:
            block = self.free_slots.popleft()
            self.free_slots_cnt -= 1
            blocks.append(block)
        while len(blocks) < block_cnt:
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
            self.write_block_occupancy = 0

        def get_indexes_and_occupancy(self):
            indexes = list(self.commited_blocks) # list so we dont accidentaly modify state
            last_block_occp = self.block_size
            if self.write_block is not None:
                indexes.append(self.write_block)
                last_block_occp = self.write_block_occupancy
            return indexes, last_block_occp

        def commit_write_block(self):
            self.write_block = self.kv_cache.radix_insert(self.uuid, self.write_block, self.write_block_toks)
            self.commited_blocks.append(self.write_block)
            self.write_block = None
            self.write_block_occupancy = 0
            self.write_block_toks = []

        def alloc_write_block(self):
            if self.write_block is None:
                self.write_block = self.kv_cache.allocate_blocks()[0]
                self.write_block_occupancy = 0
                self.write_block_toks = []

        def insert_single(self, tok, k,v):
            if self.write_block_occupancy == self.kv_cache.block_size:
                self.commit_write_block()
                self.alloc_write_block()

            self.kv_cache.K[self.write_block][:, self.write_block_occupancy, :] = k 
            self.kv_cache.V[self.write_block][:, self.write_block_occupancy, :] = v
            self.write_block_toks.append(int(tok.item()) if torch.is_tensor(tok) else int(tok))
            self.write_block_occupancy+=1

        def insert_seq(self, toks, k, v):
            src_start = 0
            remaining_tokens = toks.len()
            while remaining_tokens > 0:
                if self.write_block is None:
                    self.alloc_write_block()

                if self.write_block_occupancy == self.kv_cache.block_size:
                    self.commit_write_block()
                    self.alloc_write_block()

                fill_tokens = min(remaining_tokens, self.block_size - self.write_block_occupancy)
                dst_start = self.write_block_occupancy
                dst_end = dst_start + fill_tokens
                src_end = src_start + fill_tokens

                self.kv_cache.K[self.write_block][:, dst_start:dst_end, :] = k[:, src_start:src_end, :]
                self.kv_cache.V[self.write_block][:, dst_start:dst_end, :] = v[:, src_start:src_end, :]
                self.write_block_toks.extend(int(tok) for tok in toks[src_start:src_end].detach().cpu().tolist())
                self.write_block_occupancy = dst_end

                src_start += fill_tokens
                remaining_tokens -= fill_tokens


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

        self.prefix_trie = PrefixBlockTrie()

    # might return different index if block is already present, then this block will be auto freed, and found block lock_count will be incremented
    def trie_insert(self, uuid, block_idx, toks):
        new_node = self.prefix_trie.insert_after_uuid(uuid, block_idx, toks) 
        new_block_idx = new_node.page_id
        self.block_to_node[new_block_idx] = new_node
        
        if new_block_idx != block_idx:
            self.block_allocator.free_blocks([block_idx])
            self.block_allocator.incr_lock_count([new_block_idx])
        
        return new_block_idx

    # returns longest list of pages that match prefix, update locked_slots if found pages
    def trie_search(self, uuid, toks):
        nodes = self.prefix_trie.search(toks)
        pages = [node.page_id for node in nodes]
        self.block_allocator.incr_lock_count(pages)
        self.prefix_trie.set_last_node(uuid, nodes[-1] if nodes else self.prefix_trie.root)
        return pages, len(nodes)

    def release_uuid(self, uuid):
        uuid_state = self.uuids.pop(uuid, None)
        if uuid_state is None:
            self.prefix_trie.remove_uuid(uuid)
            return
        self.prefix_trie.remove_uuid(uuid)

        blocks = uuid_state.commited_blocks + [uuid_state.write_block]
        self.block_allocator.free_blocks(blocks)

    def allocate_blocks(self, block_cnt=1):
        blocks = self.block_allocator.get_free_blocks(block_cnt)
        # if they were in tree remove them
        for block in blocks:
            if block in self.block_to_node:
                node = self.block_to_node.pop(block)
                removed_nodes = self.prefix_trie.remove_subtree(node)
                for n in removed_nodes:
                    self.block_to_node.pop(n.page_id, None)
        self.block_allocator.incr_lock_count(blocks)
        return blocks

    def get_pages(self, indexes):
        K_ = [self.K[ind] for ind in indexes]
        V_ = [self.V[ind] for ind in indexes]
        return K_, V_

    def get_blocks_available(self):
        reserved = 0
        for uuid_state in self.uuids.values():
            reserved += self.max_requests_per_uuid - uuid_state.blocks_count
        return self.block_allocator.free_slots_cnt + len(self.block_allocator.evictable_blocks) - reserved

    def tokens_to_blocks(self, toks):
        return [
            toks[i:i + self.block_size]
            for i in range(0, len(toks) - len(toks) % self.block_size, self.block_size)
        ]

    # first step in prefill, serach for pages matching longest prefix of toks
    def init(self, uuid, toks):
        blocks, blocks_cnt = self.trie_search(uuid, self.tokens_to_blocks(toks))
        self.uuids[uuid] = self.UUIDState(
                uuid,
                self,
                self.block_size,
                blocks_count=blocks_cnt,
                commited_blocks=blocks,
        )

    # k, v as actuall tensors without masking for seq_dim 
    def append_prefill(self, uuid, toks, k, v):
        uuid_state = self.uuids[uuid]
        uuid_state.insert_seq(toks, k,v)

    def append_decode(self, uuid, k, v, tok):        
        uuid_state = self.uuids[uuid]
        uuid_state.insert_single(tok, k, v)
