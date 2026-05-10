import torch

'''
So how this works?

For each prefill we first make sure it's present in cache,
and if its not then we run prefill on what's needed and pin all those pages that given prompt requires
'''

# key : list of tokens
# value : block of k/v
# children : list



class Page:
    def __init__(self, k_mem, v_mem, size:int, d_model:int, device:str, dtype):
        self.size = size
        self.elem_size = d_model * dtype().itemsize
        self.k_mem = k_mem
        self.v_mem = v_mem
        self.next_write_pos = 0

    def full(self):
        return self.next_write_pos >= self.size
    
    def insert(self, k_value, v_value):
        if self.full():
            return False
        self.k_mem[self.next_write_pos] = k_value
        self.v_mem[self.next_write_pos] = v_value
        self.next_write_pos += self.elem_size

# ops
# we can prefill

# we can append single token, this either means creating new subnode or updating existing one?
# ok we dont update existing ones and we commit block only once its finished
# so yes append but not update

# we can clean pages up to certain point if trie becomes full


class Trie:
    def __init__(self, key_size: int, key: list[int], value:Page, children: list[Trie|None] = None):
        self.parent = None
        self.key = key
        self.value = value
        self.children = children or []
        self.key_size = key_size 

    # insert new node into the trie
    def append(self, node: Trie):
        self.children.append(node)
        node.parent = self

    # called by reverse mapping when page is evicted, so we can remove it from trie
    def delete(self):
        for child in self.children:
            child.delete()
        if self.parent is not None:
            self.parent.children.remove(self)

    def fetch(self, keys: list[int]):
        def _fetch(node:Trie, blocks:list[Page], missing_keys:list[int]):
            key = keys[:node.key_size]
            if key != node.key:
                return blocks, missing_keys
            for child in node.children:
                if child.key == key:
                    blocks.append(node.value)
                    return _fetch(child, blocks, missing_keys[node.key_size:])
            # case where there is no child
            return blocks, missing_keys

        blocks = []
        return _fetch(self, blocks, keys)

# store pages, reverse mapping ie page to key, so we can evict them when needed
# store info such as page size, key size ie how many toks is single page
# lock pages so that they can't be evicted
# track last access time for blocks
# since there is single prompt, tries will have single root
class KVCache:
    def __init__(self, num_pages:int, page_size:int, d_model, device: str, dtype, max_seq_len:int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.d_model = d_model
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        self.k_cache = torch.zeros((num_pages, page_size, d_model), device=device, dtype=dtype)
        self.v_cache = torch.zeros((num_pages, page_size, d_model), device=device, dtype=dtype)

        self.free_pages = [Page(self.k_cache[i], self.v_cache[i], page_size, d_model, device, dtype) for i in range(num_pages)]

    # todo
    def append_and_fetch(self):
        pass
    def commit_page(self):
        pass
    def evict_page(self):
        pass
    def get_free_page(self):
        pass