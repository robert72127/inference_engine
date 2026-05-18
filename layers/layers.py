from einops import rearrange
import torch
import math

from kvcache import KVCache

class Embedding:
    def __init__(self, weights:torch.Tensor):
        self.embedding = torch.nn.Embedding.from_pretrained(weights, freeze=True)
    
    def __call__(self, x:torch.Tensor):
        return self.embedding(x)

class Linear:
    def __init__(self, W:torch.Tensor, b:torch.Tensor|None = None):
        self.weights = W
        self.bias = b

    def __call__(self, x:torch.Tensor):
        return torch.nn.functional.linear(x, self.weights, self.bias)

class SwiGLUMlp:
    def __init__(self, down_proj_weights:torch.Tensor, up_proj_weights:torch.Tensor, gate_proj_weights:torch.Tensor):
        self.down_proj = Linear(down_proj_weights)
        self.up_proj =  Linear(up_proj_weights)
        self.gate_proj = Linear(gate_proj_weights)

    def __call__(self, x:torch.Tensor):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        
class RMSNorm:
    def __init__(self, gamma_weights:torch.Tensor, eps=1e-6):
        self.gamma = gamma_weights
        self.eps = eps

    def __call__(self, x:torch.Tensor):
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return self.gamma * x * torch.rsqrt(rms + self.eps)

class RoPe:
    def __init__(self, head_dim:int, max_seq_len:int, base:float, device: torch.device):
        self.head_dim = head_dim
        dim_half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_half, device=device, dtype=torch.float32) / dim_half))
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)   # [L, D/2]
        self.cos = freqs.cos()    # [L, D/2]
        self.sin = freqs.sin()

    def __call__(self, x:torch.Tensor, position=None):
        # x: [B, H, L, D]
        B, H, L, D = x.shape
        half = D // 2
        if position is None:
            position = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        else:
            position = torch.as_tensor(position, device=x.device)
            if position.shape != (B, L):
                raise ValueError(f"Unsupported RoPE position shape {tuple(position.shape)} for x shape {tuple(x.shape)}")

        cos = self.cos[position].to(x.device)[:, None, :, :]
        sin = self.sin[position].to(x.device)[:, None, :, :]

        # upcast x for RoPE math
        x_f = x.to(torch.float32)
        x1 = x_f[..., :half]
        x2 = x_f[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)

class MultiQueryAttention:
    def __init__(self, num_kv_heads, num_attn_heads, max_seq_len, rope,
                 q_weights, q_bias,
                 k_weights, k_bias,
                 v_weights, v_bias,
                 out_proj_weights, out_proj_bias,
                 cache_max_requests=100,
                 cache_max_seq_len=4096):
        q_out_shape, _ = q_weights.shape
        kv_out_shape, _ = k_weights.shape
        self.num_kv_heads = num_kv_heads
        self.num_attn_heads= num_attn_heads
        self.rope = rope
        self.head_dim = q_out_shape // self.num_attn_heads
        self.max_seq_len = max_seq_len
        self.KV_cache = KVCache(
            d_model=kv_out_shape,
            max_requests=cache_max_requests,
            max_request_len=cache_max_seq_len,
            device=q_weights.device,
            dtype=q_weights.dtype,
        )
        
        self.q_weights = Linear(q_weights)
        self.k_weights = Linear(k_weights)
        self.v_weights = Linear(v_weights)
        self.outproj   = Linear(out_proj_weights, out_proj_bias)
        self.q_bias = None if q_bias is None else q_bias
        self.k_bias = None if k_bias is None else k_bias
        self.v_bias = None if v_bias is None else v_bias

    # todo fix this to properly operate on block ie directly writing on them etc
    def __call__(self, x:torch.Tensor, prefill, mask, uuid):
        if prefill: return self._prefill(x, mask, uuid)
        else: return self._generate(x, uuid)

    def _prefill(self, x:torch.Tensor, mask, uuid):
        Q = self.q_weights(x)
        K = self.k_weights(x)
        V = self.v_weights(x)
        Q = Q + self.q_bias if self.q_bias is not None else Q
        K = K + self.k_bias if self.k_bias is not None else K
        V = V + self.v_bias if self.v_bias is not None else V
        # prefills cache from given K,V
        self.KV_cache.prefill(uuid, K, V, mask)
        
        Qh = rearrange(Q, "b l (h d) -> b h l d", h=self.num_attn_heads)
        Kh = rearrange(K, "b l (h d) -> b h l d", h=self.num_kv_heads)
        Vh = rearrange(V, "b l (h d) -> b h l d", h=self.num_kv_heads)

        if self.rope is not None:
            positions = torch.arange(Qh.size(2), device=Qh.device).unsqueeze(0).expand(Qh.size(0), -1)
            Qh = self.rope(Qh, position=positions)
            Kh = self.rope(Kh, position=positions)

        B, H_q, L, D = Qh.shape
        _, H_kv, _, _ = Kh.shape

        group = H_q // H_kv   # number of query heads per kv head

        Qg = Qh.view(B, H_kv, group, L, D)
        Kg = Kh.unsqueeze(2)
        Vg = Vh.unsqueeze(2)

        attn_scores = torch.matmul(Qg, Kg.transpose(-1, -2)) / math.sqrt(D)
        
        B, H_kv, G, L, _ = attn_scores.shape

        # [L, L] upper-triangular mask where j > i (future positions)
        causal = torch.triu(
            torch.ones(L, L, device=attn_scores.device, dtype=torch.bool),
            diagonal=1,
        )
        
        attn_scores = attn_scores.masked_fill(causal, float("-inf"))
        if mask is not None:
            mask = mask.to(device=attn_scores.device, dtype=torch.bool)
            key_padding = ~mask[:, None, None, None, :]
            attn_scores = attn_scores.masked_fill(key_padding, float("-inf"))
        attn = attn_scores.softmax(dim=-1)

        out_g = torch.matmul(attn, Vg)

        out_heads = rearrange(out_g, "b h_kv g l d -> b l (h_kv g d)")
        out = self.outproj(out_heads)
        if mask is not None:
            out = out * mask[..., None].to(out.dtype)
        return out


    def _generate(self, x:torch.Tensor, uuid=None):
        Q = self.q_weights(x)
        K = self.k_weights(x)
        V = self.v_weights(x)
        Q = Q + self.q_bias if self.q_bias is not None else Q
        K = K + self.k_bias if self.k_bias is not None else K
        V = V + self.v_bias if self.v_bias is not None else V
        K, V, q_positions, k_positions, kv_mask = self.KV_cache.append_and_fetch(uuid, K, V)
        Qh = rearrange(Q, "b l (h d) -> b h l d", h=self.num_attn_heads)
        Kh = rearrange(K, "b l (h d) -> b h l d", h=self.num_kv_heads)
        Vh = rearrange(V, "b l (h d) -> b h l d", h=self.num_kv_heads)
        if self.rope is not None:
            Qh = self.rope(Qh, position=q_positions)
            Kh = self.rope(Kh, position=k_positions)

        B, H_q, L, D = Qh.shape
        _, H_kv, _, _ = Kh.shape
        group = H_q // H_kv   # number of query heads per kv head
        Qg = Qh.view(B, H_kv, group, L, D)
        Kg = Kh.unsqueeze(2)
        Vg = Vh.unsqueeze(2)
        
        attn_scores = torch.matmul(Qg, Kg.transpose(-1, -2)) / math.sqrt(D)
        B, H_kv, G, L, _ = attn_scores.shape

        causal = torch.triu(torch.ones(L, L, device=attn_scores.device, dtype=torch.bool),diagonal=1,)
        attn_scores = attn_scores.masked_fill(causal, float("-inf"))
        if kv_mask is not None:
            kv_mask = kv_mask.to(device=attn_scores.device, dtype=torch.bool)
            key_padding = ~kv_mask[:, None, None, None, :]
            attn_scores = attn_scores.masked_fill(key_padding, float("-inf"))
        attn = attn_scores.softmax(dim=-1)
        out_g = torch.matmul(attn, Vg)
        out_heads = rearrange(out_g, "b h_kv g l d -> b l (h_kv g d)")
        return self.outproj(out_heads)

class TransformerBlock:
    def __init__(self, mlp, pre_norm, attention, post_norm):
        self.mlp = mlp
        self.pre_norm = pre_norm
        self.attention = attention
        self.post_norm = post_norm

    def __call__(self, x:torch.Tensor, prefill, mask, uuid):
        attn_in = self.pre_norm(x)
        x = x + self.attention(attn_in, prefill, mask, uuid)
        mlp_in = self.post_norm(x)
        return  x + self.mlp(mlp_in)

# todo modify model processor to admit reject new requests based on usage of cache
# before admiting new req: needed_blocks = ceil((prompt_len + max_new_tokens) / block_size)

from collections import deque

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
                    K[i][j] = self.kv_cache.K[blck][j]
                    V[i][j] = self.kv_cache.V[blck][j]
                    i+= 1
            for j in range(self.write_block_occupancy):
                K[i][j] = self.kv_cache.K[self.write_block][j]
                V[i][j] = self.kv_cache.V[self.write_block][j]

    def __init__(self, d_model, num_blocks, block_size, device, dtype):
        self.d_model = d_model
        self.blocks_cnt = num_blocks
        self.block_size = block_size

        self.uuids = {}

        self.free_slots = deque([i for i in range (num_blocks)])
        self.fr// soee_slots_cnt = num_blocks

        self.device = device
        self.dtype = dtype
        self.K = torch.zeros((num_blocks, block_size, d_model), device=device, dtype=dtype)
        self.V = torch.zeros((num_blocks, block_size, d_model), device=device, dtype=dtype)

    # can be like computed by how much blocks total can current request count take
    def get_occupancy_info(self):
        return self.size, self.free_slots_cnt
    
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

        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        return K, V, mask
    
    def release(self, uuids):
        for uuid in uuids:
            self._free_blocks(uuid)