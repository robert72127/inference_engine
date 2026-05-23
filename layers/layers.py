from einops import rearrange
import torch
import math

from kvcache import PagedKVCache
from triton_kernels import paged_mqa_decode

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
                 cache_blocks_cnt,
                 cache_block_size,
                ):
        q_out_shape, _ = q_weights.shape
        kv_out_shape, _ = k_weights.shape
        self.num_kv_heads = num_kv_heads
        self.num_attn_heads= num_attn_heads
        self.rope = rope
        self.head_dim = q_out_shape // self.num_attn_heads
        self.max_seq_len = max_seq_len
        self.KV_cache = PagedKVCache(
            d_head=self.head_dim, 
            head_cnt=self.num_kv_heads,
            num_blocks = cache_blocks_cnt,
            block_size = cache_block_size,
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
        
        Qh = rearrange(Q, "b l (h d) -> b h l d", h=self.num_attn_heads)
        Kh = rearrange(K, "b l (h d) -> b h l d", h=self.num_kv_heads)
        Vh = rearrange(V, "b l (h d) -> b h l d", h=self.num_kv_heads)

        if self.rope is not None:
            positions = torch.arange(Qh.size(2), device=Qh.device).unsqueeze(0).expand(Qh.size(0), -1)
            Qh = self.rope(Qh, position=positions)
            Kh = self.rope(Kh, position=positions)

        # prefills cache from given K,V
        self.KV_cache.prefill(uuid, Kh, Vh, mask)
        
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

    def _generate(self, x:torch.Tensor, uuid):
        Q = self.q_weights(x)
        K = self.k_weights(x)
        V = self.v_weights(x)
        Q = Q + self.q_bias if self.q_bias is not None else Q
        K = K + self.k_bias if self.k_bias is not None else K
        V = V + self.v_bias if self.v_bias is not None else V

        Qh = rearrange(Q, "b l (h d) -> b h l d", h=self.num_attn_heads)
        Kh = rearrange(K, "b l (h d) -> b h l d", h=self.num_kv_heads)
        Vh = rearrange(V, "b l (h d) -> b h l d", h=self.num_kv_heads)
        q_positions = self.KV_cache.get_q_position(uuid)
        if self.rope is not None:
            Qh = self.rope(Qh, position=q_positions)
            Kh = self.rope(Kh, position=q_positions)

        pages_info = self.KV_cache.append_and_fetch(uuid, Kh, Vh)
        seq_len_per_page = self.KV_cache.block_size
        out = torch.zeros(
                    x.shape[0],
                    self.num_attn_heads,
                    self.head_dim,
                    dtype=Q.dtype,
                    device=Q.device,
                )
        page_indexes = [pm["indexes"] for pm in pages_info]

        if x.device == torch.device("cuda"):
            max_len = max(len(x) for x in page_indexes)
            padded_indexes = torch.full(
                (len(page_indexes), max_len),
                fill_value=-1,   # padding value
                dtype=torch.long,
                device=x.device,
            )

            attn_out = paged_mqa_decode(
                q=Qh, K_cache=self.KV_cache.K, V_cache=self.KV_cache.V, out=out,
                page_indexes = page_indexes,
                KV_cache_block_cnt=self.KV_cache.blocks_cnt,
                batch_size = len(uuid),
                tok_cnt= torch.tensor([page["token_count"] for page in pages_info], device=Q.device),
                seq_len_per_page=seq_len_per_page,
                d_head=self.head_dim,
                num_attn_heads=self.num_attn_heads,
                num_kv_heads=self.num_kv_heads,
            )

        else:
            self._paged_mqa_cpu(pages_info, Qh, out)

        out = rearrange(out, "b h d -> b 1 (h d)")
        return self.outproj(out)
    
    def _paged_mqa_cpu(self, pages_info, Qh, out):
        group_size = self.num_attn_heads // self.num_kv_heads
        scale = 1 / math.sqrt(self.head_dim)
        kv_head_idx = torch.arange(self.num_attn_heads, device=Q.device) // group_size

        for idx, state in enumerate(pages_info):
            q = Qh[idx].squeeze(dim=1).to(torch.float32)
            token_count = state["token_count"]
            m_max = torch.full((self.num_attn_heads,), float("-inf"), device=Q.device)
            l = torch.zeros((self.num_attn_heads,), device=Q.device)
            acc = torch.zeros(
                self.num_attn_heads,
                self.head_dim,
                dtype=torch.float32,
                device=Qh.device,
            )
            tokens_seen = 0
            K_pages,V_pages = self.KV_cache._get_pages(state["indexes"])
            for k_block, v_block in zip(K_pages, V_pages):
                valid_tokens = min(self.KV_cache.block_size, token_count - tokens_seen)
                if valid_tokens <= 0:
                    break
                k_block = k_block[:, :valid_tokens, :][kv_head_idx].to(torch.float32)
                v_block = v_block[:, :valid_tokens, :][kv_head_idx].to(torch.float32)
                s = torch.einsum("h d, h s d -> h s", q, k_block) * scale
                m_block = s.max(dim=-1).values
                m_new = torch.maximum(m_max, m_block)
                old_rescale = torch.exp(m_max - m_new)
                m_max = m_new
                weight = torch.exp(s - m_max[:, None])
                l = l * old_rescale + torch.sum(weight, dim=-1)
                acc = acc * old_rescale[:, None] + torch.einsum("h s, h s d -> h d", weight, v_block)
                tokens_seen += valid_tokens
            out[idx] = (acc / l[:, None]).to(out.dtype)

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
