from einops import rearrange
import torch
import math

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
    def __init__(self, head_dim:int, max_seq_len:int, base:float):
        self.head_dim = head_dim
        dim_half = head_dim // 2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_half, dtype=torch.float32) / dim_half))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)   # [L, D/2]
        self.cos = freqs.cos()    # [L, D/2]
        self.sin = freqs.sin()

    def __call__(self, x:torch.Tensor, position=None):
        # x: [B, H, L, D]
        B, H, L, D = x.shape
        half = D // 2
        if position is None: position = torch.arange(L, device=x.device)
        cos = self.cos[position].to(x.device)[None,None,:,:]
        sin = self.sin[position].to(x.device)[None,None,:,:]
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
                 out_proj_weights, out_proj_bias):
        self.KV_cache = STUBKVCache()
        q_out_shape, _ = q_weights.shape
        kv_out_shape, _ = k_weights.shape
        self.num_kv_heads = num_kv_heads
        self.num_attn_heads= num_attn_heads
        self.rope = rope
        self.head_dim = q_out_shape // self.num_attn_heads
        
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
        self.KV_cache.prefill(uuid, K, V)
        
        Qh = rearrange(Q, "b l (h d) -> b h l d", h=self.num_attn_heads)
        Kh = rearrange(K, "b l (h d) -> b h l d", h=self.num_kv_heads)
        Vh = rearrange(V, "b l (h d) -> b h l d", h=self.num_kv_heads)

        if self.rope is not None:
            Qh = self.rope(Qh)
            Kh = self.rope(Kh)

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


    def _generate(self, x:torch.Tensor, prefill=False, uuid=None):
        Q = self.q_weights(x)
        K = self.k_weights(x)
        V = self.v_weights(x)
        Q = Q + self.q_bias if self.q_bias is not None else Q
        K = K + self.k_bias if self.k_bias is not None else K
        V = V + self.v_bias if self.v_bias is not None else V
        K,V = self.KV_cache.append_and_fetch(uuid, K, V)
        Qh = rearrange(Q, "b l (h d) -> b h l d", h=self.num_attn_heads)
        Kh = rearrange(K, "b l (h d) -> b h l d", h=self.num_kv_heads)
        Vh = rearrange(V, "b l (h d) -> b h l d", h=self.num_kv_heads)
        if self.rope is not None:
            Qh = self.rope(Qh)
            Kh = self.rope(Kh)

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
    
class STUBKVCache:
    def __init__(self):
        self.uuid_to_tensors = {}

    def prefill(self, uuid, K, V):
        self.uuid_to_tensors[uuid] = [K, V]

    def append_and_fetch(self, uuid, K, V):
        K_old, V_old = self.uuid_to_tensors[uuid]
        if K.dim() == K_old.dim() - 1:
            K = K.unsqueeze(0)
            V = V.unsqueeze(0)

        K_new = torch.cat([K_old, K], dim=0)
        V_new = torch.cat([V_old, V], dim=0)

        self.uuid_to_tensors[uuid] = [K_new, V_new]
        return K_new, V_new
