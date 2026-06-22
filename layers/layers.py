from einops import rearrange
import torch
from torch.nn.utils.rnn import pad_sequence
import math

from kvcache import RadixKVCache
from layers.triton_kernels import paged_mqa_decode, residual_rms
from models.model import PrefillStateBuff, DecodeStateBuff

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

class ResidualRMSNorm:
    def __init__(self, gamma_weights:torch.Tensor, eps=1e-6):
        self.rms_norm = RMSNorm(gamma_weights, eps)

    def __call__(self, x:torch.Tensor, y:torch.Tensor):
        total = x + y
        return total, self.rms_norm(total)

class ResidualRMSNormCuda:
    def __init__(self, gamma_weights:torch.Tensor):
        self.gamma = gamma_weights

    def __call__(self, x:torch.Tensor, y:torch.Tensor):
        return residual_rms(x, y, self.gamma)

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
                 max_requests_per_uuid,
                ):
        q_out_shape, _ = q_weights.shape
        kv_out_shape, _ = k_weights.shape
        self.num_kv_heads = num_kv_heads
        self.num_attn_heads= num_attn_heads
        self.rope = rope
        self.head_dim = q_out_shape // self.num_attn_heads
        self.max_seq_len = max_seq_len
        self.KV_cache = RadixKVCache(
            d_head=self.head_dim, 
            head_cnt=self.num_kv_heads,
            num_blocks = cache_blocks_cnt,
            block_size = cache_block_size,
            max_requests_per_uuid=max_requests_per_uuid,
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


    def attention_prologue(self, x, rope_positions):
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
            Qh = self.rope(Qh, position=rope_positions)
            Kh = self.rope(Kh, position=rope_positions)

        return Qh, Kh, Vh


    def prefill(self, x, input_meta:PrefillStateBuff):
        request_slots = input_meta.request_slots
        tokens = input_meta.tokens
        batch_size = x.size(0)
        seq_len = int(input_meta.seq_lens[0].item())
        x = x[:, :seq_len, :]
        tokens = tokens[:, :seq_len]
        positions = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        rope_positions = input_meta.offsets[:, None] + positions[None, :]
        
        Qh, Kh, Vh = self.attention_prologue(x, rope_positions)

        page_indexes = []
        for idx in range(0, batch_size):
            request_slot = int(request_slots[idx].item())
            self.KV_cache.append_prefill(
                request_slot,
                tokens[idx],
                Kh[idx],
                Vh[idx],
            )
            page_indexes.append(self.KV_cache.get_indexes(request_slot))

        out = torch.zeros(
            batch_size,
            self.num_attn_heads,
            seq_len,
            self.head_dim,
            dtype=x.dtype,
            device=x.device,
        )
        self._paged_prefill_cpu(input_meta.offsets, input_meta.seq_lens, page_indexes, batch_size, Qh, out)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.outproj(out)


    def decode(self, x:torch.Tensor, input_meta:DecodeStateBuff):
        request_slots = input_meta.request_slots
        batch_size = x.shape[0]
        tokens = input_meta.tokens       
        rope_positions = input_meta.offsets[:, None]
        Qh, Kh, Vh = self.attention_prologue(x, rope_positions)

        page_indexes = []
        for idx in range(0, batch_size):
            request_slot = int(request_slots[idx].item())
            self.KV_cache.append_decode(request_slot, Kh[idx, :, 0, :], Vh[idx, :, 0, :], tokens[idx])
            page_indexes.append(self.KV_cache.get_indexes(request_slot))

        token_counts = input_meta.offsets + 1
        out = torch.zeros(
                    batch_size,
                    self.num_attn_heads,
                    self.head_dim,
                    dtype=x.dtype,
                    device=x.device,
                )

        if x.device.type == "cuda":
            page_index_tensors = [
                torch.tensor(indexes, dtype=torch.int32, device=x.device)
                for indexes in page_indexes
            ]
            padded_indexes = pad_sequence(page_index_tensors, batch_first=True, padding_value=-1)
            paged_mqa_decode(
                q=Qh, K_cache=self.KV_cache.K, V_cache=self.KV_cache.V, out=out,
                page_indexes = padded_indexes,
                page_index_stride = padded_indexes.stride(0),
                batch_size = batch_size,
                tok_cnt= token_counts,
                seq_len_per_page=self.KV_cache.block_size,
                d_head=self.head_dim,
                num_attn_heads=self.num_attn_heads,
                num_kv_heads=self.num_kv_heads,
            )
        else:
            self._paged_decode_cpu(token_counts, page_indexes, batch_size, Qh, out)
        out = rearrange(out, "b h d -> b 1 (h d)")
        return self.outproj(out)
    
    def _paged_decode_cpu(self, token_counts, page_indexes, batch_size, Qh, out):
        group_size = self.num_attn_heads // self.num_kv_heads
        scale = 1 / math.sqrt(self.head_dim)
        kv_head_idx = torch.arange(self.num_attn_heads, device=Qh.device) // group_size

        for idx in range(batch_size):
            q = Qh[idx, :, 0, :].to(torch.float32)
            token_count = int(token_counts[idx].item())
            m_max = torch.full((self.num_attn_heads,), float("-inf"), device=Qh.device)
            l = torch.zeros((self.num_attn_heads,), device=Qh.device)
            acc = torch.zeros(
                self.num_attn_heads,
                self.head_dim,
                dtype=torch.float32,
                device=Qh.device,
            )
            tokens_seen = 0
            K_pages, V_pages = self.KV_cache.get_pages(page_indexes[idx])
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

    def _paged_prefill_cpu(self, offsets, seq_lens, page_indexes, batch_size, Qh, out):
        group_size = self.num_attn_heads // self.num_kv_heads
        scale = 1 / math.sqrt(self.head_dim)
        kv_head_idx = torch.arange(self.num_attn_heads, device=Qh.device) // group_size

        for idx in range(batch_size):
            offset = int(offsets[idx].item())
            seq_len = int(seq_lens[idx].item())
            token_count = offset + seq_len
            if seq_len <= 0:
                continue

            q = Qh[idx, :, :seq_len, :].to(torch.float32)
            query_limits = offset + torch.arange(seq_len, device=Qh.device)
            m_max = torch.full((self.num_attn_heads, seq_len), float("-inf"), device=Qh.device)
            l = torch.zeros((self.num_attn_heads, seq_len), device=Qh.device)
            acc = torch.zeros(
                self.num_attn_heads,
                seq_len,
                self.head_dim,
                dtype=torch.float32,
                device=Qh.device,
            )

            tokens_seen = 0
            K_pages, V_pages = self.KV_cache.get_pages(page_indexes[idx])
            for k_block, v_block in zip(K_pages, V_pages):
                valid_tokens = min(self.KV_cache.block_size, token_count - tokens_seen)
                if valid_tokens <= 0:
                    break

                k_block = k_block[:, :valid_tokens, :][kv_head_idx].to(torch.float32)
                v_block = v_block[:, :valid_tokens, :][kv_head_idx].to(torch.float32)
                scores = torch.einsum("h q d, h s d -> h q s", q, k_block) * scale

                key_positions = tokens_seen + torch.arange(valid_tokens, device=Qh.device)
                causal_mask = key_positions[None, :] <= query_limits[:, None]
                scores = scores.masked_fill(~causal_mask[None, :, :], float("-inf"))

                m_block = scores.max(dim=-1).values
                m_new = torch.maximum(m_max, m_block)
                old_rescale = torch.exp(m_max - m_new)
                m_max = m_new
                weight = torch.exp(scores - m_max[:, :, None])
                l = l * old_rescale + torch.sum(weight, dim=-1)
                acc = acc * old_rescale[:, :, None] + torch.einsum("h q s, h s d -> h q d", weight, v_block)
                tokens_seen += valid_tokens

            out[idx, :, :seq_len, :] = (acc / l[:, :, None]).to(out.dtype)

class TransformerBlock:
    def __init__(self, mlp, pre_norm, attention, residual_rms):
        self.mlp = mlp
        self.pre_norm = pre_norm
        self.attention = attention
        self.residual_rms = residual_rms

    def decode(self, x:torch.Tensor, input_meta:DecodeStateBuff):
        attn_in = self.pre_norm(x)
        x, mlp_in = self.residual_rms(x, self.attention.decode(attn_in, input_meta))
        return x + self.mlp(mlp_in)

    def prefill(self, x, input_meta:PrefillStateBuff):
        seq_len = int(input_meta.seq_lens[0].item())
        x = x[:, :seq_len, :]
        attn_in = self.pre_norm(x)
        x, mlp_in = self.residual_rms(x, self.attention.prefill(attn_in, input_meta))
        return x + self.mlp(mlp_in)


