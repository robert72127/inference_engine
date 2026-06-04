import triton
import triton.language as tl
import torch


def paged_mqa_decode(
            q, K_cache, V_cache, out, 
            page_indexes,
            page_index_stride,
            batch_size,
            tok_cnt,
            seq_len_per_page,
            d_head,
            num_attn_heads,
            num_kv_heads,
        ):
    
    grid = (batch_size, num_attn_heads)
    paged_mqa_decode_kernel[grid](
        q, K_cache, V_cache, out, 
        page_indexes, tok_cnt,
        page_index_stride,
        seq_len_per_page, 
        d_head=d_head,
        num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads,
        BLOCK_D=triton.next_power_of_2(d_head),
        BLOCK_T=triton.next_power_of_2(seq_len_per_page),
    )

@triton.jit
def paged_mqa_decode_kernel(
    q_ptr, K_cache, V_cache, out_ptr,
    page_indexes, tok_cnt,
    page_index_stride,
    seq_len_per_page: tl.constexpr,
    d_head: tl.constexpr,
    num_attn_heads: tl.constexpr,
    num_kv_heads : tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    group_size = num_attn_heads // num_kv_heads
    kv_head_idx = head_idx // group_size

    offst_t = tl.arange(0, BLOCK_T)

    tok_cnt = tl.load(tok_cnt + batch_idx)
    pages_cnt = tl.cdiv(tok_cnt, seq_len_per_page)

    q_block_ptr = tl.make_block_ptr(
        q_ptr + (batch_idx * num_attn_heads + head_idx) * d_head,
        shape=(d_head,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    out_block_ptr = tl.make_block_ptr(
        out_ptr + (batch_idx * num_attn_heads + head_idx) * d_head,
        shape=(d_head,),
        strides=(1,),
        offsets=(0,), 
        block_shape=(BLOCK_D,),
        order=(0,),
    )
    q_block = tl.load(q_block_ptr, boundary_check=(0,), padding_option="zeros")

    m_max = -float("inf")
    l = 0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    scale = 1.0 / tl.sqrt(d_head + 0.0)

    i = 0
    while i < pages_cnt:
        idx = tl.load(page_indexes + batch_idx * page_index_stride + i)
        token_base = i * seq_len_per_page
        valid_mask = token_base + offst_t < tok_cnt

        in_cache_offset = idx * num_kv_heads * d_head * seq_len_per_page + kv_head_idx * d_head * seq_len_per_page
        k_block_ptr = tl.make_block_ptr(
            base=K_cache + in_cache_offset,
            shape=(seq_len_per_page, d_head),
            strides=(d_head, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_T, BLOCK_D),
            order=(1, 0),
        )

        v_block_ptr = tl.make_block_ptr(
            base=V_cache + in_cache_offset,
            shape=(seq_len_per_page, d_head),
            strides=(d_head, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_T, BLOCK_D),
            order=(1, 0),
        )
        k_block = tl.load(k_block_ptr, boundary_check=(0,), padding_option="zeros")
        v_block = tl.load(v_block_ptr, boundary_check=(0,), padding_option="zeros")

        scores = tl.sum(k_block * q_block[None, :], axis=1) * scale
        scores = tl.where(valid_mask, scores, -float("inf"))

        m_block = tl.max(scores, axis=0)
        m_new = tl.maximum(m_max, m_block)

        old_rescale = tl.exp(m_max - m_new)
        
        m_max = m_new
        weight = tl.exp(scores - m_max)
        l = l * old_rescale + tl.sum(weight, axis=0)
        acc = acc * old_rescale + tl.sum(weight[:, None] * v_block, axis=0)

        i += 1

    out = acc / l
    
    tl.store(
        out_block_ptr,
        out,
        boundary_check=(0,),
    )

def residual_rms(x, y, gamma):
    batch, seq_len, hidden_dim = x.shape
    grid = (batch, seq_len)
    out_rms = torch.empty_like(x)
    out = torch.empty_like(x)
    residual_rms_kernel[grid](x, y, gamma, out, out_rms, hidden_dim)
    return out, out_rms

@triton.jit
def residual_rms_kernel(x_ptr, y_ptr, rms_weight, out_ptr, out_rms_ptr, HIDDEN_DIM: tl.constexpr):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    seq_len = tl.num_programs(1)
    offset =  HIDDEN_DIM * (batch_idx * seq_len + seq_idx)

    x_block_ptr = tl.make_block_ptr(
        x_ptr + offset,
        shape=(HIDDEN_DIM,),
        strides=(1,),
        offsets=(0,),
        block_shape=(HIDDEN_DIM,),
        order=(0,),
    )
    y_block_ptr = tl.make_block_ptr(
        y_ptr + offset,
        shape=(HIDDEN_DIM,),
        strides=(1,),
        offsets=(0,),
        block_shape=(HIDDEN_DIM,),
        order=(0,),
    )
    gamma_block_ptr = tl.make_block_ptr(
        rms_weight,
        shape=(HIDDEN_DIM,),
        strides=(1,),
        offsets=(0,),
        block_shape=(HIDDEN_DIM,),
        order=(0,),
    )    

    out_block_ptr = tl.make_block_ptr(
        out_ptr + offset,
        shape=(HIDDEN_DIM,),
        strides=(1,),
        offsets=(0,),
        block_shape=(HIDDEN_DIM,),
        order=(0,),
    )
    out_rms_block_ptr = tl.make_block_ptr(
        out_rms_ptr + offset,
        shape=(HIDDEN_DIM,),
        strides=(1,),
        offsets=(0,),
        block_shape=(HIDDEN_DIM,),
        order=(0,),
    )

    x_block = tl.load(x_block_ptr).to(tl.float32)
    y_block = tl.load(y_block_ptr).to(tl.float32)
    gamma_block = tl.load(gamma_block_ptr).to(tl.float32)
    total = x_block + y_block
    square = total * total
    norm = tl.sqrt( (tl.sum(square) / HIDDEN_DIM ) + 1e-6)
    rms = total / norm * gamma_block

    tl.store(out_block_ptr, total)
    tl.store(out_rms_block_ptr, rms)