import triton
import triton.language as tl
import torch


def write_mqa_decode_kv(
            K_cache, V_cache, Kh, Vh,
            page_indexes,
            page_index_stride,
            offsets,
            batch_size,
            seq_len_per_page,
            d_head,
            num_kv_heads,
        ):
    grid = (batch_size, num_kv_heads)
    write_mqa_decode_kv_kernel[grid](
        K_cache, V_cache, Kh, Vh,
        page_indexes, offsets,
        page_index_stride,
        seq_len_per_page,
        d_head=d_head,
        num_kv_heads=num_kv_heads,
        BLOCK_D=triton.next_power_of_2(d_head),
    )

@triton.jit
def write_mqa_decode_kv_kernel(
    K_cache, V_cache, Kh, Vh,
    page_indexes, offsets,
    page_index_stride,
    seq_len_per_page: tl.constexpr,
    d_head: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_D)
    mask = offs_d < d_head

    offset = tl.load(offsets + batch_idx)
    page_slot = offset // seq_len_per_page
    page_offset = offset - page_slot * seq_len_per_page
    page_idx = tl.load(page_indexes + batch_idx * page_index_stride + page_slot)

    src_base = ((batch_idx * num_kv_heads + kv_head_idx) * d_head)
    dst_base = (((page_idx * num_kv_heads + kv_head_idx) * seq_len_per_page + page_offset) * d_head)

    k = tl.load(Kh + src_base + offs_d, mask=mask, other=0.0)
    v = tl.load(Vh + src_base + offs_d, mask=mask, other=0.0)
    tl.store(K_cache + dst_base + offs_d, k, mask=mask)
    tl.store(V_cache + dst_base + offs_d, v, mask=mask)


def write_mqa_prefill_kv(
        K_cache, V_cache, Kh, Vh,
        page_indexes,
        page_index_stride,
        offsets,
        seq_lens,
        batch_size,
        seq_len_per_page,
        chunk_size,
        d_head,
        num_kv_heads
):   

    grid = (batch_size, num_kv_heads, chunk_size)
    write_mqa_prefill_kv_kernel[grid](
        K_cache, V_cache, Kh, Vh,
        page_indexes, offsets, seq_lens,
        page_index_stride,
        seq_len_per_page,
        chunk_size=chunk_size,
        d_head=d_head,
        num_kv_heads=num_kv_heads,
        BLOCK_D=triton.next_power_of_2(d_head),
    )

@triton.jit
def write_mqa_prefill_kv_kernel(
    K_cache, V_cache, Kh, Vh,
    page_indexes, offsets, seq_lens,
    page_index_stride,
    seq_len_per_page: tl.constexpr,
    chunk_size: tl.constexpr,
    d_head: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    token_idx = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_D)
    mask_head = offs_d < d_head

    valid_token = token_idx < tl.load(seq_lens + batch_idx)
    offset = tl.load(offsets + batch_idx) + token_idx
    page_slot = offset // seq_len_per_page
    page_offset = offset - page_slot * seq_len_per_page
    page_idx = tl.load(page_indexes + batch_idx * page_index_stride + page_slot)

    mask_page_present = page_idx != -1
    load_store_mask = valid_token & mask_page_present & mask_head

    src_base = (((batch_idx * num_kv_heads + kv_head_idx) * chunk_size + token_idx) * d_head)
    dst_base = (((page_idx * num_kv_heads + kv_head_idx) * seq_len_per_page + page_offset) * d_head)

    k = tl.load(Kh + src_base + offs_d, mask=load_store_mask, other=0.0)
    v = tl.load(Vh + src_base + offs_d, mask=load_store_mask, other=0.0)
    tl.store(K_cache + dst_base + offs_d, k, mask=load_store_mask)
    tl.store(V_cache + dst_base + offs_d, v, mask=load_store_mask)

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
        page_indexes,
        page_index_stride,
        tok_cnt,
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
    page_indexes,
    page_index_stride,
    tok_cnt,
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

    offs_d = tl.arange(0, BLOCK_D)
    q_base = (batch_idx * num_attn_heads + head_idx) * d_head
    q_block = tl.load(q_ptr + q_base + offs_d, mask=offs_d < d_head, other=0.0)

    m_max = tl.full((), -float("inf"), dtype=tl.float32)
    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    scale = 1.0 / tl.sqrt(d_head + 0.0)

    i = 0
    while i < pages_cnt:
        idx = tl.load(page_indexes + batch_idx * page_index_stride + i)
        token_base = i * seq_len_per_page
        valid_mask = token_base + offst_t < tok_cnt

        in_cache_offset = idx * num_kv_heads * d_head * seq_len_per_page + kv_head_idx * d_head * seq_len_per_page
        
        cache_offsets = offst_t[:, None] * d_head + offs_d[None, :]
        cache_mask = ((offst_t[:, None] < seq_len_per_page) &
                      (offs_d[None, :] < d_head))
        k_block = tl.load(K_cache + in_cache_offset + cache_offsets, mask=cache_mask, other=0.0)
        v_block = tl.load(V_cache + in_cache_offset + cache_offsets, mask=cache_mask, other=0.0)

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
    
    out_base = (batch_idx * num_attn_heads + head_idx) * d_head
    tl.store(
        out_ptr + out_base + offs_d,
        out.to(out_ptr.dtype.element_ty),
        mask=offs_d < d_head,
    )

def paged_mqa_prefill(
            q, 
            K_cache, V_cache, 
            out, 
            page_indexes,
            page_index_stride,
            batch_size,
            offsets,
            seq_lens,
            chunk_size,
            seq_len_per_page,
            d_head,
            num_attn_heads,
            num_kv_heads,
        ):

    BLOCK_SEQ = 4
    grid = (batch_size, num_attn_heads, triton.cdiv(chunk_size, BLOCK_SEQ))
    paged_mqa_prefill_kernel[grid](
        q, K_cache, V_cache, out, 
        page_indexes,
        page_index_stride,
        offsets,
        seq_lens,
        chunk_size,
        seq_len_per_page,
        d_head=d_head,
        num_attn_heads=num_attn_heads,
        num_kv_heads=num_kv_heads,
        BLOCK_D=triton.next_power_of_2(d_head),
        BLOCK_T=triton.next_power_of_2(seq_len_per_page),
        BLOCK_SEQ=BLOCK_SEQ
    )

@triton.jit
def paged_mqa_prefill_kernel(
    q_ptr, K_cache, V_cache, out_ptr,
    page_indexes,
    page_index_stride,
    offsets,
    seq_lens,
    chunk_size: tl.constexpr,
    seq_len_per_page: tl.constexpr,
    d_head: tl.constexpr,
    num_attn_heads: tl.constexpr,
    num_kv_heads : tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    query_block_idx = tl.program_id(2)

    group_size = num_attn_heads // num_kv_heads
    kv_head_idx = head_idx // group_size

    query_offsets = query_block_idx * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dim_offsets = tl.arange(0, BLOCK_D)
    query_valid = query_offsets < tl.load(seq_lens + batch_idx)
    request_offset = tl.load(offsets + batch_idx)
    tok_cnt = request_offset + tl.load(seq_lens + batch_idx)
    pages_cnt = tl.cdiv(tok_cnt, seq_len_per_page)

    q_offsets = (((batch_idx * num_attn_heads + head_idx) * chunk_size +
                  query_offsets[:, None]) * d_head + dim_offsets[None, :])
    q_block = tl.load(
        q_ptr + q_offsets,
        mask=query_valid[:, None] & (dim_offsets[None, :] < d_head),
        other=0.0,
    )

    m_max = tl.full((BLOCK_SEQ,), -float("inf"), dtype=tl.float32)
    l = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SEQ, BLOCK_D), dtype=tl.float32)
    scale = 1.0 / tl.sqrt(d_head + 0.0)
    i = 0
    while i < pages_cnt:
        idx = tl.load(page_indexes + batch_idx * page_index_stride + i)
        key_offsets = tl.arange(0, BLOCK_T)
        key_positions = i * seq_len_per_page + key_offsets

        in_cache_offset = idx * num_kv_heads * d_head * seq_len_per_page + kv_head_idx * d_head * seq_len_per_page
        cache_offsets = key_offsets[:, None] * d_head + dim_offsets[None, :]
        cache_mask = ((key_offsets[:, None] < seq_len_per_page) &
                      (dim_offsets[None, :] < d_head))
        k_block = tl.load(K_cache + in_cache_offset + cache_offsets, mask=cache_mask, other=0.0)
        v_block = tl.load(V_cache + in_cache_offset + cache_offsets, mask=cache_mask, other=0.0)

        scores = tl.sum(q_block[:, None, :] * k_block[None, :, :], axis=2) * scale
        query_positions = request_offset + query_offsets
        attention_mask = (query_valid[:, None] &
                          (key_positions[None, :] < tok_cnt) &
                          (key_positions[None, :] <= query_positions[:, None]))
        scores = tl.where(attention_mask, scores, -float("inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_max, m_block)
        old_rescale = tl.exp(m_max - m_new)
        m_max = m_new
        weight = tl.where(attention_mask, tl.exp(scores - m_max[:, None]), 0.0)
        l = l * old_rescale + tl.sum(weight, axis=1)
        acc = acc * old_rescale[:, None] + tl.sum(weight[:, :, None] * v_block[None, :, :], axis=1)

        i += 1

    out = acc / l[:, None]
    tl.store(
        out_ptr + q_offsets,
        out.to(out_ptr.dtype.element_ty),
        mask=query_valid[:, None] & (dim_offsets[None, :] < d_head),
    )

def residual_rms(x, y, gamma):
    batch, seq_len, hidden_dim = x.shape
    grid = (batch, seq_len)
    out_rms = torch.empty_like(x)
    out = torch.empty_like(x)
    residual_rms_kernel[grid](x, y, gamma, out, out_rms, hidden_dim, BLOCK_H=triton.next_power_of_2(hidden_dim))
    return out, out_rms

@triton.jit
def residual_rms_kernel(
    x_ptr,
    y_ptr,
    rms_weight,
    out_ptr,
    out_rms_ptr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    seq_len = tl.num_programs(1)
    offset =  HIDDEN_DIM * (batch_idx * seq_len + seq_idx)
    range_offsets = tl.arange(0, BLOCK_H)
    mask = range_offsets < HIDDEN_DIM

    x_block = tl.load(x_ptr + offset + range_offsets, mask=mask, other=0.0).to(tl.float32)
    y_block = tl.load(y_ptr + offset + range_offsets, mask=mask, other=0.0).to(tl.float32)
    gamma_block = tl.load(rms_weight + range_offsets, mask=mask, other=0.0).to(tl.float32)
    total = x_block + y_block
    square = total * total
    norm = tl.sqrt( (tl.sum(square) / HIDDEN_DIM ) + 1e-6)
    rms = total / norm * gamma_block

    tl.store(out_ptr + offset + range_offsets, total.to(out_ptr.dtype.element_ty), mask=mask)
    tl.store(out_rms_ptr + offset + range_offsets, rms.to(out_rms_ptr.dtype.element_ty), mask=mask)
