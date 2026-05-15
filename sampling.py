import torch

@torch.inference_mode()
def top_k_top_p_sample(
    logits: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Accepts logits shaped [B, V].
    Returns next token ids shaped [B].
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B,V], got {tuple(logits.shape)}")

    if temperature != 1.0:
        logits = logits / temperature

    V = logits.size(-1)

    # Top-K
    if top_k is not None and 0 < top_k < V:
        kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits = torch.where(
            logits < kth_vals, torch.full_like(logits, float("-inf")), logits
        )

    # Top-P
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    if top_p < 1.0:
        mask = cumulative_probs > top_p
        mask[..., 0] = False
        sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    sampled = torch.multinomial(sorted_probs, 1)  # [B,1]
    next_token = sorted_indices.gather(-1, sampled)  # [B,1]
    return next_token.squeeze(-1)  # [B]
