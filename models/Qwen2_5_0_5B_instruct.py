from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import json
import os
from pathlib import Path

from models.model import Model, ModelForwardState
from kvcache import blocks_for_tokens
from layers.layers import (
    Embedding,
    RMSNorm,
    ResidualRMSNorm,
    ResidualRMSNormCuda,
    SwiGLUMlp,
    MultiQueryAttention,
    RoPe,
    Linear,
    TransformerBlock,
)


import torch

repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
model_dir = Path(snapshot_download(repo_id=repo_id))  # user-agnostic
CACHE_BLOCK_SIZE = 256

def get_model_dir():
    return model_dir

config = model_dir / "config.json"
model_file = model_dir / "model.safetensors"

with open(config, "r") as f:
    cfg = json.load(f)

def print_model():
    model = model_dir / "model.safetensors"
    tensors = load_file(model)
    for name, arr in tensors.items():
        print(name, arr.shape, arr.dtype)

#print_model()

class ModelConfig:
    def __init__(self, cfg):
        self.dtype = None
        dtype = cfg["torch_dtype"]
        match dtype:
            case "bfloat16":
                self.dtype = torch.bfloat16
            case "float16":
                self.dtype = torch.float16
  
        self.n_kv_heads = cfg["num_key_value_heads"]
        self.n_attn_heads = cfg["num_attention_heads"]
        self.n_layers = cfg["num_hidden_layers"]
        self.hidden_size = cfg["hidden_size"]
        self.rms_eps = cfg['rms_norm_eps']
        self.rope_theta = cfg['rope_theta']
        self.max_seq_len = cfg['sliding_window']

    @property
    def dtype_size(self):
        return torch.empty((), dtype=self.dtype).element_size()

def parse_mlp(mlp_tensors):
    down_weights = gate_weights = up_weights = None
    
    for it in mlp_tensors:
        name = it["layer"]
        arr = it["array"]

        match name[0]:
            case "down_proj": down_weights = arr
            case "gate_proj": gate_weights = arr
            case "up_proj": up_weights = arr
            case "_": raise Exception("unknown layer, aborting")

    return SwiGLUMlp(down_proj_weights=down_weights, gate_proj_weights=gate_weights, up_proj_weights=up_weights)

def parse_attn(
    attn_tensors,
    num_kv_heads,
    num_attn_heads,
    max_seq_len,
    rope,
    cache_blocks_cnt,
    cache_block_size,
    max_requests_per_uuid,
):
    k_proj_bias = k_proj_weight = None
    v_proj_bias = v_proj_weight = None
    q_proj_bias = q_proj_weight = None
    o_proj_bias = o_proj_weight = None

    for it in attn_tensors:
        name = it["layer"]
        arr  = it["array"]

        match name[0]:
            case "k_proj":
                if name[1] == "weight": k_proj_weight = arr
                elif name[1] == "bias": k_proj_bias = arr
                else: raise Exception("unknown layer, aborting")
            case "q_proj":
                if name[1] == "weight": q_proj_weight = arr
                elif name[1] == "bias": q_proj_bias = arr
                else: raise Exception("unknown layer, aborting")
            case "v_proj":
                if name[1] == "weight": v_proj_weight = arr
                elif name[1] == "bias": v_proj_bias = arr
                else: raise Exception("unknown layer, aborting")
            case "o_proj":
                if name[1] == "weight": o_proj_weight = arr
                elif name[1] == "bias": o_proj_bias = arr
                else: raise Exception("unknown layer, aborting")
            case "_": raise Exception("unknown layer, aborting")

    return MultiQueryAttention(
        num_kv_heads=num_kv_heads, num_attn_heads=num_attn_heads,
        max_seq_len=max_seq_len, rope=rope,
        q_weights=q_proj_weight, q_bias=q_proj_bias,
        k_weights=k_proj_weight, k_bias=k_proj_bias,
        v_weights=v_proj_weight, v_bias=v_proj_bias,
        out_proj_weights=o_proj_weight, out_proj_bias=o_proj_bias,
        cache_blocks_cnt=cache_blocks_cnt,
        cache_block_size=cache_block_size,
        max_requests_per_uuid=max_requests_per_uuid,
    )

def cache_blocks_cnt_from_budget(
    cfg: ModelConfig,
    tensors: dict[str, torch.Tensor],
    cache_block_size: int,
    cache_max_requests: int,
    cache_max_seq_len: int,
    memory_available: int,
):
    max_blocks_cnt = cache_max_requests * blocks_for_tokens(cache_max_seq_len, cache_block_size)
    model_bytes = sum(arr.numel() * arr.element_size() for arr in tensors.values())
    input_bytes = cache_max_seq_len * cfg.hidden_size * cfg.dtype_size
    memory_left = memory_available - model_bytes - input_bytes
    kv_dim = cfg.hidden_size * cfg.n_kv_heads // cfg.n_attn_heads
    bytes_per_block = 2 * cfg.n_layers * cache_block_size * kv_dim * cfg.dtype_size
    if bytes_per_block <= 0:
        raise RuntimeError("Invalid cache block geometry")
    if memory_left <= 0:
        raise RuntimeError("Not enough memory left for KV cache")
    return int(min(max_blocks_cnt, memory_left // bytes_per_block))

# generates tensor processing graph from safetensor
def parse_model(
    model_dir: Path,
    cfg: ModelConfig,
    device: torch.device,
    cache_max_requests: int,
    cache_max_seq_len: int,
    memory_available: int,
):
    cache_block_size = CACHE_BLOCK_SIZE
    max_requests_per_uuid = blocks_for_tokens(cache_max_seq_len, cache_block_size)
    tensors = load_file(model_dir, device=str(device))
    cache_blocks_cnt = cache_blocks_cnt_from_budget(
        cfg,
        tensors,
        cache_block_size,
        cache_max_requests,
        cache_max_seq_len,
        memory_available,
    )
    print("Cache blocks count:", cache_blocks_cnt)
    if cache_blocks_cnt == 0:
        raise RuntimeError("KV cache budget allows zero cache blocks")

    hidden_layers = {}
    model = []
    kv_caches = []
    rope_head_dim = None

    for name, arr in tensors.items():
        chunks = name.split(".")
        if chunks[0] != "model":
            raise Exception("Unexpected")
                 
        name = chunks[1]
        rest = chunks[2:]

        match name:
            case "embed_tokens":
                model += [(Embedding(arr), False)]
                output_embed = Linear(arr)
            case "norm" :
                output_norm = RMSNorm(arr, cfg.rms_eps)
            case "layers": # add to layers
                pos = int(rest[0])
                layer = rest[1]
                rest = rest[2:]
                if pos not in hidden_layers:
                    hidden_layers[pos] = {}
                hidden_layers[pos].setdefault(layer, []).append({"layer": rest, "array": arr})
                if layer == "self_attn" and rest[0] == "q_proj" and rest[1] == "weight" and rope_head_dim is None:
                    rope_head_dim = arr.shape[0] // cfg.n_attn_heads
            case _ : raise Exception("Unknown layer, aborting")

    if rope_head_dim is None:
        raise Exception("Missing q_proj weights, cannot initialize RoPE")

    rope = RoPe(rope_head_dim, cfg.max_seq_len, cfg.rope_theta, device)

    # parse layers
    for layer in dict(sorted(hidden_layers.items())).values():
        input_layernorm, post_attention_layernorm, mlp, self_attn = (None, None, None, None)
        for sub_layer in layer:
            match sub_layer:
                case "input_layernorm":
                    input_layernorm = RMSNorm(layer[sub_layer][0]['array'], eps=cfg.rms_eps) 
                case "post_attention_layernorm":
                    if device.type == "cuda":
                        residual_rms = ResidualRMSNormCuda(layer[sub_layer][0]['array'])
                    else:
                        residual_rms = ResidualRMSNorm(layer[sub_layer][0]['array'], eps=cfg.rms_eps) 
                case "mlp":
                    mlp = parse_mlp(layer[sub_layer])
                case "self_attn":
                    self_attn = parse_attn(
                        layer[sub_layer],
                        cfg.n_kv_heads,
                        cfg.n_attn_heads,
                        cfg.max_seq_len,
                        rope,
                        cache_blocks_cnt,
                        cache_block_size,
                        max_requests_per_uuid,
                    )
                    kv_caches.append(self_attn.KV_cache)
                case _: raise Exception("Unknown layer, aborting")
        model += [(TransformerBlock(mlp, input_layernorm, self_attn, residual_rms), True)] 

    output_layers = [output_norm, output_embed]


    return model, output_layers, kv_caches

class Qwen2_5_0_5B_Instruct(Model):
    model_dir = model_dir
    
    def __init__(
        self,
        device: torch.device,
        cache_max_requests: int,
        cache_max_seq_len: int,
        memory_available: int,
    ):
        model_file = model_dir / "model.safetensors"
        config = model_dir / "config.json"
        with open(config, "r") as f: cfg = json.load(f)
        model_cfg = ModelConfig(cfg)
        self.device = torch.device(device)
        self.model_dir = model_dir 
        self.layers, self.output_layers, self.kv_caches = parse_model(
            model_file,
            model_cfg,
            self.device,
            cache_max_requests=cache_max_requests,
            cache_max_seq_len=cache_max_seq_len,
            memory_available=memory_available,
        )
  
    @torch.inference_mode()
    def __call__(self, input: torch.Tensor, state: ModelForwardState) -> torch.Tensor:
        tokens = state.tokens.to(self.device)
        mask = None if state.mask is None else state.mask.to(self.device)
        out = input.to(self.device)
        for layer in self.layers: 
            layer, is_transformer = layer
            out = layer(out, state) if is_transformer else layer(out)
        if state.prefill_state is not None:
            out = torch.stack(
                [
                    out[idx, int(prefill.length.item()) - 1]
                    for idx, prefill in enumerate(state.prefill_state)
                ]
            )
        else:
            out = out[:, -1, :]
        if state.prefill:
            out = out[[prefill.last_chunk for prefill in state.prefill_state]]
        for layer in self.output_layers:
            out = layer(out)
        return out

    def get_model_dir(self):
        return self.model_dir

if __name__ == '__main__':
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    memory_available = pages * page_size
    model = Qwen2_5_0_5B_Instruct(
        torch.device("cpu"),
        cache_max_requests=100,
        cache_max_seq_len=4096,
        memory_available=memory_available,
    )
    print_model()
