from huggingface_hub import snapshot_download
from safetensors.torch import load_file

import json
from pathlib import Path
import os

from model import Model
'''
from layers.layers import (
    Embedding,
    RMSNorm,
    MLP,
    MultiQueryAttention,
    RoPe,
    AsModelInput,
    Linear,
    TransformerBlock,
)
'''
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
model_dir = Path(snapshot_download(repo_id=repo_id))  # user-agnostic

config = model_dir / "config.json"
model_file = model_dir / "model.safetensors"

print(model_dir)

print(config)

with open(config, "r") as f:
    cfg = json.load(f)

print(cfg)

def print_model():

    model = model_dir / "model.safetensors"

    tensors = load_file(model)

    for name, arr in tensors.items():
        print(name, arr.shape, arr.dtype)

#print_model()
'''
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

    return MLP(down_weights=down_weights, gate_weights=gate_weights, up_weights=up_weights)

def parse_attn(attn_tensors, num_kv_heads, num_attn_heads, max_seq_len, rope_theta):
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

    if not hasattr(parse_attn, "rope"):
        head_dim = q_proj_weight.shape[0] // num_attn_heads
        parse_attn.rope = RoPe(head_dim, max_seq_len, rope_theta)

    return MultiQueryAttention(
        num_kv_heads=num_kv_heads, num_attn_heads=num_attn_heads,
        max_seq_len=max_seq_len, rope=parse_attn.rope,
        q_weights=q_proj_weight, q_bias=q_proj_bias,
        k_weights=k_proj_weight, k_bias=k_proj_bias,
        v_weights=v_proj_weight, v_bias=v_proj_bias,
        out_proj_weights=o_proj_weight, out_proj_bias=o_proj_bias,
    )

# generates tensor processing graph from safetensor
def parse_model(model_dir:Path, cfg:ModelConfig):

    tensors = load_file(model_dir)

    hidden_layers = {}
    model = []
    KV_caches = []

    for name, arr in tensors.items():
        chunks = name.split(".")
        if chunks[0] != "model":
            raise Exception("Unexpected")
                 
        name = chunks[1]
        rest = chunks[2:]

        match name:
            case "embed_tokens":
                model += [Embedding(arr)]
                output_embed = AsModelInput(Linear(arr))
            case "norm" :
                output_norm = AsModelInput(RMSNorm(arr, cfg.rms_eps))
            case "layers": # add to layers
                pos = int(rest[0])
                layer = rest[1]
                rest = rest[2:]
                if pos not in hidden_layers:
                    hidden_layers[pos] = {}
                hidden_layers[pos].setdefault(layer, []).append({"layer": rest, "array": arr})
            case _ : raise Exception("Unknown layer, aborting")

    # parse layers
    for layer in dict(sorted(hidden_layers.items())).values():
        input_layernorm, post_attention_layernorm, mlp, self_attn = (None, None, None, None)
        for sub_layer in layer:
            match sub_layer:
                case "input_layernorm":
                    input_layernorm = RMSNorm(layer[sub_layer][0]['array'], eps=cfg.rms_eps) 
                case "post_attention_layernorm":
                    post_attention_layernorm = RMSNorm(layer[sub_layer][0]['array'], eps=cfg.rms_eps) 
                case "mlp":
                    mlp = parse_mlp(layer[sub_layer])
                case "self_attn":
                    self_attn = parse_attn(layer[sub_layer], cfg.n_kv_heads, cfg.n_attn_heads, cfg.max_seq_len, cfg.rope_theta)
                    KV_caches.append(self_attn.KV_CACHE)
                case _: raise Exception("Unknown layer, aborting")
        model += [TransformerBlock(mlp, input_layernorm, self_attn, post_attention_layernorm)] 

    model += [output_norm, output_embed]

    return model, KV_caches

class QwenModel2505(Model):
    def __init__(self):
        model_file = model_dir / "model.safetensors"
        config = model_dir / "config.json"
        with open(config, "r") as f:
            cfg = json.load(f)

        model_cfg = ModelConfig(cfg)
        
        self.layers, self.kv_caches = parse_model(model_file, model_cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
  
    def __call__(self, input):
        out = input
        for layer in self.layers:
            out = layer(out)
        return out

'''

if __name__ == '__main__':
    print_model()