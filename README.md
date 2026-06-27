# Inference Engine

Minimal OpenAI-compatible LLM serving engine.

## Features

- Paged attention
- Continuous batching across prefill and decode
- Prefix KV cache
- Chunked prefill
- Custom Triton kernels and CUDA graph replay on the CUDA backend

Supported model:
- `qwen2_5_0_5b_instruct` with 300 + toks/sec for single request on rtx5070

Supported backends:
- `cpu`
- `cuda`

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

## Run

Start the API server with `uvicorn` through `uv`:

```bash
uv run uvicorn api_server:app --host 0.0.0.0 --port 8000
```

Optional environment variables:

```bash
MODEL_NAME=qwen2_5_0_5b_instruct
MODEL_BACKEND=cpu
MODEL_MAX_WORKERS=1
MODEL_MAX_PROC_REQ=100
MODEL_MAX_CACHE_SEQ_LEN=4096
MODEL_CPU_RESERVE_GB=6
MODEL_CUDA_MEMORY_FRACTION=1.0
MODEL_PREFILL_CHUNK_SIZE=256
```

Example:

running on cpu:

```bash
MODEL_BACKEND=cpu MODEL_CPU_RESERVE_GB=8 uv run uvicorn api_server:app --host 0.0.0.0 --port 8000
```

running on gpu:

```bash
MODEL_BACKEND=cuda uv run uvicorn api_server:app --host 0.0.0.0 --port 8000
```


## API

- `GET /v1/models`
- `POST /v1/chat/completions`

Streaming chat completion example:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2_5_0_5b_instruct",
    "messages": [
      {"role": "user", "content": "Count from 1 to 5."}
    ],
    "max_tokens": 32,
    "stream": true
  }'
```
