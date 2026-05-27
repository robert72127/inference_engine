import os
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from engine import BACKEND, Engine, GenDelta, GenStart, GenEnd
from models import MODEL
from utils.logger import Logger


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: str | list[str] | None = None


def validate_sampling_params(temperature: float, top_p: float):
    if temperature < 0:
        raise HTTPException(status_code=400, detail="temperature must be >= 0")
    if not 0 < top_p <= 1:
        raise HTTPException(status_code=400, detail="top_p must be in (0, 1]")


def parse_model(model_name: str) -> MODEL:
    try:
        return MODEL(model_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}") from exc


def build_prompt(messages: list[ChatMessage]) -> str:
    prompt = "\n".join(msg.content for msg in messages if msg.role == "user")
    if not prompt:
        raise HTTPException(status_code=400, detail="At least one user message is required")
    return prompt




def chunk_payload(completion_id: str, created: int, model: str, delta: dict, finish_reason=None) -> dict:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def get_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = parse_model(os.getenv("MODEL_NAME", MODEL.QWEN_2_5_0_5B_INSTRUCT.value))
    backend = BACKEND(os.getenv("MODEL_BACKEND", BACKEND.CPU.value))
    max_workers = int(os.getenv("MODEL_MAX_WORKERS", "1"))
    max_proc_req = int(os.getenv("MODEL_MAX_PROC_REQ", "100"))
    max_cache_seq_len = int(os.getenv("MODEL_MAX_CACHE_SEQ_LEN", "4096"))
    cpu_reserve_gb = float(os.getenv("MODEL_CPU_RESERVE_GB", "6"))
    cuda_memory_fraction = float(os.getenv("MODEL_CUDA_MEMORY_FRACTION", "1.0"))
    prefill_chunk_size = os.getenv("MODEL_PREFILL_CHUNK_SIZE")
    app.state.engine = Engine(
        model=model,
        backend=backend,
        max_workers=max_workers,
        max_proc_req=max_proc_req,
        max_cache_seq_len=max_cache_seq_len,
        cpu_reserve_gb=cpu_reserve_gb,
        cuda_memory_fraction=cuda_memory_fraction,
        prefill_chunk_size=None if prefill_chunk_size is None else int(prefill_chunk_size),
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": model.value, "object": "model", "created": 0, "owned_by": "local"}
            for model in MODEL
        ],
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatCompletionRequest):
    engine = app.state.engine
    model = parse_model(req.model)
    if model != engine.model:
        raise HTTPException(
            status_code=400,
            detail=f"Server loaded model '{engine.model.value}', requested '{req.model}'",
        )

    prompt = build_prompt(req.messages)
    validate_sampling_params(req.temperature, req.top_p)
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    Logger.info("Chat completion request id=%s stream=%s max_tokens=%d", completion_id, req.stream, req.max_tokens)

    if req.stream:
        async def event_stream():
            yield f"data: {json.dumps(chunk_payload(completion_id, created, req.model, {'role': 'assistant'}))}\n\n"

            prompt_tokens = 0
            completion_tokens = 0

            async for ev in engine.generate_stream(prompt, req.max_tokens, temperature=req.temperature, top_p=req.top_p):
                match ev:
                    case GenStart():
                        prompt_tokens = ev.prompt_tokens
                    case GenEnd():
                        completion_tokens = ev.completion_tokens 
                    case GenDelta():
                        yield f"data: {json.dumps(chunk_payload(completion_id, created, req.model, {'content': ev.text}))}\n\n"
            
            yield f"data: {json.dumps(chunk_payload(completion_id, created, req.model, {}, 'stop') | {"usage": get_usage(prompt_tokens, completion_tokens)})}\n\n"
            yield "data: [DONE]\n\n"
            
            Logger.info("Chat completion streamed id=%s", completion_id)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    text, prompt_tokens, completion_tokens = await engine.generate(prompt, req.max_tokens, temperature=req.temperature, top_p=req.top_p)
    Logger.info("Chat completion finished id=%s completion_tokens=%d", completion_id, completion_tokens)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": get_usage(prompt_tokens, completion_tokens),
    }
