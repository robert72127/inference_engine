from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from pydantic import BaseModel
from typing import Literal, Optional, List

import time


app = FastAPI()

# request

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False
    stop: Optional[str | list[str]] = None

# /v1/models response

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# non-streaming response

class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"]
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: Literal["stop", "length"]

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

# streaming response

class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None

class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

# post, get API

@app.get("/v1/models")
async def list_models():
    data_model = ModelCard(id="qwen-custom", object="model", created=0, owned_by="local")
    model_list = ModelList(object="list", data = [data_model])
    return model_list


async def generate_full(text:str):
    return "text"

@app.post("/v1/chat/completions")
async def chat(req : ChatCompletionRequest):
    #prompt = messages_to_prompt(req.messages)

    if req.stream:
        chunk_choice = ChatCompletionChunkChoice(index=0, delta=DeltaMessage())
        chunk = ChatCompletionChunk(id=0,object="chat.completion.chunk", created=0, model="qwen", choices=[chunk_choice])        
        return StreamingResponse(
            chunk,
            media_type="text/event-stream",
        )

    text = await generate_full(req)

    return ChatCompletionResponse(
        id="chatcmpl-123",
        created=int(time.time()),
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": text},
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )