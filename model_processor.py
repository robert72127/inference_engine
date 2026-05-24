import asyncio
import multiprocessing as mp
import queue
import threading
import uuid
from dataclasses import dataclass
from enum import Enum

import msgpack
import torch
import zmq
import zmq.asyncio
from torch.nn.utils.rnn import pad_sequence

from sampling import top_k_top_p_sample
from kvcache import blocks_for_tokens

class OP(Enum):
    PREFILL = 1
    GENERATE = 2

class ServerHandle:
    def __init__(
        self,
        handle_id: int,
        tokens: torch.Tensor,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ):
        self.id = handle_id
        self.tokens = tokens
        self.temperature = temperature
        self.top_p = top_p
        self.cache_len = int(tokens.size(0))
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = 0

        #  previous token for generation.
        self.input_token: int | None = None
        self.token_q: queue.Queue[int] = queue.Queue()
        self.finished = False
        self.cancelled = False

@dataclass(frozen=True)
class RemoteHandle:
    id: int

class RemoteModelProcessorHandle:
    def __init__(self, endpoint: str, model_constructor, device, eos_token_id: int, max_request_len: int):
        self.ctx = zmq.asyncio.Context.instance()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.connect(endpoint)

        self.pending: dict[str, asyncio.Future] = {}
        self.process = mp.get_context("spawn").Process(
            target=start_model_process,
            args=(endpoint, model_constructor, device, eos_token_id, max_request_len),
            daemon=True,
        )
        self.process.start()

        # spawn reader
        self.reader_task = asyncio.get_running_loop().create_task(self._reader_loop())

    async def _reader_loop(self):
        while True:
            raw = await self.socket.recv()
            msg = msgpack.unpackb(raw, raw=False)
            fut = self.pending.pop(msg["request_id"], None)
            if fut is not  None:
                if msg["ok"]:
                    fut.set_result(msg["payload"])
                else:
                    fut.set_exception(RuntimeError(msg["error"]))

    async def call(self, op: str, payload: dict):
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut = loop.create_future()

        self.pending[request_id] = fut

        await self.socket.send(
            msgpack.packb({"request_id": request_id,"op": op,"payload": payload,},
            use_bin_type=True,)
        )
        return await fut

    async def prefill(
        self,
        tokens: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int | None = None,
    ) -> RemoteHandle:
        resp = await self.call(
            "PREFILL",
            {
                "tokens": tokens.tolist(),
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
            },
        )
        return RemoteHandle(id=resp["handle_id"])

    async def next_token(self, handle: RemoteHandle) -> int:
        resp = await self.call("NEXT_TOKEN",{"handle_id": handle.id,},)
        return resp["token"]

    async def release(self, handle: RemoteHandle):
        await self.call("RELEASE",{"handle_id": handle.id,},)

    async def close(self):
        self.socket.close(linger=0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)

async def run_model_server(endpoint: str, processor: "ModelProcessor"):
    ctx = zmq.asyncio.Context.instance()
    socket = ctx.socket(zmq.ROUTER)
    socket.bind(endpoint)

    while True:
        client_id, raw = await socket.recv_multipart()
        msg = msgpack.unpackb(raw, raw=False)

        op = msg["op"]
        payload = msg["payload"]

        if op == "PREFILL":
            result = await processor.prefill(
                payload["tokens"],
                temperature=payload.get("temperature", 1.0),
                top_p=payload.get("top_p", 1.0),
                max_new_tokens=payload.get("max_new_tokens"),
            )

        elif op == "NEXT_TOKEN":result = await processor.next_token(payload["handle_id"])

        elif op == "RELEASE": result = await processor.release(payload["handle_id"])

        response = {"request_id": msg["request_id"], "ok": True, "payload": result,}

        await socket.send_multipart([client_id, msgpack.packb(response, use_bin_type=True),])

def start_model_process(endpoint: str, model_constructor, device, eos_token_id: int, max_request_len: int):
    processor = ModelProcessor(
        model_constructor=model_constructor,
        device=device,
        eos_token_id=eos_token_id,
        max_request_len=max_request_len,
    )
    asyncio.run(run_model_server(endpoint, processor))

class ModelProcessor:
    def __init__(
        self,
        model_constructor,
        device: torch.device,
        eos_token_id: int,
        max_request_len: int,
        max_batch_prefill: int = 8,
        max_batch_generate: int = 8,
    ):
        self.device = torch.device(device)
        self.model = model_constructor(self.device)
        self.kv_caches = list(getattr(self.model, "kv_caches", []))
        self.max_request_len = max_request_len
        self.eos_token_id = eos_token_id

        self.max_batch_prefill = max_batch_prefill
        self.max_batch_generate = max_batch_generate

        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

        self.next_handle_id = 0
        self.handles: dict[int, ServerHandle] = {}

        self.prefill_waiting: list[ServerHandle] = []
        self.generating: list[ServerHandle] = []
        self.in_flight: set[int] = set()

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,
        )
        self.worker.start()

    async def prefill(
        self,
        tokens: list[int],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int | None = None,
    ):
        if isinstance(tokens, torch.Tensor):
            tokens_t = tokens.detach().clone().to(device=self.device, dtype=torch.long)
        else:
            tokens_t = torch.tensor(tokens, device=self.device, dtype=torch.long)

        prompt_len = int(tokens_t.size(0))
        if prompt_len > self.max_request_len:
            raise RuntimeError(
                f"Prompt length {prompt_len} exceeds KV cache capacity {self.max_request_len}"
            )
        allowed_new_tokens = self.max_request_len - prompt_len
        max_new_tokens = allowed_new_tokens if max_new_tokens is None else min(max_new_tokens, allowed_new_tokens)
       
        with self.cv:
            handle_id = self.next_handle_id
            self.next_handle_id += 1

            handle = ServerHandle(
                handle_id,
                tokens_t,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )
            self.handles[handle_id] = handle
            self.prefill_waiting.append(handle)

            self.cv.notify()

        return handle_id

    async def next_token(self, handle_id: int):
        with self.lock:
            handle = self.handles.get(handle_id)
        if handle is None:
            return self.eos_token_id

        loop = asyncio.get_running_loop()
        tok = await loop.run_in_executor(None,handle.token_q.get,)
        return tok

    async def release(self, handle_id: int):
        with self.cv:
            handle = self.handles.get(handle_id)
            self.prefill_waiting = [h for h in self.prefill_waiting if h.id != handle_id]
            self.generating = [h for h in self.generating if h.id != handle_id]
            if handle is not None:
                handle.cancelled = True
                if handle_id not in self.in_flight:
                    self.handles.pop(handle_id, None)
                    self._release_caches(handle.id)
            self.cv.notify()

    def _make_batch(self, batch: list[ServerHandle], op: OP):
        handle_ids = [h.id for h in batch]
        if op == OP.PREFILL:
            seqs = [h.tokens for h in batch]
            lengths = torch.tensor([t.size(0) for t in seqs], device=self.device)
            input_ids = pad_sequence(
                seqs,
                batch_first=True,
                padding_value=0,
            )
            mask = torch.arange(input_ids.size(1), device=self.device)[None, :] < lengths[:, None]
            return handle_ids, input_ids, mask, lengths

        else:
            input_ids = torch.tensor(
                [h.input_token for h in batch],
                device=self.device,
                dtype=torch.long,
            ).unsqueeze(1)
            return handle_ids, input_ids, None, None

    def worker_loop(self):
        next_op = OP.PREFILL

        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.prefill_waiting or self.generating)

                if self.prefill_waiting and (next_op == OP.PREFILL or not self.generating):
                    available_slots = self._get_max_available_occupancy_left()
                    if available_slots == 0:
                        if not self.generating:
                            continue
                        next_op = OP.GENERATE
                        continue
                    op = OP.PREFILL
                    batch_size = min(self.max_batch_prefill, available_slots)
                    batch = self.prefill_waiting[:batch_size]
                    self.prefill_waiting = self.prefill_waiting[batch_size:]
                else:
                    max_batch_size = self._get_max_generate_batch_size()
                    if max_batch_size == 0:
                        next_op = OP.PREFILL
                        continue
                    op = OP.GENERATE
                    batch = self.generating[:max_batch_size]

            with self.cv:
                batch = [handle for handle in batch if handle.id in self.handles]
                if not batch:
                    next_op = OP.PREFILL if op == OP.GENERATE else OP.GENERATE
                    continue
                self.in_flight.update(handle.id for handle in batch)

            with torch.inference_mode():
                handle_ids, input_ids, mask, lengths = self._make_batch(batch, op)
                request_key = tuple(handle_ids)
                logits = self.model(
                    input_ids,
                    input_ids,
                    op == OP.PREFILL,
                    mask,
                    lengths,
                    request_key,
                    prefill_last_chunk=True,
                )
                results = self._decode_batch(logits, batch)
                results = results.tolist()

            with self.cv:
                for handle, tok in zip(batch, results):
                    self.in_flight.discard(handle.id)
                    released = handle.cancelled or handle.id not in self.handles
                    if op == OP.GENERATE:
                        handle.cache_len += 1
                        handle.generated_tokens += 1
                    handle.input_token = tok
                    reached_limit = (
                        handle.cache_len >= self.max_request_len
                        or handle.generated_tokens >= handle.max_new_tokens
                    )
                    if tok == self.eos_token_id or released or reached_limit:
                        handle.finished = True
                        self.generating = [
                            h for h in self.generating
                            if h.id != handle.id
                        ]
                        self.handles.pop(handle.id, None)
                        self._release_caches(handle.id)
                    elif op == OP.PREFILL:
                        self.generating.append(handle)

                    if not released:
                        handle.token_q.put_nowait(tok)
                next_op = OP.GENERATE if op == OP.PREFILL else OP.PREFILL

    def _decode_batch(self, logits: torch.Tensor, batch: list[ServerHandle]) -> torch.Tensor:
        tokens = []
        for idx, handle in enumerate(batch):
            if handle.temperature == 0.0:
                token = logits[idx].argmax(dim=-1)
            else:
                token = top_k_top_p_sample(
                    logits[idx:idx + 1],
                    top_k=None,
                    top_p=handle.top_p,
                    temperature=handle.temperature,
                ).squeeze(0)
            tokens.append(token)
        return torch.stack(tokens)

    def _get_max_available_occupancy_left(self):
        max_blocks_per_request = blocks_for_tokens(self.max_request_len, self.kv_caches[0].block_size)
        return max(0, self.kv_caches[0].get_blocks_available() // max_blocks_per_request)

    def _get_max_generate_batch_size(self):
        return min(self.max_batch_generate, len(self.generating))

    def _release_caches(self, handle_id: int):
        for cache in self.kv_caches:
            cache.release((handle_id,))
