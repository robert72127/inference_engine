from dataclasses import dataclass

import asyncio
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


class OP(Enum):
    WAITING = 0
    PREFILL = 1
    GENERATE = 2
    FINISHED = 3

class ServerHandle:
    def __init__(self, handle_id: int, tokens: torch.Tensor):
        self.id = handle_id
        self.tokens = tokens
        self.input = tokens

        self.token_q: queue.Queue[int] = queue.Queue()
        self.finished = False

@dataclass(frozen=True)
class RemoteHandle:
    id: int

class RemoteModelProcessorHandle:
    def __init__(self, endpoint: str):
        self.ctx = zmq.asyncio.Context.instance()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.connect(endpoint)

        self.pending: dict[str, asyncio.Future] = {}
        self.reader_task: asyncio.Task | None = None

    async def start(self):
        self.reader_task = asyncio.create_task(self._reader_loop())

    async def _reader_loop(self):
        while True:
            raw = await self.socket.recv()
            msg = msgpack.unpackb(raw, raw=False)

            fut = self.pending.pop(msg["request_id"], None)
            if fut is None:
                continue

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

    async def prefill(self, tokens: torch.Tensor) -> RemoteHandle:
        resp = await self.call( "PREFILL",{"tokens": tokens.tolist(),},)
        return RemoteHandle(id=resp["handle_id"])

    async def next_token(self, handle: RemoteHandle) -> int:
        resp = await self.call("NEXT_TOKEN",{"handle_id": handle.id,},)

        return resp["token"]

    async def release(self, handle: RemoteHandle):
        await self.call("RELEASE",{"handle_id": handle.id,},)

    async def close(self):
        if self.reader_task:
            self.reader_task.cancel()

        self.socket.close(linger=0)


class ServerHandle:
    def __init__(self, handle_id: int, tokens: torch.Tensor, max_queue_size: int = 8):
        self.id = handle_id
        self.tokens = tokens

        #  previous token for generation.
        self.input_token: int | None = None

        # produced tokens are pushed here by worker_loop.
        self.token_q: queue.Queue[int] = queue.Queue()

        self.finished = False

class ModelProcessor:
    def __init__(
        self,
        model_constructor,
        backend: str,
        eos_token_id: int,
        max_batch_prefill: int = 8,
        max_batch_generate: int = 8,
    ):
        self.model = model_constructor(backend)
        self.eos_token_id = eos_token_id

        self.max_batch_prefill = max_batch_prefill
        self.max_batch_generate = max_batch_generate

        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)

        self.next_handle_id = 0
        self.handles: dict[int, ServerHandle] = {}

        self.prefill_waiting: list[ServerHandle] = []
        self.generating: list[ServerHandle] = []

        self.worker = threading.Thread(
            target=self.worker_loop,
            daemon=True,
        )
        self.worker.start()

    async def prefill(self, tokens: list[int]):
        tokens_t = torch.tensor(tokens, dtype=torch.long)

        with self.cv:
            handle_id = self.next_handle_id
            self.next_handle_id += 1

            handle = ServerHandle(handle_id, tokens_t)
            self.handles[handle_id] = handle
            self.prefill_waiting.append(handle)

            self.cv.notify()

        return {"handle_id": handle_id,}

    async def next_token(self, handle_id: int):
        with self.lock:
            handle = self.handles.get(handle_id)

        loop = asyncio.get_running_loop()
        tok = await loop.run_in_executor(None,handle.token_q.get,)
        return {"token": tok,}

    async def release(self, handle_id: int):
        with self.cv:
            handle = self.handles.pop(handle_id, None)

    def _make_batch(self, batch: list[ServerHandle], op: OP):
        handle_ids = [h.id for h in batch]

        if op == OP.PREFILL:
            seqs = [h.tokens for h in batch]
            lengths = torch.tensor([t.size(0) for t in seqs])

            input_ids = pad_sequence(
                seqs,
                batch_first=True,
                padding_value=0,
            )

            mask = torch.arange(input_ids.size(1))[None, :] < lengths[:, None]

            return handle_ids, input_ids, mask

        else:
            input_ids = torch.tensor(
                [h.input_token for h in batch],
                dtype=torch.long,
            )

            return handle_ids, input_ids, None
    
    def worker_loop(self):
        next_op = OP.PREFILL

        while True:
            with self.cv:
                self.cv.wait_for(
                    lambda: self.prefill_waiting or self.generating
                )

                if self.prefill_waiting and (
                    next_op == OP.PREFILL or not self.generating
                ):
                    op = OP.PREFILL
                    batch = self.prefill_waiting[: self.max_batch_prefill]
                    self.prefill_waiting = self.prefill_waiting[self.max_batch_prefill :]

                else:
                    op = OP.GENERATE
                    batch = self.generating[: self.max_batch_generate]

                batch = [
                    h for h in batch
                    if not h.cancelled and h.id in self.handles
                ]

            if not batch:
                continue

            handle_ids, input_ids, mask = self._make_batch(batch, op)

            results = self.model(input_ids, op == OP.PREFILL, mask, handle_ids,)

            with self.cv:
                for handle, tok in zip(batch, results):
                    if handle.cancelled or handle.id not in self.handles:
                        continue

                    handle.input_token = tok

                    if tok == self.eos_token_id:
                        handle.finished = True
                        self.generating = [
                            h for h in self.generating
                            if h.id != handle.id
                        ]

                    elif op == OP.PREFILL:
                        self.generating.append(handle)

                    # This can block if client stops reading.
                    # That is good backpressure, but if you don't want worker blocking,
                    # use put_nowait and handle overflow.
                    handle.token_q.put(tok)

                next_op = OP.GENERATE if op == OP.PREFILL else OP.PREFILL


async def run_model_server(endpoint: str, processor: ModelProcessor):
    ctx = zmq.asyncio.Context.instance()
    socket = ctx.socket(zmq.ROUTER)
    socket.bind(endpoint)

    while True:
        client_id, raw = await socket.recv_multipart()
        msg = msgpack.unpackb(raw, raw=False)

        op = msg["op"]
        payload = msg["payload"]

        if op == "PREFILL": result = await processor.prefill(payload["tokens"])

        elif op == "NEXT_TOKEN":result = await processor.next_token(payload["handle_id"])

        elif op == "RELEASE": result = await processor.release(payload["handle_id"])

        response = {"request_id": msg["request_id"], "ok": True, "payload": result,}

        await socket.send_multipart([client_id, msgpack.packb(response, use_bin_type=True),])

def main_model_process():
    processor = ModelProcessor(
        model_constructor=make_model,
        backend="cuda",
        eos_token_id=2,
    )

    asyncio.run(
        run_model_server(
            "ipc:///tmp/model-engine.ipc",
            processor,
        )
    )