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

from models.model import PrefillStateBuff, DecodeStateBuff
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
        self.max_new_tokens = max_new_tokens
        self.generated_tokens = 0
        self.cache_pos = 0

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
                torch.tensor(payload["tokens"], dtype=torch.long),
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
        max_batch_size: int = 8,
        prefill_chunk_size: int = 64,
    ):
        self.device = torch.device(device)
        self.model = model_constructor(self.device)
        self.kv_caches = list(getattr(self.model, "kv_caches", []))
        self.max_request_len = max_request_len
        self.eos_token_id = eos_token_id

        self.supported_batch_sizes = [2**i for i in range(max_batch_size.bit_length()) if 2**i <= max_batch_size]
        self.prefill_chunk_size = prefill_chunk_size

        block_size = self.kv_caches[0].block_size
        max_pages_per_req = blocks_for_tokens(max_request_len, block_size)

        #prepare buffers for supported batch sizes
        self.prefill_buffs = {batch_size : PrefillStateBuff(
            batch_size=batch_size,
            chunk_size=self.prefill_chunk_size,
            max_pages=max_pages_per_req,
            device=self.device)
                for batch_size in self.supported_batch_sizes}
        self.decode_buffs = {batch_size: DecodeStateBuff(
            batch_size=batch_size,
            max_pages=max_pages_per_req,
            device=self.device)
                for batch_size in self.supported_batch_sizes}
        self.decode_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.decode_graph_outputs: dict[int, torch.Tensor] = {}

        self.prefill_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.prefill_graph_outputs: dict[int, torch.Tensor] = {}

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
        tokens: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_new_tokens: int | None = None,
    ):
        prompt_tokens = [int(tok) for tok in tokens.detach().cpu().tolist()]
        tokens_t = tokens.to(device=self.device)

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

            # calls KV_CACHE.init Init_uuid, prompt_tokens for each cache in model
            self.model.KV_init(handle_id, prompt_tokens)

            self.cv.notify()

        return handle_id

    async def decode(self, handle_id: int):
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

    def worker_loop(self):
        next_op = OP.PREFILL

        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.prefill_waiting or self.generating)

                if self.prefill_waiting and (next_op == OP.PREFILL or not self.generating):
                    available_slots = self._get_max_available_occupancy_left()
                    if available_slots == 0:
                        next_op = OP.GENERATE
                        continue
                    op = OP.PREFILL
                    batch_size =  self._get_batch_size( min(available_slots, len(self.prefill_waiting)))
                    batch = self.prefill_waiting[:batch_size]
                    self.prefill_waiting = self.prefill_waiting[batch_size:]
                else:
                    batch_size = self._get_batch_size(len(self.generating))
                    if batch_size == 0:
                        next_op = OP.PREFILL
                        continue
                    op = OP.GENERATE
                    batch = self.generating[:batch_size]

                self.in_flight.update(handle.id for handle in batch)

            with torch.inference_mode():
                if op == OP.PREFILL:
                    prefill_state, last_chunk = self._make_batched_input(batch, op, batch_size)
                    model_out = self.run_prefill(prefill_state, batch_size)
                    for h in batch: self.caches_commit_prefill(h.id)
                    
                    results = []
                    batch_decode = []
                    for is_last_chunk, out, handle in zip(last_chunk, model_out, batch):
                        if is_last_chunk:
                            results.append(out)
                            batch_decode.append(handle)

                    if batch_decode:
                        results = self._decode_batch(torch.stack(results), batch_decode).tolist()
                else:
                    decode_state = self._make_batched_input(batch, op, batch_size)
                    model_out = self.run_decode(decode_state, batch_size)
                    results = self._decode_batch(model_out, batch).tolist()

            with self.cv:
                result_idx = 0
                for batch_idx, handle in enumerate(batch):
                    self.in_flight.discard(handle.id)
                    released =  handle.id not in self.handles
                    if op == OP.PREFILL:
                        seq_len = int(prefill_state.seq_lens[batch_idx].item())
                        handle.cache_pos += seq_len
                        if not last_chunk[batch_idx]:
                            if not released:
                                self.prefill_waiting.append(handle)
                            continue
                        
                        tok = results[result_idx]
                        result_idx += 1
                    else:
                        tok = results[batch_idx]
                        handle.cache_pos += 1
                    
                    handle.generated_tokens += 1
                    handle.input_token = tok
                    reached_limit = (
                        handle.cache_pos >= self.max_request_len
                        or handle.generated_tokens >= handle.max_new_tokens
                    )
                    if tok == self.eos_token_id or released or reached_limit:
                        handle.finished = True
                        self.generating = [h for h in self.generating if h.id != handle.id]
                        self.handles.pop(handle.id, None)
                        self._release_caches(handle.id)
                    elif op == OP.PREFILL:
                        self.generating.append(handle)

                    if not released:
                        handle.token_q.put_nowait(tok)
                next_op = OP.GENERATE if op == OP.PREFILL else OP.PREFILL

    def _make_batched_input(self, batch: list[ServerHandle], op: OP, batch_size:int):
        if op == OP.PREFILL:
            buf = self.prefill_buffs[batch_size]
            last_chunk = []
            # todo needs fix, maybe -1? cause zero is valid page index
            buf.page_indexes.fill_(-1)
            for i, handle in enumerate(batch):
                end = min(handle.cache_pos + self.prefill_chunk_size, handle.tokens.size(0))
                length = end - handle.cache_pos

                tokens = handle.tokens[handle.cache_pos:end]
                buf.tokens[i].zero_()
                buf.tokens[i, :length].copy_(tokens)

                buf.request_slots[i] = handle.id
                buf.offsets[i] = handle.cache_pos
                buf.seq_lens[i] = length
                last_chunk.append(end == handle.tokens.size(0))
                # prepare space for k,v in each cache
                self.caches_init_prefill(handle.id, tokens)
                # set page indexes
                indexes = self.kv_caches[0].get_indexes(handle.id)
                buf.page_indexes[i, :len(indexes)] =  torch.as_tensor(indexes, dtype=torch.int32, device=self.device)

            return buf, last_chunk
        else:
            buf = self.decode_buffs[batch_size]
            buf.page_indexes.fill_(-1)
            for i, handle in enumerate(batch):
                buf.tokens[i,0] = handle.input_token
                buf.request_slots[i] = handle.id
                buf.offsets[i] = handle.cache_pos
                # prepare space for k,v in each cache
                self.caches_insert_decode(handle.id, handle.input_token)
                # set page indexes
                indexes = self.kv_caches[0].get_indexes(handle.id)
                buf.page_indexes[i, :len(indexes)] =  torch.as_tensor(indexes, dtype=torch.int32, device=self.device)
            return buf


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

    def run_decode(self, decode_state: DecodeStateBuff, batch_size: int) -> torch.Tensor:
        return self._run_model(decode_state, batch_size,
                               graphs=self.decode_graphs,
                               graph_outputs=self.decode_graph_outputs,
                               process=self.model.decode)


    def run_prefill(self, prefill_state: DecodeStateBuff, batch_size: int) -> torch.Tensor:
        return self._run_model(prefill_state, batch_size,
                               graphs=self.prefill_graphs,
                               graph_outputs=self.prefill_graph_outputs,
                               process=self.model.prefill)


    def _run_model(self, state, batch_size:int, graphs, graph_outputs, process) -> torch.Tensor:
        if self.device.type != "cuda":
            return process(state)
        
        graph = graphs.get(batch_size)
        if graph is not None:
            graph.replay()
            return graph_outputs[batch_size]

        with torch.cuda.device(self.device):
            warmup_stream = torch.cuda.Stream()
            warmup_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    process(state)
            torch.cuda.current_stream().wait_stream(warmup_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = process(state)

        graphs[batch_size] = graph
        graph_outputs[batch_size] = static_out
        return static_out

    def _get_max_available_occupancy_left(self):
        max_blocks_per_request = blocks_for_tokens(self.max_request_len, self.kv_caches[0].block_size)
        return max(0, self.kv_caches[0].get_blocks_available() // max_blocks_per_request)

    def _get_batch_size(self, reqs_len):
        if reqs_len == 0: return 0
        return max(b_size for b_size in self.supported_batch_sizes if b_size <= reqs_len)

    def caches_insert_decode(self, uuid, tok):
        for kv_cache in self.kv_caches:
            kv_cache.append_decode(uuid, tok)

    def caches_init_prefill(self, uuid, toks):
        for kv_cache in self.kv_caches:
            kv_cache.init_prefill(uuid, toks)

    def caches_commit_prefill(self, uuid):
        for kv_cache in self.kv_caches:
            kv_cache.finish_prefill(uuid)

    def _release_caches(self, handle_id: int):
        for cache in self.kv_caches:
            cache.release(handle_id)
