import asyncio
import threading 

import torch
from torch.nn.utils.rnn import pad_sequence

from enum import Enum
from utils.remote import subprocess_worker

class OP(Enum):
    WAITING = 0
    PREFILL = 1
    GENERATE = 2
    FINISHED = 3

class Handle:
    def __init__(self, uuid: int, tokens: torch.Tensor, mask: torch.Tensor):
        self.uuid = uuid
        self.tokens = tokens
        self.token_queue = asyncio.Queue()
        self.input = None

    def set_next_token_gen(self, tok):
        self.input = tok

    async def get_next_token(self):
        tok = await self.token_queue.get()
        return tok

@subprocess_worker
class ModelProcessor:
    
    def __init__(self, model_constructor: callable, backend:str, max_jobs=16, max_batch_prefill=8, max_batch_generate=8):
        self.lock = threading.Lock()
        self.backend = backend
        self.model = model_constructor(backend)

        self.waiting = []
        self.generate = []

        self.cv = threading.Condition()

        self.next_uuid = 0

        self.next_op = OP.PREFILL
        self.max_jobs = max_jobs
        self.max_batch_prefill = max_batch_prefill
        self.max_batch_generate = max_batch_generate

        self.job_count = 0

    def _has_place(self): return self.job_count < self.max_jobs
    def _add_jobs(self, cnt): self.job_count += cnt
    def _remove_jobs(self, cnt): self.job_count -= cnt
    
    def _take_batch_handle(self):
        if self.next_op == OP.GENERATE:
            max_batch_size = self.max_batch_generate
            in_q = self.waiting
        else:
            max_batch_size = self.max_batch_prefill
            in_q = self.prefill
        batch = []
        while len(batch) < max_batch_size and in_q and self.job_count < self.max_jobs :
            batch.append(q.pop())
            self._add_jobs(1)
        return batch

    def _make_batch(self, batch_handle:list[Handle]):
        uuid = [handle.uuid for handle in batch_handle]
        if self.next_op == OP.GENERATE:
            mask = None
            input = torch.tensor([handle.input for handle in batch_handle])
        else:
            input = [handle.input for handle in batch_handle]
            lengths = torch.tensor([t.size(0) for t in input])
            input = pad_sequence(
                input,
                batch_first=True,
                padding_value=0
            )
            mask = torch.arange(input.size(1))[None, :] < lengths[:, None]
        return uuid,input, mask

    def worker_loop(self):
        while True:
            with self.cv:
                # sleep until ANY queue has work
                self.cv.wait_for(
                    lambda: self.waiting or self.prefill
                )
                with self.lock:
                    self.next_op = OP.PREFILL if self.waiting and self._has_place() else OP.GENERATE
                    batch = self._take_batch_handle()
                
                uuid, input, mask = self._make_batch(batch)
                results = self.model(input, self.next_op == OP.PREFILL, mask, uuid)
                if self.next_op == OP.PREFILL:
                    for i, handle in enumerate(batch):
                        self.prefill.append(handle)
                        handle.next_token_gen = results[i]
                        handle.token_queue.put(results[i])
                else:
                    for handle in batch:
                        handle.next_token_gen = results[i]
                        handle.token_queue.put(results[i])
                        if handle.eos:
                            self.generate.remove(handle)
                            self._remove_jobs(1)
                self.next_op = OP.GENERATE if self.next_op == OP.PREFILL else OP.PREFILL

    async def prefill(self, tokens: torch.Tensor):
        handle = Handle(self.next_uuid, tokens)
        uuid = 0
        with self.lock:
            uuid = self.next_uuid
            self.next_uuid += 1
            self.waiting += (handle,)
        next_token = await handle.get_next_token()
        return handle

    async def single_step(self, handle: Handle):
        next_token = await handle.get_next_token()
        return next_token 
