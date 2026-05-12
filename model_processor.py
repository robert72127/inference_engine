import asyncio
import threading 

import torch
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
        self.mask = mask
        self.token_queue = asyncio.Queue()

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

    def take_batch(self, q, max_batch_size):
        batch = []
        while len(batch) < max_batch_size and q:
            batch.append(q.pop())
        return batch

    def has_place(self): return self.job_count < self.max_jobs
    def add_jobs(self, cnt): self.job_count += cnt
    def remove_jobs(self, cnt): self.job_count -= cnt

    def worker_loop(self):
        while True:
            with self.cv:
                # sleep until ANY queue has work
                self.cv.wait_for(
                    lambda: self.waiting or self.prefill
                )

                self.next_op = OP.PREFILL if self.waiting else OP.GENERATE
                with self.lock:
                    if self.next_op == OP.PREFILL:
                        batch = self.take_batch(self.waiting, self.max_batch_prefill)
                    else:
                        batch = self.take_batch(self.generate, self.max_batch_generate)
                
                input_batch = [handle.input for handle in batch]
                uuid_batch = [handle.uuid for handle in batch]
                results = self.model(input_batch, self.next_op == OP.PREFILL, uuid_batch)
                if self.next_op == OP.PREFILL:
                    for i, handle in enumerate(batch):
                        self.prefill.append(handle)
                        handle.token_queue.put(results[i])
                else:
                    for handle in batch:
                        handle.token_queue.put(results[i])
                        if handle.eos:
                            self.generate.remove(handle)
                            self.remove_jobs(1)
                self.next_op = OP.GENERATE if self.next_op == OP.PREFILL else OP.PREFILL

    async def prefill(self, tokens: torch.Tensor, mask:torch.Tensor):
        handle = Handle(self.next_uuid, tokens, mask)
        uuid = 0
        with self.lock:
            uuid = self.next_uuid
            self.next_uuid += 1
            self.waiting += (Handle(uuid, tokens, mask),)
        next_token = await handle.get_next_token()
        return handle, next_token

    async def single_step(self, handle: Handle):
        next_token = await handle.get_next_token()
        return next_token 
