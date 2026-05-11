import asyncio
import threading 
from collections import deque

import torch
from enum import Enum
from utils.remote import subprocess_worker


class OP(Enum):
    PREFILL = 0
    GENERATE = 1


class Handle:
    def __init__(self, uuid: int, tokens: torch.Tensor, mask: torch.Tensor):
        self.uuid = uuid
        self.tokens = tokens
        self.mask = mask

@subprocess_worker
class ModelProcessor:
    
    def __init__(self, model_constructor: callable, backend:str, max_jobs=16, max_batch_prefill=8, max_batch_generate=8):
        self.lock = threading.Lock()
        self.backend = backend
        self.model = model_constructor(backend)

        self.waiting = []
        self.prefill = []
        self.generate = []

        self.cv = threading.Condition()

        self.next_uuid = 0

        self.next_op = OP.PREFILL
        self.max_jobs = max_jobs
        self.max_batch_prefill = max_batch_prefill
        self.max_batch_generate = max_batch_generate

    def take_batch(self, q, max_batch_size):
        batch = []
        while len(batch) < max_batch_size and q:
            batch.append(q.popleft())
        return batch

    def worker_loop(self):
        while True:
            with self.cv:
                # sleep until ANY queue has work
                self.cv.wait_for(
                    lambda: self.waiting or self.prefill or self.generate
                )
                if self.next_op == OP.PREFILL and self.waiting:
                    batch = self.take_batch(self.waiting, self.max_batch_prefill)
                    self.next_op = OP.GENERATE
                    self.model.compute(batch)
                    for handle in batch:
                        self.prefill.append(handle)
                        self.generate.remove(handle)
                        # notifuy waiting handles that prefill is done
                elif self.next_op == OP.GENERATE and self.prefill:
                    batch = self.take_batch(self.generate, self.max_batch_generate)
                    self.model.compute(batch)
                    for handle in batch:
                        if handle.eos:
                            self.generate.remove(handle)
                        # notify waiting handles that generate step is done
                    self.next_op = OP.PREFILL


    async def prefill(self, tokens: torch.Tensor, mask:torch.Tensor):
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        handle = Handle(self.next_uuid, tokens, mask)
        uuid = 0
        with self.lock:
            uuid = self.next_uuid
            self.next_uuid += 1
            self.waiting += (Handle(uuid, tokens, mask),)
        self.prefill = (handle,)
        await 
        return handle    

    def single_step(self, input: torch.Tensor):
        pass
