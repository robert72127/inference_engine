from enum import Enum
import os
import importlib
from functools import partial
import torch

from utils.logger import Logger
from tokenizer import Tokenizer
from model_processor import ModelProcessor
from models import MODEL, models

CPU_SYSTEM_RESERVE_GB = 4

class BACKEND(Enum):
    CPU = "cpu"
    CUDA = "cuda"

    def get_device_count(self, max_workers=1):
        match self:
            case BACKEND.CPU: return min(max_workers, os.cpu_count())
            case BACKEND.CUDA: return min(max_workers, torch.cuda.device_count())

    def is_available(self):
        match self:
            case BACKEND.CPU: return True
            case BACKEND.CUDA: return torch.cuda.is_available()

class Engine:
    def __init__(
        self, 
        model: MODEL,
        backend: BACKEND,
        max_workers=1,
        max_proc_req=100,
        max_cache_seq_len=4096,
    ):
        # detect backend and init
        self.backend = backend if backend.is_available() else BACKEND.CPU
        self.model = model
        
        self.model_workers_cnt = self.backend.get_device_count(max_workers)

        module = importlib.import_module(f"models.{models[model]['module']}")
        model_dir = module.get_model_dir() 
        model_constructor = getattr(module, models[model]['constructor'])
        model_memory_available = self.get_model_memory_available()
        model_factory = partial(
            model_constructor,
            cache_max_requests=max_proc_req,
            cache_max_seq_len=max_cache_seq_len,
            memory_available=model_memory_available,
        )

        Logger.info(f"Launching tokenizer server for model : {self.model.value}")
        self.tokenizer =  Tokenizer(model_dir=model_dir)
        eos_token_id = self.tokenizer.tokenizer.eos_token_id

        Logger.info(f"Launching model : {self.model.value}, with backend : {self.backend.value}, num workers : {self.model_workers_cnt}") 
        self.workers = [
            ModelProcessor(model_factory,
                torch.device(self.backend.value if self.backend == BACKEND.CPU else f"{self.backend.value}:{worker_index}"),
                eos_token_id=eos_token_id,
                max_request_len=max_cache_seq_len,
            )
            for worker_index in range(self.model_workers_cnt)
        ]

        self.next_worker = 0

    def get_model_memory_available(self):
        match self.backend:
            case BACKEND.CPU:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                total_memory = pages * page_size
                reserve = CPU_SYSTEM_RESERVE_GB * 1024**3
                return max(0, total_memory - reserve) // self.model_workers_cnt
            case BACKEND.CUDA:
                return min(
                    torch.cuda.get_device_properties(worker_index).total_memory
                    for worker_index in range(self.model_workers_cnt)
                )

    def apply_prompt_template(self, prompt: str):
        return f"<|im_start|>system\nYou are a helpful assistant answer .<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    def schedule(self):
        worker = self.workers[self.next_worker]
        self.next_worker = (self.next_worker + 1) % self.model_workers_cnt
        return worker

    async def generate(self, message, max_tokens, temperature=1.0, top_p=1.0):
        out = []
        async for delta in self.generate_stream(message, max_tokens, temperature=temperature, top_p=top_p):
            out.append(delta)
        return "".join(out)

    async def generate_stream(self, message, max_tokens, temperature=1.0, top_p=1.0):
        prompt = self.apply_prompt_template(message)
        tokenized = await self.tokenizer.tokenize([prompt])
        tokens = tokenized["input_ids"][0]

        backend = self.schedule()
        Logger.debug("Scheduled request prompt_tokens=%d max_tokens=%d", tokens.size(0), max_tokens)

        handle = await backend.prefill(tokens, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens)

        for _ in range(max_tokens):
            tok = await backend.next_token(handle)

            if tok == self.tokenizer.tokenizer.eos_token_id:
                break

            yield self.tokenizer.detokenize_sync([[tok]])[0]

        await backend.release(handle)
