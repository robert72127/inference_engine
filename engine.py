from enum import Enum
from dataclasses import dataclass
import os
import importlib
from time import sleep
import zmq
import torch

from utils.logger import Logger
from tokenizer import Tokenizer
from model_processor import ModelProcessor
from models import MODEL, models
from sampling import top_k_top_p_sample

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
    def __init__(self, model: MODEL, backend: BACKEND, max_workers=1, max_proc_req=100):
        # detect backend and init
        self.backend = backend if backend.is_available() else BACKEND.CPU
        self.model = model
        self.model_workers_cnt = self.backend.get_device_count(max_workers)

        module = importlib.import_module(f"models.{models[model]['module']}")
        model_dir = module.get_model_dir() 
        model_constructor = getattr(module, models[model]['constructor'])

        self.tokenizer_pool = ThreadPoolExecutor(max_workers=1)
        Logger.info(f"Launching tokenizer server for model : {self.model.value}")
        self.tokenizer =  Tokenizer(model_dir=model_dir)

        Logger.info(f"Launching model : {self.model.value}, with backend : {self.backend.value}, num workers : {self.model_workers_cnt}") 
        self.workers = [ModelProcessor(model_constructor, self.backend, endpoint="tcp://127.0.0.1:6000") for i in range(self.model_workers_cnt)]

        self.next_worker = 0

        self.pro

    def apply_prompt_template(self, prompt: str):
        return f"<|im_start|>system\nYou are a helpful assistant anser .<|im_end|>\n<|im_start|>user\n{{prompt}}<|im_end|>\n<|im_start|>assistant\n"

    def schedule(self):
        worker = self.workers[self.next_worker]
        self.next_worker = (self.next_worker + 1) % self.model_workers_cnt
        return worker

    async def generate(self, message, max_tokens):
        out = []
        async for delta in self.generate_stream(message, max_tokens):
            out.append(delta)
        return "".join(out)

    async def generate_stream(self, message, max_tokens):
        prompt = self.apply_prompt_template(message)
        tokenized = await self.tokenizer.tokenize([prompt])
        tokens = tokenized["input_ids"]

        backend = self.schedule()

        handle = await self.backend.prefill(tokens)

        generated = []
        for _ in range(max_tokens):
            next_token = await backend.single_step(handle)
            if next_token == self.tokenizer.tokenizer.eos_token_id:
                break
            generated.append(next_token)
            text = self.tokenizer.tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )

            yield text