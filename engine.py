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


class OP(Enum):
    PREFILL = 0
    GENERATE = 1

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

class Scheduler:
    def __init__(self, max_prefill, max_single_step):
        self.next_user_id = 0
        self.max_prefill = max_prefill
        self.max_single_step = max_single_step
        
        self.next_op = OP.PREFILL

        self.waiting_prefill = set()
        self.waiting_single_step = set()

    async def add_request(self, text: str):
        pass

class Engine:
    def __init__(self, model: MODEL, backend: BACKEND, max_workers=1, max_proc_req=100):
        # detect backend and init
        self.backend = backend if backend.is_available() else BACKEND.CPU
        self.model = model
        self.model_workers_cnt = self.backend.get_device_count(max_workers)

        module = importlib.import_module(f"models.{models[model]['module']}")
        model_dir = module.get_model_dir() 
        model_constructor = getattr(module, models[model]['constructor'])

        self.scheduler = Scheduler(max_proc_req//2, max_proc_req//2)

        Logger.info(f"Launching tokenizer server for model : {self.model.value}")
        self.tokenizer =  Tokenizer(model_dir=model_dir)

        Logger.info(f"Launching model : {self.model.value}, with backend : {self.backend.value}, num workers : {self.model_workers_cnt}") 
        self.workers = [ModelProcessor(model_constructor, self.backend, endpoint="tcp://127.0.0.1:6000") for i in range(self.model_workers_cnt)]

    def call(self):
        toks = self.tokenizer.tokenize(["Hello world!"])
        #todo needs to pass mask returned by tokenizer
        res = self.workers[0].call(toks["input_ids"])
        toks = top_k_top_p_sample(res, top_k=50, top_p=0.95)
        print(toks)
        #return self.tokenizer.detokenize(res)

if __name__ == '__main__':
    engine = Engine(MODEL.QWEN_2_5_0_5B_INSTRUCT, BACKEND.CPU, max_workers=1)
    while True: 
        engine.call()
        sleep(1)