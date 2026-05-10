from enum import Enum
from dataclasses import dataclass
import os
import importlib

import zmq
import torch

from utils.logger import Logger
from tokenizer import Tokenizer
from model_processor import ModelProcessor
from models import MODEL, models

class OP(Enum):
    PREFILL = 0
    GENERATE = 1

class BACKEND(Enum):
    CPU = "cpu"
    CUDA = "cuda"

    def get_device_count(self, max_workers=1):
        match self:
            case BACKEND.CPU: return os.cpu_count()
            case BACKEND.CUDA: return torch.cuda.device_count()

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
        self.model_workers_cnt = self.backend.get_device_count(max_workers)

        module = importlib.import_module(f"models.{models[model]['module']}")
        model_dir = module.QwenModel2505.get_model_dir() 
        model_constructor = getattr(module, models[model]['constructor'])

        self.scheduler = Scheduler(max_proc_req//2, max_proc_req//2)

        Logger.info(f"Launching tokenizer server for model : {self.model.value}")
        self.tokenizer =  Tokenizer(model_dir=model_dir)

        Logger.info(f"Launching model : {self.model.value}, with backend : {self.backend.value}, num workers : {self.model_workers_cnt}") 
        self.workers = [ModelProcessor(model_constructor, self.backend, id=i, endpoint="tcp://127.0.0.1:6000") for i in range(self.model_workers_cnt)]
