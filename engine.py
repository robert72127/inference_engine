from enum import Enum
from dataclasses import dataclass
import torch
import os

from multiprocessing import Process
import zmq

from logger import Logger
from tokenizer import TokenizerHandle


class OP(Enum):
    PREFILL = 0
    SINGLES_STEP = 1


class BACKEND(Enum):
    CPU = 0
    CUDA = 1

@dataclass
class ModelSettings:
    model: str
    tokenizer: str

model_settings = {
    MODEL.QWEN_2 : ModelSettings(model="qwen_2", tokenizer="tiktoken")
}

class MODEL(Enum):
    QWEN_2 = 0

class GraphWorkerHandler:
    def __init__(self, model:str, backend:str, worker_id:int, endpoint:str):
       self.worker_id = worker_id
       self.endpoint = endpoint
       self.process = None

       ctx = zmq.Context()

       # receive jobs
       job_receiver = ctx 

    async def process(batch:torch.Tensor, operation:OP):
        pass

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
        match backend:
            case BACKEND.CPU:
                self.backend = "cpu"
                self.device_count = min(max_workers, os.cpu_count())
            case BACKEND.CUDA:
                if torch.cuda.is_available(): self.backend = "cuda"
                else: raise RuntimeError("Cuda is not available")
                self.device_count = min(max_workers, torch.cuda.device_count())

        self.scheduler = Scheduler(max_proc_req//2, max_proc_req//2)

        self.tokenizer = model_settings[model].tokenizer
        self.model = model_settings[model].model

        Logger.info(f"Launching tokenizer server with tokenizer : {self.tokenizer}")
        self.tokenizer =  TokenizerHandle(model=self.tokenizer)

        Logger.info(f"Launching model : {self.model}, with backend : {self.backend}, num workers : {self.device_count}") 
        self.workers = [GraphWorkerHandler(model, self.backend, id=i, endpoint="tcp://127.0.0.1:6000") for i in range(self.device_count)]

