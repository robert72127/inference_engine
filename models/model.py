from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

class PrefillStateBuff:
    def __init__(self, batch_size: int, chunk_size: int, max_pages:int, device: torch.device):
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.tokens = torch.empty((batch_size, chunk_size), device=device, dtype=torch.long)
        self.request_slots = torch.empty((batch_size,), device=device, dtype=torch.int32)
        self.offsets = torch.empty((batch_size,), device=device, dtype=torch.int32)
        self.seq_lens = torch.empty((batch_size,), device=device, dtype=torch.int32)
        self.mask = torch.empty((batch_size, chunk_size), device=device, dtype=torch.bool)
        self.last_chunk = torch.empty((batch_size, chunk_size), device=device, dtype=torch.bool)
        self.page_indexes = torch.empty((batch_size, max_pages), device=device, dtype=torch.int32)

class DecodeStateBuff:
    def __init__(self, batch_size: int, max_pages:int, device: torch.device):
        self.batch_size = batch_size
        self.tokens = torch.empty((batch_size, 1), device=device, dtype=torch.long)
        self.request_slots = torch.empty((batch_size,), device=device, dtype=torch.int32)
        self.offsets = torch.empty((batch_size,), device=device, dtype=torch.int32)
        self.page_indexes = torch.empty((batch_size, max_pages), device=device, dtype=torch.int32)

class Model(ABC):
    def __init__(self, device: torch.device):
        pass

    @abstractmethod
    def prefill(self, state_buff:PrefillStateBuff):
        pass

    @abstractmethod
    def decode(self, state_buff:DecodeStateBuff):
        pass

    @abstractmethod
    def get_model_dir(self):
        pass


    @abstractmethod
    def KV_init(self, handle_id, prompt_tokens):
        pass