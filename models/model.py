from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

class Model(ABC):
    def __init__(self, device: torch.device):
        pass
    @abstractmethod
    def __call__(self, input, tokens, prefill, mask, lengths, uuid, prefill_last_chunk):
        pass

    @abstractmethod
    def get_model_dir(self):
        pass
