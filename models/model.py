from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class PrefillState:
    offset: int
    last_chunk: bool
    prompt_tokens: list[int]
    length: torch.Tensor | None = None

@dataclass(frozen=True)
class ModelForwardState:
    tokens: torch.Tensor
    uuid: tuple[int, ...]
    mask: torch.Tensor | None = None
    prefill_state : list[PrefillState] | None = None



class Model(ABC):
    def __init__(self, device: torch.device):
        pass

    @abstractmethod
    def prefill(self,input:torch.Tensor, state:ModelForwardState, batch_size:int):
        pass

    @abstractmethod
    def decode(self,input:torch.Tensor, state:ModelForwardState, batch_size:int):
        pass

    @abstractmethod
    def get_model_dir(self):
        pass


    @abstractmethod
    def KV_init(self, handle_id, prompt_tokens):
        pass