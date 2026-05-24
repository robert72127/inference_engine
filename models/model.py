from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class ModelForwardState:
    tokens: torch.Tensor
    prefill: bool
    mask: torch.Tensor | None
    lengths: torch.Tensor | None
    uuid: tuple[int, ...]
    prefill_last_chunk: torch.Tensor | None = None

class Model(ABC):
    def __init__(self, device: torch.device):
        pass

    @abstractmethod
    def __call__(self, input: torch.Tensor, state: ModelForwardState):
        pass

    @abstractmethod
    def get_model_dir(self):
        pass
