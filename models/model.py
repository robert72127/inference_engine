from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class PrefillState:
    offset: int
    first_chunk: bool
    last_chunk: bool
    prompt_tokens: torch.Tensor
    prompt_length: torch.Tensor
    length: torch.Tensor | None = None

@dataclass(frozen=True)
class ModelForwardState:
    tokens: torch.Tensor
    prefill: bool
    uuid: tuple[int, ...]
    mask: torch.Tensor | None = None
    prefill_state : list[PrefillState] | None = None



class Model(ABC):
    def __init__(self, device: torch.device):
        pass

    @abstractmethod
    def __call__(self, input: torch.Tensor, state: ModelForwardState) -> torch.Tensor:
        pass

    @abstractmethod
    def get_model_dir(self):
        pass
