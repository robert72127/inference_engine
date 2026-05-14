from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

class Model(ABC):
    def __init__(self, backend:str):
        pass
    def __call__(self, input, prefill, mask, uuid):
        pass
    def get_model_dir(self):
        pass
