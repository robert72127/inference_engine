import torch

from utils.remote import subprocess_worker

@subprocess_worker
class ModelProcessor:
    def __init__(self, model_constructor: callable, backend:str):
        self.backend = backend
        self.model = model_constructor(backend)

    def call(self, input: torch.Tensor):
        return self.model(input)

    def prefill(self, input: torch.Tensor):
        pass

    def single_step(self, input: torch.Tensor):
        pass
