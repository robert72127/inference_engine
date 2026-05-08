import torch
from utils.remote import subprocess_worker

@subprocess_worker()
class Graph:
    def __init__(self, model: str, backend:str, endpoint:str):
        self.model = model
        self.backend = backend

    def serve(self):
        while True:
            pass
    
