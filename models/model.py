from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, backend:str):
        pass
    def __call__(self, input):
        pass
    def get_model_dir(self):
        pass