from transformers import AutoTokenizer
import torch

from remote import subprocess_worker

@subprocess_worker()
class Tokenizer:
    def __init__(self, model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    def tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    def detokenize(self, toks: torch.Tensor):
        return self.tokenizer.batch_decode(
            toks,
            skip_special_tokens=True,
        )