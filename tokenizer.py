from transformers import AutoTokenizer
import torch

from utils.remote import subprocess_worker

@subprocess_worker
class Tokenizer:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )

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
