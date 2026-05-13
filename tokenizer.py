from concurrent.futures import ThreadPoolExecutor
import asyncio

from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_dir: str, max_workers=1):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True
        )
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def _tokenize_sync(self, texts: list[str]):
        return self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    async def tokenize(self, texts: list[str]):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.pool,
            self._tokenize_sync,
            texts,
        )

    def detokenize_sync(self, toks):
        return self.tokenizer.batch_decode(
            toks,
            skip_special_tokens=True,
        )

    def __exit__(self, exc_type, exc, tb):
        self.close()