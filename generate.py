import torch
import torch.nn.functional as F

from model.gpt2 import GPT2
from tokenizer.base import BaseTokenizer


class Generator:
    """
    Wraps a trained model for autoregressive text generation.

    Three strategies: greedy (deterministic), top-k (sample from k best),
    top-p / nucleus (sample from smallest set summing to p).
    Temperature scales logits before softmax — lower = more conservative.
    """

    def __init__(self, model: GPT2, tokenizer: BaseTokenizer, device: torch.device = None):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or next(model.parameters()).device
        self.model.eval()

    @torch.no_grad()
    def greedy(self, prompt: str, max_new_tokens: int = 100) -> str:
        idx = self._encode(prompt)
        for _ in range(max_new_tokens):
            logits, _ = self.model(self._crop(idx))
            idx = torch.cat([idx, logits[:, -1, :].argmax(dim=-1, keepdim=True)], dim=1)
        return self._decode(idx)

    @torch.no_grad()
    def top_k(self, prompt: str, max_new_tokens: int = 100, k: int = 40, temperature: float = 1.0) -> str:
        idx = self._encode(prompt)
        for _ in range(max_new_tokens):
            logits, _ = self.model(self._crop(idx))
            logits    = self._apply_top_k(logits[:, -1, :] / temperature, k)
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return self._decode(idx)

    @torch.no_grad()
    def top_p(self, prompt: str, max_new_tokens: int = 100, p: float = 0.9, temperature: float = 1.0) -> str:
        idx = self._encode(prompt)
        for _ in range(max_new_tokens):
            logits, _ = self.model(self._crop(idx))
            logits    = self._apply_top_p(logits[:, -1, :] / temperature, p)
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return self._decode(idx)

    @torch.no_grad()
    def sample(
        self,
        prompt:         str,
        max_new_tokens: int   = 100,
        temperature:    float = 1.0,
        top_k:          int   = None,
        top_p:          float = None,
    ) -> str:
        idx = self._encode(prompt)
        for _ in range(max_new_tokens):
            logits, _ = self.model(self._crop(idx))
            logits    = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = self._apply_top_k(logits, top_k)
            if top_p is not None:
                logits = self._apply_top_p(logits, top_p)
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return self._decode(idx)

    @staticmethod
    def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        k         = min(k, logits.size(-1))
        threshold = torch.topk(logits, k)[0][:, -1].unsqueeze(-1)
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # shift right so we keep the token that first pushes cumsum over p
        to_remove = (cumulative - F.softmax(sorted_logits, dim=-1)) > p
        sorted_logits[to_remove] = float("-inf")
        return logits.scatter(1, sorted_idx, sorted_logits)

    def _encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.tokenizer.encode(text)], dtype=torch.long, device=self.device)

    def _decode(self, idx: torch.Tensor) -> str:
        return self.tokenizer.decode(idx[0].tolist())

    def _crop(self, idx: torch.Tensor) -> torch.Tensor:
        m = self.model.config.block_size
        return idx if idx.size(1) <= m else idx[:, -m:]
