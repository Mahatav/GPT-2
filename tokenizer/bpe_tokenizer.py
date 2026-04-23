"""
BPE tokenizer built from scratch.

Train on raw text: start with individual characters, repeatedly merge the
most frequent adjacent pair until the vocab hits the target size. Produces
shorter sequences than char-level and handles any text without OOV tokens.
"""

import json
from collections import Counter
from tokenizer.base import BaseTokenizer


class BPETokenizer(BaseTokenizer):

    def __init__(self):
        self._merges:     dict[tuple[str, str], str] = {}  # ordered: rank = insertion order
        self._encoder:    dict[str, int] = {}
        self._decoder:    dict[int, str] = {}
        self._merge_rank: dict[tuple[str, str], int] = {}
        self._trained = False

    @property
    def vocab_size(self) -> int:
        return len(self._encoder)

    def train(self, text: str, vocab_size: int, verbose: bool = True) -> None:
        base_chars = sorted(set(text))
        if vocab_size <= len(base_chars):
            raise ValueError(
                f"vocab_size ({vocab_size}) must be larger than unique chars ({len(base_chars)})"
            )

        self._encoder    = {c: i for i, c in enumerate(base_chars)}
        self._decoder    = {i: c for c, i in self._encoder.items()}
        self._merges     = {}
        self._merge_rank = {}

        tokens: list[str] = list(text)
        n_merges = vocab_size - len(base_chars)

        if verbose:
            print(f"  BPE: {len(base_chars)} base chars, learning {n_merges} merges")

        for idx in range(n_merges):
            pairs = self._count_pairs(tokens)
            if not pairs:
                break

            best   = max(pairs, key=pairs.get)
            merged = best[0] + best[1]
            new_id = len(self._encoder)

            self._encoder[merged]  = new_id
            self._decoder[new_id]  = merged
            self._merges[best]     = merged
            self._merge_rank[best] = idx

            tokens = self._apply_merge(tokens, best, merged)

            if verbose and (idx + 1) % 100 == 0:
                print(f"    merge {idx+1:4d}/{n_merges}  '{best[0]}'+'{best[1]}'->'{merged}'  freq={pairs[best]}")

        self._trained = True
        if verbose:
            print(f"  BPE done. vocab size: {self.vocab_size}")

    def encode(self, text: str) -> list[int]:
        if not self._trained:
            raise RuntimeError("call .train() first")

        tokens = [c for c in text if c in self._encoder]

        while len(tokens) > 1:
            pairs    = {(tokens[i], tokens[i+1]) for i in range(len(tokens) - 1)}
            eligible = [(self._merge_rank[p], p) for p in pairs if p in self._merge_rank]
            if not eligible:
                break
            _, best = min(eligible)
            tokens = self._apply_merge(tokens, best, self._merges[best])

        return [self._encoder[t] for t in tokens]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._decoder.get(i, "") for i in ids)

    @staticmethod
    def _count_pairs(tokens: list[str]) -> Counter:
        return Counter((tokens[i], tokens[i+1]) for i in range(len(tokens) - 1))

    @staticmethod
    def _apply_merge(tokens: list[str], pair: tuple[str, str], merged: str) -> list[str]:
        result, i = [], 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                result.append(merged)
                i += 2
            else:
                result.append(tokens[i])
                i += 1
        return result

    def save(self, path: str) -> None:
        data = {
            "encoder": self._encoder,
            "merges":  {f"{a}\x00{b}": m for (a, b), m in self._merges.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok._encoder = {k: int(v) for k, v in data["encoder"].items()}
        tok._decoder = {int(v): k for k, v in data["encoder"].items()}
        tok._merges  = {}
        tok._merge_rank = {}
        for rank, (key, merged) in enumerate(data["merges"].items()):
            a, b = key.split("\x00", 1)
            tok._merges[(a, b)]     = merged
            tok._merge_rank[(a, b)] = rank
        tok._trained = True
        return tok

    def vocab_table(self, max_rows: int = 30) -> str:
        lines = [f"{'ID':>6}  TOKEN"]
        for i in range(min(self.vocab_size, max_rows)):
            lines.append(f"{i:>6}  {repr(self._decoder[i])}")
        if self.vocab_size > max_rows:
            lines.append(f"  ... ({self.vocab_size - max_rows} more)")
        return "\n".join(lines)
