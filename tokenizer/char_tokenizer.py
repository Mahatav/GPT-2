import json
from tokenizer.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """
    Maps each unique character to an integer id.

    Simplest possible tokenizer — tiny vocab, long sequences, zero OOV.
    Good for quick experiments and demos.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self._vocab_size = len(chars)
        self._stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self._itos: dict[int, str] = {i: c for c, i in self._stoi.items()}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        return [self._stoi[c] for c in text if c in self._stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos.get(i, "") for i in ids)

    def vocab_list(self) -> list[str]:
        return [self._itos[i] for i in range(self._vocab_size)]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self._stoi}, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        inst = cls.__new__(cls)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        inst._stoi = data["stoi"]
        inst._itos = {int(v): k for k, v in inst._stoi.items()}
        inst._vocab_size = len(inst._stoi)
        return inst
