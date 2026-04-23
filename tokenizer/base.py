from abc import ABC, abstractmethod


class BaseTokenizer(ABC):
    """text <-> integer token ids"""

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @abstractmethod
    def encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str: ...

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def roundtrip(self, text: str) -> bool:
        return self.decode(self.encode(text)) == text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size})"
