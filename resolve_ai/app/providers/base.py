"""Provider interfaces (FR-025). All provider-specific code lives behind these."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(Exception):
    """Upstream provider failed (timeout, 5xx, bad credentials, malformed response)."""


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ContextPassage:
    source_id: str
    title: str
    page: int | None
    section: str | None
    text: str
    score: float = 0.0


class EmbeddingProvider(ABC):
    dim: int
    recommended_min_score: float = 0.30

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system: str, history: list[ChatTurn], question: str,
                 passages: list[ContextPassage]) -> str: ...
