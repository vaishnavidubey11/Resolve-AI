"""Deterministic offline providers for local demos and tests (NFR-007)."""
import hashlib
import math
from collections import Counter

from app.providers.base import ChatTurn, ContextPassage, EmbeddingProvider, LLMProvider
from app.providers.text import sentences, tokens


class MockEmbeddingProvider(EmbeddingProvider):
    """Hashed bag-of-words (+ bigrams). Not semantic, but stable and good enough to test retrieval plumbing."""

    recommended_min_score = 0.18

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        toks = tokens(text)
        feats = Counter(toks) + Counter(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        v = [0.0] * self.dim
        for feat, count in feats.items():
            h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
            sign = 1.0 if (h >> 64) & 1 else -1.0
            v[h % self.dim] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class MockLLMProvider(LLMProvider):
    """Extractive 'LLM': returns the most question-relevant sentences from the supplied context only."""

    def generate(self, system: str, history: list[ChatTurn], question: str,
                 passages: list[ContextPassage]) -> str:
        q = set(tokens(question))
        sents: list[tuple[int, int, str, set[str]]] = []  # (passage idx, sentence idx, text, tokens)
        for pi, p in enumerate(passages):
            body = p.text[len(p.section) + 1:] if p.section and p.text.startswith(p.section + "\n") else p.text
            for si, s in enumerate(sentences(body)):
                sents.append((pi, si, s, set(tokens(s))))
        if not sents:
            return "I don't have enough information to answer that."
        df = Counter(t for *_, toks in sents for t in toks)  # rarer shared terms are more informative
        ranked = sorted(sents, key=lambda x: (-sum(1.0 / df[t] for t in q & x[3]), x[0], x[1]))
        top = [x for x in ranked[:2] if q & x[3]] or [ranked[0]]
        return " ".join(x[2] for x in sorted(top, key=lambda x: (x[0], x[1])))
