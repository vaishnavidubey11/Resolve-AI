"""OpenAI-compatible chat + embeddings (works with OpenAI and compatible gateways)."""
import logging
import time

import httpx

from app.config import Settings
from app.providers.base import ChatTurn, ContextPassage, EmbeddingProvider, LLMProvider, ProviderError
from app.providers.prompts import build_user_prompt

log = logging.getLogger("resolve.provider")


class _Http:
    def __init__(self, s: Settings):
        if not s.openai_api_key:
            raise ProviderError("OPENAI_API_KEY is not configured.")
        self.s = s
        self.client = httpx.Client(
            base_url=s.openai_base_url.rstrip("/"), timeout=s.provider_timeout_seconds,
            headers={"Authorization": f"Bearer {s.openai_api_key}"},
        )

    def post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.s.provider_max_retries + 1):
            try:
                r = self.client.post(path, json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise ProviderError(f"upstream status {r.status_code}")
                if r.status_code >= 400:
                    raise ProviderError(f"upstream rejected request ({r.status_code})")
                return r.json()
            except (httpx.TimeoutException, httpx.TransportError, ProviderError, ValueError) as e:
                last = e
                if isinstance(e, ProviderError) and "rejected" in str(e):
                    break  # 4xx other than 429: retrying will not help
                time.sleep(min(2 ** attempt * 0.5, 4))
        log.warning("provider call failed", extra={"outcome": type(last).__name__})
        raise ProviderError("The AI provider is temporarily unavailable.") from last


class OpenAIEmbeddingProvider(EmbeddingProvider):
    recommended_min_score = 0.30

    def __init__(self, s: Settings):
        self.http, self.model, self.dim = _Http(s), s.openai_embedding_model, s.embedding_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            data = self.http.post("/embeddings", {
                "model": self.model, "input": texts[i:i + 64], "dimensions": self.dim,
            })
            try:
                out.extend(d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"]))
            except (KeyError, TypeError) as e:
                raise ProviderError("Malformed embedding response.") from e
        return out


class OpenAILLMProvider(LLMProvider):
    def __init__(self, s: Settings):
        self.http, self.model = _Http(s), s.openai_chat_model

    def generate(self, system: str, history: list[ChatTurn], question: str,
                 passages: list[ContextPassage]) -> str:
        messages = [{"role": "system", "content": system}]
        messages += [{"role": t.role, "content": t.content} for t in history]
        messages.append({"role": "user", "content": build_user_prompt(question, passages)})
        data = self.http.post("/chat/completions", {"model": self.model, "messages": messages, "temperature": 0.1})
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ProviderError("Malformed LLM response.") from e
