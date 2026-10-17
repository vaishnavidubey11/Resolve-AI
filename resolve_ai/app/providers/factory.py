from app.config import Settings
from app.providers.base import EmbeddingProvider, LLMProvider


def build_providers(s: Settings) -> tuple[LLMProvider, EmbeddingProvider]:
    if s.llm_provider == "openai":
        from app.providers.openai_compat import OpenAILLMProvider
        llm: LLMProvider = OpenAILLMProvider(s)
    else:
        from app.providers.mock import MockLLMProvider
        llm = MockLLMProvider()
    if s.embedding_provider == "openai":
        from app.providers.openai_compat import OpenAIEmbeddingProvider
        emb: EmbeddingProvider = OpenAIEmbeddingProvider(s)
    else:
        from app.providers.mock import MockEmbeddingProvider
        emb = MockEmbeddingProvider(s.embedding_dim)
    return llm, emb
