"""Provider selection and clients shared by generation and RAGAS."""

import config

_gemini_rate_limiter = None


def active_provider() -> str | None:
    selected_provider = config.LLM_PROVIDER
    if selected_provider == "ollama":
        return "ollama"
    if selected_provider == "gemini":
        return "gemini" if config.GEMINI_API_KEY else None
    if selected_provider == "openai":
        return "openai" if config.OPENAI_API_KEY else None
    if selected_provider != "auto":
        raise ValueError(f"Unsupported LLM_PROVIDER: {selected_provider}")
    if config.GEMINI_API_KEY:
        return "gemini"
    if config.OPENAI_API_KEY:
        return "openai"
    return None


def wait_for_gemini_request() -> None:
    """Serialize Gemini requests below the configured per-minute quota."""
    if active_provider() != "gemini":
        return
    _get_gemini_rate_limiter().acquire()


def chat_model_name() -> str:
    provider = active_provider()
    if provider == "gemini":
        return config.GEMINI_MODEL
    if provider == "openai":
        return config.OPENAI_MODEL
    if provider == "ollama":
        return config.OLLAMA_MODEL
    raise RuntimeError(
        "Set LLM_PROVIDER and the matching credentials/model to enable the LLM."
    )


def normalize_abstention(answer: str | None) -> str:
    if not answer or not answer.strip():
        raise ValueError("The LLM returned an empty answer.")
    normalized = answer.strip().casefold().rstrip(".!?")
    if normalized in {
        "không tìm thấy",
        "không tìm thấy thông tin",
        "không tìm thấy đủ thông tin",
    }:
        return "Ngữ cảnh hiện có không cung cấp đủ thông tin để trả lời câu hỏi."
    return answer.strip()


def create_chat_client():
    """Create an OpenAI SDK client, using Gemini's official compatible endpoint."""
    provider = active_provider()
    if provider is None:
        return None

    from openai import OpenAI

    if provider == "gemini":
        return OpenAI(
            api_key=config.GEMINI_API_KEY,
            base_url=config.GEMINI_OPENAI_BASE_URL,
            timeout=60,
            max_retries=0,
        )
    if provider == "ollama":
        return OpenAI(
            api_key="ollama",
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            timeout=120,
            max_retries=0,
        )
    return OpenAI(api_key=config.OPENAI_API_KEY, timeout=60, max_retries=1)


def create_ragas_models():
    """Build provider-specific LangChain models required by RAGAS."""
    provider = active_provider()
    if provider == "gemini":
        from langchain_core.embeddings import Embeddings
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
            GoogleGenerativeAIEmbeddings,
        )

        limiter = _get_gemini_rate_limiter()
        raw_embeddings = GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBEDDING_MODEL,
            google_api_key=config.GEMINI_API_KEY,
        )

        class RateLimitedGeminiEmbeddings(Embeddings):
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                limiter.acquire()
                return raw_embeddings.embed_documents(texts)

            def embed_query(self, text: str) -> list[float]:
                limiter.acquire()
                return raw_embeddings.embed_query(text)

        return (
            ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                google_api_key=config.GEMINI_API_KEY,
                temperature=0,
                max_retries=0,
                rate_limiter=limiter,
            ),
            RateLimitedGeminiEmbeddings(),
            "gemini",
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        return (
            ChatOpenAI(
                model=config.OPENAI_MODEL,
                temperature=0,
                api_key=config.OPENAI_API_KEY,
            ),
            OpenAIEmbeddings(
                model=config.OPENAI_EMBEDDING_MODEL,
                api_key=config.OPENAI_API_KEY,
            ),
            "openai",
        )
    if provider == "ollama":
        import httpx
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_openai import ChatOpenAI

        return (
            ChatOpenAI(
                model=config.OLLAMA_MODEL,
                temperature=0,
                max_tokens=2048,
                api_key="ollama",
                base_url=f"{config.OLLAMA_BASE_URL}/v1",
                timeout=120,
                max_retries=0,
                http_async_client=httpx.AsyncClient(
                    limits=httpx.Limits(max_keepalive_connections=0),
                ),
            ),
            OllamaEmbeddings(
                model=config.OLLAMA_EMBEDDING_MODEL,
                base_url=config.OLLAMA_BASE_URL,
            ),
            "ollama",
        )
    raise RuntimeError(
        "Set LLM_PROVIDER and the matching credentials/model to enable RAGAS."
    )


def _get_gemini_rate_limiter():
    global _gemini_rate_limiter
    if _gemini_rate_limiter is None:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        interval = config.GEMINI_MIN_REQUEST_INTERVAL_SECONDS
        requests_per_second = 1.0 / interval if interval > 0 else 1000.0
        _gemini_rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=min(0.1, interval or 0.1),
            max_bucket_size=1,
        )
    return _gemini_rate_limiter
