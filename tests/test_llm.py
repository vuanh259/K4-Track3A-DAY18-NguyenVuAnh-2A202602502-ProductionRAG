"""Provider-selection tests; these do not make network requests."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src import llm


def test_gemini_is_preferred(monkeypatch):
    import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "openai-test-key")
    assert llm.active_provider() == "gemini"
    assert llm.chat_model_name() == config.GEMINI_MODEL


def test_ollama_selection_overrides_api_keys(monkeypatch):
    import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "openai-test-key")

    assert llm.active_provider() == "ollama"
    assert llm.chat_model_name() == config.OLLAMA_MODEL


def test_normalize_abstention_for_evaluation():
    abstention = "Ngữ cảnh hiện có không cung cấp đủ thông tin để trả lời câu hỏi."
    assert llm.normalize_abstention("Không tìm thấy.") == abstention
    assert llm.normalize_abstention(abstention) == abstention
    with pytest.raises(ValueError, match="empty answer"):
        llm.normalize_abstention("  ")


def test_ollama_client_uses_local_openai_compatible_endpoint(monkeypatch):
    import config

    created = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    llm.create_chat_client()

    assert created["api_key"] == "ollama"
    assert created["base_url"] == f"{config.OLLAMA_BASE_URL}/v1"


def test_ollama_ragas_models_use_local_chat_and_embeddings(monkeypatch):
    import config

    calls = []
    community_embeddings = ModuleType("langchain_community.embeddings")
    langchain_openai = ModuleType("langchain_openai")
    community_embeddings.OllamaEmbeddings = lambda **kwargs: (
        calls.append(("embeddings", kwargs)) or "local-embeddings"
    )
    langchain_openai.ChatOpenAI = lambda **kwargs: (
        calls.append(("chat", kwargs)) or "local-chat"
    )
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", community_embeddings)
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai)

    chat, embeddings, provider = llm.create_ragas_models()

    assert (chat, embeddings, provider) == (
        "local-chat",
        "local-embeddings",
        "ollama",
    )
    assert calls[0][0] == "chat"
    assert calls[0][1]["base_url"] == f"{config.OLLAMA_BASE_URL}/v1"
    assert calls[1] == (
        "embeddings",
        {
            "model": config.OLLAMA_EMBEDDING_MODEL,
            "base_url": config.OLLAMA_BASE_URL,
        },
    )


def test_gemini_client_uses_official_compatible_endpoint(monkeypatch):
    import config

    created = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    llm.create_chat_client()

    assert created["api_key"] == "gemini-test-key"
    assert created["base_url"] == config.GEMINI_OPENAI_BASE_URL


def test_gemini_ragas_models_use_gemini_provider(monkeypatch):
    import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-test-key")
    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(llm, "_get_gemini_rate_limiter", lambda: SimpleNamespace(acquire=lambda: None))
    calls = []
    fake_module = ModuleType("langchain_google_genai")

    def fake_chat(**kwargs):
        calls.append(("chat", kwargs))
        return "chat-model"

    class FakeEmbeddings:
        def embed_documents(self, texts):
            return [[1.0] for _ in texts]

        def embed_query(self, text):
            return [1.0]

    def fake_embeddings(**kwargs):
        calls.append(("embeddings", kwargs))
        return FakeEmbeddings()

    fake_module.ChatGoogleGenerativeAI = fake_chat
    fake_module.GoogleGenerativeAIEmbeddings = fake_embeddings
    monkeypatch.setitem(sys.modules, "langchain_google_genai", fake_module)

    chat, embeddings, provider = llm.create_ragas_models()

    assert chat == "chat-model"
    assert embeddings.embed_query("query") == [1.0]
    assert embeddings.embed_documents(["doc"]) == [[1.0]]
    assert provider == "gemini"
    assert all(call[1]["google_api_key"] == "gemini-test-key" for call in calls)


def test_gemini_chat_uses_shared_rate_limiter(monkeypatch):
    import config

    monkeypatch.setattr(config, "LLM_PROVIDER", "auto")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "gemini-test-key")
    calls = []
    limiter = SimpleNamespace(acquire=lambda: calls.append("acquire"))
    monkeypatch.setattr(llm, "_get_gemini_rate_limiter", lambda: limiter)

    llm.wait_for_gemini_request()

    assert calls == ["acquire"]
