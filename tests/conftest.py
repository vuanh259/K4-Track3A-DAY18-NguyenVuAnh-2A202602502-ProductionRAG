"""Unit tests never call a configured cloud/local LLM accidentally."""
import pytest


@pytest.fixture(autouse=True)
def isolate_llm_credentials(monkeypatch):
    import config

    monkeypatch.setattr(config, 'LLM_PROVIDER', 'auto')
    monkeypatch.setattr(config, 'GEMINI_API_KEY', '')
    monkeypatch.setattr(config, 'OPENAI_API_KEY', '')
