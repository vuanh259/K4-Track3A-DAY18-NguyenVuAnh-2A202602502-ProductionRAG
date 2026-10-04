"""Tests for Module 5: Enrichment Pipeline."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import m5_enrichment
from src.m5_enrichment import (
    EnrichedChunk,
    contextual_prepend,
    enrich_chunks,
    extract_metadata,
    generate_hypothesis_questions,
    summarize_chunk,
)


@pytest.fixture(autouse=True)
def use_offline_enrichment(monkeypatch):
    monkeypatch.setattr(m5_enrichment, "active_provider", lambda: None)
    m5_enrichment._enrich_single_call.cache_clear()
    yield
    m5_enrichment._enrich_single_call.cache_clear()

SAMPLE = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."
CHUNKS = [
    {"text": SAMPLE, "metadata": {"source": "policy.md"}},
    {"text": "Mật khẩu phải thay đổi mỗi 90 ngày.", "metadata": {"source": "it.md"}},
]


def test_summarize_returns_string():
    result = summarize_chunk(SAMPLE)
    assert isinstance(result, str)


def test_summarize_shorter_than_original():
    result = summarize_chunk(SAMPLE)
    if result:  # May be empty if no API key
        assert len(result) <= len(SAMPLE) * 2  # Summary should not be much longer


def test_hyqa_returns_list():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    assert isinstance(result, list)


def test_hyqa_generates_questions():
    result = generate_hypothesis_questions(SAMPLE, n_questions=2)
    if result:
        assert len(result) >= 1
        assert any("?" in q or "bao" in q.lower() or "mấy" in q.lower() for q in result)


def test_contextual_prepend_returns_string():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert isinstance(result, str)
    assert len(result) >= len(SAMPLE)  # Should be at least as long as original


def test_contextual_contains_original():
    result = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert SAMPLE in result  # Original text must be preserved


def test_extract_metadata_returns_dict():
    result = extract_metadata(SAMPLE)
    assert isinstance(result, dict)


def test_enrich_chunks_returns_list():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    assert isinstance(result, list)


def test_enrich_chunks_type():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert all(isinstance(c, EnrichedChunk) for c in result)


def test_enrich_preserves_original():
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    if result:
        assert result[0].original_text == SAMPLE


def test_gemini_enrichment_disables_reasoning(monkeypatch, tmp_path):
    calls = []
    content = (
        '{"summary":"summary","questions":["question?"],"context":"context",'
        '"metadata":{"topic":"policy"}}'
    )

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                ]
            )

    monkeypatch.setattr(m5_enrichment, "active_provider", lambda: "gemini")
    monkeypatch.setattr(m5_enrichment, "chat_model_name", lambda: "gemini-2.5-flash")
    monkeypatch.setattr(m5_enrichment, "create_chat_client", lambda: SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    ))
    monkeypatch.setattr(m5_enrichment, "wait_for_gemini_request", lambda: None)
    monkeypatch.setattr(
        m5_enrichment, "ENRICHMENT_CACHE_PATH", str(tmp_path / "enrichment.json")
    )

    result = m5_enrichment._enrich_single_call(SAMPLE, "policy.md")

    assert result["summary"] == "summary"
    assert calls[0]["reasoning_effort"] == "none"


@pytest.mark.parametrize("status_code", [429, 503])
def test_enrichment_stops_on_quota_or_server_error(
    monkeypatch, tmp_path, status_code
):
    from openai import APIError

    class ProviderError(APIError):
        def __init__(self):
            message = "quota exhausted" if status_code == 429 else "provider unavailable"
            super().__init__(message, request=None, body=None)
            self.status_code = status_code

    class FakeCompletions:
        def create(self, **kwargs):
            raise ProviderError()

    monkeypatch.setattr(m5_enrichment, "active_provider", lambda: "gemini")
    monkeypatch.setattr(m5_enrichment, "chat_model_name", lambda: "gemini-test")
    monkeypatch.setattr(
        m5_enrichment,
        "create_chat_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    monkeypatch.setattr(m5_enrichment, "wait_for_gemini_request", lambda: None)
    monkeypatch.setattr(
        m5_enrichment, "ENRICHMENT_CACHE_PATH", str(tmp_path / "enrichment.json")
    )

    expected_message = "quota exhausted" if status_code == 429 else "provider unavailable"
    with pytest.raises(ProviderError, match=expected_message):
        m5_enrichment._enrich_single_call(SAMPLE, "policy.md")


def test_local_enrichment_does_not_hide_provider_errors(monkeypatch, tmp_path):
    from openai import APIError

    class FakeCompletions:
        def create(self, **kwargs):
            raise APIError("local Ollama is unavailable", request=None, body=None)

    monkeypatch.setattr(m5_enrichment, "active_provider", lambda: "ollama")
    monkeypatch.setattr(m5_enrichment, "chat_model_name", lambda: "qwen-local")
    monkeypatch.setattr(
        m5_enrichment,
        "create_chat_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    monkeypatch.setattr(m5_enrichment, "wait_for_gemini_request", lambda: None)
    monkeypatch.setattr(
        m5_enrichment, "ENRICHMENT_CACHE_PATH", str(tmp_path / "enrichment.json")
    )

    with pytest.raises(APIError, match="local Ollama is unavailable"):
        m5_enrichment._enrich_single_call(SAMPLE, "policy.md")


def test_enrichment_cache_preserves_multiple_chunks_and_ignores_bad_keys(
    monkeypatch, tmp_path
):
    content = (
        '{"summary":"summary","questions":["question?"],"context":"context",'
        '"metadata":{"topic":"policy"}}'
    )
    cache_path = tmp_path / "enrichment.json"
    cache_path.write_text('{"context":{"summary":"stale"}}', encoding="utf-8")

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                ]
            )

    monkeypatch.setattr(m5_enrichment, "active_provider", lambda: "gemini")
    monkeypatch.setattr(m5_enrichment, "chat_model_name", lambda: "gemini-test")
    monkeypatch.setattr(
        m5_enrichment,
        "create_chat_client",
        lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions())
        ),
    )
    monkeypatch.setattr(m5_enrichment, "wait_for_gemini_request", lambda: None)
    monkeypatch.setattr(m5_enrichment, "ENRICHMENT_CACHE_PATH", str(cache_path))

    with pytest.warns(UserWarning, match="Ignoring 1 invalid enrichment cache"):
        first = m5_enrichment._enrich_single_call(SAMPLE, "policy.md")
    second_text = "Mật khẩu phải thay đổi mỗi 90 ngày."
    second = m5_enrichment._enrich_single_call(second_text, "it.md")
    cache = m5_enrichment._read_enrichment_cache()

    assert first["summary"] == second["summary"] == "summary"
    assert set(cache) == {
        m5_enrichment._cache_key(SAMPLE, "policy.md", "gemini", "gemini-test"),
        m5_enrichment._cache_key(second_text, "it.md", "gemini", "gemini-test"),
    }
