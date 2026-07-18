import httpx
import pytest

from pdf_summarizer.config import SummarizerConfig
from pdf_summarizer.llm import (
    CachedLLMClient,
    OllamaClient,
    OutputLimitReached,
    SUMMARY_COMPLETION_MARKER,
    create_llm_client,
    preflight_ollama,
)


class CountingClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def summarize(self, prompt: str, *, max_output_tokens: int | None = None) -> str:
        self.calls += 1
        return self.responses.pop(0)


def test_ollama_client_sends_context_and_thinking_options(
    monkeypatch,
    tmp_path,
) -> None:
    requests = []

    def fake_post(url, *, json, timeout):
        requests.append((url, json, timeout))
        return httpx.Response(
            200,
            json={"response": "summary"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = SummarizerConfig(
        pdf_file_path=tmp_path / "input.pdf",
        chunk_length=12000,
        chunk_overlap=800,
        max_context_length=48000,
        compression_ratio=0.22,
        max_compression_retries=2,
        llm_model="heretic-model",
        llm_temperature=0.0,
        llm_max_output_tokens=4096,
        llm_context_window=65536,
        llm_think=False,
        llm_timeout_seconds=600.0,
        llm_system_prompt="Summarize.",
        output_file_path=tmp_path / "summary.txt",
        ollama_base_url="http://localhost:11434",
        token_encoding="cl100k_base",
    )

    result = OllamaClient.from_config(config).summarize("source", max_output_tokens=2048)

    assert result == "summary"
    assert requests
    _, payload, timeout = requests[0]
    assert timeout == 600.0
    assert payload["model"] == "heretic-model"
    assert payload["think"] is False
    assert payload["keep_alive"] == -1
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_predict"] == 2048
    assert payload["options"]["num_ctx"] == 65536


def test_cached_client_reuses_completed_request(tmp_path) -> None:
    underlying = CountingClient(["cached summary"])
    client = CachedLLMClient(underlying, tmp_path / "cache", "model-settings")

    first = client.summarize("same prompt", max_output_tokens=100)
    second = client.summarize("same prompt", max_output_tokens=100)

    assert first == second == "cached summary"
    assert underlying.calls == 1


def test_cached_client_does_not_store_summary_missing_required_marker(
    tmp_path,
) -> None:
    completed = f"complete summary\n{SUMMARY_COMPLETION_MARKER}"
    underlying = CountingClient(["incomplete summary", completed])
    client = CachedLLMClient(underlying, tmp_path / "cache", "model-settings")
    prompt = f"Finish with {SUMMARY_COMPLETION_MARKER}"

    assert client.summarize(prompt, max_output_tokens=100) == "incomplete summary"
    assert client.summarize(prompt, max_output_tokens=100) == completed
    assert underlying.calls == 2


def test_cached_client_ignores_cached_final_missing_required_marker(
    tmp_path,
) -> None:
    first = f"first complete\n{SUMMARY_COMPLETION_MARKER}"
    regenerated = f"regenerated complete\n{SUMMARY_COMPLETION_MARKER}"
    underlying = CountingClient([first, regenerated])
    cache_dir = tmp_path / "cache"
    client = CachedLLMClient(underlying, cache_dir, "model-settings")
    prompt = f"Finish with {SUMMARY_COMPLETION_MARKER}"

    assert client.summarize(prompt, max_output_tokens=100) == first
    cache_file = next(cache_dir.glob("*.txt"))
    cache_file.write_text("cached but incomplete", encoding="utf-8")

    assert client.summarize(prompt, max_output_tokens=100) == regenerated
    assert underlying.calls == 2


def test_cached_client_invalidates_on_prompt_output_or_namespace_change(tmp_path) -> None:
    underlying = CountingClient(["one", "two", "three", "four"])
    cache_dir = tmp_path / "cache"
    client = CachedLLMClient(underlying, cache_dir, "namespace-a")

    assert client.summarize("prompt-a", max_output_tokens=100) == "one"
    assert client.summarize("prompt-b", max_output_tokens=100) == "two"
    assert client.summarize("prompt-a", max_output_tokens=200) == "three"
    assert (
        CachedLLMClient(underlying, cache_dir, "namespace-b").summarize(
            "prompt-a",
            max_output_tokens=100,
        )
        == "four"
    )
    assert underlying.calls == 4


def test_cached_client_ignores_corrupt_empty_entry(tmp_path) -> None:
    underlying = CountingClient(["first", "regenerated"])
    cache_dir = tmp_path / "cache"
    client = CachedLLMClient(underlying, cache_dir, "settings")
    assert client.summarize("prompt", max_output_tokens=100) == "first"
    cache_file = next(cache_dir.glob("*.txt"))
    cache_file.write_text("", encoding="utf-8")

    assert client.summarize("prompt", max_output_tokens=100) == "regenerated"
    assert underlying.calls == 2


def test_create_llm_client_respects_cache_setting(tmp_path) -> None:
    config = _make_config(tmp_path)
    enabled = config.__class__(
        **{**config.__dict__, "llm_cache_enabled": True, "llm_cache_dir": tmp_path}
    )
    disabled = config.__class__(
        **{**config.__dict__, "llm_cache_enabled": False}
    )

    assert isinstance(create_llm_client(enabled), CachedLLMClient)
    assert isinstance(create_llm_client(disabled), OllamaClient)


def test_ollama_preflight_checks_configured_model(monkeypatch, tmp_path) -> None:
    requests = []

    def fake_post(url, *, json, timeout):
        requests.append((url, json, timeout))
        return httpx.Response(200, json={"details": {}}, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _make_config(tmp_path)

    preflight_ollama(config)

    assert requests == [
        (
            "http://localhost:11434/api/show",
            {"model": "heretic-model", "verbose": False},
            10.0,
        )
    ]


def test_ollama_preflight_reports_missing_model(monkeypatch, tmp_path) -> None:
    def fake_post(url, *, json, timeout):
        return httpx.Response(
            404,
            json={"error": "model not found"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="ollama pull heretic-model"):
        preflight_ollama(_make_config(tmp_path))


def test_ollama_preflight_reports_unreachable_service(monkeypatch, tmp_path) -> None:
    def fake_post(url, *, json, timeout):
        raise httpx.ConnectError(
            "connection refused",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(RuntimeError, match="Start `ollama serve`"):
        preflight_ollama(_make_config(tmp_path))


@pytest.mark.parametrize("response_text", ["", "   "])
def test_ollama_client_rejects_empty_responses(
    monkeypatch,
    tmp_path,
    response_text: str,
) -> None:
    def fake_post(url, *, json, timeout):
        return httpx.Response(
            200,
            json={"response": response_text, "done_reason": "stop"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient.from_config(_make_config(tmp_path))

    with pytest.raises(RuntimeError, match="empty text response"):
        client.summarize("source")


def test_ollama_client_reports_output_limit_with_partial_text(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_post(url, *, json, timeout):
        return httpx.Response(
            200,
            json={"response": "unfinished summary", "done_reason": "length"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient.from_config(_make_config(tmp_path))

    with pytest.raises(OutputLimitReached) as exc_info:
        client.summarize("source")

    assert exc_info.value.partial_text == "unfinished summary"


def test_ollama_client_treats_unfinished_response_as_output_limit(
    monkeypatch,
    tmp_path,
) -> None:
    def fake_post(url, *, json, timeout):
        return httpx.Response(
            200,
            json={"response": "unfinished summary", "done": False},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    client = OllamaClient.from_config(_make_config(tmp_path))

    with pytest.raises(OutputLimitReached):
        client.summarize("source")


def test_ollama_client_retries_transient_transport_failure(
    monkeypatch,
    tmp_path,
) -> None:
    request = httpx.Request("POST", "http://localhost:11434/api/generate")
    outcomes = [
        httpx.ConnectError("temporarily unavailable", request=request),
        httpx.Response(200, json={"response": "summary"}, request=request),
    ]
    sleeps: list[float] = []

    def fake_post(url, *, json, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("pdf_summarizer.llm.time.sleep", sleeps.append)
    config = _make_config(tmp_path)
    config = config.__class__(
        **{
            **config.__dict__,
            "llm_request_retries": 2,
            "llm_retry_backoff_seconds": 0.25,
        }
    )

    assert OllamaClient.from_config(config).summarize("source") == "summary"
    assert sleeps == [0.25]
    assert outcomes == []


def test_ollama_client_retries_server_error_but_not_client_error(
    monkeypatch,
    tmp_path,
) -> None:
    statuses = [503, 400]
    calls = 0

    def fake_post(url, *, json, timeout):
        nonlocal calls
        calls += 1
        return httpx.Response(
            statuses.pop(0),
            json={"error": "failure"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr("pdf_summarizer.llm.time.sleep", lambda _seconds: None)
    config = _make_config(tmp_path)
    config = config.__class__(
        **{
            **config.__dict__,
            "llm_request_retries": 3,
            "llm_retry_backoff_seconds": 0.0,
        }
    )

    with pytest.raises(RuntimeError, match="400 Bad Request"):
        OllamaClient.from_config(config).summarize("source")

    assert calls == 2


def _make_config(tmp_path) -> SummarizerConfig:
    return SummarizerConfig(
        pdf_file_path=tmp_path / "input.pdf",
        chunk_length=12000,
        chunk_overlap=800,
        max_context_length=48000,
        compression_ratio=0.22,
        max_compression_retries=2,
        llm_model="heretic-model",
        llm_temperature=0.0,
        llm_max_output_tokens=4096,
        llm_context_window=65536,
        llm_think=False,
        llm_timeout_seconds=600.0,
        llm_system_prompt="Summarize.",
        output_file_path=tmp_path / "summary.txt",
        ollama_base_url="http://localhost:11434",
        token_encoding="cl100k_base",
    )
