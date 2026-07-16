from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Protocol
from uuid import uuid4

import httpx

from pdf_summarizer.config import SummarizerConfig


class LLMClient(Protocol):
    def summarize(self, prompt: str, *, max_output_tokens: int | None = None) -> str:
        """Return a summary for the prompt."""


class OutputLimitReached(RuntimeError):
    """Raised when Ollama stops because it exhausted the generation budget."""

    def __init__(self, partial_text: str) -> None:
        super().__init__("Ollama reached the output-token limit before finishing.")
        self.partial_text = partial_text


@dataclass(frozen=True)
class CachedLLMClient:
    """Persist completed LLM calls so interrupted document runs can resume."""

    client: LLMClient
    cache_dir: Path
    namespace: str

    def summarize(self, prompt: str, *, max_output_tokens: int | None = None) -> str:
        identity = json.dumps(
            {
                "namespace": self.namespace,
                "prompt": prompt,
                "max_output_tokens": max_output_tokens,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        key = hashlib.sha256(identity).hexdigest()
        cache_path = self.cache_dir / f"{key}.txt"

        try:
            cached = cache_path.read_text(encoding="utf-8").strip()
            if cached:
                return cached
        except (FileNotFoundError, OSError, UnicodeError):
            pass

        summary = self.client.summarize(
            prompt,
            max_output_tokens=max_output_tokens,
        )
        self._store(cache_path, summary)
        return summary

    def _store(self, cache_path: Path, summary: str) -> None:
        temporary_path: Path | None = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_path.with_name(
                f".{cache_path.name}.{uuid4().hex}.tmp"
            )
            temporary_path.write_text(summary, encoding="utf-8")
            temporary_path.replace(cache_path)
        except OSError:
            # Caching must never turn a successful model response into a failed run.
            return
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


@dataclass(frozen=True)
class OllamaClient:
    model: str
    system_prompt: str
    temperature: float
    max_output_tokens: int
    context_window: int
    think: bool
    keep_alive: int | str
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 300.0
    request_retries: int = 0
    retry_backoff_seconds: float = 0.0

    @classmethod
    def from_config(cls, config: SummarizerConfig) -> "OllamaClient":
        return cls(
            model=config.llm_model,
            system_prompt=config.llm_system_prompt,
            temperature=config.llm_temperature,
            max_output_tokens=config.llm_max_output_tokens,
            context_window=config.llm_context_window,
            think=config.llm_think,
            keep_alive=config.llm_keep_alive,
            base_url=config.ollama_base_url,
            timeout_seconds=config.llm_timeout_seconds,
            request_retries=config.llm_request_retries,
            retry_backoff_seconds=config.llm_retry_backoff_seconds,
        )

    def summarize(self, prompt: str, *, max_output_tokens: int | None = None) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        token_limit = max_output_tokens or self.max_output_tokens
        payload = {
            "model": self.model,
            "system": self.system_prompt,
            "prompt": prompt,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": token_limit,
                "num_ctx": self.context_window,
            },
        }

        response: httpx.Response | None = None
        for attempt in range(self.request_retries + 1):
            try:
                response = httpx.post(url, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                break
            except httpx.HTTPError as exc:
                retryable = _is_retryable_http_error(exc)
                if not retryable or attempt >= self.request_retries:
                    raise RuntimeError(f"Ollama request failed: {exc}") from exc
                if self.retry_backoff_seconds:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))

        if response is None:  # Defensive: the loop either succeeds or raises.
            raise RuntimeError("Ollama request failed without a response.")

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Ollama returned an invalid JSON response.") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Ollama returned an unexpected response structure.")
        summary = data.get("response")
        if not isinstance(summary, str):
            raise RuntimeError("Ollama response did not include a text response.")
        summary = summary.strip()
        if not summary:
            raise RuntimeError("Ollama returned an empty text response.")
        if data.get("done_reason") == "length":
            raise OutputLimitReached(summary)
        return summary


def create_llm_client(config: SummarizerConfig) -> LLMClient:
    client: LLMClient = OllamaClient.from_config(config)
    if not config.llm_cache_enabled:
        return client

    namespace = json.dumps(
        {
            "model": config.llm_model,
            "system_prompt": config.llm_system_prompt,
            "temperature": config.llm_temperature,
            "context_window": config.llm_context_window,
            "think": config.llm_think,
        },
        sort_keys=True,
    )
    return CachedLLMClient(client, config.llm_cache_dir, namespace)


def preflight_ollama(config: SummarizerConfig) -> None:
    """Fail early when Ollama or the configured model is unavailable."""
    url = f"{config.ollama_base_url.rstrip('/')}/api/show"
    try:
        response = httpx.post(
            url,
            json={"model": config.llm_model, "verbose": False},
            timeout=min(10.0, config.llm_timeout_seconds),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise RuntimeError(
                f"Ollama model is not available locally: {config.llm_model}. "
                f"Run `ollama pull {config.llm_model}` before summarizing."
            ) from exc
        raise RuntimeError(f"Ollama preflight failed: {exc}") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {config.ollama_base_url}. Start `ollama serve` "
            "and confirm OLLAMA_BASE_URL or --ollama-url."
        ) from exc


def _is_retryable_http_error(exc: httpx.HTTPError) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return False
