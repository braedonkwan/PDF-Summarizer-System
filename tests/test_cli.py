from pathlib import Path

from pdf_summarizer.__main__ import build_config, main


def test_build_config_uses_profile_defaults_without_arguments() -> None:
    config = build_config([])

    assert config.pdf_file_path == Path("input.pdf")
    assert config.output_file_path == Path("summary.txt")
    assert config.llm_cache_enabled is True


def test_build_config_applies_command_line_overrides(tmp_path: Path) -> None:
    input_path = tmp_path / "document.pdf"
    output_path = tmp_path / "result.md"
    cache_path = tmp_path / "cache"

    config = build_config(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--model",
            "custom-model",
            "--ollama-url",
            "http://ollama.example:11434",
            "--cache-dir",
            str(cache_path),
            "--no-cache",
            "--skip-preflight",
        ]
    )

    assert config.pdf_file_path == input_path
    assert config.output_file_path == output_path
    assert config.llm_model == "custom-model"
    assert config.ollama_base_url == "http://ollama.example:11434"
    assert config.llm_cache_dir == cache_path
    assert config.llm_cache_enabled is False
    assert config.ollama_preflight is False


def test_main_passes_cli_config_to_client_and_summarizer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    import pdf_summarizer.__main__ as module

    input_path = tmp_path / "document.pdf"
    captured = {}
    fake_client = object()

    def fake_create_client(config):
        captured["client_config"] = config
        return fake_client

    def fake_summarize(config, client, *, preflight=None):
        captured["summary_config"] = config
        captured["client"] = client
        captured["preflight"] = preflight

    monkeypatch.setattr(module, "create_llm_client", fake_create_client)
    monkeypatch.setattr(module, "summarize_pdf", fake_summarize)

    main(["--input", str(input_path), "--no-cache"])

    assert captured["client_config"] is captured["summary_config"]
    assert captured["summary_config"].pdf_file_path == input_path
    assert captured["summary_config"].llm_cache_enabled is False
    assert captured["client"] is fake_client
    assert callable(captured["preflight"])
