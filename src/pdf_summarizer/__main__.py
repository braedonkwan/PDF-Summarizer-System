from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from pdf_summarizer.config import SummarizerConfig
from pdf_summarizer.llm import create_llm_client, preflight_ollama
from pdf_summarizer.summarizer import summarize_pdf


def build_config(argv: Sequence[str] | None = None) -> SummarizerConfig:
    parser = argparse.ArgumentParser(
        prog="pdf-summarizer",
        description="Create a detailed, recursively compressed PDF summary with Ollama.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="pdf_file_path",
        type=Path,
        help="PDF to summarize (default: input.pdf).",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file_path",
        type=Path,
        help="Summary destination (default: summary.txt).",
    )
    parser.add_argument(
        "--model",
        dest="llm_model",
        help="Ollama model name; overrides the RTX 6000 Ada default.",
    )
    parser.add_argument(
        "--ollama-url",
        dest="ollama_base_url",
        help="Ollama server base URL.",
    )
    parser.add_argument(
        "--cache-dir",
        dest="llm_cache_dir",
        type=Path,
        help="Directory used for resumable model-response caching.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading and writing the model-response cache for this run.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the Ollama service/model check before PDF extraction.",
    )
    args = parser.parse_args(argv)

    config = SummarizerConfig.from_constants()
    overrides = {
        field: value
        for field, value in vars(args).items()
        if field not in {"no_cache", "skip_preflight"} and value is not None
    }
    if args.no_cache:
        overrides["llm_cache_enabled"] = False
    if args.skip_preflight:
        overrides["ollama_preflight"] = False
    return replace(config, **overrides)


def main(argv: Sequence[str] | None = None) -> None:
    config = build_config(argv)
    client = create_llm_client(config)
    preflight = (lambda: preflight_ollama(config)) if config.ollama_preflight else None
    summarize_pdf(config, client, preflight=preflight)


if __name__ == "__main__":
    main()
