"""Local Ollama PDF summarizer package."""

from pdf_summarizer.config import SummarizerConfig
from pdf_summarizer.summarizer import summarize_pdf

__all__ = ["SummarizerConfig", "summarize_pdf"]

