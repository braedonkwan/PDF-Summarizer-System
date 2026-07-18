from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ExtractedPDFText(str):
    """Extracted text plus page-coverage information.

    This is a ``str`` subclass to preserve the original public API while making
    partial extraction visible to the summarization pipeline.
    """

    total_pages: int
    extracted_pages: tuple[int, ...]
    unextractable_pages: tuple[int, ...]

    def __new__(
        cls,
        text: str,
        *,
        total_pages: int,
        extracted_pages: Iterable[int],
        unextractable_pages: Iterable[int],
    ) -> "ExtractedPDFText":
        instance = super().__new__(cls, text)
        instance.total_pages = total_pages
        instance.extracted_pages = tuple(extracted_pages)
        instance.unextractable_pages = tuple(unextractable_pages)
        return instance

    @property
    def coverage_ratio(self) -> float:
        if self.total_pages == 0:
            return 0.0
        return len(self.extracted_pages) / self.total_pages


def extract_pdf_text(pdf_path: Path) -> ExtractedPDFText:
    try:
        reader = PdfReader(str(pdf_path))
    except (OSError, PdfReadError) as exc:
        raise ValueError(f"Could not read PDF: {pdf_path}: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        try:
            unlocked = reader.decrypt("")
        except (PdfReadError, TypeError, ValueError) as exc:
            raise ValueError(
                f"PDF is encrypted and cannot be opened without a password: {pdf_path}"
            ) from exc
        if not unlocked:
            raise ValueError(
                f"PDF is encrypted and cannot be opened without a password: {pdf_path}"
            )

    page_text: dict[int, str] = {}
    extracted_pages: list[int] = []
    unextractable_pages: list[int] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            # Layout mode retains columns and table-like spacing that the plain
            # extractor tends to flatten. Compare against plain extraction because
            # layout mode can omit rotated or otherwise unusual text.
            layout_text = page.extract_text(extraction_mode="layout")
            plain_text = page.extract_text()
            text = _prefer_more_complete_text(layout_text, plain_text)
        except TypeError:
            text = page.extract_text()
        except (PdfReadError, ValueError, KeyError):
            text = None
        if text and text.strip():
            page_text[page_number] = text.strip()
            extracted_pages.append(page_number)
        else:
            unextractable_pages.append(page_number)

    pages = [
        f"[Page {page_number}]\n{page_text[page_number]}"
        for page_number in sorted(page_text)
    ]

    return ExtractedPDFText(
        "\n\n".join(pages),
        total_pages=len(reader.pages),
        extracted_pages=extracted_pages,
        unextractable_pages=unextractable_pages,
    )


def _prefer_more_complete_text(
    layout_text: str | None,
    plain_text: str | None,
) -> str | None:
    if not layout_text or not layout_text.strip():
        return plain_text
    if not plain_text or not plain_text.strip():
        return layout_text

    layout_characters = len(re.sub(r"\s+", "", layout_text))
    plain_characters = len(re.sub(r"\s+", "", plain_text))
    # Prefer layout unless plain mode finds substantially more real content.
    return plain_text if plain_characters > layout_characters * 1.2 else layout_text
