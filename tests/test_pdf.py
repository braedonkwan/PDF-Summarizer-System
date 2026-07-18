from pathlib import Path

from pypdf import PdfWriter

from pdf_summarizer.pdf import ExtractedPDFText, extract_pdf_text


def test_extract_pdf_text_ignores_pages_without_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as file:
        writer.write(file)

    assert extract_pdf_text(pdf_path) == ""


def test_extract_pdf_text_preserves_physical_page_numbers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pdf_summarizer.pdf as module

    class FakePage:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self) -> str | None:
            return self.text

    class FakeReader:
        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("First"), FakePage(None), FakePage("Third")]

    monkeypatch.setattr(module, "PdfReader", FakeReader)

    assert extract_pdf_text(tmp_path / "input.pdf") == (
        "[Page 1]\nFirst\n\n[Page 3]\nThird"
    )


def test_extract_pdf_text_reports_page_coverage_and_uses_layout_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pdf_summarizer.pdf as module

    modes: list[str | None] = []

    class FakePage:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self, *, extraction_mode: str | None = None) -> str | None:
            modes.append(extraction_mode)
            return self.text

    class FakeReader:
        is_encrypted = False

        def __init__(self, _path: str) -> None:
            self.pages = [FakePage("A  B"), FakePage(None), FakePage("C")]

    monkeypatch.setattr(module, "PdfReader", FakeReader)

    result = extract_pdf_text(tmp_path / "input.pdf")

    assert isinstance(result, ExtractedPDFText)
    assert result.total_pages == 3
    assert result.extracted_pages == (1, 3)
    assert result.unextractable_pages == (2,)
    assert result.coverage_ratio == 2 / 3
    assert modes == ["layout", None, "layout", None, "layout", None]
