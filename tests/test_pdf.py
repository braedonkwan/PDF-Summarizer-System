from pathlib import Path

from pypdf import PdfWriter

from pdf_summarizer.pdf import extract_pdf_text


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
