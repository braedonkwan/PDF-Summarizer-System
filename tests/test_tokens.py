from pdf_summarizer.tokens import Tokenizer


def test_chunk_text_short_text_returns_one_chunk() -> None:
    tokenizer = Tokenizer("cl100k_base")

    assert tokenizer.chunk_text("one two", chunk_length=10, chunk_overlap=0) == [
        "one two"
    ]


def test_chunk_text_without_overlap() -> None:
    tokenizer = Tokenizer("cl100k_base")
    text = "one two three four five six"

    chunks = tokenizer.chunk_text(text, chunk_length=2, chunk_overlap=0)

    assert [tokenizer.count(chunk) for chunk in chunks] == [2, 2, 2]


def test_chunk_text_with_overlap() -> None:
    tokenizer = Tokenizer("cl100k_base")
    text = "one two three four five"

    chunks = tokenizer.chunk_text(text, chunk_length=3, chunk_overlap=1)

    assert len(chunks) == 2
    assert tokenizer.encode(chunks[0])[-1:] == tokenizer.encode(chunks[1])[:1]


def test_chunk_text_exact_boundary() -> None:
    tokenizer = Tokenizer("cl100k_base")
    text = "one two three four"

    chunks = tokenizer.chunk_text(text, chunk_length=4, chunk_overlap=1)

    assert chunks == [text]


def test_chunk_text_prefers_paragraph_boundary_near_limit() -> None:
    tokenizer = Tokenizer("cl100k_base")
    first_paragraph = "Alpha beta gamma delta.\n\n"
    text = first_paragraph + "Second paragraph continues with several more words."
    boundary_tokens = tokenizer.count(first_paragraph)

    chunks = tokenizer.chunk_text(
        text,
        chunk_length=boundary_tokens + 2,
        chunk_overlap=0,
    )

    assert len(chunks) >= 2
    assert chunks[0].endswith("\n\n")
    assert "".join(chunks) == text


def test_chunk_text_prefers_sentence_boundary_when_no_paragraph_exists() -> None:
    tokenizer = Tokenizer("cl100k_base")
    first_sentence = "This is the first complete sentence."
    text = first_sentence + " Another sentence continues with several more words."
    boundary_tokens = tokenizer.count(first_sentence)

    chunks = tokenizer.chunk_text(
        text,
        chunk_length=boundary_tokens + 2,
        chunk_overlap=0,
    )

    assert len(chunks) >= 2
    assert chunks[0] == first_sentence
    assert "".join(chunks) == text


def test_chunk_text_falls_back_to_hard_limit_for_unstructured_text() -> None:
    tokenizer = Tokenizer("cl100k_base")
    text = " word" * 20

    chunks = tokenizer.chunk_text(text, chunk_length=5, chunk_overlap=0)

    assert all(tokenizer.count(chunk) <= 5 for chunk in chunks)
    assert "".join(chunks) == text
