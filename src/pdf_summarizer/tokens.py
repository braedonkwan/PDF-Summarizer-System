from __future__ import annotations

import re

import tiktoken


_SENTENCE_END = re.compile(r"[.!?][\"')\]}]*\s*$")


class Tokenizer:
    def __init__(self, encoding_name: str) -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)

    def chunk_text(
        self,
        text: str,
        *,
        chunk_length: int,
        chunk_overlap: int,
    ) -> list[str]:
        if chunk_length <= 0:
            raise ValueError("chunk_length must be positive.")
        if not 0 <= chunk_overlap < chunk_length:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_length.")

        tokens = self.encode(text)
        if not tokens:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(tokens):
            nominal_end = min(len(tokens), start + chunk_length)
            if nominal_end == len(tokens):
                end = nominal_end
            else:
                end = self._preferred_boundary(
                    tokens,
                    start=start,
                    nominal_end=nominal_end,
                    chunk_length=chunk_length,
                    chunk_overlap=chunk_overlap,
                )

            chunks.append(self.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start = end - chunk_overlap
        return chunks

    def _preferred_boundary(
        self,
        tokens: list[int],
        *,
        start: int,
        nominal_end: int,
        chunk_length: int,
        chunk_overlap: int,
    ) -> int:
        """Prefer paragraph, line, then sentence ends near the token limit."""
        minimum_size = max(
            chunk_overlap + 1,
            int(chunk_length * 0.75),
        )
        minimum_end = min(nominal_end, start + minimum_size)
        pieces = [
            self.decode([token])
            for token in tokens[minimum_end - 1 : nominal_end]
        ]

        def latest_boundary(predicate) -> int | None:
            for offset in range(len(pieces) - 1, -1, -1):
                if predicate(offset):
                    return minimum_end + offset
            return None

        paragraph_end = latest_boundary(
            lambda offset: "\n\n" in pieces[offset]
            or "".join(pieces[max(0, offset - 1) : offset + 1]).endswith("\n\n")
        )
        if paragraph_end is not None:
            return paragraph_end

        line_end = latest_boundary(lambda offset: "\n" in pieces[offset])
        if line_end is not None:
            return line_end

        sentence_end = latest_boundary(
            lambda offset: bool(_SENTENCE_END.search(pieces[offset]))
        )
        return sentence_end if sentence_end is not None else nominal_end
