from __future__ import annotations

import re
from dataclasses import dataclass

from activation_rag.schema import ChunkRecord, DocumentRecord, stable_hash


@dataclass(frozen=True)
class ChunkerSettings:
    chunk_size: int = 384
    chunk_overlap: int = 64
    name: str = "sentence_recursive"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


@dataclass(frozen=True)
class _TokenSpan:
    text: str
    start: int
    end: int


class Chunker:
    def __init__(self, settings: ChunkerSettings | None = None) -> None:
        self.settings = settings or ChunkerSettings()

    def split(self, documents: list[DocumentRecord]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: DocumentRecord) -> list[ChunkRecord]:
        tokens = self._token_spans(document.text)
        if not tokens:
            return []

        chunks: list[ChunkRecord] = []
        start_token = 0
        ordinal = 0
        while start_token < len(tokens):
            end_token = min(start_token + self.settings.chunk_size, len(tokens))
            char_start = tokens[start_token].start
            char_end = tokens[end_token - 1].end
            chunk_text = document.text[char_start:char_end]
            text_hash = DocumentRecord.hash_text(chunk_text)
            chunk_id = stable_hash(
                "\n".join(
                    [
                        document.document_id,
                        self.settings.name,
                        str(self.settings.chunk_size),
                        str(self.settings.chunk_overlap),
                        str(char_start),
                        str(char_end),
                        text_hash,
                    ]
                ),
                32,
            )
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    ordinal=ordinal,
                    text=chunk_text,
                    text_hash=text_hash,
                    char_start=char_start,
                    char_end=char_end,
                    token_count_estimate=end_token - start_token,
                    chunker=self.settings.name,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
            )
            if end_token == len(tokens):
                break
            start_token = max(end_token - self.settings.chunk_overlap, start_token + 1)
            ordinal += 1
        return chunks

    def _token_spans(self, text: str) -> list[_TokenSpan]:
        return [_TokenSpan(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]

