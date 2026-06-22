from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from typing import Protocol

from activation_rag.schema import ChunkRecord, EmbeddingRecord, l2_normalize


class EmbeddingProvider(Protocol):
    model_id: str

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        ...

    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[EmbeddingRecord]:
        ...


class HashEmbeddingProvider:
    def __init__(self, dimension: int = 256, model_id: str = "hash-bow-v1") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.model_id = model_id

    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[EmbeddingRecord]:
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        return [
            EmbeddingRecord(chunk_id=chunk.chunk_id, model_id=self.model_id, vector=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        return l2_normalize(vector)


class SentenceTransformerEmbeddingProvider:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        model: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.model_id = f"sentence-transformers:{model_name}"
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self._model = model

    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[EmbeddingRecord]:
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        return [
            EmbeddingRecord(chunk_id=chunk.chunk_id, model_id=self.model_id, vector=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        vectors = self._load_model().encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=self.show_progress_bar,
        )
        return [tuple(float(value) for value in vector) for vector in vectors]

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for --embedding-provider sentence-transformers. "
                    "Install activation-rag with the 'embeddings' extra or install sentence-transformers."
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model


class CommandEmbeddingProvider:
    def __init__(
        self,
        command: list[str] | tuple[str, ...],
        *,
        model_id: str,
        timeout_seconds: int = 3600,
        work_dir: str | Path | None = None,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        if not command:
            raise ValueError("command must not be empty")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_backoff_seconds < 0.0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        self.command = tuple(command)
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self.work_dir = Path(work_dir) if work_dir else None
        self.max_attempts = max_attempts
        self.retry_backoff_seconds = retry_backoff_seconds

    def embed_chunks(self, chunks: list[ChunkRecord]) -> list[EmbeddingRecord]:
        vectors = self._embed_rows([(chunk.chunk_id, chunk.text) for chunk in chunks])
        return [
            EmbeddingRecord(chunk_id=chunk.chunk_id, model_id=self.model_id, vector=vectors[chunk.chunk_id])
            for chunk in chunks
        ]

    def embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        ids_and_texts = [(f"text-{index}", text) for index, text in enumerate(texts)]
        vectors = self._embed_rows(ids_and_texts)
        return [vectors[row_id] for row_id, _ in ids_and_texts]

    def _embed_rows(self, ids_and_texts: list[tuple[str, str]]) -> dict[str, tuple[float, ...]]:
        if not ids_and_texts:
            return {}
        with tempfile.TemporaryDirectory(dir=self.work_dir) as tmp:
            root = Path(tmp)
            input_path = root / "embedding-input.jsonl"
            output_path = root / "embedding-output.jsonl"
            with input_path.open("w", encoding="utf-8") as handle:
                for row_id, text in ids_and_texts:
                    handle.write(json.dumps({"id": row_id, "text": text}, ensure_ascii=True, sort_keys=True) + "\n")
            self._run_command(input_path=input_path, output_path=output_path)
            rows = _read_embedding_jsonl(output_path)
        vectors = {str(row["id"]): tuple(float(value) for value in row["vector"]) for row in rows}
        missing = [row_id for row_id, _ in ids_and_texts if row_id not in vectors]
        if missing:
            raise ValueError(f"embedding command did not return vectors for ids: {missing[:5]}")
        return vectors

    def _run_command(self, *, input_path: Path, output_path: Path) -> None:
        replacements = {
            "input_jsonl": str(input_path),
            "output_jsonl": str(output_path),
        }
        command = [part.format(**replacements) for part in self.command]
        last_message = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(self.work_dir) if self.work_dir else None,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except OSError as exc:
                last_message = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_attempts and _is_transient_command_failure(last_message):
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(f"embedding command failed before execution: {last_message}") from exc
            if completed.returncode == 0:
                if not output_path.exists():
                    raise FileNotFoundError(f"embedding command did not create {output_path}")
                return
            last_message = completed.stderr.strip() or completed.stdout.strip()
            if attempt < self.max_attempts and _is_transient_command_failure(last_message):
                self._sleep_before_retry(attempt)
                continue
            raise RuntimeError(
                "embedding command failed "
                f"with code {completed.returncode}: {last_message}"
            )
        raise RuntimeError(f"embedding command failed after {self.max_attempts} attempts: {last_message}")

    def _sleep_before_retry(self, attempt: int) -> None:
        if self.retry_backoff_seconds:
            time.sleep(self.retry_backoff_seconds * attempt)


def _read_embedding_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _is_transient_command_failure(message: str) -> bool:
    lowered = message.lower()
    transient_markers = (
        "resource temporarily unavailable",
        "blockingioerror",
        "errno 35",
        "temporarily unavailable",
        "temporary failure",
        "connection reset",
        "connection timed out",
        "ssh_exchange_identification",
    )
    return any(marker in lowered for marker in transient_markers)
