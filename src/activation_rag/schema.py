from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).strip()


@dataclass(frozen=True)
class DocumentRecord:
    source_uri: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    document_id: str = ""
    text_hash: str = ""

    def __post_init__(self) -> None:
        normalized = normalize_text(self.text)
        text_hash = self.text_hash or self.hash_text(normalized)
        document_id = self.document_id or stable_hash(f"{self.source_uri}\n{text_hash}", 24)
        object.__setattr__(self, "text", normalized)
        object.__setattr__(self, "text_hash", text_hash)
        object.__setattr__(self, "document_id", document_id)

    @staticmethod
    def hash_text(text: str) -> str:
        return stable_hash(normalize_text(text), 32)

    @classmethod
    def from_text(
        cls,
        source_uri: str,
        title: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> "DocumentRecord":
        return cls(source_uri=source_uri, title=title, text=text, metadata=metadata or {})


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    text_hash: str
    char_start: int
    char_end: int
    token_count_estimate: int
    chunker: str
    chunk_size: int
    chunk_overlap: int
    schema_version: str = "activation_rag.chunk.v1"


@dataclass(frozen=True)
class EmbeddingRecord:
    chunk_id: str
    model_id: str
    vector: tuple[float, ...]
    normalized: bool = True
    schema_version: str = "activation_rag.embedding.v1"

    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass(frozen=True)
class ActivationRecord:
    chunk_id: str
    document_id: str
    capture_run_id: str
    provider_id: str
    model_id: str
    site_id: str
    current_em_state: dict[str, float]
    neutral_baseline_state: dict[str, float]
    prior_current_state: dict[str, float]
    delta_vs_neutral: dict[str, float]
    delta_vs_current: dict[str, float]
    saturation: dict[str, float]
    residual_headroom: dict[str, float]
    positive_mass: float
    negative_mass: float
    total_mass: float
    signed_balance: float
    sae_feature_values: dict[str, float]
    sae_delta_vs_neutral: dict[str, float]
    sae_delta_vs_current: dict[str, float]
    sae_feature_mask: dict[str, bool]
    model_hash: str | None = None
    tokenizer_hash: str | None = None
    hook_name: str | None = None
    layer_index: int | None = None
    layer_selection_policy: str = "unspecified"
    prompt_template_id: str = "unspecified"
    prompt_template_hash: str | None = None
    normalization_policy: str = "unspecified"
    token_start: int = 0
    token_end: int = 0
    aggregation: str = "mean_over_chunk"
    sae_novelty: float | None = None
    sae_overlap_with_memory: float | None = None
    telemetry_valid: bool = True
    invalid_reason: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "activation_rag.activation_record.v1"

    def activation_vector(self) -> tuple[float, ...]:
        values: list[float] = []
        for key in sorted(self.delta_vs_neutral):
            values.append(self.delta_vs_neutral[key])
        for key in sorted(self.delta_vs_current):
            values.append(self.delta_vs_current[key])
        values.extend([self.positive_mass, self.negative_mass, self.signed_balance])
        for key in sorted(self.sae_feature_values):
            values.append(self.sae_feature_values[key])
        return tuple(values)

    def to_json_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self, default=lambda obj: obj.__dict__, sort_keys=True))


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    strategy: str
    score: float
    rank: int
    component_scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchComparison:
    query: str
    dense_results: list[RetrievalResult]
    activation_results: list[RetrievalResult]
    activation_reranked_results: list[RetrievalResult]
    dense_activation_overlap: tuple[str, ...]
    dense_rerank_overlap: tuple[str, ...]
    notes: tuple[str, ...] = ()


def l2_normalize(vector: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(0.0 for _ in vector)
    return tuple(value / norm for value in vector)
