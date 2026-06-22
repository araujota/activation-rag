from __future__ import annotations

from dataclasses import replace

from activation_rag.chunking import Chunker
from activation_rag.embedding import EmbeddingProvider
from activation_rag.retrieval import ActivationMatchingConfig, rank_activation, rank_activation_with_strategy, rank_dense, rerank_with_activation
from activation_rag.schema import ActivationRecord, ChunkRecord, DocumentRecord, RetrievalResult, SearchComparison, stable_hash
from activation_rag.telemetry import TelemetryProvider


class RagEngine:
    def __init__(
        self,
        chunker: Chunker,
        embedder: EmbeddingProvider,
        telemetry_provider: TelemetryProvider,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.telemetry_provider = telemetry_provider
        self.chunks: dict[str, ChunkRecord] = {}
        self.embeddings = []
        self.activation_records: list[ActivationRecord] = []

    def ingest(
        self,
        documents: list[DocumentRecord],
        *,
        embed: bool = True,
        capture_telemetry: bool = True,
    ) -> list[ChunkRecord]:
        chunks = self.chunker.split(documents)
        for chunk in chunks:
            self.chunks[chunk.chunk_id] = chunk
        if embed:
            self.embeddings.extend(self.embedder.embed_chunks(chunks))
        if capture_telemetry:
            self.activation_records.extend(self.telemetry_provider.capture_prefill(chunks))
        return chunks

    def search_dense(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.embeddings:
            raise ValueError("dense search requires embeddings")
        query_vector = self.embedder.embed_texts([query])[0]
        return rank_dense(query_vector, self.embeddings, top_k)

    def search_activation(
        self,
        query: str,
        top_k: int = 5,
        *,
        config: ActivationMatchingConfig | None = None,
    ) -> list[RetrievalResult]:
        return self.search_activation_knn(query, top_k, config=config)

    def search_activation_knn(
        self,
        query: str,
        top_k: int = 5,
        *,
        config: ActivationMatchingConfig | None = None,
    ) -> list[RetrievalResult]:
        if not self.activation_records:
            raise ValueError("activation search requires captured telemetry")
        query_record = self._capture_query_activation(query)
        if config is None:
            return rank_activation(query_record, self.activation_records, top_k)
        return rank_activation_with_strategy(query_record, self.activation_records, top_k, config=config)

    def search_dense_then_activation_rerank(
        self,
        query: str,
        *,
        candidate_k: int = 20,
        top_k: int = 5,
        config: ActivationMatchingConfig | None = None,
    ) -> list[RetrievalResult]:
        dense_candidates = self.search_dense(query, top_k=candidate_k)
        return self.rerank_with_activation(query, dense_candidates, top_k=top_k, config=config)

    def rerank_with_activation(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int = 5,
        config: ActivationMatchingConfig | None = None,
    ) -> list[RetrievalResult]:
        if not self.activation_records:
            raise ValueError("activation reranking requires captured telemetry")
        query_record = self._capture_query_activation(query)
        return rerank_with_activation(query_record, self.activation_records, candidates, top_k, config=config)

    def compare_search_methods(
        self,
        query: str,
        *,
        top_k: int = 5,
        candidate_k: int | None = None,
        config: ActivationMatchingConfig | None = None,
    ) -> SearchComparison:
        candidate_count = candidate_k or max(top_k, 20)
        dense_for_display = self.search_dense(query, top_k=top_k)
        dense_for_rerank = self.search_dense(query, top_k=candidate_count)
        activation_results = self.search_activation_knn(query, top_k=top_k, config=config)
        activation_reranked = self.rerank_with_activation(query, dense_for_rerank, top_k=top_k, config=config)
        dense_ids = {result.chunk_id for result in dense_for_display}
        activation_ids = {result.chunk_id for result in activation_results}
        rerank_ids = {result.chunk_id for result in activation_reranked}
        notes = (
            "activation_results are full-index KNN over stored activation records",
            "activation_reranked_results are restricted to the dense candidate pool",
        )
        return SearchComparison(
            query=query,
            dense_results=dense_for_display,
            activation_results=activation_results,
            activation_reranked_results=activation_reranked,
            dense_activation_overlap=tuple(sorted(dense_ids & activation_ids)),
            dense_rerank_overlap=tuple(sorted(dense_ids & rerank_ids)),
            notes=notes,
        )

    def _capture_query_activation(self, query: str) -> ActivationRecord:
        query_chunk = ChunkRecord(
            chunk_id=stable_hash(f"query\n{query}", 32),
            document_id="query",
            ordinal=0,
            text=query,
            text_hash=stable_hash(query, 32),
            char_start=0,
            char_end=len(query),
            token_count_estimate=max(1, len(query.split())),
            chunker="query",
            chunk_size=max(1, len(query.split())),
            chunk_overlap=0,
        )
        query_record = self.telemetry_provider.capture_prefill([query_chunk])[0]
        return replace(query_record, chunk_id=query_chunk.chunk_id, document_id="query")

    def get_chunk(self, chunk_id: str) -> ChunkRecord:
        return self.chunks[chunk_id]
