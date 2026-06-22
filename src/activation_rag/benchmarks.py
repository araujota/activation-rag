from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from activation_rag.chunking import Chunker
from activation_rag.embedding import EmbeddingProvider
from activation_rag.pipeline import RagEngine
from activation_rag.retrieval import (
    ActivationMatchingConfig,
    activation_strategy_label,
    rank_activation_with_strategy,
    rank_dense,
    rerank_with_activation,
)
from activation_rag.schema import DocumentRecord, RetrievalResult
from activation_rag.telemetry import TelemetryProvider


@dataclass(frozen=True)
class BenchmarkDataset:
    name: str
    split: str
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    metric_profile: str


@dataclass(frozen=True)
class BenchmarkRunSummary:
    dataset_name: str
    split: str
    query_count: int
    corpus_count: int
    approaches: tuple[str, ...]
    metrics_by_approach: dict[str, dict[str, float]]
    candidate_k: int
    top_k: int
    started_at: float
    finished_at: float
    notes: tuple[str, ...] = ()
    schema_version: str = "activation_rag.benchmark_run.v1"

    @property
    def duration_seconds(self) -> float:
        return self.finished_at - self.started_at

    def to_json_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "query_count": self.query_count,
            "corpus_count": self.corpus_count,
            "approaches": list(self.approaches),
            "metrics_by_approach": self.metrics_by_approach,
            "candidate_k": self.candidate_k,
            "top_k": self.top_k,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "notes": list(self.notes),
        }

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assert_benchmark_telemetry_allowed(
    telemetry_provider: TelemetryProvider,
    *,
    fixture: bool,
    allow_mock_telemetry: bool,
) -> None:
    provider_id = str(getattr(telemetry_provider, "provider_id", ""))
    if fixture or allow_mock_telemetry:
        return
    if "mock" in provider_id.lower():
        raise ValueError(
            "non-fixture benchmark runs require real prefill telemetry, not mock telemetry; "
            "pass --allow-mock-telemetry only for harness smoke tests"
        )


def mean_reciprocal_rank(ranked_doc_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
    for index, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def recall_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
    if not relevant:
        return 0.0
    found = relevant & set(ranked_doc_ids[:k])
    return len(found) / len(relevant)


def ndcg_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        grade = qrels.get(doc_id, 0)
        if grade > 0:
            dcg += ((2**grade) - 1) / math.log2(index + 1)
    ideal_grades = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)[:k]
    ideal = sum(((2**grade) - 1) / math.log2(index + 1) for index, grade in enumerate(ideal_grades, start=1))
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def evaluate_dataset(
    dataset: BenchmarkDataset,
    *,
    chunker: Chunker,
    embedder: EmbeddingProvider,
    telemetry_provider: TelemetryProvider,
    top_k: int,
    candidate_k: int,
    activation_matching_config: ActivationMatchingConfig | None = None,
) -> BenchmarkRunSummary:
    started_at = time.time()
    documents: list[DocumentRecord] = []
    doc_id_by_document_id: dict[str, str] = {}
    for doc_id, text in dataset.corpus.items():
        document = DocumentRecord.from_text(
            source_uri=f"benchmark://{dataset.name}/{doc_id}",
            title=doc_id,
            text=text,
            metadata={"benchmark_doc_id": doc_id},
        )
        documents.append(document)
        doc_id_by_document_id[document.document_id] = doc_id
    engine = RagEngine(chunker=chunker, embedder=embedder, telemetry_provider=telemetry_provider)
    chunks = engine.ingest(documents, embed=True, capture_telemetry=True)
    chunk_to_doc_id = {
        chunk.chunk_id: doc_id_by_document_id[chunk.document_id]
        for chunk in chunks
    }

    activation_matching_config = activation_matching_config or ActivationMatchingConfig()
    activation_approach = activation_strategy_label(activation_matching_config.strategy)
    rerank_approach = (
        "dense+activation-rerank"
        if activation_approach == "activation-sim"
        else f"dense+{activation_approach}-rerank"
    )
    results_by_approach: dict[str, dict[str, list[str]]] = {
        "dense": {},
        activation_approach: {},
        rerank_approach: {},
    }
    query_items = list(dataset.queries.items())
    query_dense_vectors = embedder.embed_texts([query for _, query in query_items])
    for (query_id, query), query_dense_vector in zip(query_items, query_dense_vectors, strict=True):
        dense = rank_dense(query_dense_vector, engine.embeddings, top_k=max(top_k, candidate_k))
        query_activation = engine._capture_query_activation(query)
        activation = rank_activation_with_strategy(
            query_activation,
            engine.activation_records,
            top_k=top_k,
            config=activation_matching_config,
        )
        reranked = rerank_with_activation(
            query_activation,
            engine.activation_records,
            dense,
            top_k=top_k,
            config=activation_matching_config,
        )
        results_by_approach["dense"][query_id] = _results_to_doc_ids(dense[:top_k], chunk_to_doc_id)
        results_by_approach[activation_approach][query_id] = _results_to_doc_ids(activation, chunk_to_doc_id)
        results_by_approach[rerank_approach][query_id] = _results_to_doc_ids(reranked, chunk_to_doc_id)

    metrics_by_approach = {
        approach: _aggregate_metrics(query_results, dataset.qrels, top_k)
        for approach, query_results in results_by_approach.items()
    }
    notes = (*_profile_notes(dataset.metric_profile), f"activation_strategy={activation_matching_config.strategy}")
    return BenchmarkRunSummary(
        dataset_name=dataset.name,
        split=dataset.split,
        query_count=len(dataset.queries),
        corpus_count=len(dataset.corpus),
        approaches=tuple(results_by_approach),
        metrics_by_approach=metrics_by_approach,
        candidate_k=candidate_k,
        top_k=top_k,
        started_at=started_at,
        finished_at=time.time(),
        notes=notes,
    )


def _results_to_doc_ids(results: list[RetrievalResult], chunk_to_doc_id: dict[str, str]) -> list[str]:
    doc_ids: list[str] = []
    for result in results:
        doc_id = chunk_to_doc_id.get(result.chunk_id)
        if doc_id is not None:
            doc_ids.append(doc_id)
    return doc_ids


def _aggregate_metrics(query_results: dict[str, list[str]], qrels_by_query: dict[str, dict[str, int]], top_k: int) -> dict[str, float]:
    if not query_results:
        return {}
    mrr_values: list[float] = []
    recall_values: list[float] = []
    ndcg_values: list[float] = []
    for query_id, ranked_doc_ids in query_results.items():
        qrels = qrels_by_query.get(query_id, {})
        mrr_values.append(mean_reciprocal_rank(ranked_doc_ids, qrels, min(10, top_k)))
        recall_values.append(recall_at_k(ranked_doc_ids, qrels, top_k))
        ndcg_values.append(ndcg_at_k(ranked_doc_ids, qrels, min(10, top_k)))
    return {
        f"mrr@{min(10, top_k)}": sum(mrr_values) / len(mrr_values),
        f"recall@{top_k}": sum(recall_values) / len(recall_values),
        f"ndcg@{min(10, top_k)}": sum(ndcg_values) / len(ndcg_values),
    }


def _profile_notes(metric_profile: str) -> tuple[str, ...]:
    if metric_profile == "msmarco_passage":
        return ("MS MARCO passage standard headline metric is MRR@10.",)
    if metric_profile == "beir":
        return ("BEIR standard headline metric is nDCG@10; Recall@100 is commonly reported.",)
    if metric_profile == "hotpotqa_supporting_evidence":
        return ("HotpotQA here is supporting-evidence retrieval, not full multi-hop QA.",)
    return ()


def load_beir_dataset(root: str | Path, *, name: str, split: str = "test") -> BenchmarkDataset:
    root_path = Path(root)
    corpus_path = root_path / "corpus.jsonl"
    queries_path = root_path / "queries.jsonl"
    qrels_path = root_path / "qrels" / f"{split}.tsv"
    if not corpus_path.exists():
        raise FileNotFoundError(corpus_path)
    if not queries_path.exists():
        raise FileNotFoundError(queries_path)
    if not qrels_path.exists():
        raise FileNotFoundError(qrels_path)
    qrels = _load_beir_qrels(qrels_path)
    queries = _load_beir_queries(queries_path)
    judged_queries = {query_id: queries[query_id] for query_id in qrels if query_id in queries}
    return BenchmarkDataset(
        name=name,
        split=split,
        corpus=_load_beir_corpus(corpus_path),
        queries=judged_queries,
        qrels=qrels,
        metric_profile="beir",
    )


def _load_beir_corpus(path: Path) -> dict[str, str]:
    corpus: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            corpus[str(row["_id"])] = f"{title}\n\n{text}".strip() if title else text
    return corpus


def _load_beir_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[str(row["_id"])] = str(row["text"])
    return queries


def _load_beir_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        expected = ["query-id", "corpus-id", "score"]
        if header != expected:
            raise ValueError(f"Unexpected BEIR qrels header {header}, expected {expected}")
        for line in handle:
            if not line.strip():
                continue
            query_id, corpus_id, score = line.rstrip("\n").split("\t")
            qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return qrels
