#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.embedding import CommandEmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider
from activation_rag.schema import ChunkRecord, EmbeddingRecord, stable_hash


SCHEMA_VERSION = "activation_rag.vertical_reranker_group.v1"


@dataclass(frozen=True)
class RetrievalComponents:
    dataset_name: str
    queries: dict[str, str]
    corpus: dict[str, str]
    qrels: dict[str, dict[str, float]]


@dataclass(frozen=True)
class RerankingComponents(RetrievalComponents):
    top_ranked: dict[str, list[str]]


PRESETS: dict[str, dict[str, str]] = {
    "mleb-legal-rag-bench": {
        "mode": "hf-retrieval",
        "dataset": "isaacus/mleb-legal-rag-bench",
        "queries_config": "queries",
        "queries_split": "queries",
        "corpus_config": "corpus",
        "corpus_split": "corpus",
        "qrels_config": "default",
        "qrels_split": "test",
        "dataset_name": "mleb-legal-rag-bench",
    },
    "r2med-medqa-diag": {
        "mode": "hf-retrieval",
        "dataset": "mteb/R2MEDMedQADiagRetrieval",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "qrels",
        "qrels_split": "test",
        "dataset_name": "r2med-medqa-diag",
    },
    "r2med-biology": {
        "mode": "hf-retrieval",
        "dataset": "mteb/R2MEDBiologyRetrieval",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "qrels",
        "qrels_split": "test",
        "dataset_name": "r2med-biology",
    },
    "r2med-bioinformatics": {
        "mode": "hf-retrieval",
        "dataset": "mteb/R2MEDBioinformaticsRetrieval",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "qrels",
        "qrels_split": "test",
        "dataset_name": "r2med-bioinformatics",
    },
    "r2med-medical-sciences": {
        "mode": "hf-retrieval",
        "dataset": "mteb/R2MEDMedicalSciencesRetrieval",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "qrels",
        "qrels_split": "test",
        "dataset_name": "r2med-medical-sciences",
    },
    "r2med-medxpertqa-exam": {
        "mode": "hf-retrieval",
        "dataset": "mteb/R2MEDMedXpertQAExamRetrieval",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "qrels",
        "qrels_split": "test",
        "dataset_name": "r2med-medxpertqa-exam",
    },
    "r2med-pmc-treatment": {
        "mode": "hf-retrieval",
        "dataset": "mteb/R2MEDPMCTreatmentRetrieval",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "qrels",
        "qrels_split": "test",
        "dataset_name": "r2med-pmc-treatment",
    },
    "r2med-pmc-clinical": {
        "mode": "hf-retrieval",
        "dataset": "mteb/R2MEDPMCClinicalRetrieval",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "qrels",
        "qrels_split": "test",
        "dataset_name": "r2med-pmc-clinical",
    },
    "r2med-iiyi-clinical": {
        "mode": "hf-retrieval",
        "dataset": "mteb/R2MEDIIYiClinicalRetrieval",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "qrels",
        "qrels_split": "test",
        "dataset_name": "r2med-iiyi-clinical",
    },
    "coreb-t2c-reranking": {
        "mode": "hf-reranking",
        "dataset": "mteb/coreb-t2c-reranking",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "default",
        "qrels_split": "test",
        "top_ranked_config": "top_ranked",
        "top_ranked_split": "test",
        "dataset_name": "coreb-t2c-reranking",
    },
    "coreb-c2t-reranking": {
        "mode": "hf-reranking",
        "dataset": "mteb/coreb-c2t-reranking",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "default",
        "qrels_split": "test",
        "top_ranked_config": "top_ranked",
        "top_ranked_split": "test",
        "dataset_name": "coreb-c2t-reranking",
    },
    "coreb-c2c-reranking": {
        "mode": "hf-reranking",
        "dataset": "mteb/coreb-c2c-reranking",
        "queries_config": "queries",
        "queries_split": "test",
        "corpus_config": "corpus",
        "corpus_split": "test",
        "qrels_config": "default",
        "qrels_split": "test",
        "top_ranked_config": "top_ranked",
        "top_ranked_split": "test",
        "dataset_name": "coreb-c2c-reranking",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare vertical actpred reranker groups from HF retrieval/reranking datasets.")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="Known vertical dataset preset.")
    parser.add_argument("--mode", choices=("hf-retrieval", "hf-reranking"))
    parser.add_argument("--dataset")
    parser.add_argument("--dataset-name")
    parser.add_argument("--queries-config")
    parser.add_argument("--queries-split")
    parser.add_argument("--corpus-config")
    parser.add_argument("--corpus-split")
    parser.add_argument("--qrels-config")
    parser.add_argument("--qrels-split")
    parser.add_argument("--top-ranked-config")
    parser.add_argument("--top-ranked-split")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--append-qrel-positives", action="store_true", help="Append missing positive qrel passages after dense/native candidates for supervised reranker training.")
    parser.add_argument("--embedding-command", help="Command with {input_jsonl}/{output_jsonl} placeholders for retrieval-mode dense candidates.")
    parser.add_argument("--embedding-timeout-seconds", type=int, default=7200)
    parser.add_argument("--hash-dimension", type=int, help="Use deterministic hash embeddings instead of a command embedder.")
    parser.add_argument("--dense-cache-out", help="Optional .npz cache for retrieval-mode vectors.")
    parser.add_argument("--reuse-dense-cache", help="Optional .npz cache to reuse for retrieval-mode vectors.")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--dev-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", default="activation-rag-vertical")
    args = parser.parse_args()

    config = _config_from_args(args)
    if int(args.candidate_k) <= 0:
        raise SystemExit("--candidate-k must be positive")
    if config["mode"] == "hf-retrieval":
        components = load_hf_retrieval_components(config)
        embedder = _build_embedder(args)
        groups = prepare_retrieval_groups(
            components,
            candidate_k=int(args.candidate_k),
            embedder=embedder,
            dense_cache_out=Path(args.dense_cache_out) if args.dense_cache_out else None,
            reuse_dense_cache=Path(args.reuse_dense_cache) if args.reuse_dense_cache else None,
            append_qrel_positives=bool(args.append_qrel_positives),
        )
    else:
        components = load_hf_reranking_components(config)
        groups = prepare_reranking_groups(components, candidate_k=int(args.candidate_k), append_qrel_positives=bool(args.append_qrel_positives))
    train_rows, dev_rows, test_rows = split_groups(
        groups,
        dev_fraction=float(args.dev_fraction),
        test_fraction=float(args.test_fraction),
        seed=str(args.seed),
    )
    _write_jsonl(Path(args.train_out), train_rows)
    _write_jsonl(Path(args.dev_out), dev_rows)
    _write_jsonl(Path(args.test_out), test_rows)
    print(
        json.dumps(
            {
                "schema_version": "activation_rag.vertical_reranker_group_preparation.v1",
                "preset": args.preset,
                "mode": config["mode"],
                "dataset_name": config["dataset_name"],
                "candidate_k": int(args.candidate_k),
                "group_count": len(groups),
                "train_count": len(train_rows),
                "dev_count": len(dev_rows),
                "test_count": len(test_rows),
                "positive_in_pool_count": sum(1 for group in groups if group["positive_in_candidate_pool"]),
                "train_out": args.train_out,
                "dev_out": args.dev_out,
                "test_out": args.test_out,
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_hf_retrieval_components(config: dict[str, str]) -> RetrievalComponents:
    from datasets import load_dataset

    queries = _rows_to_text_map(load_dataset(config["dataset"], config["queries_config"], split=config["queries_split"]))
    corpus = _rows_to_text_map(load_dataset(config["dataset"], config["corpus_config"], split=config["corpus_split"]))
    qrels = _rows_to_qrels(load_dataset(config["dataset"], config["qrels_config"], split=config["qrels_split"]))
    return RetrievalComponents(dataset_name=config["dataset_name"], queries=queries, corpus=corpus, qrels=qrels)


def load_hf_reranking_components(config: dict[str, str]) -> RerankingComponents:
    from datasets import load_dataset

    retrieval = load_hf_retrieval_components(config)
    top_ranked = _rows_to_top_ranked(load_dataset(config["dataset"], config["top_ranked_config"], split=config["top_ranked_split"]))
    return RerankingComponents(
        dataset_name=retrieval.dataset_name,
        queries=retrieval.queries,
        corpus=retrieval.corpus,
        qrels=retrieval.qrels,
        top_ranked=top_ranked,
    )


def prepare_retrieval_groups(
    components: RetrievalComponents,
    *,
    candidate_k: int,
    embedder: EmbeddingProvider,
    dense_cache_out: Path | None = None,
    reuse_dense_cache: Path | None = None,
    append_qrel_positives: bool = False,
) -> list[dict[str, Any]]:
    doc_ids = list(components.corpus)
    query_ids = list(components.queries)
    doc_vectors, query_vectors = _embed_or_load_vectors(
        components=components,
        doc_ids=doc_ids,
        query_ids=query_ids,
        embedder=embedder,
        dense_cache_out=dense_cache_out,
        reuse_dense_cache=reuse_dense_cache,
    )
    groups: list[dict[str, Any]] = []
    for query_index, query_id in enumerate(query_ids):
        dense_scores = _cosine_scores(query_vectors[query_index], doc_vectors)
        top_indices = _top_indices(dense_scores, min(candidate_k, len(doc_ids)))
        candidates = []
        positives = components.qrels.get(query_id, {})
        for rank, dense_index in enumerate(top_indices, start=1):
            doc_id = doc_ids[int(dense_index)]
            score = float(positives.get(doc_id, 0.0))
            dense_score = float(dense_scores[int(dense_index)])
            candidates.append(_candidate_row(components.dataset_name, doc_id, components.corpus[doc_id], rank, dense_score, score))
        if append_qrel_positives:
            present = {candidate["doc_id"] for candidate in candidates}
            for doc_id, score in sorted(positives.items()):
                if score <= 0 or doc_id in present or doc_id not in components.corpus:
                    continue
                dense_index = doc_ids.index(doc_id)
                candidates.append(
                    _candidate_row(
                        components.dataset_name,
                        doc_id,
                        components.corpus[doc_id],
                        len(candidates) + 1,
                        float(dense_scores[dense_index]),
                        float(score),
                    )
                )
        groups.append(_group_row(components, query_id, candidate_k, candidates, dense_score_source="embedding_cosine"))
    return groups


def prepare_reranking_groups(components: RerankingComponents, *, candidate_k: int, append_qrel_positives: bool = False) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for query_id, ranked_doc_ids in components.top_ranked.items():
        if query_id not in components.queries:
            continue
        positives = components.qrels.get(query_id, {})
        candidates = []
        for rank, doc_id in enumerate(ranked_doc_ids[:candidate_k], start=1):
            if doc_id not in components.corpus:
                continue
            label_score = float(positives.get(doc_id, 0.0))
            candidates.append(_candidate_row(components.dataset_name, doc_id, components.corpus[doc_id], rank, 1.0 / rank, label_score))
        if append_qrel_positives:
            present = {candidate["doc_id"] for candidate in candidates}
            for doc_id, score in sorted(positives.items()):
                if score <= 0 or doc_id in present or doc_id not in components.corpus:
                    continue
                candidates.append(_candidate_row(components.dataset_name, doc_id, components.corpus[doc_id], len(candidates) + 1, 0.0, float(score)))
        if candidates:
            groups.append(_group_row(components, query_id, candidate_k, candidates, dense_score_source="top_ranked_reciprocal_rank"))
    return groups


def split_groups(
    groups: list[dict[str, Any]],
    *,
    dev_fraction: float,
    test_fraction: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if dev_fraction <= 0.0 or test_fraction <= 0.0 or dev_fraction + test_fraction >= 1.0:
        raise ValueError("dev_fraction and test_fraction must be positive and sum to less than 1")
    ordered = sorted(groups, key=lambda group: _split_score(str(group["query_id"]), seed))
    test_count = max(1, round(len(ordered) * test_fraction)) if ordered else 0
    dev_count = max(1, round(len(ordered) * dev_fraction)) if ordered else 0
    test_ids = {str(group["query_id"]) for group in ordered[:test_count]}
    dev_ids = {str(group["query_id"]) for group in ordered[test_count : test_count + dev_count]}
    train_rows = [_with_split(group, "train") for group in groups if str(group["query_id"]) not in test_ids and str(group["query_id"]) not in dev_ids]
    dev_rows = [_with_split(group, "dev") for group in groups if str(group["query_id"]) in dev_ids]
    test_rows = [_with_split(group, "test") for group in groups if str(group["query_id"]) in test_ids]
    return train_rows, dev_rows, test_rows


def _embed_or_load_vectors(
    *,
    components: RetrievalComponents,
    doc_ids: list[str],
    query_ids: list[str],
    embedder: EmbeddingProvider,
    dense_cache_out: Path | None,
    reuse_dense_cache: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    if reuse_dense_cache and reuse_dense_cache.exists():
        cached = np.load(reuse_dense_cache, allow_pickle=False)
        if list(cached["doc_ids"]) == doc_ids and list(cached["query_ids"]) == query_ids:
            return np.array(cached["doc_vectors"], dtype=np.float64), np.array(cached["query_vectors"], dtype=np.float64)
    chunks = [
        ChunkRecord(
            chunk_id=_chunk_id_for_doc(components.dataset_name, doc_id),
            document_id=doc_id,
            ordinal=0,
            text=components.corpus[doc_id],
            text_hash=stable_hash(components.corpus[doc_id], 32),
            char_start=0,
            char_end=len(components.corpus[doc_id]),
            token_count_estimate=max(1, len(components.corpus[doc_id].split())),
            chunker="vertical-passage-v1",
            chunk_size=0,
            chunk_overlap=0,
        )
        for doc_id in doc_ids
    ]
    doc_vectors = np.array([record.vector for record in embedder.embed_chunks(chunks)], dtype=np.float64)
    query_vectors = np.array(embedder.embed_texts([components.queries[query_id] for query_id in query_ids]), dtype=np.float64)
    if dense_cache_out:
        dense_cache_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dense_cache_out,
            doc_ids=np.array(doc_ids),
            query_ids=np.array(query_ids),
            doc_vectors=doc_vectors,
            query_vectors=query_vectors,
        )
    return doc_vectors, query_vectors


def _candidate_row(dataset_name: str, doc_id: str, text: str, rank: int, dense_score: float, label_score: float) -> dict[str, Any]:
    label = int(label_score > 0.0)
    return {
        "chunk_id": _chunk_id_for_doc(dataset_name, doc_id),
        "doc_id": doc_id,
        "text": text,
        "dense_rank": rank,
        "dense_score": dense_score,
        "label": label,
        "label_score": label_score,
        "negative_source": None if label else "candidate_negative",
        "negative_trust": None if label else "benchmark_or_retriever_negative",
        "features": {
            "dense_score": dense_score,
            "dense_rank_reciprocal": 1.0 / rank,
        },
    }


def _group_row(
    components: RetrievalComponents,
    query_id: str,
    candidate_k: int,
    candidates: list[dict[str, Any]],
    *,
    dense_score_source: str,
) -> dict[str, Any]:
    positives = {doc_id: score for doc_id, score in components.qrels.get(query_id, {}).items() if score > 0}
    query_text = components.queries[query_id]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": components.dataset_name,
        "split": "unsplit",
        "split_policy": "deterministic_internal_query_hash_split",
        "query_id": query_id,
        "query_text": query_text,
        "query_activation_chunk_id": stable_hash(f"query\n{query_text}", 32),
        "candidate_k": candidate_k,
        "positive_doc_ids": sorted(positives),
        "positive_in_candidate_pool": any(candidate["label"] > 0 for candidate in candidates),
        "false_negative_policy": "non_positive_candidate_ids_treated_as_negatives_for_training",
        "dense_score_source": dense_score_source,
        "candidates": candidates,
    }


def _chunk_id_for_doc(dataset_name: str, doc_id: str) -> str:
    return stable_hash(f"{dataset_name}\n{doc_id}", 32)


def _rows_to_text_map(rows: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        row_id = _first_present(row, ("_id", "id", "query-id", "query_id", "corpus-id", "corpus_id"))
        if row_id is None:
            raise ValueError(f"could not infer row id from columns: {sorted(row)}")
        title = str(row.get("title") or "").strip()
        text_value = _first_present(row, ("text", "question", "query", "answer"))
        if text_value is None:
            raise ValueError(f"could not infer text from columns: {sorted(row)}")
        text = str(text_value).strip()
        out[str(row_id)] = f"{title}\n\n{text}" if title and title not in text else text
    return out


def _rows_to_qrels(rows: Any) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = {}
    for row in rows:
        query_id = str(_first_present(row, ("query-id", "query_id", "qid", "queryid")))
        doc_id = str(_first_present(row, ("corpus-id", "corpus_id", "doc_id", "docid", "pid")))
        score = _first_present(row, ("score", "relevance", "label"))
        if query_id == "None" or doc_id == "None" or score is None:
            raise ValueError(f"could not infer qrel fields from columns: {sorted(row)}")
        qrels.setdefault(query_id, {})[doc_id] = float(score)
    return qrels


def _rows_to_top_ranked(rows: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        query_id = str(_first_present(row, ("query-id", "query_id", "qid", "queryid")))
        corpus_ids = _first_present(row, ("corpus-ids", "corpus_ids", "doc_ids", "docids"))
        if query_id == "None" or not isinstance(corpus_ids, list):
            raise ValueError(f"could not infer top-ranked fields from columns: {sorted(row)}")
        out[query_id] = [str(doc_id) for doc_id in corpus_ids]
    return out


def _first_present(row: dict[str, Any], names: tuple[str, ...]) -> Any | None:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _cosine_scores(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query)
    doc_norms = np.linalg.norm(docs, axis=1)
    denominator = np.maximum(query_norm * doc_norms, 1e-12)
    return docs @ query / denominator


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    unordered = np.argpartition(-scores, kth=k - 1)[:k]
    return unordered[np.argsort(-scores[unordered])]


def _with_split(group: dict[str, Any], split: str) -> dict[str, Any]:
    row = dict(group)
    row["split"] = split
    return row


def _split_score(query_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}\n{query_id}".encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _build_embedder(args: argparse.Namespace) -> EmbeddingProvider:
    if args.hash_dimension:
        return HashEmbeddingProvider(dimension=int(args.hash_dimension))
    if args.embedding_command:
        return CommandEmbeddingProvider(
            shlex.split(str(args.embedding_command)),
            model_id="command:vertical-dense-cache",
            timeout_seconds=int(args.embedding_timeout_seconds),
        )
    raise SystemExit("retrieval mode requires --embedding-command or --hash-dimension")


def _config_from_args(args: argparse.Namespace) -> dict[str, str]:
    config = dict(PRESETS.get(str(args.preset), {}))
    for key in (
        "mode",
        "dataset",
        "dataset_name",
        "queries_config",
        "queries_split",
        "corpus_config",
        "corpus_split",
        "qrels_config",
        "qrels_split",
        "top_ranked_config",
        "top_ranked_split",
    ):
        value = getattr(args, key, None)
        if value:
            config[key] = str(value)
    required = ("mode", "dataset", "dataset_name", "queries_config", "queries_split", "corpus_config", "corpus_split", "qrels_config", "qrels_split")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise SystemExit(f"missing required dataset settings: {', '.join(missing)}")
    if config["mode"] == "hf-reranking":
        missing_rerank = [key for key in ("top_ranked_config", "top_ranked_split") if not config.get(key)]
        if missing_rerank:
            raise SystemExit(f"missing required reranking settings: {', '.join(missing_rerank)}")
    return config


if __name__ == "__main__":
    main()
