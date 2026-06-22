#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by operator environment
    raise SystemExit(
        "NumPy is required for the ablation pass. Create a venv and install the research extra, "
        "for example: python3 -m venv .venv && .venv/bin/python -m pip install numpy"
    ) from exc

from activation_rag.benchmarks import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from activation_rag.chunking import Chunker, ChunkerSettings
from activation_rag.embedding import CommandEmbeddingProvider
from activation_rag.schema import DocumentRecord, stable_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Run activation retrieval ablations over a frozen telemetry cache.")
    parser.add_argument("--cache-dir", required=True, help="Telemetry cache directory containing ActivationRecord JSON rows.")
    parser.add_argument("--beir-zip", required=True, help="BEIR dataset zip, e.g. data/benchmarks/beir/scifact/scifact.zip.")
    parser.add_argument("--dataset-name", default="beir-scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="runs/activation-ablation/latest.json")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--dense-embedding-cache", help="Optional .npz cache for dense doc/query embeddings.")
    parser.add_argument("--embedding-command", help="Optional command with {input_jsonl}/{output_jsonl} placeholders.")
    parser.add_argument("--embedding-timeout-seconds", type=int, default=3600)
    parser.add_argument("--dense-integration-limit", type=int, default=12)
    args = parser.parse_args()

    started_at = time.time()
    dataset = load_beir_zip(Path(args.beir_zip), split=args.split)
    chunks, chunk_to_doc_id = build_chunks(
        dataset["corpus"],
        dataset_name=args.dataset_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    qids = [qid for qid in dataset["qrels"] if qid in dataset["queries"]]
    queries = {qid: dataset["queries"][qid] for qid in qids}

    rows_by_chunk = load_cache_rows(Path(args.cache_dir))
    doc_chunk_ids = [chunk.chunk_id for chunk in chunks if chunk.chunk_id in rows_by_chunk]
    query_chunk_ids = [stable_hash(f"query\n{queries[qid]}", 32) for qid in qids]
    present_query_pairs = [
        (qid, chunk_id)
        for qid, chunk_id in zip(qids, query_chunk_ids, strict=True)
        if chunk_id in rows_by_chunk
    ]
    qids = [qid for qid, _ in present_query_pairs]
    query_chunk_ids = [chunk_id for _, chunk_id in present_query_pairs]

    feature_names = sorted(
        {
            key
            for chunk_id in [*doc_chunk_ids, *query_chunk_ids]
            for key in rows_by_chunk[chunk_id].get("sae_feature_values", {})
        }
    )
    docs_all = rows_to_matrix(rows_by_chunk, doc_chunk_ids, feature_names)
    queries_all = rows_to_matrix(rows_by_chunk, query_chunk_ids, feature_names)

    report: dict[str, Any] = {
        "schema_version": "activation_rag.activation_ablation_report.v1",
        "started_at": started_at,
        "dataset": {
            "name": args.dataset_name,
            "split": args.split,
            "query_count": len(qids),
            "corpus_document_count": len(dataset["corpus"]),
            "chunk_count": len(chunks),
            "telemetry_doc_chunk_count": len(doc_chunk_ids),
            "telemetry_query_count": len(query_chunk_ids),
        },
        "inputs": {
            "cache_dir": str(Path(args.cache_dir)),
            "beir_zip": str(Path(args.beir_zip)),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        },
        "capture_hygiene": capture_hygiene(rows_by_chunk, doc_chunk_ids, query_chunk_ids),
        "not_runnable_from_current_cache": not_runnable_items(),
        "variants": [],
        "dense_integration": {},
        "observations": [],
    }

    feature_groups = build_feature_groups(feature_names, docs_all)
    transforms = [
        ("raw", {}),
        ("doc_center", {}),
        ("doc_zscore", {}),
        ("all_but_top", {"remove_components": 1}),
        ("all_but_top", {"remove_components": 3}),
        ("all_but_top", {"remove_components": 5}),
        ("whiten", {"dimensions": 64}),
        ("rank", {}),
    ]

    qrels = {qid: dataset["qrels"].get(qid, {}) for qid in qids}
    executed: list[dict[str, Any]] = []
    for group in feature_groups:
        mask = group["mask"]
        if int(mask.sum()) == 0:
            report["variants"].append({"name": group["name"], "status": "skipped", "reason": "empty feature group"})
            continue
        for transform_name, transform_params in transforms:
            variant_name = variant_label(group["name"], transform_name, transform_params)
            docs, query_matrix = transform_pair(docs_all[:, mask], queries_all[:, mask], transform_name, transform_params)
            if docs.shape[1] == 0:
                report["variants"].append({"name": variant_name, "status": "skipped", "reason": "transform produced zero dimensions"})
                continue
            scores = cosine_scores(query_matrix, docs)
            entry = evaluate_scores(
                name=variant_name,
                scores=scores,
                qids=qids,
                doc_chunk_ids=doc_chunk_ids,
                chunk_to_doc_id=chunk_to_doc_id,
                qrels=qrels,
                top_k=args.top_k,
                feature_count=int(mask.sum()),
                group=group["name"],
                transform=transform_name,
                transform_params=transform_params,
            )
            executed.append(entry)
            report["variants"].append(entry)

    for transform_name, transform_params in [("raw", {}), ("doc_zscore", {})]:
        docs, query_matrix = transform_pair(docs_all, queries_all, transform_name, transform_params)
        scores = csls_scores(query_matrix, docs, k=10)
        entry = evaluate_scores(
            name=variant_label("all_features", f"csls_{transform_name}", {}),
            scores=scores,
            qids=qids,
            doc_chunk_ids=doc_chunk_ids,
            chunk_to_doc_id=chunk_to_doc_id,
            qrels=qrels,
            top_k=args.top_k,
            feature_count=docs.shape[1],
            group="all_features",
            transform=f"csls_{transform_name}",
            transform_params={"k": 10},
        )
        executed.append(entry)
        report["variants"].append(entry)
        scores = nicdm_scores(query_matrix, docs, k=10)
        entry = evaluate_scores(
            name=variant_label("all_features", f"nicdm_{transform_name}", {}),
            scores=scores,
            qids=qids,
            doc_chunk_ids=doc_chunk_ids,
            chunk_to_doc_id=chunk_to_doc_id,
            qrels=qrels,
            top_k=args.top_k,
            feature_count=docs.shape[1],
            group="all_features",
            transform=f"nicdm_{transform_name}",
            transform_params={"k": 10},
        )
        executed.append(entry)
        report["variants"].append(entry)

    if args.dense_embedding_cache or args.embedding_command:
        dense_report = run_dense_integration(
            args=args,
            chunks=[chunk for chunk in chunks if chunk.chunk_id in doc_chunk_ids],
            qids=qids,
            query_texts=[queries[qid] for qid in qids],
            qrels=qrels,
            doc_chunk_ids=doc_chunk_ids,
            chunk_to_doc_id=chunk_to_doc_id,
            activation_variants=top_variants_for_dense(executed, args.dense_integration_limit),
            docs_all=docs_all,
            queries_all=queries_all,
            feature_names=feature_names,
            feature_groups=feature_groups,
        )
        report["dense_integration"] = dense_report
    else:
        report["dense_integration"] = {
            "status": "not_run",
            "reason": "pass --dense-embedding-cache or --embedding-command to evaluate dense integration variants",
        }

    report["observations"] = summarize_observations(report)
    report["finished_at"] = time.time()
    report["duration_seconds"] = report["finished_at"] - started_at
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report_summary(report), indent=2, sort_keys=True))


def load_beir_zip(path: Path, *, split: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        corpus = {}
        with archive.open(f"{prefix}/corpus.jsonl") as handle:
            for raw_line in handle:
                row = json.loads(raw_line.decode("utf-8"))
                title = str(row.get("title") or "").strip()
                text = str(row.get("text") or "").strip()
                corpus[str(row["_id"])] = f"{title}\n\n{text}".strip() if title else text
        queries = {}
        with archive.open(f"{prefix}/queries.jsonl") as handle:
            for raw_line in handle:
                row = json.loads(raw_line.decode("utf-8"))
                queries[str(row["_id"])] = str(row["text"])
        qrels = {}
        with archive.open(f"{prefix}/qrels/{split}.tsv") as handle:
            header = handle.readline().decode("utf-8").strip().split("\t")
            if header != ["query-id", "corpus-id", "score"]:
                raise ValueError(f"Unexpected qrels header: {header}")
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                query_id, corpus_id, score = raw_line.decode("utf-8").rstrip("\n").split("\t")
                qrels.setdefault(query_id, {})[corpus_id] = int(score)
    return {"corpus": corpus, "queries": queries, "qrels": qrels}


def build_chunks(corpus: dict[str, str], *, dataset_name: str, chunk_size: int, chunk_overlap: int):
    documents = [
        DocumentRecord.from_text(
            source_uri=f"benchmark://{dataset_name}/{doc_id}",
            title=doc_id,
            text=text,
            metadata={"benchmark_doc_id": doc_id},
        )
        for doc_id, text in corpus.items()
    ]
    document_id_to_doc_id = {document.document_id: document.metadata["benchmark_doc_id"] for document in documents}
    chunks = Chunker(ChunkerSettings(chunk_size=chunk_size, chunk_overlap=chunk_overlap)).split(documents)
    chunk_to_doc_id = {chunk.chunk_id: document_id_to_doc_id[chunk.document_id] for chunk in chunks}
    return chunks, chunk_to_doc_id


def load_cache_rows(cache_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(cache_dir.glob("*.json")):
        if ".bak" in path.name:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if not row.get("telemetry_valid", True):
            continue
        rows[str(row["chunk_id"])] = row
    return rows


def rows_to_matrix(rows_by_chunk: dict[str, dict[str, Any]], chunk_ids: list[str], feature_names: list[str]):
    matrix = np.zeros((len(chunk_ids), len(feature_names)), dtype=np.float64)
    for row_index, chunk_id in enumerate(chunk_ids):
        values = rows_by_chunk[chunk_id].get("sae_feature_values", {})
        for col_index, name in enumerate(feature_names):
            matrix[row_index, col_index] = float(values.get(name, 0.0))
    return matrix


def capture_hygiene(rows_by_chunk: dict[str, dict[str, Any]], doc_chunk_ids: list[str], query_chunk_ids: list[str]) -> dict[str, Any]:
    rows = [rows_by_chunk[chunk_id] for chunk_id in [*doc_chunk_ids, *query_chunk_ids] if chunk_id in rows_by_chunk]
    prompt_ids = Counter(str(row.get("prompt_template_id")) for row in rows)
    prompt_hashes = Counter(str(row.get("prompt_template_hash")) for row in rows)
    normalization = Counter(str(row.get("normalization_policy")) for row in rows)
    generation_disabled = Counter(str(row.get("generation_disabled")) for row in rows)
    section_labels = Counter(str(row.get("prompt_section_label")) for row in rows if row.get("prompt_section_label") is not None)
    provenance_section_labels = Counter(
        str((row.get("provenance") or {}).get("prompt_section_label"))
        for row in rows
        if (row.get("provenance") or {}).get("prompt_section_label") is not None
    )
    return {
        "status": "partial_cache_audit",
        "doc_rows": len(doc_chunk_ids),
        "query_rows": len(query_chunk_ids),
        "prompt_template_ids": dict(prompt_ids),
        "prompt_template_hashes": dict(prompt_hashes),
        "normalization_policies": dict(normalization),
        "generation_disabled_values": dict(generation_disabled),
        "section_labels_in_cache": dict(section_labels),
        "section_labels_in_provenance": dict(provenance_section_labels),
        "section_label_verdict": (
            "missing_from_activation_cache"
            if not section_labels and not provenance_section_labels
            else "present"
        ),
    }


def not_runnable_items() -> list[dict[str, str]]:
    return [
        {
            "ablation": "A1.capture_hygiene.strict_section_extraction",
            "status": "requires_raw_manifest_audit_or_recapture",
            "reason": "cached ActivationRecord rows do not persist prompt section boundaries",
        },
        {
            "ablation": "A2.pooling.mean_over_content_tokens",
            "status": "requires_recapture",
            "reason": "current cache stores only summary features, not token-level content activations",
        },
        {
            "ablation": "A2.pooling.last_content_token",
            "status": "requires_recapture",
            "reason": "current cache stores prefill-last summaries but not strict section-aligned content-token positions",
        },
        {
            "ablation": "A2.pooling.final_n_content_tokens",
            "status": "requires_recapture",
            "reason": "current cache does not persist token-level activation matrices",
        },
        {
            "ablation": "A2.pooling.answer_span",
            "status": "requires_supervised_answer_spans_and_recapture",
            "reason": "SciFact qrels identify relevant docs, not exact answer-span token positions",
        },
        {
            "ablation": "A4.supervised_feature_selection",
            "status": "deferred",
            "reason": "must split train/dev before qrel-based feature selection to avoid tuning on held-out test",
        },
        {
            "ablation": "A5.activation_margin_gate",
            "status": "requires_dense_candidates",
            "reason": "run with dense embeddings to evaluate candidate-preserving activation gates",
        },
    ]


def build_feature_groups(feature_names: list[str], docs: Any) -> list[dict[str, Any]]:
    names = np.array(feature_names)
    groups = [
        {"name": "all_features", "mask": np.ones(len(feature_names), dtype=bool)},
        {"name": "prefill_last_only", "mask": np.array([":prefill_last:" in name for name in feature_names])},
        {"name": "post50_mean_only", "mask": np.array([":post50_mean:" in name for name in feature_names])},
        {"name": "chunk_bins_only", "mask": np.array([name.rsplit(":", 1)[-1].startswith("chunk_") for name in feature_names])},
        {"name": "scalar_moments_only", "mask": np.array([not name.rsplit(":", 1)[-1].startswith("chunk_") for name in feature_names])},
    ]
    sites = sorted({name.split(":")[1] for name in feature_names if name.startswith("act:") and len(name.split(":")) >= 4})
    for site in sites:
        groups.append({"name": f"site_{site}", "mask": np.array([f":{site}:" in name for name in feature_names])})
    site_groups = {
        "site_early_proxy": {"emv2_p15_attn_out", "emv2_p25_attn_out", "emv2_p35_attn_out"},
        "site_middle_proxy": {"emv2_p45_attn_out", "emv2_p55_attn_out", "emv2_p65_attn_out"},
        "site_late_proxy": {"emv2_p75_attn_out", "emv2_p85_attn_out", "emv2_p92_attn_out"},
    }
    for group_name, group_sites in site_groups.items():
        groups.append({"name": group_name, "mask": np.array([name.split(":")[1] in group_sites for name in feature_names])})
    std = docs.std(axis=0)
    groups.append({"name": "drop_low_variance_bottom_10pct", "mask": std >= np.percentile(std, 10)})
    rogue_count = min(5, len(feature_names))
    rogue_indices = set(np.argsort(std)[-rogue_count:])
    groups.append({"name": "drop_rogue_variance_top_5", "mask": np.array([index not in rogue_indices for index in range(len(feature_names))])})
    groups.append({"name": "rogue_variance_top_5_only", "mask": np.array([index in rogue_indices for index in range(len(feature_names))])})
    return groups


def transform_pair(docs: Any, queries: Any, name: str, params: dict[str, Any]):
    docs = docs.astype(np.float64, copy=True)
    queries = queries.astype(np.float64, copy=True)
    if name == "raw":
        return docs, queries
    mean = docs.mean(axis=0, keepdims=True)
    if name == "doc_center":
        return docs - mean, queries - mean
    if name == "doc_zscore":
        std = docs.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        return (docs - mean) / std, (queries - mean) / std
    if name == "all_but_top":
        centered_docs = docs - mean
        centered_queries = queries - mean
        remove = min(int(params["remove_components"]), max(0, docs.shape[1] - 1))
        if remove == 0:
            return centered_docs, centered_queries
        _, _, vt = np.linalg.svd(centered_docs, full_matrices=False)
        top = vt[:remove].T
        return centered_docs - centered_docs @ top @ top.T, centered_queries - centered_queries @ top @ top.T
    if name == "whiten":
        centered_docs = docs - mean
        centered_queries = queries - mean
        cov = np.cov(centered_docs, rowvar=False)
        cov = np.atleast_2d(cov)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        keep = min(int(params["dimensions"]), docs.shape[1], len(order))
        order = order[:keep]
        scale = np.sqrt(np.maximum(eigenvalues[order], 1e-8))
        components = eigenvectors[:, order]
        return centered_docs @ components / scale, centered_queries @ components / scale
    if name == "rank":
        return row_ranks(docs), row_ranks(queries)
    raise ValueError(f"unknown transform: {name}")


def row_ranks(matrix: Any):
    order = np.argsort(matrix, axis=1)
    ranks = np.empty_like(order, dtype=np.float64)
    row_indices = np.arange(matrix.shape[0])[:, None]
    ranks[row_indices, order] = np.arange(matrix.shape[1], dtype=np.float64)
    ranks -= ranks.mean(axis=1, keepdims=True)
    std = ranks.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return ranks / std


def cosine_scores(queries: Any, docs: Any):
    return normalize_rows(queries) @ normalize_rows(docs).T


def csls_scores(queries: Any, docs: Any, *, k: int):
    sim = cosine_scores(queries, docs)
    k_docs = min(k, sim.shape[1])
    k_queries = min(k, sim.shape[0])
    query_radius = np.partition(sim, -k_docs, axis=1)[:, -k_docs:].mean(axis=1)
    doc_radius = np.partition(sim, -k_queries, axis=0)[-k_queries:, :].mean(axis=0)
    return (2.0 * sim) - query_radius[:, None] - doc_radius[None, :]


def nicdm_scores(queries: Any, docs: Any, *, k: int):
    distances = 1.0 - cosine_scores(queries, docs)
    k_docs = min(k, distances.shape[1])
    k_queries = min(k, distances.shape[0])
    query_radius = np.partition(distances, k_docs - 1, axis=1)[:, :k_docs].mean(axis=1)
    doc_radius = np.partition(distances, k_queries - 1, axis=0)[:k_queries, :].mean(axis=0)
    denominator = np.sqrt(np.maximum(query_radius[:, None] * doc_radius[None, :], 1e-12))
    return -(distances / denominator)


def normalize_rows(matrix: Any):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def evaluate_scores(
    *,
    name: str,
    scores: Any,
    qids: list[str],
    doc_chunk_ids: list[str],
    chunk_to_doc_id: dict[str, str],
    qrels: dict[str, dict[str, int]],
    top_k: int,
    feature_count: int,
    group: str,
    transform: str,
    transform_params: dict[str, Any],
) -> dict[str, Any]:
    top_indices = topk_indices(scores, top_k)
    ranked_doc_ids = indices_to_doc_ids(top_indices, doc_chunk_ids, chunk_to_doc_id)
    metrics = aggregate_metrics(qids, ranked_doc_ids, qrels, top_k)
    return {
        "name": name,
        "status": "executed",
        "group": group,
        "transform": transform,
        "transform_params": transform_params,
        "feature_count": feature_count,
        "metrics": metrics,
        "hubness": hubness(scores, top_indices),
    }


def topk_indices(scores: Any, top_k: int):
    k = min(top_k, scores.shape[1])
    unordered = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    unordered_scores = np.take_along_axis(scores, unordered, axis=1)
    order = np.argsort(-unordered_scores, axis=1)
    return np.take_along_axis(unordered, order, axis=1)


def indices_to_doc_ids(indices: Any, doc_chunk_ids: list[str], chunk_to_doc_id: dict[str, str]) -> list[list[str]]:
    result = []
    for row in indices:
        result.append([chunk_to_doc_id[doc_chunk_ids[int(index)]] for index in row])
    return result


def aggregate_metrics(qids: list[str], ranked_doc_ids: list[list[str]], qrels: dict[str, dict[str, int]], top_k: int):
    mrr_values = []
    recall_values = []
    ndcg_values = []
    for qid, ranked in zip(qids, ranked_doc_ids, strict=True):
        qrel = qrels.get(qid, {})
        mrr_values.append(mean_reciprocal_rank(ranked, qrel, min(10, top_k)))
        recall_values.append(recall_at_k(ranked, qrel, top_k))
        ndcg_values.append(ndcg_at_k(ranked, qrel, min(10, top_k)))
    return {
        f"mrr@{min(10, top_k)}": float(sum(mrr_values) / len(mrr_values)),
        f"recall@{top_k}": float(sum(recall_values) / len(recall_values)),
        f"ndcg@{min(10, top_k)}": float(sum(ndcg_values) / len(ndcg_values)),
    }


def hubness(scores: Any, top_indices: Any) -> dict[str, Any]:
    top1 = top_indices[:, 0]
    counts = Counter(int(index) for index in top1)
    top1_scores = scores[np.arange(scores.shape[0]), top1]
    top10_scores = np.take_along_axis(scores, top_indices, axis=1)
    top_hubs = counts.most_common(10)
    return {
        "unique_top1": len(counts),
        "top_hub_count": int(top_hubs[0][1]) if top_hubs else 0,
        "top_hub_share": float(top_hubs[0][1] / scores.shape[0]) if top_hubs else 0.0,
        "top_hubs_by_doc_index": [{"doc_index": index, "count": count} for index, count in top_hubs],
        "top1_score_mean": float(top1_scores.mean()),
        "top1_score_std": float(top1_scores.std()),
        "top1_score_min": float(top1_scores.min()),
        "top1_score_max": float(top1_scores.max()),
        "topk_score_mean": float(top10_scores.mean()),
        "topk_score_std": float(top10_scores.std()),
    }


def run_dense_integration(
    *,
    args: argparse.Namespace,
    chunks: list[Any],
    qids: list[str],
    query_texts: list[str],
    qrels: dict[str, dict[str, int]],
    doc_chunk_ids: list[str],
    chunk_to_doc_id: dict[str, str],
    activation_variants: list[dict[str, Any]],
    docs_all: Any,
    queries_all: Any,
    feature_names: list[str],
    feature_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_docs, dense_queries = load_or_compute_dense_embeddings(args, chunks, query_texts)
    dense_scores = cosine_scores(dense_queries, dense_docs)
    dense_top = topk_indices(dense_scores, args.top_k)
    dense_ranked = indices_to_doc_ids(dense_top, doc_chunk_ids, chunk_to_doc_id)
    dense_metrics = aggregate_metrics(qids, dense_ranked, qrels, args.top_k)
    candidate_indices = topk_indices(dense_scores, args.candidate_k)
    variants = []
    group_by_name = {group["name"]: group for group in feature_groups}
    for variant in activation_variants:
        group = group_by_name[variant["group"]]
        docs, query_matrix = transform_pair(
            docs_all[:, group["mask"]],
            queries_all[:, group["mask"]],
            variant["transform"],
            variant["transform_params"],
        )
        activation_scores = cosine_scores(query_matrix, docs)
        candidate_activation = np.take_along_axis(activation_scores, candidate_indices, axis=1)
        reranked_candidate_order = np.argsort(-candidate_activation, axis=1)[:, : args.top_k]
        reranked_indices = np.take_along_axis(candidate_indices, reranked_candidate_order, axis=1)
        reranked_doc_ids = indices_to_doc_ids(reranked_indices, doc_chunk_ids, chunk_to_doc_id)
        entry = {
            "activation_variant": variant["name"],
            "activation_rerank_metrics": aggregate_metrics(qids, reranked_doc_ids, qrels, args.top_k),
            "blend_metrics": {},
        }
        candidate_dense = np.take_along_axis(dense_scores, candidate_indices, axis=1)
        dense_z = row_zscore(candidate_dense)
        activation_z = row_zscore(candidate_activation)
        for lam in (0.01, 0.03, 0.1, 0.3, 1.0):
            blended = dense_z + (lam * activation_z)
            blend_order = np.argsort(-blended, axis=1)[:, : args.top_k]
            blend_indices = np.take_along_axis(candidate_indices, blend_order, axis=1)
            blend_doc_ids = indices_to_doc_ids(blend_indices, doc_chunk_ids, chunk_to_doc_id)
            entry["blend_metrics"][str(lam)] = aggregate_metrics(qids, blend_doc_ids, qrels, args.top_k)
        variants.append(entry)
    return {
        "status": "executed",
        "dense_metrics": dense_metrics,
        "candidate_k": args.candidate_k,
        "evaluated_activation_variant_count": len(variants),
        "variants": variants,
    }


def row_zscore(matrix: Any):
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (matrix - mean) / std


def load_or_compute_dense_embeddings(args: argparse.Namespace, chunks: list[Any], query_texts: list[str]):
    cache_path = Path(args.dense_embedding_cache) if args.dense_embedding_cache else None
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    query_ids = [f"query-{index}" for index in range(len(query_texts))]
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if list(cached["chunk_ids"]) == chunk_ids and list(cached["query_ids"]) == query_ids:
            return cached["doc_vectors"], cached["query_vectors"]
    if not args.embedding_command:
        raise ValueError("dense embedding cache missing and --embedding-command was not supplied")
    provider = CommandEmbeddingProvider(
        command=shlex.split(args.embedding_command),
        model_id="command:dense-ablation",
        timeout_seconds=args.embedding_timeout_seconds,
    )
    vectors = provider.embed_texts([chunk.text for chunk in chunks] + query_texts)
    doc_vectors = np.array(vectors[: len(chunks)], dtype=np.float64)
    query_vectors = np.array(vectors[len(chunks) :], dtype=np.float64)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            chunk_ids=np.array(chunk_ids),
            query_ids=np.array(query_ids),
            doc_vectors=doc_vectors,
            query_vectors=query_vectors,
        )
    return doc_vectors, query_vectors


def top_variants_for_dense(variants: list[dict[str, Any]], limit: int):
    return sorted(
        [
            variant
            for variant in variants
            if not str(variant["transform"]).startswith(("csls_", "nicdm_"))
        ],
        key=lambda item: item["metrics"].get("ndcg@10", 0.0),
        reverse=True,
    )[:limit]


def variant_label(group: str, transform: str, params: dict[str, Any]) -> str:
    if not params:
        return f"{group}::{transform}"
    suffix = ",".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{group}::{transform}({suffix})"


def summarize_observations(report: dict[str, Any]) -> list[str]:
    executed = [variant for variant in report["variants"] if variant.get("status") == "executed"]
    if not executed:
        return ["No activation variants executed."]
    best = max(executed, key=lambda item: item["metrics"].get("ndcg@10", -1.0))
    observations = [
        f"Best activation-only nDCG@10 variant was {best['name']} at {best['metrics'].get('ndcg@10', 0.0):.6f}.",
        f"Best activation-only unique top-1 count was {best['hubness']['unique_top1']} of {report['dataset']['query_count']} queries.",
    ]
    dense = report.get("dense_integration") or {}
    if dense.get("status") == "executed":
        observations.append(f"Dense nDCG@10 in this ablation run was {dense['dense_metrics'].get('ndcg@10', 0.0):.6f}.")
        best_blend = None
        for variant in dense["variants"]:
            for lam, metrics in variant["blend_metrics"].items():
                candidate = (metrics.get("ndcg@10", 0.0), variant["activation_variant"], lam, metrics)
                if best_blend is None or candidate[0] > best_blend[0]:
                    best_blend = candidate
        if best_blend is not None:
            observations.append(
                f"Best dense blend nDCG@10 was {best_blend[0]:.6f} using {best_blend[1]} at lambda {best_blend[2]}."
            )
    if report["capture_hygiene"]["section_label_verdict"] == "missing_from_activation_cache":
        observations.append("Section labels are missing from cached ActivationRecord rows; raw manifest audit or recapture is mandatory.")
    return observations


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    executed = [variant for variant in report["variants"] if variant.get("status") == "executed"]
    best = max(executed, key=lambda item: item["metrics"].get("ndcg@10", -1.0)) if executed else None
    return {
        "out": "written",
        "dataset": report["dataset"],
        "variant_count": len(report["variants"]),
        "best_activation_variant": None if best is None else {
            "name": best["name"],
            "metrics": best["metrics"],
            "hubness": best["hubness"],
        },
        "dense_integration_status": report["dense_integration"].get("status"),
        "observations": report["observations"],
    }


if __name__ == "__main__":
    main()
