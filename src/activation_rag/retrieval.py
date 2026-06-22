from __future__ import annotations

import math
from dataclasses import dataclass

from activation_rag.schema import ActivationRecord, EmbeddingRecord, RetrievalResult


@dataclass(frozen=True)
class ActivationMatchingConfig:
    strategy: str = "cosine"
    local_k: int = 10
    remove_components: int = 3
    whiten_dimensions: int | None = None
    site_weights: dict[str, float] | None = None


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_dense(
    query_vector: tuple[float, ...],
    embeddings: list[EmbeddingRecord],
    top_k: int,
) -> list[RetrievalResult]:
    scored = [
        (embedding.chunk_id, cosine(query_vector, embedding.vector))
        for embedding in embeddings
    ]
    return _to_results(scored, "dense", top_k)


def rank_activation(
    query_record: ActivationRecord,
    activation_records: list[ActivationRecord],
    top_k: int,
) -> list[RetrievalResult]:
    return rank_activation_with_strategy(query_record, activation_records, top_k)


def rank_activation_with_strategy(
    query_record: ActivationRecord,
    activation_records: list[ActivationRecord],
    top_k: int,
    config: ActivationMatchingConfig | None = None,
) -> list[RetrievalResult]:
    config = config or ActivationMatchingConfig()
    if not query_record.telemetry_valid:
        return []
    query_vector = query_record.activation_vector()
    records = [
        record
        for record in activation_records
        if record.telemetry_valid and len(record.activation_vector()) == len(query_vector)
    ]
    if config.strategy == "cosine":
        scored = []
        for record in records:
            score = cosine(query_vector, record.activation_vector())
            scored.append((record.chunk_id, score, {"activation-sim": score}))
        return _to_results_with_components(scored, "activation-sim", top_k)
    if config.strategy == "per_site_late_fusion":
        scored = _score_per_site_late_fusion(query_record, records, config)
        return _to_results_with_components(scored, "activation-per-site-late-fusion", top_k)
    scored = _score_matrix_strategy(query_vector, records, config)
    return _to_results_with_components(scored, _strategy_label(config.strategy), top_k)


def rerank_with_activation(
    query_record: ActivationRecord,
    activation_records: list[ActivationRecord],
    candidates: list[RetrievalResult],
    top_k: int,
    config: ActivationMatchingConfig | None = None,
) -> list[RetrievalResult]:
    config = config or ActivationMatchingConfig()
    if not query_record.telemetry_valid:
        return []
    records_by_chunk = {record.chunk_id: record for record in activation_records if record.telemetry_valid}
    candidate_records = [
        records_by_chunk[candidate.chunk_id]
        for candidate in candidates
        if candidate.chunk_id in records_by_chunk
    ]
    candidate_by_chunk = {candidate.chunk_id: candidate for candidate in candidates}
    activation_results = rank_activation_with_strategy(
        query_record,
        candidate_records,
        top_k=len(candidate_records),
        config=config,
    )
    scored = [
        (candidate_by_chunk[result.chunk_id], result.score, result.component_scores)
        for result in activation_results
        if result.chunk_id in candidate_by_chunk
    ]

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
    results: list[RetrievalResult] = []
    for index, (candidate, activation_score, activation_components) in enumerate(ranked):
        component_scores = dict(candidate.component_scores)
        component_scores.setdefault(candidate.strategy, candidate.score)
        component_scores["dense"] = component_scores.get("dense", candidate.score)
        component_scores.update(activation_components)
        component_scores.setdefault(_strategy_label(config.strategy), activation_score)
        results.append(
            RetrievalResult(
                chunk_id=candidate.chunk_id,
                strategy="dense+activation-rerank",
                score=activation_score,
                rank=index + 1,
                component_scores=component_scores,
            )
        )
    return results


def _to_results(scored: list[tuple[str, float]], strategy: str, top_k: int) -> list[RetrievalResult]:
    return _to_results_with_components(
        [(chunk_id, score, {strategy: score}) for chunk_id, score in scored],
        strategy,
        top_k,
    )


def _to_results_with_components(
    scored: list[tuple[str, float, dict[str, float]]],
    strategy: str,
    top_k: int,
) -> list[RetrievalResult]:
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
    return [
        RetrievalResult(
            chunk_id=chunk_id,
            strategy=strategy,
            score=score,
            rank=index + 1,
            component_scores=component_scores,
        )
        for index, (chunk_id, score, component_scores) in enumerate(ranked)
    ]


def _score_matrix_strategy(
    query_vector: tuple[float, ...],
    records: list[ActivationRecord],
    config: ActivationMatchingConfig,
) -> list[tuple[str, float, dict[str, float]]]:
    import numpy as np

    if not records:
        return []
    docs = np.array([record.activation_vector() for record in records], dtype=np.float64)
    query = np.array([query_vector], dtype=np.float64)
    strategy = config.strategy
    if strategy == "csls":
        scores = _csls_scores(query, docs, local_k=config.local_k)
    elif strategy == "nicdm":
        scores = _nicdm_scores(query, docs, local_k=config.local_k)
    elif strategy == "whiten_l2":
        docs_t, query_t = _whiten_pair(docs, query, dimensions=config.whiten_dimensions)
        scores = _cosine_scores(query_t, docs_t)
    elif strategy == "top_pc_removed":
        docs_t, query_t = _remove_top_components_pair(docs, query, remove_components=config.remove_components)
        scores = _cosine_scores(query_t, docs_t)
    else:
        raise ValueError(f"unknown activation matching strategy: {strategy}")
    label = _strategy_label(strategy)
    return [
        (record.chunk_id, float(scores[0, index]), {label: float(scores[0, index])})
        for index, record in enumerate(records)
    ]


def _cosine_scores(queries: object, docs: object) -> object:
    return _normalize_rows(queries) @ _normalize_rows(docs).T


def _csls_scores(queries: object, docs: object, *, local_k: int) -> object:
    import numpy as np

    sim = _cosine_scores(queries, docs)
    query_k = min(max(1, local_k), sim.shape[1])
    query_radius = np.partition(sim, -query_k, axis=1)[:, -query_k:].mean(axis=1)
    doc_doc = _cosine_scores(docs, docs)
    np.fill_diagonal(doc_doc, -np.inf)
    doc_k = min(max(1, local_k), max(1, docs.shape[0] - 1))
    doc_radius = np.partition(doc_doc, -doc_k, axis=1)[:, -doc_k:].mean(axis=1)
    return (2.0 * sim) - query_radius[:, None] - doc_radius[None, :]


def _nicdm_scores(queries: object, docs: object, *, local_k: int) -> object:
    import numpy as np

    sim = _cosine_scores(queries, docs)
    distances = 1.0 - sim
    query_k = min(max(1, local_k), distances.shape[1])
    query_radius = np.partition(distances, query_k - 1, axis=1)[:, :query_k].mean(axis=1)
    doc_distances = 1.0 - _cosine_scores(docs, docs)
    np.fill_diagonal(doc_distances, np.inf)
    doc_k = min(max(1, local_k), max(1, docs.shape[0] - 1))
    doc_radius = np.partition(doc_distances, doc_k - 1, axis=1)[:, :doc_k].mean(axis=1)
    denominator = np.sqrt(np.maximum(query_radius[:, None] * doc_radius[None, :], 1e-12))
    return -(distances / denominator)


def _whiten_pair(docs: object, queries: object, *, dimensions: int | None) -> tuple[object, object]:
    import numpy as np

    mean = docs.mean(axis=0, keepdims=True)
    centered_docs = docs - mean
    centered_queries = queries - mean
    covariance = np.atleast_2d(np.cov(centered_docs, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    keep = min(dimensions or docs.shape[1], docs.shape[1], len(order))
    order = order[:keep]
    components = eigenvectors[:, order]
    scale = np.sqrt(np.maximum(eigenvalues[order], 1e-8))
    return centered_docs @ components / scale, centered_queries @ components / scale


def _remove_top_components_pair(
    docs: object,
    queries: object,
    *,
    remove_components: int,
) -> tuple[object, object]:
    import numpy as np

    mean = docs.mean(axis=0, keepdims=True)
    centered_docs = docs - mean
    centered_queries = queries - mean
    remove = min(max(0, remove_components), max(0, docs.shape[1] - 1), docs.shape[0])
    if remove == 0:
        return centered_docs, centered_queries
    _, _, vt = np.linalg.svd(centered_docs, full_matrices=False)
    top = vt[:remove].T
    return centered_docs - centered_docs @ top @ top.T, centered_queries - centered_queries @ top @ top.T


def _normalize_rows(matrix: object) -> object:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return matrix / norms


def _score_per_site_late_fusion(
    query_record: ActivationRecord,
    records: list[ActivationRecord],
    config: ActivationMatchingConfig,
) -> list[tuple[str, float, dict[str, float]]]:
    site_weights = config.site_weights or {}
    query_values = query_record.sae_feature_values
    sites = sorted(
        {
            _site_from_feature_name(name)
            for name in query_values
            if _site_from_feature_name(name) is not None
        }
    )
    sites = [site for site in sites if site_weights.get(site, 1.0) != 0.0]
    scored: list[tuple[str, float, dict[str, float]]] = []
    for record in records:
        weighted_total = 0.0
        weight_sum = 0.0
        components: dict[str, float] = {}
        for site in sites:
            feature_names = sorted(
                name
                for name in set(query_values) | set(record.sae_feature_values)
                if _site_from_feature_name(name) == site
            )
            if not feature_names:
                continue
            query_vector = tuple(float(query_values.get(name, 0.0)) for name in feature_names)
            record_vector = tuple(float(record.sae_feature_values.get(name, 0.0)) for name in feature_names)
            score = cosine(query_vector, record_vector)
            weight = float(site_weights.get(site, 1.0))
            components[f"activation-site:{site}"] = score
            weighted_total += score * weight
            weight_sum += abs(weight)
        if weight_sum == 0.0:
            score = 0.0
        else:
            score = weighted_total / weight_sum
        components["activation-per-site-late-fusion"] = score
        scored.append((record.chunk_id, score, components))
    return scored


def _site_from_feature_name(name: str) -> str | None:
    parts = name.split(":")
    if len(parts) >= 4 and parts[0] == "act":
        return parts[1]
    return None


def _strategy_label(strategy: str) -> str:
    labels = {
        "cosine": "activation-sim",
        "csls": "activation-csls",
        "nicdm": "activation-nicdm",
        "whiten_l2": "activation-whiten-l2",
        "top_pc_removed": "activation-top-pc-removed",
        "per_site_late_fusion": "activation-per-site-late-fusion",
    }
    return labels.get(strategy, f"activation-{strategy.replace('_', '-')}")


def activation_strategy_label(strategy: str) -> str:
    return _strategy_label(strategy)
