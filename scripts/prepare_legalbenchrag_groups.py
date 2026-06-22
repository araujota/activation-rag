#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from activation_rag.embedding import CommandEmbeddingProvider, EmbeddingProvider, HashEmbeddingProvider
from activation_rag.schema import stable_hash
from scripts.prepare_vertical_reranker_groups import (
    RetrievalComponents,
    prepare_retrieval_groups,
    split_groups,
    _write_jsonl,
)


SCHEMA_VERSION = "activation_rag.legalbenchrag_preparation.v1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LegalBench-RAG span evidence groups for activation-aware reranker training.")
    parser.add_argument("--root", required=True, help="LegalBench-RAG unpacked root containing corpus/ and benchmarks/.")
    parser.add_argument("--benchmarks", default="all", help="Comma-separated benchmark stems, or 'all'.")
    parser.add_argument("--dataset-name", default="legalbenchrag")
    parser.add_argument("--window-chars", type=int, default=1600)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--append-qrel-positives", action="store_true", help="Append missing gold snippet windows after dense candidates.")
    parser.add_argument("--embedding-command", help="Command with {input_jsonl}/{output_jsonl} placeholders for dense candidates.")
    parser.add_argument("--embedding-timeout-seconds", type=int, default=7200)
    parser.add_argument("--hash-dimension", type=int, help="Use deterministic hash embeddings instead of a command embedder.")
    parser.add_argument("--dense-cache-out", help="Optional .npz cache for dense vectors.")
    parser.add_argument("--reuse-dense-cache", help="Optional .npz cache to reuse dense vectors.")
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--dev-out", required=True)
    parser.add_argument("--test-out", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", default="activation-rag-legalbenchrag")
    args = parser.parse_args()

    if args.window_chars <= 0:
        raise SystemExit("--window-chars must be positive")
    if args.candidate_k <= 0:
        raise SystemExit("--candidate-k must be positive")

    components = load_legalbenchrag_components(
        root=Path(args.root),
        dataset_name=str(args.dataset_name),
        benchmarks=_parse_benchmarks(str(args.benchmarks)),
        window_chars=int(args.window_chars),
    )
    groups = prepare_retrieval_groups(
        components,
        candidate_k=int(args.candidate_k),
        embedder=_build_embedder(args),
        dense_cache_out=Path(args.dense_cache_out) if args.dense_cache_out else None,
        reuse_dense_cache=Path(args.reuse_dense_cache) if args.reuse_dense_cache else None,
        append_qrel_positives=bool(args.append_qrel_positives),
    )
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
                "schema_version": SCHEMA_VERSION,
                "dataset_name": components.dataset_name,
                "benchmarks": sorted(_benchmark_names_from_queries(components.queries)),
                "query_count": len(components.queries),
                "evidence_passage_count": len(components.corpus),
                "candidate_k": int(args.candidate_k),
                "positive_in_pool_count": sum(1 for group in groups if group["positive_in_candidate_pool"]),
                "train_count": len(train_rows),
                "dev_count": len(dev_rows),
                "test_count": len(test_rows),
                "train_out": args.train_out,
                "dev_out": args.dev_out,
                "test_out": args.test_out,
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_legalbenchrag_components(
    *,
    root: Path,
    dataset_name: str = "legalbenchrag",
    benchmarks: set[str] | None = None,
    window_chars: int = 1600,
) -> RetrievalComponents:
    benchmarks_dir = root / "benchmarks"
    corpus_dir = root / "corpus"
    if not benchmarks_dir.exists() or not corpus_dir.exists():
        raise FileNotFoundError(f"{root} must contain benchmarks/ and corpus/")
    benchmark_paths = sorted(benchmarks_dir.glob("*.json"))
    if benchmarks:
        benchmark_paths = [path for path in benchmark_paths if path.stem in benchmarks]
        missing = sorted(benchmarks - {path.stem for path in benchmark_paths})
        if missing:
            raise FileNotFoundError(f"missing LegalBench-RAG benchmark files: {', '.join(missing)}")

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    qrels: dict[str, dict[str, float]] = {}
    text_cache: dict[str, str] = {}
    for benchmark_path in benchmark_paths:
        benchmark_name = benchmark_path.stem
        cases = _benchmark_cases(json.loads(benchmark_path.read_text(encoding="utf-8")))
        for index, case in enumerate(cases):
            query = str(case.get("query") or "").strip()
            snippets = case.get("snippets") or []
            if not query or not isinstance(snippets, list) or not snippets:
                continue
            query_id = f"{benchmark_name}:{index:05d}:{stable_hash(query, 10)}"
            queries[query_id] = query
            qrels.setdefault(query_id, {})
            for snippet_index, snippet in enumerate(snippets):
                file_path = str(snippet.get("file_path") or "")
                span = snippet.get("span")
                if not file_path or not _valid_span(span):
                    continue
                document_text = _load_corpus_text(corpus_dir=corpus_dir, file_path=file_path, text_cache=text_cache)
                span_start, span_end = int(span[0]), int(span[1])
                if span_start < 0 or span_end > len(document_text) or span_start >= span_end:
                    raise ValueError(f"invalid span {span} for {file_path} with length {len(document_text)}")
                window_start, window_end = _span_centered_window(len(document_text), span_start, span_end, window_chars)
                passage_text = document_text[window_start:window_end].strip()
                if not passage_text:
                    continue
                passage_id = f"{benchmark_name}:{file_path}:{span_start}:{span_end}:{snippet_index}"
                corpus[passage_id] = passage_text
                qrels[query_id][passage_id] = 1.0
    return RetrievalComponents(dataset_name=dataset_name, queries=queries, corpus=corpus, qrels=qrels)


def _benchmark_cases(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows: list[dict[str, Any]] = []
        for value in payload.values():
            if isinstance(value, list):
                rows.extend(row for row in value if isinstance(row, dict))
        return rows
    raise ValueError("LegalBench-RAG benchmark payload must be a list or dict of lists")


def _valid_span(span: Any) -> bool:
    return isinstance(span, list) and len(span) == 2 and all(isinstance(value, int) for value in span)


def _load_corpus_text(*, corpus_dir: Path, file_path: str, text_cache: dict[str, str]) -> str:
    if file_path not in text_cache:
        path = corpus_dir / file_path
        if not path.exists():
            raise FileNotFoundError(f"missing LegalBench-RAG corpus file: {path}")
        text_cache[file_path] = path.read_text(encoding="utf-8")
    return text_cache[file_path]


def _span_centered_window(text_length: int, span_start: int, span_end: int, window_chars: int) -> tuple[int, int]:
    if span_end - span_start >= window_chars:
        return span_start, span_end
    extra = window_chars - (span_end - span_start)
    left = extra // 2
    right = extra - left
    start = max(0, span_start - left)
    end = min(text_length, span_end + right)
    if end - start < window_chars:
        if start == 0:
            end = min(text_length, window_chars)
        elif end == text_length:
            start = max(0, text_length - window_chars)
    return start, end


def _parse_benchmarks(value: str) -> set[str] | None:
    if value.strip().lower() == "all":
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    if not names:
        raise SystemExit("--benchmarks must be 'all' or a comma-separated list")
    return names


def _benchmark_names_from_queries(queries: dict[str, str]) -> set[str]:
    return {query_id.split(":", 1)[0] for query_id in queries}


def _build_embedder(args: argparse.Namespace) -> EmbeddingProvider:
    if args.hash_dimension:
        return HashEmbeddingProvider(dimension=int(args.hash_dimension))
    if args.embedding_command:
        return CommandEmbeddingProvider(
            shlex.split(str(args.embedding_command)),
            model_id="command:legalbenchrag-dense-cache",
            timeout_seconds=int(args.embedding_timeout_seconds),
        )
    raise SystemExit("LegalBench-RAG preparation requires --embedding-command or --hash-dimension")


if __name__ == "__main__":
    main()
