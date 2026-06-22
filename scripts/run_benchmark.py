#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from activation_rag import (
    Chunker,
    ChunkerSettings,
    CommandPrefillTelemetryProvider,
    CommandEmbeddingProvider,
    HashEmbeddingProvider,
    MockTelemetryProvider,
    SentenceTransformerEmbeddingProvider,
)
from activation_rag.benchmarks import BenchmarkDataset, assert_benchmark_telemetry_allowed, evaluate_dataset, load_beir_dataset
from activation_rag.retrieval import ActivationMatchingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense, activation-KNN, and activation-rerank benchmark comparisons.")
    parser.add_argument("--fixture", action="store_true", help="Run built-in fixture benchmark")
    parser.add_argument("--dataset-json", help="Path to prepared BenchmarkDataset JSON")
    parser.add_argument("--beir-dir", help="Path to an extracted BEIR dataset directory containing corpus.jsonl, queries.jsonl, qrels/")
    parser.add_argument("--beir-name", default="beir")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="runs/benchmarks/latest.json", help="Summary output JSON")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--embedding-provider", choices=("hash", "sentence-transformers", "command"), default="hash")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-command",
        help="Command for embedding capture. May include {input_jsonl} and {output_jsonl} placeholders.",
    )
    parser.add_argument("--embedding-timeout-seconds", type=int, default=3600)
    parser.add_argument("--embedding-command-max-attempts", type=int, default=3)
    parser.add_argument("--embedding-command-retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument(
        "--telemetry-command",
        help=(
            "Command for real prefill telemetry capture. May include {input_jsonl}, "
            "{output_jsonl}, and {manifest_json} placeholders."
        ),
    )
    parser.add_argument("--telemetry-provider-id", default="sidecar-command-prefill")
    parser.add_argument("--telemetry-model-id", default="sidecar-prefill-model")
    parser.add_argument("--telemetry-site-id", default="selected_resid_pre")
    parser.add_argument("--layer-selection-policy", default="configured_selected_prefill_sites")
    parser.add_argument("--prompt-template-id", default="rag_raw_chunk_prefill_v1")
    parser.add_argument("--normalization-policy", default="configured_sidecar_prefill_normalization")
    parser.add_argument("--telemetry-timeout-seconds", type=int, default=86400)
    parser.add_argument("--telemetry-cache-dir", help="Persistent cache directory for command prefill telemetry rows")
    parser.add_argument("--allow-mock-telemetry", action="store_true", help="Allow mock telemetry for non-fixture harness smoke tests")
    parser.add_argument(
        "--activation-matching-strategy",
        choices=("cosine", "csls", "nicdm", "whiten_l2", "top_pc_removed", "per_site_late_fusion"),
        default="cosine",
        help="Activation matching strategy for activation-only and dense-first activation reranking.",
    )
    parser.add_argument("--activation-local-k", type=int, default=10, help="Local neighborhood size for CSLS/NICDM.")
    parser.add_argument("--activation-remove-components", type=int, default=3, help="Top PCs to remove for top_pc_removed.")
    parser.add_argument("--activation-whiten-dimensions", type=int, help="Whitened output dimensions for whiten_l2.")
    args = parser.parse_args()

    if args.fixture:
        dataset = fixture_dataset()
    elif args.dataset_json:
        dataset = load_dataset_json(Path(args.dataset_json))
    elif args.beir_dir:
        dataset = load_beir_dataset(Path(args.beir_dir), name=args.beir_name, split=args.split)
    else:
        raise SystemExit("Pass --fixture, --dataset-json, or --beir-dir")

    telemetry_provider = build_telemetry_provider(args)
    assert_benchmark_telemetry_allowed(
        telemetry_provider,
        fixture=bool(args.fixture),
        allow_mock_telemetry=bool(args.allow_mock_telemetry),
    )
    summary = evaluate_dataset(
        dataset,
        chunker=Chunker(ChunkerSettings(chunk_size=512, chunk_overlap=0)),
        embedder=build_embedding_provider(args),
        telemetry_provider=telemetry_provider,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        activation_matching_config=ActivationMatchingConfig(
            strategy=args.activation_matching_strategy,
            local_k=args.activation_local_k,
            remove_components=args.activation_remove_components,
            whiten_dimensions=args.activation_whiten_dimensions,
        ),
    )
    summary.save_json(args.out)
    print(json.dumps(summary.to_json_dict(), indent=2, sort_keys=True))


def build_telemetry_provider(args: argparse.Namespace):
    if args.telemetry_command:
        return CommandPrefillTelemetryProvider(
            command=shlex.split(args.telemetry_command),
            provider_id=args.telemetry_provider_id,
            model_id=args.telemetry_model_id,
            site_id=args.telemetry_site_id,
            layer_selection_policy=args.layer_selection_policy,
            prompt_template_id=args.prompt_template_id,
            normalization_policy=args.normalization_policy,
            timeout_seconds=args.telemetry_timeout_seconds,
            cache_dir=args.telemetry_cache_dir,
        )
    return MockTelemetryProvider()


def build_embedding_provider(args: argparse.Namespace):
    if args.embedding_provider == "command":
        if not args.embedding_command:
            raise ValueError("--embedding-command is required with --embedding-provider command")
        return CommandEmbeddingProvider(
            command=shlex.split(args.embedding_command),
            model_id=f"command:{args.embedding_model}",
            timeout_seconds=args.embedding_timeout_seconds,
            max_attempts=args.embedding_command_max_attempts,
            retry_backoff_seconds=args.embedding_command_retry_backoff_seconds,
        )
    if args.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbeddingProvider(
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
        )
    return HashEmbeddingProvider(dimension=256)


def fixture_dataset() -> BenchmarkDataset:
    return BenchmarkDataset(
        name="fixture",
        split="test",
        corpus={
            "doc-evidence": "Evidence should be checked carefully before finalizing the incident report.",
            "doc-health": "The repair plan is to verify the failing health check and inspect logs.",
            "doc-travel": "Employees should book travel through the approved portal.",
        },
        queries={"q1": "verification evidence health check"},
        qrels={"q1": {"doc-evidence": 1, "doc-health": 1}},
        metric_profile="beir",
    )


def load_dataset_json(path: Path) -> BenchmarkDataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkDataset(
        name=payload["name"],
        split=payload["split"],
        corpus=payload["corpus"],
        queries=payload["queries"],
        qrels=payload["qrels"],
        metric_profile=payload["metric_profile"],
    )


if __name__ == "__main__":
    main()
