#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="20260611-140007"
OUT="runs/benchmarks/scifact-fixed-zero-${STAMP}.json"
CACHE="runs/telemetry-cache/scifact-fixed-zero-${STAMP}"

mkdir -p "$(dirname "$OUT")" "$CACHE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting scifact fixed-zero resume"

PYTHONPATH=src python3 scripts/run_benchmark.py \
  --beir-dir data/benchmarks/beir/scifact/scifact \
  --beir-name beir-scifact \
  --split test \
  --out "$OUT" \
  --top-k 10 \
  --candidate-k 100 \
  --embedding-provider command \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --embedding-command "python3 scripts/embed_remote_sentence_transformers.py {input_jsonl} {output_jsonl} --host root@vicuna-host --model BAAI/bge-base-en-v1.5 --device cuda --batch-size 8" \
  --embedding-timeout-seconds 7200 \
  --telemetry-command "python3 scripts/capture_sidecar_prefill.py {input_jsonl} {output_jsonl} --host root@vicuna-host --base-url http://127.0.0.1:28080 --raw-root /tmp/activation-rag-fixed-ingest-20260611-140007/activations --request-prefix scifact-fixed-zero --progress-every 25 --timeout 180" \
  --telemetry-provider-id sidecar-command-prefill-zero \
  --telemetry-model-id DeepSeek-R1-Distill-Llama-8B-Q6_K \
  --telemetry-site-id selected_resid_pre \
  --layer-selection-policy selected_runtime_summary \
  --prompt-template-id rag_raw_chunk_prefill_v1_strict_zero \
  --normalization-policy raw_summary_values_v1 \
  --telemetry-timeout-seconds 43200 \
  --telemetry-cache-dir "$CACHE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) completed scifact fixed-zero resume"
