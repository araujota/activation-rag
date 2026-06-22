#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${1:-20260612-section-prefill}"
OUT="runs/benchmarks/scifact-section-prefill-${STAMP}.json"
CACHE="runs/telemetry-cache/scifact-section-prefill-${STAMP}"
RAW_ROOT="/tmp/activation-rag-${STAMP}/activations"

mkdir -p "$(dirname "$OUT")" "$CACHE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) starting scifact section-prefill recapture"
echo "output=$OUT"
echo "cache=$CACHE"
echo "raw_root=$RAW_ROOT"

PYTHONPATH=src python3 scripts/run_benchmark.py \
  --beir-dir data/benchmarks/beir/scifact/scifact \
  --beir-name beir-scifact \
  --split test \
  --out "$OUT" \
  --top-k 10 \
  --candidate-k 100 \
  --embedding-provider command \
  --embedding-model BAAI/bge-base-en-v1.5 \
  --embedding-command "python3 scripts/embed_remote_sentence_transformers.py {input_jsonl} {output_jsonl} --host root@vicuna-host --model BAAI/bge-base-en-v1.5 --device cuda --batch-size 128" \
  --embedding-timeout-seconds 7200 \
  --telemetry-command "python3 scripts/capture_sidecar_prefill.py {input_jsonl} {output_jsonl} --host root@vicuna-host --base-url http://127.0.0.1:28080 --raw-root ${RAW_ROOT} --request-prefix scifact-section-prefill --progress-every 25 --timeout 180 --no-reuse-existing-raw" \
  --telemetry-provider-id sidecar-command-prefill-zero-section-filtered \
  --telemetry-model-id DeepSeek-R1-Distill-Llama-8B-Q6_K \
  --telemetry-site-id selected_resid_pre \
  --layer-selection-policy selected_runtime_summary_prompt_prefill_only \
  --prompt-template-id rag_raw_chunk_prefill_v1_strict_zero_section_v2 \
  --normalization-policy raw_summary_values_v2_prompt_prefill_filtered \
  --telemetry-timeout-seconds 43200 \
  --telemetry-cache-dir "$CACHE"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) completed scifact section-prefill recapture"
