#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

STAMP="${1:-20260612-section-prefill}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
RAW_ROOT="/tmp/activation-rag-${STAMP}/activations"
CACHE="runs/telemetry-cache/scifact-section-prefill-${STAMP}"
HEARTBEAT="runs/benchmarks/scifact-section-prefill-${STAMP}.heartbeat.log"

mkdir -p "$(dirname "$HEARTBEAT")"

while true; do
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  raw_dirs="$(ssh vicuna-host "find '$RAW_ROOT' -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' '")"
  cache_rows="$(find "$CACHE" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"
  if tmux has-session -t activation-rag-section-prefill 2>/dev/null; then
    tmux_state="running"
  else
    tmux_state="missing"
  fi
  echo "${ts} raw_dirs=${raw_dirs} cache_rows=${cache_rows} tmux=${tmux_state}" | tee -a "$HEARTBEAT"
  sleep "$INTERVAL_SECONDS"
done
